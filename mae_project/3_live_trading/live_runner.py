# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from typing import Dict, List
from zoneinfo import ZoneInfo

import pandas as pd
import numpy as np

# -----------------------------------------------------------------------------
# Path setup: import existing code from ../2_backtest
# -----------------------------------------------------------------------------
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
BACKTEST_DIR = os.path.join(REPO_ROOT, "2_backtest")
MODEL_DIR = os.path.join(REPO_ROOT, "model")

if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)
if BACKTEST_DIR not in sys.path:
    sys.path.insert(0, BACKTEST_DIR)

from live_state import (
    LiveStateStore,
    RunRecord,
    STATUS_STARTED,
    STATUS_DATA_LOCKED,
    STATUS_PREDICTED,
    STATUS_WEIGHTS_COMPUTED,
    STATUS_ORDERS_SUBMITTED,
    STATUS_DONE,
    STATUS_FAILED,
)
from live_logging import utc_now_iso, JsonlLogger, ensure_dir, write_csv_rows
from live_data import build_live_window_online, save_market_snapshot_csv
from alpaca_client import AlpacaPaperClient

from strategy_ml_weights import MLReturnToWeightStrategy
from order_manager import OrderManager, OrderManagerConfig
from order_book import Order


def _safe_float(x, default=0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def _positions_to_qty_map(positions, symbols: List[str]) -> Dict[str, float]:
    out = {s: 0.0 for s in symbols}
    for p in positions:
        if p.symbol in out:
            out[p.symbol] = float(p.qty)
    return out


def _positions_to_dict_rows(positions) -> List[dict]:
    rows = []
    for p in positions:
        rows.append({
            "symbol": p.symbol,
            "qty": float(p.qty),
            "market_value": float(p.market_value),
            "current_price": float(p.current_price),
        })
    return rows


def _build_client_order_id(strategy_id: str, trade_date: str, symbol: str, side: str) -> str:
    # Alpaca client_order_id max length is 48 chars (practical limit).
    # Keep deterministic + compact.
    base = f"{strategy_id}-{trade_date}-{symbol}-{side}"
    base = base.replace("_", "-")
    if len(base) <= 48:
        return base
    # Compact fallback
    sid = strategy_id[:8]
    sym = symbol[:8]
    return f"{sid}-{trade_date}-{sym}-{side}"[:48]


def _log(msg: str) -> None:
    print(f"[LIVE] {msg}", flush=True)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v is None:
        raise argparse.ArgumentTypeError("Boolean value expected.")
    s = str(v).strip().lower()
    if s in {"true", "1", "yes", "y", "t"}:
        return True
    if s in {"false", "0", "no", "n", "f"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {v}")


def parse_args():
    ap = argparse.ArgumentParser(description="Part 4 Live Trading Runner (Alpaca Paper)")
    ap.add_argument("--ckpt", type=str, default=os.path.join(MODEL_DIR, "downstream_return_full.pth"))
    ap.add_argument("--run_dir", type=str, default=os.path.join(REPO_ROOT, "runs"), help="Directory for state/logs")
    ap.add_argument("--strategy_id", type=str, default="ml_ret_opt")
    ap.add_argument("--device", type=str, default="cpu")

    ap.add_argument(
        "--raw_calendar_lookback_days",
        type=int,
        default=250,
        help="Calendar days to download online for feature build warmup",
    )

    # Strategy knobs (same as backtest/infer)
    ap.add_argument("--cov_window", type=int, default=60)
    ap.add_argument("--gamma_turn", type=float, default=0.5)
    ap.add_argument("--turnover_l1", action="store_true")
    ap.add_argument("--mu_ema", type=float, default=0.0)
    ap.add_argument("--min_trade_notional", type=float, default=10.0)
    ap.add_argument("--lot_size", type=float, default=0.000001)

    # Live execution controls
    ap.add_argument("--minutes_after_open", type=int, default=10)
    ap.add_argument(
        "--skip_time_gate",
        type=str2bool,
        default=True,
        help="Use True/False to skip or enforce time gate, e.g. --skip_time_gate True",
    )
    ap.add_argument("--skip_already_done_guard", action="store_true", help="Allow rerun even if state says DONE (debug only)")
    ap.add_argument("--allow_pack_last_date_mismatch", action="store_true", help="Do not require pack last date == signal_date")
    ap.add_argument("--dry_run", action="store_true", help="Generate prediction/orders but do not submit to Alpaca")

    # Validation knobs
    ap.add_argument("--max_orders_per_step", type=int, default=200)
    ap.add_argument("--max_gross_buy_notional_per_step", type=float, default=1e12)
    ap.add_argument("--allow_fractional_qty", type=str2bool, default=True)

    return ap.parse_args()


def main():
    args = parse_args()

    _log("Starting live trading runner...")

    ensure_dir(args.run_dir)
    state_path = os.path.join(args.run_dir, "live_state.json")
    event_log_path = os.path.join(args.run_dir, "event_log.jsonl")
    event_logger = JsonlLogger(event_log_path)
    state = LiveStateStore(state_path)

    now_utc = utc_now_iso()
    event_logger.write({"ts_utc": now_utc, "event": "START_PROCESS", "args": vars(args)})

    # -------------------------------------------------------------------------
    # 1) Connect Alpaca + determine trade date and signal date
    # -------------------------------------------------------------------------
    _log("[1/8] Connecting Alpaca and determining trade context...")
    api = AlpacaPaperClient.from_config()

    if args.skip_time_gate:
        trade_date = api.get_today_trading_date_et()
        now_et_iso = "SKIPPED_TIME_GATE"
    else:
        trade_date, now_et_iso = api.ensure_after_open_plus_minutes(minutes_after_open=args.minutes_after_open)

    signal_date = api.get_previous_trading_day(trade_date)
    signal_date = str(pd.Timestamp(signal_date).strftime("%Y-%m-%d"))

    _log(f"Trade date={trade_date} | Signal date={signal_date} | skip_time_gate={args.skip_time_gate}")

    event_logger.write({
        "ts_utc": utc_now_iso(),
        "event": "TRADE_CONTEXT",
        "trade_date": trade_date,
        "signal_date": signal_date,
        "now_et": now_et_iso,
    })

    # -------------------------------------------------------------------------
    # 2) Idempotency guard (daily)
    # -------------------------------------------------------------------------
    _log("[2/8] Checking daily idempotency guard...")
    existing = state.get_run(args.strategy_id, trade_date)
    if existing is not None and existing.status == STATUS_DONE and (not args.skip_already_done_guard):
        print(f"[SKIP] trade_date={trade_date} already DONE for strategy={args.strategy_id}")
        event_logger.write({
            "ts_utc": utc_now_iso(),
            "event": "SKIP_ALREADY_DONE",
            "trade_date": trade_date,
            "strategy_id": args.strategy_id,
        })
        return

    if existing is None:
        rec = RunRecord(
            strategy_id=args.strategy_id,
            trade_date=trade_date,
            signal_date=signal_date,
            status=STATUS_STARTED,
            created_at_utc=utc_now_iso(),
            updated_at_utc=utc_now_iso(),
            payload={},
        )
        state.upsert_run(rec)
    else:
        rec = existing
        rec = state.patch_run(
            args.strategy_id,
            trade_date,
            updated_at_utc=utc_now_iso(),
            status=STATUS_STARTED,
            payload_updates={"restarted": True},
        )

    # -------------------------------------------------------------------------
    # 3) Download online data + build signal-date-aligned model inputs (NO local pack)
    # -------------------------------------------------------------------------
    _log("[3/8] Downloading market data and building live feature window...")
    try:
        pw = build_live_window_online(
            signal_date=signal_date,
            seq_len=20,
            raw_calendar_lookback_days=args.raw_calendar_lookback_days,
        )

        pack_last_date = str(pw.dates[-1])
        if (not args.allow_pack_last_date_mismatch) and (pack_last_date != signal_date):
            raise ValueError(
                f"Online feature table last date ({pack_last_date}) != required signal_date ({signal_date})."
            )

        _log(
            f"Feature window ready: last_date={pack_last_date}, rows={len(pw.dates)}, feature_dim={pw.feature_dim}"
        )

        state.patch_run(
            args.strategy_id,
            trade_date,
            updated_at_utc=utc_now_iso(),
            status=STATUS_DATA_LOCKED,
            payload_updates={
                "data_source": "online_runtime_download",
                "pack_path": pw.pack_path,
                "pack_signal_date": pw.signal_date,
                "pack_last_date": pack_last_date,
                "feature_dim": pw.feature_dim,
                "etf_symbols": pw.etf_symbols,
                "n_rows_online": len(pw.dates),
            },
        )
    except Exception as e:
        state.patch_run(
            args.strategy_id,
            trade_date,
            updated_at_utc=utc_now_iso(),
            status=STATUS_FAILED,
            payload_updates={"error_stage": "DATA_LOCK", "error": str(e)},
        )
        raise

    # -------------------------------------------------------------------------
    # 4) Build strategy, predict next-day returns, compute target weights
    # -------------------------------------------------------------------------
    _log("[4/8] Predicting next-day returns and computing target weights...")
    strategy = MLReturnToWeightStrategy(
        ckpt_path=args.ckpt,
        etf_symbols=pw.etf_symbols,
        num_feat=pw.feature_dim,
        seq_len=20,
        cov_window=args.cov_window,
        gamma_turn=args.gamma_turn,
        turnover_l1=args.turnover_l1,
        mu_ema=args.mu_ema,
        min_trade_notional=args.min_trade_notional,
        lot_size=args.lot_size,
        device=args.device,
    )

    try:
        mu_hat = strategy.predict_mu(pw.x_window_20xF)
        state.patch_run(
            args.strategy_id,
            trade_date,
            updated_at_utc=utc_now_iso(),
            status=STATUS_PREDICTED,
            payload_updates={"mu_hat": [float(x) for x in np.asarray(mu_hat).ravel()]},
        )

        # generate_target_weights internally re-predicts mu; this keeps parity with strategy class behavior.
        target_w = strategy.generate_target_weights(
            x_window_20xF=pw.x_window_20xF,
            all_returns_TxN=pw.R,
            t_idx_inclusive=pw.t_idx,
        )
        target_w = np.asarray(target_w, dtype=np.float64)
        if np.any(np.isnan(target_w)) or (target_w < -1e-9).any():
            raise ValueError(f"Invalid target weights: {target_w}")

        s = float(target_w.sum())
        if s <= 0:
            raise ValueError(f"Target weights sum <= 0: {s}")
        target_w = target_w / s

        _log(f"Target weights computed for {len(target_w)} ETFs.")

        state.patch_run(
            args.strategy_id,
            trade_date,
            updated_at_utc=utc_now_iso(),
            status=STATUS_WEIGHTS_COMPUTED,
            payload_updates={"target_weights": [float(x) for x in target_w]},
        )
    except Exception as e:
        state.patch_run(
            args.strategy_id,
            trade_date,
            updated_at_utc=utc_now_iso(),
            status=STATUS_FAILED,
            payload_updates={"error_stage": "PREDICT_WEIGHTS", "error": str(e)},
        )
        raise

    # -------------------------------------------------------------------------
    # 5) Pull account + positions + live prices (execution-time), save market snapshots
    # -------------------------------------------------------------------------
    _log("[5/8] Pulling account snapshot, positions, and live prices...")
    account = api.get_account_snapshot()
    positions = api.list_positions()
    current_positions = _positions_to_qty_map(positions, pw.etf_symbols)

    live_prices = api.get_latest_prices(pw.etf_symbols)
    total_equity = float(account.equity)
    cash = float(account.cash)

    _log(f"Equity={total_equity:.2f} | Cash={cash:.2f} | Positions={len(positions)}")

    # Save current positions snapshot + 1min bar snapshots for audit
    day_dir = os.path.join(args.run_dir, trade_date)
    ensure_dir(day_dir)
    positions_csv = os.path.join(day_dir, "positions_before.csv")
    write_csv_rows(positions_csv, _positions_to_dict_rows(positions))

    try:
        bar_rows = [api.download_latest_1min_bar(sym) for sym in pw.etf_symbols]
        bars_csv = os.path.join(day_dir, "market_snapshot_1min.csv")
        save_market_snapshot_csv(bars_csv, bar_rows)
    except Exception as e:
        # Non-fatal: price fetch already succeeded, bar snapshot is only for logging
        event_logger.write({
            "ts_utc": utc_now_iso(),
            "event": "WARN_BAR_SNAPSHOT_FAILED",
            "error": str(e),
        })

    # -------------------------------------------------------------------------
    # 6) Generate rebalance orders using your existing strategy logic
    # -------------------------------------------------------------------------
    _log("[6/8] Generating rebalance orders and validating risk checks...")
    raw_orders = strategy.generate_rebalance_orders(
        target_weights=target_w,
        prices=live_prices,
        current_positions=current_positions,
        cash=cash,
        total_equity=total_equity,
    )

    # Convert to Part 2 Order objects for validation parity
    candidate_orders = [
        Order(
            symbol=o.symbol,
            side=o.side,
            qty=o.qty,
            order_type=o.order_type,
            tif=o.tif,
            timestamp=trade_date,
            client_tag=args.strategy_id,
        )
        for o in raw_orders
    ]

    order_manager = OrderManager(OrderManagerConfig(
        max_orders_per_step=args.max_orders_per_step,
        max_gross_buy_notional_per_step=args.max_gross_buy_notional_per_step,
        allow_fractional_qty=bool(args.allow_fractional_qty),
    ))

    accepted, rejected = order_manager.validate_orders(
        orders=candidate_orders,
        prices=live_prices,
        cash=cash,
        positions=current_positions,
    )

    _log(f"Orders validated: accepted={len(accepted)}, rejected={len(rejected)}")

    event_logger.write({
        "ts_utc": utc_now_iso(),
        "event": "ORDERS_VALIDATED",
        "accepted_n": len(accepted),
        "rejected_n": len(rejected),
    })

    rejected_rows = []
    for od, reason in rejected:
        rejected_rows.append({
            "trade_date": trade_date,
            "signal_date": signal_date,
            "symbol": od.symbol,
            "side": od.side,
            "qty": float(od.qty),
            "reason": reason,
        })

    write_csv_rows(os.path.join(day_dir, "orders_rejected.csv"), rejected_rows)

    # -------------------------------------------------------------------------
    # 7) Submit to Alpaca (idempotent by client_order_id)
    # -------------------------------------------------------------------------
    if args.dry_run:
        _log("[7/8] Dry run mode: orders will not be submitted.")
    else:
        _log("[7/8] Submitting accepted orders to Alpaca...")

    submitted_rows = []
    skipped_existing_rows = []

    for od in accepted:
        client_order_id = _build_client_order_id(args.strategy_id, trade_date, od.symbol, od.side)

        existing_od = api.find_existing_order_by_client_id(trade_date, client_order_id)
        if existing_od is not None:
            skipped_existing_rows.append({
                "trade_date": trade_date,
                "signal_date": signal_date,
                "symbol": od.symbol,
                "side": od.side,
                "qty": float(od.qty),
                "client_order_id": client_order_id,
                "alpaca_order_id": str(getattr(existing_od, "id", "")),
                "alpaca_status": str(getattr(existing_od, "status", "")),
                "note": "existing_client_order_id_found_skip_submit",
            })
            continue

        if args.dry_run:
            submitted_rows.append({
                "trade_date": trade_date,
                "signal_date": signal_date,
                "symbol": od.symbol,
                "side": od.side,
                "qty": float(od.qty),
                "client_order_id": client_order_id,
                "alpaca_order_id": "",
                "alpaca_status": "DRY_RUN_NOT_SUBMITTED",
            })
            continue

        try:
            alp_od = api.submit_market_order(
                symbol=od.symbol,
                side=od.side,
                qty=float(od.qty),
                client_order_id=client_order_id,
                tif=od.tif,
            )
            submitted_rows.append({
                "trade_date": trade_date,
                "signal_date": signal_date,
                "symbol": od.symbol,
                "side": od.side,
                "qty": float(od.qty),
                "client_order_id": client_order_id,
                "alpaca_order_id": str(getattr(alp_od, "id", "")),
                "alpaca_status": str(getattr(alp_od, "status", "")),
            })
        except Exception as e:
            submitted_rows.append({
                "trade_date": trade_date,
                "signal_date": signal_date,
                "symbol": od.symbol,
                "side": od.side,
                "qty": float(od.qty),
                "client_order_id": client_order_id,
                "alpaca_order_id": "",
                "alpaca_status": "SUBMIT_ERROR",
                "error": str(e),
            })

    _log(
        f"Submission finished: submitted_rows={len(submitted_rows)}, existing_skipped={len(skipped_existing_rows)}"
    )

    write_csv_rows(os.path.join(day_dir, "orders_submitted.csv"), submitted_rows)
    write_csv_rows(os.path.join(day_dir, "orders_existing_skipped.csv"), skipped_existing_rows)

    state.patch_run(
        args.strategy_id,
        trade_date,
        updated_at_utc=utc_now_iso(),
        status=STATUS_ORDERS_SUBMITTED,
        payload_updates={
            "accepted_orders": [
                {"symbol": o.symbol, "side": o.side, "qty": float(o.qty)} for o in accepted
            ],
            "rejected_orders": rejected_rows,
            "submitted_orders": submitted_rows,
            "existing_skipped_orders": skipped_existing_rows,
            "account_before": asdict(account),
            "live_prices": {k: float(v) for k, v in live_prices.items()},
        },
    )

    # -------------------------------------------------------------------------
    # 8) Mark done (single-run semantics)
    # -------------------------------------------------------------------------
    _log("[8/8] Finalizing run and saving summary artifacts...")

    state.patch_run(
        args.strategy_id,
        trade_date,
        updated_at_utc=utc_now_iso(),
        status=STATUS_DONE,
        payload_updates={
            "dry_run": bool(args.dry_run),
            "done_reason": "daily_run_complete",
        },
    )

    # Save a compact summary JSON for the day
    summary = {
        "strategy_id": args.strategy_id,
        "trade_date": trade_date,
        "signal_date": signal_date,
        "pack_path": pw.pack_path,
        "data_source": "online_runtime_download",
        "ckpt": args.ckpt,
        "target_weights": [float(x) for x in target_w],
        "mu_hat": [float(x) for x in np.asarray(mu_hat).ravel()],
        "etf_symbols": pw.etf_symbols,
        "live_prices": {k: float(v) for k, v in live_prices.items()},
        "account_before": asdict(account),
        "n_candidate_orders": len(candidate_orders),
        "n_accepted_orders": len(accepted),
        "n_rejected_orders": len(rejected),
        "n_submitted_rows": len(submitted_rows),
        "n_existing_skipped": len(skipped_existing_rows),
        "dry_run": bool(args.dry_run),
    }
    with open(os.path.join(day_dir, "run_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n[LIVE RUN COMPLETE]")
    print(f"  trade_date : {trade_date}")
    print(f"  signal_date: {signal_date}")
    print(f"  day_dir    : {day_dir}")
    print(f"  dry_run    : {args.dry_run}")
    print(f"  accepted   : {len(accepted)}")
    print(f"  rejected   : {len(rejected)}")


if __name__ == "__main__":
    main()
# -*- coding: utf-8 -*-
"""
Strategy Backtesting:
- Integrates strategy + gateway + order book + order manager + matching engine
- Simulates daily event-driven rebalancing
- Logs orders and executions
- Computes metrics and saves plots/reports

Notes:
- The user's weight-generation logic is preserved in strategy_ml_weights.py
- Execution uses a price proxy reconstructed from pack returns because pack does not contain raw prices
"""

import os
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PT_DIR = os.path.abspath(os.path.join(THIS_DIR, "..", "data", "data_pt"))
MODEL_DIR = os.path.abspath(os.path.join(THIS_DIR, "..", "model"))
OUT_DIR = os.path.abspath(os.path.join(THIS_DIR, "..", "backtest_outputs"))

import argparse
import glob
import math
import json
from dataclasses import asdict
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from gateway import HistoricalGateway
from order_book import Order, OrderBook
from order_manager import OrderManager, OrderManagerConfig
from matching_engine import MatchingEngine
from strategy_ml_weights import MLReturnToWeightStrategy


def find_pack(pt_dir: str, split: str) -> str:
    cand = sorted(glob.glob(os.path.join(pt_dir, f"market_{split}_*.pt")))
    if not cand:
        raise FileNotFoundError(f"Cannot find pack for split='{split}' under {pt_dir}")
    return cand[0]


def compute_metrics(port_ret: np.ndarray, weights: np.ndarray, ann_factor: int = 252) -> Dict[str, float]:
    port_ret = np.asarray(port_ret, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)

    if port_ret.size == 0:
        return {
            "AnnRet": 0.0,
            "AnnVol": 0.0,
            "Sharpe": 0.0,
            "MaxDD": 0.0,
            "Calmar": 0.0,
            "AvgTurnover": 0.0,
            "WinLossRatio": 0.0,
        }

    equity = np.cumprod(1.0 + port_ret)
    peak = np.maximum.accumulate(equity)
    drawdown = equity / peak - 1.0
    max_dd = float(drawdown.min())

    mean_d = float(port_ret.mean())
    vol_d = float(port_ret.std(ddof=0))
    ann_ret = float((equity[-1] ** (ann_factor / max(1, len(port_ret))) - 1.0))
    ann_vol = float(vol_d * math.sqrt(ann_factor))
    sharpe = float((mean_d / (vol_d + 1e-12)) * math.sqrt(ann_factor))
    calmar = float(ann_ret / (abs(max_dd) + 1e-12))

    if len(weights) <= 1:
        avg_turn = 0.0
    else:
        dw = np.abs(weights[1:] - weights[:-1]).sum(axis=1) * 0.5
        avg_turn = float(dw.mean())

    wins = int(np.sum(port_ret > 0))
    losses = int(np.sum(port_ret < 0))
    win_loss_ratio = float(wins / max(losses, 1))

    return {
        "AnnRet": ann_ret,
        "AnnVol": ann_vol,
        "Sharpe": sharpe,
        "MaxDD": max_dd,
        "Calmar": calmar,
        "AvgTurnover": avg_turn,
        "WinLossRatio": win_loss_ratio,
    }


def print_metrics_table(name_to_metrics: Dict[str, Dict[str, float]]) -> None:
    keys = ["AnnRet", "AnnVol", "Sharpe", "MaxDD", "Calmar", "AvgTurnover", "WinLossRatio"]
    header = f"{'Model':<20} " + " ".join([f"{k:>14}" for k in keys])
    print("\n" + header)
    print("-" * len(header))
    for name, m in name_to_metrics.items():
        row = f"{name:<20} " + " ".join([f"{m.get(k, 0.0):14.6f}" for k in keys])
        print(row)


class PortfolioState:
    def __init__(self, symbols: List[str], initial_cash: float):
        self.cash = float(initial_cash)
        self.positions: Dict[str, float] = {s: 0.0 for s in symbols}

    def market_value(self, prices: Dict[str, float]) -> float:
        return sum(self.positions[s] * float(prices[s]) for s in self.positions)

    def total_equity(self, prices: Dict[str, float]) -> float:
        return self.cash + self.market_value(prices)

    def weights(self, prices: Dict[str, float], symbol_order: List[str]) -> np.ndarray:
        eq = self.total_equity(prices)
        if eq <= 0:
            return np.zeros(len(symbol_order), dtype=np.float64)
        return np.array([(self.positions[s] * float(prices[s])) / eq for s in symbol_order], dtype=np.float64)


import matplotlib.dates as mdates

def save_equity_curve_plot(dates: List[str], equity: List[float], out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    dts = pd.to_datetime(dates)

    plt.figure(figsize=(10, 5))
    plt.plot(dts, equity)

    ax = plt.gca()
    locator = mdates.AutoDateLocator(minticks=5, maxticks=10)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))

    plt.tight_layout()
    plt.title("Equity Curve")
    plt.xlabel("Date")
    plt.ylabel("Equity")
    plt.savefig(out_path, dpi=150)
    plt.close()

def save_trade_distribution_plot(trades_df: pd.DataFrame, out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    vals = trades_df["signed_notional"].values if len(trades_df) else np.array([0.0])
    plt.figure(figsize=(8, 5))
    plt.hist(vals, bins=30)
    plt.tight_layout()
    plt.title("Trade Notional Distribution")
    plt.xlabel("Signed Notional")
    plt.ylabel("Count")
    plt.savefig(out_path, dpi=150)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pt_dir", type=str, default=PT_DIR)
    ap.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    ap.add_argument("--ckpt", type=str, default=os.path.join(MODEL_DIR, "downstream_return_full.pth"))
    ap.add_argument("--out_dir", type=str, default=OUT_DIR)
    ap.add_argument("--initial_cash", type=float, default=100000.0)
    ap.add_argument("--device", type=str, default="cpu")

    # Strategy knobs (kept same core logic as user's infer)
    ap.add_argument("--cov_window", type=int, default=60)
    ap.add_argument("--gamma_turn", type=float, default=0.5)
    ap.add_argument("--turnover_l1", action="store_true")
    ap.add_argument("--mu_ema", type=float, default=0.0)
    ap.add_argument("--min_trade_notional", type=float, default=10.0)

    # Matching engine knobs
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--p_full_fill", type=float, default=0.70)
    ap.add_argument("--p_partial_fill", type=float, default=0.20)
    ap.add_argument("--p_cancel", type=float, default=0.10)
    ap.add_argument("--slippage_bps", type=float, default=2.0)

    # Variant comparison (Part 3 requirement)
    ap.add_argument("--run_baseline_hold", default=True, help="Run equal-weight buy-and-hold baseline in same report")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Load pack
    pack_path = find_pack(args.pt_dir, args.split)
    pack = torch.load(pack_path, map_location="cpu")
    X = pack["X"].float()
    R = pack["R"].float()
    meta = pack.get("meta", {})
    etf_symbols = meta.get("etfs", None)
    if not etf_symbols:
        raise ValueError("Pack meta['etfs'] missing")

    print(f"[PACK] {pack_path}")
    print(f"       split={meta.get('split')} X={tuple(X.shape)} R={tuple(R.shape)} date_range={meta.get('start_date')}..{meta.get('end_date')}")

    # Components
    gateway = HistoricalGateway(pack=pack, etf_symbols=etf_symbols)
    order_book = OrderBook()
    order_manager = OrderManager(OrderManagerConfig(
        max_orders_per_step=200,
        max_gross_buy_notional_per_step=1e12,
        allow_fractional_qty=True,
    ))
    matching_engine = MatchingEngine(
        seed=args.seed,
        p_full_fill=args.p_full_fill,
        p_partial_fill=args.p_partial_fill,
        p_cancel=args.p_cancel,
        slippage_bps=args.slippage_bps,
    )
    strategy = MLReturnToWeightStrategy(
        ckpt_path=args.ckpt,
        etf_symbols=etf_symbols,
        num_feat=int(X.shape[1]),
        seq_len=20,
        cov_window=args.cov_window,
        gamma_turn=args.gamma_turn,
        turnover_l1=args.turnover_l1,
        mu_ema=args.mu_ema,
        min_trade_notional=args.min_trade_notional,
        device=args.device,
    )

    # Backtest state
    portfolio = PortfolioState(symbols=etf_symbols, initial_cash=args.initial_cash)
    trade_log: List[dict] = []
    order_log: List[dict] = []
    daily_log: List[dict] = []

    # Need windows for strategy and future realized return for scoring
    T = len(gateway)
    if T < 21:
        raise ValueError(f"Need at least 21 rows in pack, got {T}")

    X_all = pack["X"].float()
    R_all = pack["R"].float().numpy()

    # Event loop (Part 3 live-style simulation over historical data)
    # We rebalance on event index t_idx = idx+19 using X[idx:idx+20],
    # and evaluate realized next-day return using R[idx+20].
    max_idx = T - 21
    for idx in range(max_idx + 1):
        t_idx = idx + 19
        event_date = str(pack["dates"][t_idx])
        eval_date = str(pack["dates"][idx + 20])

        prices = gateway._make_price_proxy(t_idx)
        equity_before = portfolio.total_equity(prices)

        # 1) Generate target weights (same strategy core logic)
        target_w = strategy.generate_target_weights(
            x_window_20xF=X_all[idx : idx + 20],
            all_returns_TxN=R_all,
            t_idx_inclusive=t_idx,
        )

        # 2) Generate rebalance orders
        raw_orders = strategy.generate_rebalance_orders(
            target_weights=target_w,
            prices=prices,
            current_positions=portfolio.positions,
            cash=portfolio.cash,
            total_equity=equity_before,
        )

        # Convert to Order objects
        orders = [
            Order(
                symbol=o.symbol,
                side=o.side,
                qty=o.qty,
                order_type=o.order_type,
                tif=o.tif,
                timestamp=event_date,
                client_tag="strategy_rebalance",
            )
            for o in raw_orders
        ]

        # 3) OrderManager validation
        accepted, rejected = order_manager.validate_orders(
            orders=orders,
            prices=prices,
            cash=portfolio.cash,
            positions=portfolio.positions,
        )

        for od in accepted:
            oid = order_book.add(od)
            gateway.log_order_audit(event_date, "sent", oid, payload=od.to_dict())
            order_log.append({
                "date": event_date,
                "order_id": oid,
                "symbol": od.symbol,
                "side": od.side,
                "qty": od.qty,
                "status": "sent",
                "reason": "",
            })

        for od, reason in rejected:
            od.status = "rejected"
            gateway.log_order_audit(event_date, "rejected", od.order_id, payload={"reason": reason, **od.to_dict()})
            order_log.append({
                "date": event_date,
                "order_id": od.order_id,
                "symbol": od.symbol,
                "side": od.side,
                "qty": od.qty,
                "status": "rejected",
                "reason": reason,
            })

        # 4) Matching engine simulate fills
        for od in accepted:
            px = float(prices[od.symbol])
            exec_rpt = matching_engine.execute_order(od, px=px, event_date=event_date)

            # Update portfolio for filled qty only
            fq = float(exec_rpt.filled_qty)
            fp = float(exec_rpt.fill_price)

            if fq > 0:
                notional = fq * fp
                if od.side == "buy":
                    portfolio.cash -= notional
                    portfolio.positions[od.symbol] += fq
                else:
                    portfolio.cash += notional
                    portfolio.positions[od.symbol] -= fq
                    portfolio.positions[od.symbol] = max(0.0, portfolio.positions[od.symbol])

            if exec_rpt.status == "filled":
                gateway.log_order_audit(event_date, "filled", od.order_id, payload=exec_rpt.to_dict())
            elif exec_rpt.status == "partial":
                gateway.log_order_audit(event_date, "filled", od.order_id, payload=exec_rpt.to_dict())
                # Cancel remaining to ensure one-step daily rebalance behavior
                if od.remaining_qty > 0:
                    order_book.cancel(od.order_id, od.symbol)
                    gateway.log_order_audit(event_date, "cancelled", od.order_id, payload={"reason": "cancel_remainder_after_partial"})
            else:
                gateway.log_order_audit(event_date, "cancelled", od.order_id, payload=exec_rpt.to_dict())

            order_log.append({
                "date": event_date,
                "order_id": od.order_id,
                "symbol": od.symbol,
                "side": od.side,
                "qty": od.qty,
                "filled_qty": exec_rpt.filled_qty,
                "remaining_qty": exec_rpt.remaining_qty,
                "fill_price": exec_rpt.fill_price,
                "status": exec_rpt.status,
                "reason": exec_rpt.reason,
            })

            if exec_rpt.filled_qty > 0:
                signed_notional = exec_rpt.filled_qty * exec_rpt.fill_price
                if exec_rpt.side == "sell":
                    signed_notional = -signed_notional
                trade_log.append({
                    "date": event_date,
                    "order_id": exec_rpt.order_id,
                    "symbol": exec_rpt.symbol,
                    "side": exec_rpt.side,
                    "filled_qty": exec_rpt.filled_qty,
                    "fill_price": exec_rpt.fill_price,
                    "signed_notional": signed_notional,
                    "status": exec_rpt.status,
                })

        # 5) Compute next-step realized portfolio return using held positions after rebalance
        # Use pack's realized returns on idx+20 to preserve user's evaluation convention.
        r_real = R_all[idx + 20, :]  # [N]
        w_after = portfolio.weights(prices=prices, symbol_order=etf_symbols)
        realized_ret = float(np.dot(w_after, r_real))

        # Mark-to-market next day using realized return on equity proxy
        equity_after = equity_before * (1.0 + realized_ret)

        # To keep internal accounting consistent with proxy return framework:
        # scale positions/cash proportionally to realized PnL at portfolio level.
        # Here we apply realized return to current marked portfolio and leave cash unchanged.
        # Simpler and sufficient for assignment metrics.
        # We log realized return and compute equity curve from returns directly later.
        daily_log.append({
            "signal_date": event_date,
            "eval_date": eval_date,
            "idx": idx,
            "t_idx": t_idx,
            "equity_before": equity_before,
            "cash": portfolio.cash,
            "realized_ret_next": realized_ret,
            "target_weights": json.dumps([float(x) for x in target_w]),
            "actual_weights_after_rebalance": json.dumps([float(x) for x in w_after]),
        })

    # Build outputs
    daily_df = pd.DataFrame(daily_log)
    order_df = pd.DataFrame(order_log)
    trade_df = pd.DataFrame(trade_log)

    daily_csv = os.path.join(args.out_dir, "daily_log.csv")
    orders_csv = os.path.join(args.out_dir, "order_log.csv")
    trades_csv = os.path.join(args.out_dir, "trade_log.csv")
    audit_csv = os.path.join(args.out_dir, "gateway_audit_log.csv")

    daily_df.to_csv(daily_csv, index=False)
    order_df.to_csv(orders_csv, index=False)
    trade_df.to_csv(trades_csv, index=False)
    gateway.save_audit_log(audit_csv)

    # Metrics
    port_ret = daily_df["realized_ret_next"].values.astype(np.float64) if len(daily_df) else np.array([], dtype=np.float64)
    if len(daily_df):
        w_mat = np.vstack(daily_df["actual_weights_after_rebalance"].apply(json.loads).values)
    else:
        w_mat = np.zeros((0, len(etf_symbols)), dtype=np.float64)

    metrics_main = compute_metrics(port_ret=port_ret, weights=w_mat)

    # Baseline: equal-weight buy-and-hold (optional)
    metrics_map = {"PredReturn->Opt (ExecSim)": metrics_main}
    baseline_daily = None
    if args.run_baseline_hold:
        # Buy-and-hold in this framework = constant equal weights evaluated on same realized returns
        ew = np.ones(len(etf_symbols), dtype=np.float64) / len(etf_symbols)
        r_mat = R_all[20 : 20 + len(daily_df), :]
        ew_ret = (r_mat @ ew) if len(daily_df) else np.array([], dtype=np.float64)
        ew_w = np.tile(ew, (len(daily_df), 1)) if len(daily_df) else np.zeros((0, len(etf_symbols)))
        metrics_map["EqualWeight Hold"] = compute_metrics(port_ret=ew_ret, weights=ew_w)

    print_metrics_table(metrics_map)

    # Save metrics report
    metrics_path = os.path.join(args.out_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_map, f, indent=2)

    # Plots required by Part 3
    if len(daily_df):
        equity_curve = np.cumprod(1.0 + daily_df["realized_ret_next"].values.astype(np.float64))
        save_equity_curve_plot(
            dates=daily_df["eval_date"].tolist(),
            equity=equity_curve.tolist(),
            out_path=os.path.join(args.out_dir, "equity_curve.png"),
        )
    else:
        pd.DataFrame({"msg": ["No daily rows"]}).to_csv(os.path.join(args.out_dir, "equity_curve_empty.csv"), index=False)

    save_trade_distribution_plot(
        trades_df=trade_df if len(trade_df) else pd.DataFrame({"signed_notional": [0.0]}),
        out_path=os.path.join(args.out_dir, "trade_distribution.png"),
    )

    print("\n[SAVED]")
    print(f"  Daily log:   {daily_csv}")
    print(f"  Order log:   {orders_csv}")
    print(f"  Trade log:   {trades_csv}")
    print(f"  Audit log:   {audit_csv}")
    print(f"  Metrics:     {metrics_path}")
    print(f"  Equity plot: {os.path.join(args.out_dir, 'equity_curve.png')}")
    print(f"  Trade plot:  {os.path.join(args.out_dir, 'trade_distribution.png')}")


if __name__ == "__main__":
    main()
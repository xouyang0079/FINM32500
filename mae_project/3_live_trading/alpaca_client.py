# -*- coding: utf-8 -*-
from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

import pandas as pd

try:
    import alpaca_trade_api as tradeapi
except Exception as e:  # pragma: no cover
    tradeapi = None
    _IMPORT_ERR = e
else:
    _IMPORT_ERR = None


ET = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")


@dataclass
class AccountSnapshot:
    equity: float
    cash: float
    buying_power: float


@dataclass
class PositionSnapshot:
    symbol: str
    qty: float
    market_value: float
    current_price: float


class AlpacaPaperClient:
    """
    Thin wrapper around alpaca_trade_api REST client.
    Uses paper endpoint only.
    """

    def __init__(self, api_key: str, api_secret: str, base_url: str = "https://paper-api.alpaca.markets"):
        if tradeapi is None:
            raise ImportError(
                "alpaca_trade_api is not installed. Run: pip install alpaca-trade-api"
            ) from _IMPORT_ERR
        self.api = tradeapi.REST(api_key, api_secret, base_url, api_version="v2")

    @classmethod
    def from_env(cls) -> "AlpacaPaperClient":
        key = os.getenv("ALPACA_API_KEY_ID", "").strip()
        secret = os.getenv("ALPACA_API_SECRET_KEY", "").strip()
        base = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets").strip()
        if not key or not secret:
            raise ValueError("Missing ALPACA_API_KEY_ID / ALPACA_API_SECRET_KEY environment variables")
        return cls(api_key=key, api_secret=secret, base_url=base)

    def get_clock(self):
        return self.api.get_clock()

    def get_account_snapshot(self) -> AccountSnapshot:
        a = self.api.get_account()
        return AccountSnapshot(
            equity=float(a.equity),
            cash=float(a.cash),
            buying_power=float(a.buying_power),
        )

    def list_positions(self) -> List[PositionSnapshot]:
        out: List[PositionSnapshot] = []
        for p in self.api.list_positions():
            # Alpaca returns strings for many fields
            current_price = float(getattr(p, "current_price", 0.0) or 0.0)
            out.append(
                PositionSnapshot(
                    symbol=str(p.symbol),
                    qty=float(p.qty),
                    market_value=float(p.market_value),
                    current_price=current_price,
                )
            )
        return out

    def get_latest_price(self, symbol: str) -> float:
        # Try multiple SDK methods for compatibility.
        # 1) get_latest_trade (v2)
        try:
            t = self.api.get_latest_trade(symbol)
            px = float(getattr(t, "price", None))
            if px > 0:
                return px
        except Exception:
            pass

        # 2) get_last_trade (older style)
        try:
            t = self.api.get_last_trade(symbol)
            px = float(getattr(t, "price", None))
            if px > 0:
                return px
        except Exception:
            pass

        # 3) get_bars fallback
        try:
            bars = self.api.get_bars(symbol, "1Min", limit=1).df
            if not bars.empty:
                # MultiIndex possible: (symbol, timestamp)
                if "close" in bars.columns:
                    return float(bars["close"].iloc[-1])
        except Exception:
            pass

        raise RuntimeError(f"Unable to fetch latest price for {symbol}")

    def get_latest_prices(self, symbols: List[str]) -> Dict[str, float]:
        return {s: self.get_latest_price(s) for s in symbols}

    def submit_market_order(
        self,
        symbol: str,
        side: str,
        qty: float,
        client_order_id: str,
        tif: str = "day",
    ):
        return self.api.submit_order(
            symbol=symbol,
            qty=qty,
            side=side,
            type="market",
            time_in_force=tif,
            client_order_id=client_order_id,
        )

    def list_orders_for_date(self, trade_date_et: str, status: str = "all", limit: int = 500):
        """
        Pull orders around the ET trade date; Alpaca timestamps are UTC.
        """
        d = date.fromisoformat(trade_date_et)
        start_et = datetime(d.year, d.month, d.day, 0, 0, 0, tzinfo=ET)
        end_et = start_et + timedelta(days=1)
        after_utc = start_et.astimezone(UTC).isoformat()
        until_utc = end_et.astimezone(UTC).isoformat()

        return self.api.list_orders(status=status, limit=limit, after=after_utc, until=until_utc, nested=True)

    def find_existing_order_by_client_id(self, trade_date_et: str, client_order_id: str):
        for od in self.list_orders_for_date(trade_date_et=trade_date_et, status="all", limit=500):
            if str(getattr(od, "client_order_id", "")) == client_order_id:
                return od
        return None

    def get_previous_trading_day(self, trade_date_et: str) -> str:
        """
        Use Alpaca calendar to get the previous trading day before trade_date_et.
        """
        d = date.fromisoformat(trade_date_et)
        start = (d - timedelta(days=10)).isoformat()
        end = d.isoformat()
        cal = self.api.get_calendar(start=start, end=end)
        days = [c.date if hasattr(c, "date") else str(c["date"]) for c in cal]
        days = [str(x) for x in days]
        days = sorted(set(days))
        prev = [x for x in days if x < trade_date_et]
        if not prev:
            raise RuntimeError(f"Cannot determine previous trading day for {trade_date_et}")
        return prev[-1]

    def get_today_trading_date_et(self) -> str:
        """
        Use Alpaca clock timestamp, converted to ET, as the current trade date candidate.
        """
        clk = self.get_clock()
        ts = getattr(clk, "timestamp", None)
        if ts is None:
            raise RuntimeError("Clock timestamp missing")
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=UTC)
        return ts.astimezone(ET).date().isoformat()

    def ensure_after_open_plus_minutes(self, minutes_after_open: int = 10) -> Tuple[str, str]:
        """
        Returns (trade_date_et, now_et_iso). Raises if market not open or time too early.
        """
        clk = self.get_clock()
        is_open = bool(getattr(clk, "is_open", False))
        if not is_open:
            raise RuntimeError("Market is not open now")

        ts = clk.timestamp
        next_open = getattr(clk, "next_open", None)
        # We need today's open. Alpaca clock also has next_open/next_close, but not always previous_open.
        # Safer: use calendar for today.
        trade_date_et = self.get_today_trading_date_et()
        cal = self.api.get_calendar(start=trade_date_et, end=trade_date_et)
        if not cal:
            raise RuntimeError(f"No calendar row for trade date {trade_date_et}")

        c0 = cal[0]
        open_dt = getattr(c0, "open", None)
        if open_dt is None:
            raise RuntimeError("Calendar open time unavailable")

        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=UTC)
        now_et = ts.astimezone(ET)

        open_et = open_dt.astimezone(ET)
        ready_et = open_et + timedelta(minutes=int(minutes_after_open))

        if now_et < ready_et:
            raise RuntimeError(
                f"Too early to execute. Now={now_et.isoformat()} ET, "
                f"required>={ready_et.isoformat()} ET"
            )

        return trade_date_et, now_et.isoformat()

    def download_latest_1min_bar(self, symbol: str) -> Dict[str, object]:
        """
        Optional market-data snapshot for Part 4 logging/saving.
        """
        bars = self.api.get_bars(symbol, "1Min", limit=1).df
        if bars.empty:
            raise RuntimeError(f"No 1Min bars for {symbol}")
        row = bars.iloc[-1]
        idx = bars.index[-1]
        return {
            "symbol": symbol,
            "bar_ts": str(idx),
            "open": float(row["open"]),
            "high": float(row["high"]),
            "low": float(row["low"]),
            "close": float(row["close"]),
            "volume": float(row["volume"]),
        }

    @classmethod
    def from_config(cls) -> "AlpacaPaperClient":
        from live_config import ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_BASE_URL
        key = str(ALPACA_API_KEY).strip()
        secret = str(ALPACA_SECRET_KEY).strip()
        base = str(ALPACA_BASE_URL).strip()

        if not key or not secret or "YOUR_ALPACA" in key or "YOUR_ALPACA" in secret:
            raise ValueError("Please set real Alpaca keys in live_config.py")
        return cls(api_key=key, api_secret=secret, base_url=base)

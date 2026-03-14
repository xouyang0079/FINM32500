# -*- coding: utf-8 -*-
"""
Gateway for Data Ingestion:
- Streams historical market data row by row (daily in this project)
- Provides event records to the backtest runner
- Maintains an order audit log
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Iterator, List, Optional

import json
import os
import pandas as pd
import torch


@dataclass
class MarketEvent:
    idx: int
    date: str
    features: torch.Tensor   # [F]
    returns: torch.Tensor    # [N] realized next-period label in pack convention
    prices: Dict[str, float] # execution prices for this event step (e.g., open or proxy)
    extra: Dict[str, float]


class HistoricalGateway:
    """
    Streams events from the pack:
      pack["X"] : [T, F]
      pack["R"] : [T, N]
      pack["dates"] : [T]
    """

    def __init__(
        self,
        pack: dict,
        etf_symbols: List[str],
        price_proxy: str = "close_proxy",
    ):
        self.pack = pack
        self.etf_symbols = etf_symbols
        self.price_proxy = price_proxy

        self.X = pack["X"].float()
        self.R = pack["R"].float()
        self.dates = pack["dates"]

        if len(self.X) != len(self.R) or len(self.X) != len(self.dates):
            raise ValueError("Pack X/R/dates length mismatch")

        self.audit_log: List[dict] = []

    def __len__(self) -> int:
        return len(self.dates)

    def _make_price_proxy(self, idx: int) -> Dict[str, float]:
        """
        Pack does not contain raw prices. For backtest framework integration we need executable prices.
        We use a normalized proxy price process per ETF from cumulative returns:
            P_t = 100 * prod(1 + r_0..t)
        This is acceptable for assignment framework simulation and keeps strategy logic unchanged.
        """
        prices: Dict[str, float] = {}
        for j, sym in enumerate(self.etf_symbols):
            r_hist = self.R[: idx + 1, j].cpu().numpy()
            p = 100.0
            for rv in r_hist:
                p *= (1.0 + float(rv))
            prices[sym] = max(p, 1e-6)
        return prices

    def stream(self) -> Iterator[MarketEvent]:
        for i in range(len(self.dates)):
            yield MarketEvent(
                idx=i,
                date=str(self.dates[i]),
                features=self.X[i],
                returns=self.R[i],
                prices=self._make_price_proxy(i),
                extra={},
            )

    def log_order_audit(
        self,
        event_date: str,
        action: str,   # sent / modified / cancelled / filled / rejected
        order_id: str,
        payload: Optional[dict] = None,
    ) -> None:
        rec = {
            "date": event_date,
            "action": action,
            "order_id": order_id,
            "payload": payload or {},
        }
        self.audit_log.append(rec)

    def save_audit_log(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        pd.DataFrame(self.audit_log).to_csv(path, index=False)
# -*- coding: utf-8 -*-
"""
Matching Engine Simulator:
- Simulates random fill / partial fill / cancel
- Returns execution details
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional
import random

from order_book import Order


@dataclass
class ExecutionReport:
    order_id: str
    symbol: str
    side: str
    status: str            # filled / partial / cancelled / rejected
    requested_qty: float
    filled_qty: float
    remaining_qty: float
    fill_price: float
    event_date: str
    reason: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


class MatchingEngine:
    """
    Randomized execution model for assignment simulation.
    """

    def __init__(
        self,
        seed: int = 42,
        p_full_fill: float = 0.70,
        p_partial_fill: float = 0.20,
        p_cancel: float = 0.10,
        partial_fill_min: float = 0.20,
        partial_fill_max: float = 0.90,
        slippage_bps: float = 2.0,
    ):
        s = p_full_fill + p_partial_fill + p_cancel
        if abs(s - 1.0) > 1e-9:
            raise ValueError("Fill probabilities must sum to 1.0")
        self.rng = random.Random(seed)
        self.p_full_fill = p_full_fill
        self.p_partial_fill = p_partial_fill
        self.p_cancel = p_cancel
        self.partial_fill_min = partial_fill_min
        self.partial_fill_max = partial_fill_max
        self.slippage_bps = slippage_bps

    def _slipped_price(self, mid_price: float, side: str) -> float:
        slip = self.slippage_bps / 10000.0
        if side == "buy":
            return mid_price * (1.0 + slip)
        return mid_price * (1.0 - slip)

    def execute_order(self, order: Order, px: float, event_date: str) -> ExecutionReport:
        u = self.rng.random()

        if u < self.p_full_fill:
            fill_ratio = 1.0
            status = "filled"
        elif u < self.p_full_fill + self.p_partial_fill:
            fill_ratio = self.rng.uniform(self.partial_fill_min, self.partial_fill_max)
            status = "partial"
        else:
            fill_ratio = 0.0
            status = "cancelled"

        fill_price = self._slipped_price(px, order.side)
        filled_qty = min(order.remaining_qty, order.remaining_qty * fill_ratio)
        remaining_qty = max(0.0, order.remaining_qty - filled_qty)

        if status == "filled":
            order.status = "filled"
            order.remaining_qty = 0.0
        elif status == "partial":
            order.status = "partial"
            order.remaining_qty = remaining_qty
        else:
            order.status = "cancelled"
            order.remaining_qty = 0.0

        return ExecutionReport(
            order_id=order.order_id,
            symbol=order.symbol,
            side=order.side,
            status=status,
            requested_qty=float(order.qty),
            filled_qty=float(filled_qty),
            remaining_qty=float(order.remaining_qty),
            fill_price=float(fill_price),
            event_date=event_date,
            reason="randomized_matching_sim",
        )
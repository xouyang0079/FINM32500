# -*- coding: utf-8 -*-
"""
Order Manager:
- Validates orders before they enter the book
- Capital sufficiency checks
- Risk limits:
    * max orders per timestamp
    * max gross buy notional per timestamp
    * max gross sell qty by current position
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

from order_book import Order


@dataclass
class OrderManagerConfig:
    max_orders_per_step: int = 50
    max_gross_buy_notional_per_step: float = 1e12
    max_symbol_weight: float = 1.0
    allow_fractional_qty: bool = True


class OrderManager:
    def __init__(self, config: OrderManagerConfig):
        self.cfg = config

    def validate_orders(
        self,
        orders: List[Order],
        prices: Dict[str, float],
        cash: float,
        positions: Dict[str, float],
    ) -> Tuple[List[Order], List[Tuple[Order, str]]]:
        """
        Returns:
            accepted_orders
            rejected_orders_with_reason
        """
        accepted: List[Order] = []
        rejected: List[Tuple[Order, str]] = []

        if len(orders) > self.cfg.max_orders_per_step:
            # Keep first N, reject rest
            keep_n = self.cfg.max_orders_per_step
        else:
            keep_n = len(orders)

        # Sequential validation (sell first expected by caller)
        est_cash = float(cash)
        tmp_positions = dict(positions)

        for i, od in enumerate(orders):
            if i >= keep_n:
                rejected.append((od, "too_many_orders_this_step"))
                continue

            if od.symbol not in prices:
                rejected.append((od, "missing_price"))
                continue

            px = float(prices[od.symbol])
            if px <= 0:
                rejected.append((od, "non_positive_price"))
                continue

            if od.qty <= 0:
                rejected.append((od, "non_positive_qty"))
                continue

            if (not self.cfg.allow_fractional_qty) and (abs(od.qty - round(od.qty)) > 1e-12):
                rejected.append((od, "fractional_qty_not_allowed"))
                continue

            if od.side == "sell":
                cur_qty = float(tmp_positions.get(od.symbol, 0.0))
                if od.qty > cur_qty + 1e-12:
                    rejected.append((od, "insufficient_position_qty"))
                    continue
                tmp_positions[od.symbol] = max(0.0, cur_qty - od.qty)
                est_cash += od.qty * px
                accepted.append(od)

            elif od.side == "buy":
                notional = od.qty * px
                if notional > self.cfg.max_gross_buy_notional_per_step:
                    rejected.append((od, "buy_notional_exceeds_step_limit"))
                    continue
                if notional > est_cash + 1e-12:
                    rejected.append((od, "insufficient_cash"))
                    continue
                est_cash -= notional
                tmp_positions[od.symbol] = float(tmp_positions.get(od.symbol, 0.0)) + od.qty
                accepted.append(od)

            else:
                rejected.append((od, "invalid_side"))

        return accepted, rejected
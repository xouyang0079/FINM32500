# Goal: Block trades that exceed position or order-size limits.

from typing import Dict
from order import Order, OrderState


class RiskEngine:
    def __init__(self, max_order_size: int = 1000, max_position: int = 2000):
        self.max_order_size = int(max_order_size)
        self.max_position = int(max_position)
        self.positions: Dict[str, int] = {}

    def check(self, order: Order):
        if order.qty <= 0:
            raise ValueError("Quantity must be positive")

        if order.qty > self.max_order_size:
            raise ValueError(f"Order size too large: {order.qty} > {self.max_order_size}")

        if order.side not in {"1", "2"}:
            raise ValueError(f"Invalid side: {order.side} (expected '1' buy or '2' sell)")

        signed_qty = order.qty if order.side == "1" else -order.qty
        cur = self.positions.get(order.symbol, 0)
        projected = cur + signed_qty

        if abs(projected) > self.max_position:
            raise ValueError(
                f"Position limit exceeded for {order.symbol}: {projected} (limit {self.max_position})"
            )

        return True

    def update_position(self, order: Order):
        # Requirement says update after a fill.
        if order.state != OrderState.FILLED:
            return

        signed_qty = order.qty if order.side == "1" else -order.qty
        self.positions[order.symbol] = self.positions.get(order.symbol, 0) + signed_qty
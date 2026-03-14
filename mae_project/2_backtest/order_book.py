# -*- coding: utf-8 -*-
"""
Order Book Implementation:
- Add / modify / cancel orders
- Price-time priority bookkeeping (simplified)
- Supports market/limit fields for assignment completeness
"""

from __future__ import annotations

import heapq
import itertools
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional


_id_counter = itertools.count(1)


@dataclass
class Order:
    symbol: str
    side: str               # "buy" / "sell"
    qty: float
    order_type: str = "market"
    tif: str = "day"
    limit_price: Optional[float] = None
    timestamp: Optional[str] = None
    client_tag: Optional[str] = None

    order_id: str = field(default_factory=lambda: f"ORD{next(_id_counter):08d}")
    remaining_qty: float = 0.0
    status: str = "new"

    def __post_init__(self):
        self.qty = float(self.qty)
        self.remaining_qty = float(self.qty)

    def to_dict(self) -> dict:
        return asdict(self)


class SymbolOrderBook:
    """
    Simplified order book:
    - stores active orders by id
    - maintains buy/sell heaps for price-time ordering
    - market orders are assigned top priority
    """

    def __init__(self, symbol: str):
        self.symbol = symbol
        self.orders: Dict[str, Order] = {}
        self.buy_heap: List[tuple] = []   # max-heap via negative price
        self.sell_heap: List[tuple] = []  # min-heap via positive price
        self.seq = 0

    def _priority(self, order: Order):
        self.seq += 1
        # Market orders first
        if order.order_type == "market":
            px_key = float("-inf") if order.side == "sell" else float("inf")
        else:
            if order.limit_price is None:
                raise ValueError("limit order requires limit_price")
            px_key = float(order.limit_price)

        if order.side == "buy":
            # higher price first; market before limit
            score = (-1e18, self.seq) if order.order_type == "market" else (-px_key, self.seq)
            return score
        # sell: lower price first; market before limit
        score = (-1e18, self.seq) if order.order_type == "market" else (px_key, self.seq)
        return score

    def add_order(self, order: Order) -> str:
        if order.symbol != self.symbol:
            raise ValueError(f"Order symbol mismatch: {order.symbol} vs book {self.symbol}")
        order.status = "open"
        self.orders[order.order_id] = order
        pri = self._priority(order)
        if order.side == "buy":
            heapq.heappush(self.buy_heap, (pri, order.order_id))
        else:
            heapq.heappush(self.sell_heap, (pri, order.order_id))
        return order.order_id

    def modify_order(self, order_id: str, new_qty: Optional[float] = None, new_limit_price: Optional[float] = None) -> None:
        if order_id not in self.orders:
            raise KeyError(f"Order not found: {order_id}")
        order = self.orders[order_id]
        if order.status not in {"open", "partial"}:
            raise ValueError(f"Cannot modify order in status={order.status}")

        if new_qty is not None:
            new_qty = float(new_qty)
            if new_qty <= 0:
                raise ValueError("new_qty must be > 0")
            filled_qty = order.qty - order.remaining_qty
            if new_qty < filled_qty:
                raise ValueError("new_qty cannot be less than already filled qty")
            order.qty = new_qty
            order.remaining_qty = new_qty - filled_qty

        if new_limit_price is not None:
            order.limit_price = float(new_limit_price)

        # Reinsert to heap to refresh priority; stale entries ignored later
        pri = self._priority(order)
        if order.side == "buy":
            heapq.heappush(self.buy_heap, (pri, order.order_id))
        else:
            heapq.heappush(self.sell_heap, (pri, order.order_id))
        order.status = "open"

    def cancel_order(self, order_id: str) -> None:
        if order_id not in self.orders:
            raise KeyError(f"Order not found: {order_id}")
        order = self.orders[order_id]
        if order.status in {"filled", "cancelled", "rejected"}:
            return
        order.status = "cancelled"
        order.remaining_qty = 0.0

    def get_order(self, order_id: str) -> Order:
        return self.orders[order_id]

    def open_orders(self) -> List[Order]:
        return [o for o in self.orders.values() if o.status in {"open", "partial"}]


class OrderBook:
    def __init__(self):
        self.books: Dict[str, SymbolOrderBook] = {}

    def _get_book(self, symbol: str) -> SymbolOrderBook:
        if symbol not in self.books:
            self.books[symbol] = SymbolOrderBook(symbol)
        return self.books[symbol]

    def add(self, order: Order) -> str:
        return self._get_book(order.symbol).add_order(order)

    def modify(self, order_id: str, symbol: str, new_qty: Optional[float] = None, new_limit_price: Optional[float] = None) -> None:
        self._get_book(symbol).modify_order(order_id, new_qty=new_qty, new_limit_price=new_limit_price)

    def cancel(self, order_id: str, symbol: str) -> None:
        self._get_book(symbol).cancel_order(order_id)

    def get(self, order_id: str, symbol: str) -> Order:
        return self._get_book(symbol).get_order(order_id)

    def all_open_orders(self) -> List[Order]:
        out: List[Order] = []
        for b in self.books.values():
            out.extend(b.open_orders())
        return out
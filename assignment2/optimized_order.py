import heapq
from order import Order
from order_book import OrderBookBase


class OptimizedOrderBook(OrderBookBase):
    def __init__(self):
        self.orders_by_id = {}

        self.price_levels = {}

        self._bid_heap = []  # store -price
        self._ask_heap = []  # store +price

    def _key(self, side, price):
        if side not in ("bid", "ask"):
            raise ValueError(f"Invalid side: {side}")
        return (side, price)

    def add_order(self, order_dict):
        order_id = int(order_dict["order_id"])
        if order_id in self.orders_by_id:
            raise ValueError(f"Duplicate order_id: {order_id}")

        o = Order(
            order_id=order_id,
            price=float(order_dict["price"]),
            quantity=int(order_dict["quantity"]),
            side=order_dict["side"],
        )
        if o.side not in ("bid", "ask"):
            raise ValueError(f"Invalid side: {o.side}")

        self.orders_by_id[order_id] = o

        k = self._key(o.side, o.price)
        level = self.price_levels.get(k)

        if level is None:
            level = {}
            self.price_levels[k] = level
            if o.side == "bid":
                heapq.heappush(self._bid_heap, -o.price)
            else:
                heapq.heappush(self._ask_heap, o.price)

        level[order_id] = o

    def amend_order(self, order_id, new_quantity):
        order_id = int(order_id)
        new_quantity = int(new_quantity)

        o = self.orders_by_id.get(order_id)
        if o is None:
            return False

        o.quantity = new_quantity
        if o.quantity <= 0:
            self.delete_order(order_id)
        return True

    def delete_order(self, order_id):
        order_id = int(order_id)
        o = self.orders_by_id.get(order_id)
        if o is None:
            return False

        k = self._key(o.side, o.price)
        level = self.price_levels.get(k)

        if level is not None and order_id in level:
            del level[order_id]
            if not level:
                del self.price_levels[k]

        del self.orders_by_id[order_id]
        return True

    def lookup_by_id(self, order_id):
        return self.orders_by_id.get(int(order_id))

    def get_orders_at_price(self, price, side=None):
        price = float(price)
        out = []

        if side is None or side == "bid":
            lvl = self.price_levels.get(("bid", price))
            if lvl:
                out.extend(lvl.values())

        if side is None or side == "ask":
            lvl = self.price_levels.get(("ask", price))
            if lvl:
                out.extend(lvl.values())

        return out

    def best_bid(self):
        while self._bid_heap:
            price = -self._bid_heap[0]
            lvl = self.price_levels.get(("bid", price))
            if lvl:
                return next(iter(lvl.values()))
            heapq.heappop(self._bid_heap)
        return None

    def best_ask(self):
        while self._ask_heap:
            price = self._ask_heap[0]
            lvl = self.price_levels.get(("ask", price))
            if lvl:
                return next(iter(lvl.values()))
            heapq.heappop(self._ask_heap)
        return None

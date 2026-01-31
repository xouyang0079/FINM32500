from order import Order
from order_book import OrderBookBase


class NaiveOrderBook(OrderBookBase):
    def __init__(self):
        self.bids = []
        self.asks = []

    def _sort_books(self):
        self.bids.sort(key=lambda o: o.price, reverse=True)
        self.asks.sort(key=lambda o: o.price)

    def add_order(self, order_dict):
        o = Order(
            order_id=order_dict["order_id"],
            price=order_dict["price"],
            quantity=order_dict["quantity"],
            side=order_dict["side"],
        )

        if o.side == "bid":
            self.bids.append(o)
        elif o.side == "ask":
            self.asks.append(o)
        else:
            raise ValueError(f"Invalid side: {o.side}")

        self._sort_books()

    def amend_order(self, order_id, new_quantity):
        order_id = int(order_id)
        new_quantity = int(new_quantity)

        for o in self.bids:
            if o.order_id == order_id:
                o.quantity = new_quantity
                self._sort_books()
                return True

        for o in self.asks:
            if o.order_id == order_id:
                o.quantity = new_quantity
                self._sort_books()
                return True

        return False

    def delete_order(self, order_id):
        order_id = int(order_id)

        for i, o in enumerate(self.bids):
            if o.order_id == order_id:
                del self.bids[i]
                self._sort_books()
                return True

        for i, o in enumerate(self.asks):
            if o.order_id == order_id:
                del self.asks[i]
                self._sort_books()
                return True

        return False

    def lookup_by_id(self, order_id):
        order_id = int(order_id)

        for o in self.bids:
            if o.order_id == order_id:
                return o

        for o in self.asks:
            if o.order_id == order_id:
                return o

        return None

    def get_orders_at_price(self, price, side=None):
        price = float(price)
        out = []

        if side is None or side == "bid":
            out.extend([o for o in self.bids if o.price == price])

        if side is None or side == "ask":
            out.extend([o for o in self.asks if o.price == price])

        return out

    def best_bid(self):
        if not self.bids:
            return None
        return self.bids[0]

    def best_ask(self):
        if not self.asks:
            return None
        return self.asks[0]

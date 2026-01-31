from abc import ABC, abstractmethod


class OrderBookBase(ABC):
    @abstractmethod
    def add_order(self, order_dict):
        raise NotImplementedError

    @abstractmethod
    def amend_order(self, order_id, new_quantity):
        raise NotImplementedError

    @abstractmethod
    def delete_order(self, order_id):
        raise NotImplementedError

    @abstractmethod
    def lookup_by_id(self, order_id):
        raise NotImplementedError

    @abstractmethod
    def get_orders_at_price(self, price, side=None):
        raise NotImplementedError

    @abstractmethod
    def best_bid(self):
        raise NotImplementedError

    @abstractmethod
    def best_ask(self):
        raise NotImplementedError

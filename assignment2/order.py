

class Order:
    def __init__(self, order_id, price, quantity, side):
        self.order_id = int(order_id)
        self.price = float(price)
        self.quantity = int(quantity)
        self.side = str(side)
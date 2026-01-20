from dataclasses import dataclass
import datetime

@dataclass(frozen=True)
class MarketDataPoint:
    timestamp: datetime.datetime
    symbol: str
    price: float

class Order: 
    # Instantiate Order objects.
    def __init__(self, symbol, quantity, price, status):
        self.symbol = symbol
        self.quantity = quantity
        self.price = price
        self.status = status

    # Validate Order objects
    def validate(self):
        if not self.symbol:
            raise OrderError("Order symbol must be non-empty.")

        if self.quantity <= 0:
            raise OrderError("Order quantity must be positive.")

        if self.price <= 0:
            raise OrderError("Order price must be positive.")

        
class OrderError(Exception): pass

class ExecutionError(Exception): pass
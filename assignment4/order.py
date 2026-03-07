# Goal: Represent an order’s journey from creation to completion.

from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Optional


class OrderState(Enum):
    NEW = auto()
    ACKED = auto()
    FILLED = auto()
    CANCELED = auto()
    REJECTED = auto()


@dataclass
class Order:
    symbol: str
    qty: int
    side: str  # FIX: "1"=Buy, "2"=Sell
    state: OrderState = field(default=OrderState.NEW)

    def transition(self, new_state: OrderState, logger: Optional[object] = None):
        allowed = {
            OrderState.NEW: {OrderState.ACKED, OrderState.REJECTED},
            OrderState.ACKED: {OrderState.FILLED, OrderState.CANCELED},
            OrderState.FILLED: set(),
            OrderState.CANCELED: set(),
            OrderState.REJECTED: set(),
        }

        ok = new_state in allowed.get(self.state, set())
        if not ok:
            msg = f"Transition not allowed: {self.state.name} -> {new_state.name} ({self.symbol})"
            print(msg)
            if logger is not None:
                logger.log("BadTransition", {"symbol": self.symbol, "from": self.state.name, "to": new_state.name})
            return

        self.state = new_state
        print(f"Order {self.symbol} is now {self.state.name}")
        if logger is not None:
            logger.log("StateChange", {"symbol": self.symbol, "state": self.state.name})
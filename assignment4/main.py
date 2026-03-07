# FIX → Parser → Order → RiskEngine → Logger

from fix_parser import FixParser
from order import Order, OrderState
from risk_engine import RiskEngine
from logger import Logger


def handle_message(raw: str, fix: FixParser, risk: RiskEngine, log: Logger):
    msg = fix.parse(raw)

    if msg.get("35") != "D":
        log.log("Ignored", {"reason": "not an order", "raw": raw})
        return

    order = Order(msg["55"], int(msg["38"]), msg["54"])
    log.log("OrderCreated", msg)

    try:
        risk.check(order)  # must be before ACK
        order.transition(OrderState.ACKED, logger=log)

        # fill order
        order.transition(OrderState.FILLED, logger=log)
        risk.update_position(order)  # update after fill

        log.log("OrderFilled", {"symbol": order.symbol, "qty": order.qty, "pos": risk.positions.get(order.symbol, 0)})

    except ValueError as e:
        order.transition(OrderState.REJECTED, logger=log)
        log.log("OrderRejected", {"reason": str(e), "symbol": order.symbol})


def main():
    fix = FixParser()
    risk = RiskEngine()
    log = Logger(path="events.json")

    messages = [
        "8=FIX.4.2|35=D|55=AAPL|54=1|38=500|40=2|44=190.0|10=128",
        "8=FIX.4.2|35=D|55=AAPL|54=1|38=5000|40=2|44=190.0|10=128",  # should reject (order size)
        "8=FIX.4.2|35=S|55=AAPL|10=999",  # quote example (ignored)
    ]

    for raw in messages:
        handle_message(raw, fix, risk, log)

    log.save()


if __name__ == "__main__":
    main()
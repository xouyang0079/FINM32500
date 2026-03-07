from order import Order, OrderState


class DummyLogger:
    def __init__(self):
        self.events = []

    def log(self, t, d):
        self.events.append((t, d))


def test_transition_ok():
    o = Order("AAPL", 10, "1")
    log = DummyLogger()
    o.transition(OrderState.ACKED, logger=log)
    assert o.state == OrderState.ACKED


def test_transition_not_allowed_does_not_change_state():
    o = Order("AAPL", 10, "1")
    log = DummyLogger()
    o.transition(OrderState.FILLED, logger=log)  # invalid from NEW
    assert o.state == OrderState.NEW
    assert any(t == "BadTransition" for t, _ in log.events)
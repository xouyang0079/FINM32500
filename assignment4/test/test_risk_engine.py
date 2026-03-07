import pytest
from order import Order, OrderState
from risk_engine import RiskEngine


def test_reject_large_order():
    r = RiskEngine(max_order_size=1000, max_position=2000)
    o = Order("AAPL", 5000, "1")
    with pytest.raises(ValueError):
        r.check(o)


def test_position_update_after_fill_only():
    r = RiskEngine(max_order_size=1000, max_position=2000)
    o = Order("AAPL", 100, "1")

    r.check(o)
    r.update_position(o)
    assert r.positions.get("AAPL", 0) == 0  # not filled yet

    o.state = OrderState.FILLED
    r.update_position(o)
    assert r.positions["AAPL"] == 100
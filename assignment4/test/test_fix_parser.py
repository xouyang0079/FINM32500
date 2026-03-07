import pytest
from fix_parser import FixParser


def test_parse_order_ok():
    msg = "8=FIX.4.2|35=D|55=AAPL|54=1|38=100|40=1|10=128"
    out = FixParser().parse(msg)
    assert out["55"] == "AAPL"
    assert out["54"] == "1"
    assert out["38"] == "100"


def test_missing_required_tag_raises():
    msg = "8=FIX.4.2|35=D|55=AAPL|38=100|40=1|10=128"  # missing 54
    with pytest.raises(ValueError):
        FixParser().parse(msg)


def test_limit_order_requires_price():
    msg = "8=FIX.4.2|35=D|55=AAPL|54=1|38=100|40=2|10=128"  # no 44
    with pytest.raises(ValueError):
        FixParser().parse(msg)
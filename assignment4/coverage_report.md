# Assignment 4: Mini Trading System

This report includes:
1) Unit test results  
2) Code coverage results  


---

## 1) Unit tests

### Command

```powershell
python -m pytest -q
```

### Results

```txt
9 passed in 0.10s
```

---

## 2) Coverage and unit tests results

### Commands

```powershell
python -m coverage run -m pytest
python -m coverage report -m
```

### Results

```txt
Name                       Stmts   Miss  Cover   Missing
--------------------------------------------------------
fix_parser.py                 40     13    68%   9, 16, 18, 23, 33, 45-52, 56-57
logger.py                     23      0   100%
order.py                      28      0   100%
risk_engine.py                25      3    88%   15, 21, 28
test\conftest.py               5      1    80%   6
test\test_fix_parser.py       16      0   100%
test\test_logger.py           15      0   100%
test\test_order.py            17      0   100%
test\test_risk_engine.py      17      0   100%
--------------------------------------------------------
TOTAL                        186     17    91%
```

---


## 3) Main program results

### Command

```powershell
python main.py
```

### Results

```txt
[LOG] OrderCreated → {'8': 'FIX.4.2', '35': 'D', '55': 'AAPL', '54': '1', '38': '500', '40': '2', '44': '190.0', '10': '128'}
Order AAPL is now ACKED
[LOG] StateChange → {'symbol': 'AAPL', 'state': 'ACKED'}
Order AAPL is now FILLED
[LOG] StateChange → {'symbol': 'AAPL', 'state': 'FILLED'}
[LOG] OrderFilled → {'symbol': 'AAPL', 'qty': 500, 'pos': 500}
[LOG] OrderCreated → {'8': 'FIX.4.2', '35': 'D', '55': 'AAPL', '54': '1', '38': '5000', '40': '2', '44': '190.0', '10': '128'}
Order AAPL is now REJECTED
[LOG] StateChange → {'symbol': 'AAPL', 'state': 'REJECTED'}
[LOG] OrderRejected → {'reason': 'Order size too large: 5000 > 1000', 'symbol': 'AAPL'}
[LOG] Ignored → {'reason': 'not an order', 'raw': '8=FIX.4.2|35=S|55=AAPL|10=999'}
```

This run generates:
- `events.json`

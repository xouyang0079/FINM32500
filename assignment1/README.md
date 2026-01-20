# Assignment 1: CSV-Based Algorithmic Trading Backtester
## Group 14

A simple modular backtester that loads tick data from a CSV, runs strategies to produce signals, simulates execution (with optional random failures), tracks cash/positions/equity, unit test, and generates a Markdown report (`performance.md`).

## Requirements
- Python 3.x
- Pandas
- Matplotlib

## Run
```bash
python main.py --csv market_data.csv --report performance.md
```

Common options:
```bash
python main.py --mac_short 5 --mac_long 20 --mom_lookback 5
python main.py --qty 1 --fail_rate 0.02 --seed 42 --cash 100000
```

## Output
- `performance.md`: summary metrics + ASCII equity curve + brief notes/errors.

## Code Layout
- `data_loader.py` — parse CSV → `MarketDataPoint` list
- `models.py` — `MarketDataPoint` (frozen), `Order` (mutable), `OrderError`, `ExecutionError`
- `strategies.py` — `Strategy` + `MAC`, `Momentum` (signals: `(action, symbol, qty, price)`)
- `engine.py` — run loop, signals→orders, execute, update portfolio, record equity, log errors
- `reporting.py` — metrics + report writer
- `main.py` — CLI + orchestration
- `performance.ipynb` — optional notebook for tests/analysis (if not writing unit tests)

## Unit Test
Demonstrate (`performance.md`: unit tests or notebook):
1) CSV parsing into frozen dataclass (`MarketDataPoint` immutable)  
2) `Order` is mutable (e.g., status changes)  
3) exceptions raised/handled (`OrderError`, `ExecutionError`)

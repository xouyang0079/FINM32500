# Performance Optimization of an Order Book

This repository implements and benchmarks two versions of a simple limit order book:

- **NaiveOrderBook**: stores bids/asks in Python lists and sorts after operations.
- **OptimizedOrderBook**: uses hash maps for O(1) lookup by order id, price-level indexing, and heaps for best bid/ask (with lazy cleanup).

It also includes a plotting script to compare runtimes across workloads on log scales.

## Repository Contents

- `main.py` — generates synthetic orders, benchmarks operations, and writes CSV results.
- `naive_order.py` — baseline list-based order book implementation.
- `optimized_order.py` — optimized order book implementation.
- `order.py` — `Order` object.
- `order_book.py` — abstract base class / interface.
- `plot_compare.py` — reads the result CSVs and creates comparison charts.
- `Performance Optimization of an Order Book.pdf` — assignment specification.

## Requirements

- Python 3.9+ 
- `matplotlib` 

## How to Run the Benchmark

`main.py` benchmarks operations for a chosen order book implementation and saves results as a CSV in the project directory.

### 1) Run naive benchmarks

```bash
python main.py --book naive
```

This produces:

- `benchmark_results_naive.csv`

### 2) Run optimized benchmarks

```bash
python main.py --book optimized
```

This produces:

- `benchmark_results_optimized.csv`


## How to Generate Comparison Charts

```bash
python plot_compare.py \
  --naive_csv benchmark_results_naive.csv \
  --opt_csv benchmark_results_optimized.csv \
  --out_dir out_compare
```

This creates PNG charts in `out_compare/`, one per operation.

Charts use log-log axes:

- X-axis: number of operations (log)
- Y-axis: total time in seconds (log)
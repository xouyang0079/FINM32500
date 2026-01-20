import os
from data_loader import load_data

import argparse

from strategies import MAC, Momentum
from engine import Engine
from reporting import compute_metrics, write_report


def main(args):
    base_path = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_path, args.csv)

    data = load_data(csv_path)

    strategies = [MAC(short_win=args.mac_short, long_win=args.mac_long), 
    Momentum(lookback=args.mom_lookback)]

    engine = Engine(
        strategies = strategies,
        default_qty = args.qty,
        fail_rate = args.fail_rate,
        seed=args.seed,
        initial_cash = args.cash
    )

    engine.run(data)

    # report
    metrics = compute_metrics(engine.equity_curve)
    report_path = os.path.join(base_path, args.report)
    write_report(report_path, metrics, engine.equity_curve, engine.errors)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="assignment 1")
    parser.add_argument("--csv", type=str, default="market_data.csv", help="Path to input CSV file with market data")
    parser.add_argument("--report", type=str, default="performance.md", help="Output path for performance report (Markdown)")

    # Strategy params
    parser.add_argument("--mac_short", type=int, default=5, help="Short window length for MAC strategy")
    parser.add_argument("--mac_long", type=int, default=20, help="Long window length for MAC strategy")
    parser.add_argument("--mom_lookback", type=int, default=5, help="Lookback window for Momentum strategy")

    # Engine params
    parser.add_argument("--qty", type=int, default=1, help="Default order quantity per trade")
    parser.add_argument("--fail_rate", type=float, default=0.02, help="Execution failure rate [0,1]")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--cash", type=float, default=100000.0, help="Initial cash balance")

    args = parser.parse_args()

    main(args)
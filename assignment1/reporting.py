import math


def series_returns(equity_curve):
    rets = []
    for i in range(1, len(equity_curve)):
        prev = float(equity_curve[i - 1][1])
        cur = float(equity_curve[i][1])
        if prev <= 0:
            rets.append(0.0)
        else:
            rets.append(cur / prev - 1.0)
    return rets


def sharpe_ratio(returns, periods_per_year=252):
    if not returns:
        return 0.0
    mean = sum(returns) / len(returns)
    var = sum((r - mean) ** 2 for r in returns) / (len(returns) - 1) if len(returns) > 1 else 0.0
    std = math.sqrt(var)
    if std == 0.0:
        return 0.0
    return (mean / std) * math.sqrt(periods_per_year)


def max_drawdown(equity_curve) -> float:
    if not equity_curve:
        return 0.0
    peak = float(equity_curve[0][1])
    mdd = 0.0
    for _, eq in equity_curve:
        eq = float(eq)
        if eq > peak:
            peak = eq
        dd = (eq / peak - 1.0) if peak > 0 else 0.0
        if dd < mdd:
            mdd = dd
    return mdd


def compute_metrics(equity_curve):
    """
    Compute:
      - total return
      - periodic returns
      - sharpe ratio
      - max drawdown
    """
    if not equity_curve:
        return {
            "total_return": 0.0,
            "returns": [],
            "sharpe": 0.0,
            "max_drawdown": 0.0,
        }

    start = float(equity_curve[0][1])
    end = float(equity_curve[-1][1])
    total_return = (end / start - 1.0) if start > 0 else 0.0

    rets = series_returns(equity_curve)
    sr = sharpe_ratio(rets, 252)
    mdd = max_drawdown(equity_curve)

    return {
        "total_return": total_return,
        "returns": rets,
        "sharpe": sr,
        "max_drawdown": mdd,
        "start_equity": start,
        "end_equity": end,
    }


def ascii_equity_curve(equity_curve, width = 60, height = 12):
    """
    Produce a simple ASCII equity curve.
    """
    vals = [float(x[1]) for x in equity_curve]
    vmin, vmax = min(vals), max(vals)
    if vmax == vmin:
        return "(flat equity)\n"

    # Sample to width points
    step = max(1, len(vals) // width)
    sampled = vals[::step]
    sampled = sampled[:width]

    smin, smax = min(sampled), max(sampled)
    scale = (smax - smin) if smax != smin else 1.0

    grid = [[" " for _ in range(len(sampled))] for _ in range(height)]
    for x, v in enumerate(sampled):
        y = int((v - smin) / scale * (height - 1))
        y = (height - 1) - y
        grid[y][x] = "*"

    lines = ["".join(row) for row in grid]
    return "\n".join(lines) + "\n"


def write_report(out_path, metrics, equity_curve, errors):
    """
    Generate performance.md with:
      - tables summarizing metrics
      - equity-curve plot (ASCII)
      - short narrative interpretation
    """
    total_return = metrics.get("total_return", 0.0)
    sharpe = metrics.get("sharpe", 0.0)
    mdd = metrics.get("max_drawdown", 0.0)
    start_eq = metrics.get("start_equity", 0.0)
    end_eq = metrics.get("end_equity", 0.0)

    eq_ascii = ascii_equity_curve(equity_curve)

    narrative = []
    narrative.append("This backtest ran over the provided tick data and executed signals from the configured strategies.")
    narrative.append("Total return summarizes overall growth in equity from start to end of the test period.")
    narrative.append("Sharpe ratio (annualized, risk-free assumed 0) provides a risk-adjusted performance view.")
    narrative.append("Maximum drawdown captures the worst peak-to-trough decline in equity.")
    if total_return > 0 and sharpe > 1:
        narrative.append("The strategy set produced positive returns with relatively favorable risk-adjusted performance.")
    elif total_return > 0:
        narrative.append("The strategy set produced positive returns, though risk-adjusted performance may be modest.")
    else:
        narrative.append("The strategy set did not produce positive returns over the test period.")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("# Performance Report\n\n")

        f.write("## Summary Metrics\n\n")
        f.write("| Metric | Value |\n")
        f.write("|---|---:|\n")
        f.write(f"| Start Equity | {start_eq:,.2f} |\n")
        f.write(f"| End Equity | {end_eq:,.2f} |\n")
        f.write(f"| Total Return | {total_return:.4f} |\n")
        f.write(f"| Sharpe Ratio | {sharpe:.4f} |\n")
        f.write(f"| Max Drawdown | {mdd:.4f} |\n\n")

        f.write("## Equity Curve (ASCII)\n\n")
        f.write("```\n")
        f.write(eq_ascii)
        f.write("```\n\n")

        f.write("## Narrative\n\n")
        for line in narrative:
            f.write(f"- {line}\n")
        f.write("\n")

        f.write("## Errors (logged, backtest continued)\n\n")
        if not errors:
            f.write("No errors were logged.\n")
        else:
            for e in errors[:200]:
                f.write(f"- {e}\n")
            if len(errors) > 200:
                f.write(f"- ... ({len(errors) - 200} more)\n")
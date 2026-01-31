import argparse
import csv
import os
import math

import matplotlib.pyplot as plt


def resolve_under_script_dir(path):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    if os.path.isabs(path):
        return path
    return os.path.join(base_dir, path)


def read_results(csv_path):
    data = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            op = row["operation"].strip()
            n = int(row["n"])
            total = float(row["total_sec"])
            if op not in data:
                data[op] = {}
            data[op][n] = total
    return data


def align_ns_union(naive, opt, op):
    return sorted(set(naive.get(op, {}).keys()) | set(opt.get(op, {}).keys()))


def plot_compare_allow_missing(out_path, title, xs, ys_naive, ys_opt):
    plt.figure()
    plt.plot(xs, ys_naive, marker="o", label="naive")
    plt.plot(xs, ys_opt, marker="o", label="optimized")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("number of operations (log)")
    plt.ylabel("total time (seconds, log)")
    plt.title(title)
    plt.legend()
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--naive_csv", type=str, default="benchmark_results_naive.csv",
                        help="Path to benchmark_results_naive.csv")
    parser.add_argument("--opt_csv", type=str, default="benchmark_results_optimized.csv",
                        help="Path to benchmark_results_optimized.csv")
    parser.add_argument("--out_dir", type=str, default="out_compare",
                        help="Output directory")
    args = parser.parse_args()

    naive_csv = resolve_under_script_dir(args.naive_csv)
    opt_csv = resolve_under_script_dir(args.opt_csv)

    out_dir_abs = resolve_under_script_dir(args.out_dir)
    os.makedirs(out_dir_abs, exist_ok=True)

    naive = read_results(naive_csv)
    opt = read_results(opt_csv)

    all_ops = sorted(set(naive.keys()) | set(opt.keys()))

    preferred = ["insert", "amend", "delete", "lookup_by_id", "get_orders_at_price", "best_bid", "best_ask"]
    ops_ordered = [op for op in preferred if op in all_ops] + [op for op in all_ops if op not in preferred]

    for op in ops_ordered:
        xs = align_ns_union(naive, opt, op)
        if not xs:
            print(f"Skip {op}: no n values found.")
            continue

        ys_naive = [naive.get(op, {}).get(n, math.nan) for n in xs]
        ys_opt = [opt.get(op, {}).get(n, math.nan) for n in xs]

        if all(math.isnan(y) for y in ys_naive) and all(math.isnan(y) for y in ys_opt):
            print(f"Skip {op}: both series missing.")
            continue

        out_path = os.path.join(out_dir_abs, f"{op}_naive_vs_optimized.png")
        plot_compare_allow_missing(
            out_path,
            f"{op.replace('_', ' ').title()} Performance: naive vs optimized",
            xs,
            ys_naive,
            ys_opt
        )


if __name__ == "__main__":
    main()

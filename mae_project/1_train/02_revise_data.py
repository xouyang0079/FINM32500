# -*- coding: utf-8 -*-
import os
import argparse
from typing import Dict, List, Optional, Set

import pandas as pd
import numpy as np


# -------------------------
# Helpers
# -------------------------
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def parse_years_arg(s: Optional[str]) -> Optional[List[int]]:
    """
    Accept:
      - None
      - "2020-2026"
      - "2020,2021,2022"
    """
    if s is None:
        return None
    s = s.strip()
    if not s:
        return None
    if "-" in s:
        a, b = s.split("-", 1)
        y0, y1 = int(a.strip()), int(b.strip())
        if y1 < y0:
            raise ValueError(f"Invalid years range: {s}")
        return list(range(y0, y1 + 1))
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def list_year_dirs(base_dir: str) -> List[int]:
    years = []
    if not os.path.exists(base_dir):
        return years
    for name in os.listdir(base_dir):
        p = os.path.join(base_dir, name)
        if os.path.isdir(p) and name.isdigit():
            years.append(int(name))
    return sorted(years)


def read_csv_safe(path: str) -> Optional[pd.DataFrame]:
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        df = df.sort_index()
        df = df[~df.index.duplicated(keep="first")]
        return df
    except Exception as e:
        print(f"[ERR] Failed to read {path}: {e}")
        return None


def save_csv_same_structure(df: pd.DataFrame, out_base_dir: str, year: int, category: str, name: str) -> None:
    out_dir = os.path.join(out_base_dir, str(year), category)
    ensure_dir(out_dir)
    out_path = os.path.join(out_dir, f"{name}.csv")
    df.to_csv(out_path, float_format="%.6f")


def get_file_path(base_dir: str, year: int, category: str, name: str) -> str:
    return os.path.join(base_dir, str(year), category, f"{name}.csv")


def has_any_nan_row(df: pd.DataFrame) -> pd.Series:
    """
    Return a boolean Series indexed by date: True if the row has any NaN/null.
    """
    if df is None or df.empty:
        return pd.Series(dtype=bool)
    return df.isna().any(axis=1)


def infer_index_names_for_year(base_dir: str, year: int) -> List[str]:
    """
    Fallback discovery when --index_names is not provided.
    """
    folder = os.path.join(base_dir, str(year), "index")
    if not os.path.isdir(folder):
        return []
    names = []
    for fn in os.listdir(folder):
        if fn.lower().endswith(".csv"):
            names.append(os.path.splitext(fn)[0])
    return sorted(names)


# -------------------------
# Main
# -------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Pre-clean yearly CSV data before pack building: "
                    "bfill index files, drop all dates where any ETF row has NaN."
    )
    parser.add_argument("--in_base_dir", type=str, default="../data/data_org", help="Input yearly CSV base directory.")
    parser.add_argument("--out_base_dir", type=str, default="../data/data_revise", help="Output cleaned CSV base directory.")
    parser.add_argument("--years", type=str, default=None, help='Optional, e.g. "2020-2026" or "2020,2021".')

    parser.add_argument("--etfs", type=str, default="GLD,SPY,TLT,UUP,XLE")
    parser.add_argument(
        "--index_names",
        type=str,
        default="cpi,dgs1,dgs2,dgs5,dgs10,dgs30,dxy,fedfunds,gdp,hyspread,unrate,vix",
        help="Comma-separated index/macro names. If empty, auto-discover per year from index folder.",
    )

    parser.add_argument(
        "--index_fill_mode",
        type=str,
        default="bfill_then_ffill",
        choices=["bfill", "bfill_then_ffill"],
        help="How to fill index/macro missing values. 'bfill' uses next value only. "
             "'bfill_then_ffill' also fills trailing NaNs with previous values.",
    )
    args = parser.parse_args()

    in_base_dir = args.in_base_dir
    out_base_dir = args.out_base_dir
    ensure_dir(out_base_dir)

    years = parse_years_arg(args.years)
    if years is None:
        years = list_year_dirs(in_base_dir)

    if not years:
        raise ValueError(f"No year folders found under: {in_base_dir}")

    etfs = [x.strip() for x in args.etfs.split(",") if x.strip()]
    configured_index_names = [x.strip() for x in args.index_names.split(",") if x.strip()]

    print(f"[INFO] Input base dir : {os.path.abspath(in_base_dir)}")
    print(f"[INFO] Output base dir: {os.path.abspath(out_base_dir)}")
    print(f"[INFO] Years          : {years}")
    print(f"[INFO] ETFs           : {etfs}")
    print(f"[INFO] Index fill mode: {args.index_fill_mode}")

    total_removed_rows = 0

    for year in years:
        print(f"\n=== Processing year {year} ===")

        # -------------------------
        # Load ETF files for this year
        # -------------------------
        etf_data: Dict[str, pd.DataFrame] = {}
        for t in etfs:
            p = get_file_path(in_base_dir, year, "etf", t)
            df = read_csv_safe(p)
            if df is None:
                print(f"[WARN] Missing ETF file: {p}")
                continue
            etf_data[t] = df

        if not etf_data:
            print(f"[WARN] No ETF files found for year={year}. Skipping year.")
            continue

        # -------------------------
        # Find bad dates from ETFs (any ETF row has any NaN)
        # -------------------------
        bad_dates: Set[pd.Timestamp] = set()
        for t, df in etf_data.items():
            bad_mask = has_any_nan_row(df)
            if len(bad_mask) > 0:
                bad_idx = set(df.index[bad_mask])
                if bad_idx:
                    print(f"[INFO] ETF {t}: found {len(bad_idx)} date(s) with NaN rows.")
                bad_dates |= bad_idx

        if bad_dates:
            print(f"[INFO] Union bad ETF dates to drop across all files: {len(bad_dates)}")
        else:
            print("[INFO] No ETF NaN rows found for this year.")

        # Drop bad dates from all ETF files
        for t, df in list(etf_data.items()):
            before = len(df)
            if bad_dates:
                df = df.loc[~df.index.isin(bad_dates)].copy()
            after = len(df)
            removed = before - after
            total_removed_rows += max(0, removed)
            if removed > 0:
                print(f"[INFO] ETF {t}: removed {removed} row(s) due to ETF NaN-date purge.")
            etf_data[t] = df

        # -------------------------
        # Load and clean index/macro files for this year
        # -------------------------
        if configured_index_names:
            index_names = configured_index_names
        else:
            index_names = infer_index_names_for_year(in_base_dir, year)

        index_data: Dict[str, pd.DataFrame] = {}
        for n in index_names:
            p = get_file_path(in_base_dir, year, "index", n)
            df = read_csv_safe(p)
            if df is None:
                # Keep silent-ish for missing index files (common in some years)
                continue

            # Fill missing values using next available value
            df = df.bfill()

            # Optional fallback to fill trailing NaNs at the end of the year
            if args.index_fill_mode == "bfill_then_ffill":
                df = df.ffill()

            # Drop the same ETF-bad dates across index files too
            if bad_dates:
                before = len(df)
                df = df.loc[~df.index.isin(bad_dates)].copy()
                removed = before - len(df)
                if removed > 0:
                    print(f"[INFO] INDEX {n}: removed {removed} row(s) due to ETF NaN-date purge.")

            index_data[n] = df

        # -------------------------
        # Save cleaned files in the same folder structure
        # -------------------------
        for t, df in etf_data.items():
            if df is None or df.empty:
                print(f"[WARN] ETF {t} became empty after cleaning for year={year}; still saving empty file skipped.")
                continue
            save_csv_same_structure(df, out_base_dir, year, "etf", t)

        for n, df in index_data.items():
            if df is None or df.empty:
                print(f"[WARN] INDEX {n} became empty after cleaning for year={year}; still saving empty file skipped.")
                continue
            save_csv_same_structure(df, out_base_dir, year, "index", n)

        print(f"[DONE] Year {year} saved to output folder.")

    print("\nALL DONE")
    print(f"[INFO] Cleaned CSV base directory: {os.path.abspath(out_base_dir)}")
    print(f"[INFO] Total ETF rows removed (sum across ETF files): {total_removed_rows}")


if __name__ == "__main__":
    main()
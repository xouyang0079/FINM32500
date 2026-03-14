# -*- coding: utf-8 -*-
import os
import time
import argparse
from typing import Dict, Optional

import yfinance as yf
import pandas as pd
from pandas_datareader import data as pdr


# -------------------------
# Helpers
# -------------------------
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def get_save_path(base_dir: str, year: int, category: str, name: str) -> str:
    """
    category: "etf" or "index"
    """
    folder = os.path.join(base_dir, str(year), category)
    ensure_dir(folder)
    return os.path.join(folder, f"{name}.csv")


def standardize_and_save_by_year(
    df: pd.DataFrame,
    base_dir: str,
    category: str,
    name: str,
    prefer_price_col: Optional[str] = None,
) -> None:
    """
    Save raw data as-is (no normalization). Split by df.index.year.

    prefer_price_col:
      - For index series like VIX downloaded from yfinance, you may want "Adj Close" (or "Close").
      - If provided and exists, keep full df but this is just a hook if you later want to select column.
    """
    if df is None or df.empty:
        print(f"[WARN] {name} is empty. Skip.")
        return

    # Flatten MultiIndex columns from yfinance
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.copy()
    df.index.name = "Date"
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    # (Optional) you can enforce a column subset here if you ever want.
    _ = prefer_price_col  # kept for extensibility

    for year, grp in df.groupby(df.index.year):
        save_path = get_save_path(base_dir, int(year), category, name)
        grp.to_csv(save_path, float_format="%.6f")
    print(f"[OK] saved {name} into yearly folders under {category}/")


def download_yf(
    ticker: str,
    start: str,
    end_exclusive: str,
    threads: bool = False,
) -> pd.DataFrame:
    return yf.download(
        ticker,
        start=start,
        end=end_exclusive,   # yfinance end is typically exclusive
        progress=False,
        threads=threads,
        auto_adjust=False,
    )


def download_fred_series(
    fred_id: str,
    start: str,
    end_inclusive: str,
) -> pd.DataFrame:
    # pandas_datareader FRED end is inclusive-like (date filter)
    return pdr.get_data_fred(fred_id, start=start, end=end_inclusive)


# -------------------------
# Main
# -------------------------
def main():
    parser = argparse.ArgumentParser(description="Download ETF + macro(index) data and save by year.")
    parser.add_argument("--start", type=str, default="2020-01-01")
    parser.add_argument("--end", type=str, default="2026-03-03", help="Inclusive intent")
    parser.add_argument("--base_dir", type=str, default="../data/data_org")

    parser.add_argument("--etfs", type=str, default="GLD,SPY,TLT,UUP,XLE")
    parser.add_argument("--vix_id", type=str, default="^VIX")

    # Macro FRED ids in "key=FRED_ID" comma-separated format
    parser.add_argument(
        "--fred",
        type=str,
        default="cpi=CPIAUCSL,dgs1=DGS1,dgs2=DGS2,dgs5=DGS5,dgs10=DGS10,dgs30=DGS30,dxy=DTWEXBGS,fedfunds=FEDFUNDS,gdp=GDP,hyspread=BAMLH0A0HYM2,unrate=UNRATE",
    )

    parser.add_argument("--sleep", type=float, default=0.5, help="Sleep seconds between yfinance calls.")
    parser.add_argument("--threads", action="store_true", help="Allow yfinance threads (default off).")
    args = parser.parse_args()

    start = args.start
    end_inclusive = args.end
    end_exclusive = (pd.to_datetime(end_inclusive) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    base_dir = args.base_dir
    ensure_dir(base_dir)

    etfs = [x.strip() for x in args.etfs.split(",") if x.strip()]
    fred_items = [x.strip() for x in args.fred.split(",") if x.strip()]
    fred_map: Dict[str, str] = {}
    for item in fred_items:
        if "=" not in item:
            raise ValueError(f"Bad --fred item: {item}. Expected key=FRED_ID.")
        k, v = item.split("=", 1)
        fred_map[k.strip()] = v.strip()

    print("\n=== Step 1: Download ETFs (yfinance) ===")
    for t in etfs:
        print(f"Downloading ETF: {t} ...")
        try:
            df = download_yf(t, start, end_exclusive, threads=args.threads)
            standardize_and_save_by_year(df, base_dir, "etf", t)
        except Exception as e:
            print(f"[ERR] ETF {t}: {e}")
        time.sleep(args.sleep)

    print("\n=== Step 2: Download macro/index (FRED) ===")
    for name, fred_id in fred_map.items():
        print(f"Downloading FRED: {name} ({fred_id}) ...")
        try:
            df = download_fred_series(fred_id, start, end_inclusive)
            # FRED returns a single column DataFrame
            standardize_and_save_by_year(df, base_dir, "index", name)
        except Exception as e:
            print(f"[ERR] FRED {name}: {e}")

    print("\n=== Step 3: Download VIX (yfinance) ===")
    print(f"Downloading VIX: {args.vix_id} ...")
    try:
        vix = download_yf(args.vix_id, start, end_exclusive, threads=args.threads)
        standardize_and_save_by_year(vix, base_dir, "index", "vix")
    except Exception as e:
        print(f"[ERR] VIX: {e}")

    print("\nDONE: Downloaded and saved by year.")
    print(f"Base directory: {os.path.abspath(base_dir)}")


if __name__ == "__main__":
    main()
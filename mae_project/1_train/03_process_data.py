# -*- coding: utf-8 -*-
import os
import json
import argparse
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch


# -------------------------
# Parsing helpers
# -------------------------
def parse_year_range(s: str) -> Tuple[int, int]:
    """
    Accept formats:
      - "2008-2024"
      - "2008:2024"
      - "2008,2024"
    """
    s = s.strip()
    for sep in ["-", ":", ","]:
        if sep in s:
            a, b = s.split(sep, 1)
            return int(a.strip()), int(b.strip())
    raise ValueError(f"Bad year range: {s}. Use like 2008-2024.")


def years_in_range(y0: int, y1: int) -> List[int]:
    if y1 < y0:
        raise ValueError(f"Invalid year range: {y0}-{y1}")
    return list(range(y0, y1 + 1))


def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path, index_col=0, parse_dates=True).sort_index()


def find_existing_year_files(base_dir: str, years: List[int], category: str, name: str) -> List[str]:
    """
    category: "etf" or "index"
    path: base_dir/{year}/{category}/{name}.csv
    """
    out = []
    for y in years:
        p = os.path.join(base_dir, str(y), category, f"{name}.csv")
        if os.path.exists(p):
            out.append(p)
    return out


# -------------------------
# Feature engineering
# -------------------------
def pick_price_series(df: pd.DataFrame) -> Optional[pd.Series]:
    """
    Prefer Adj Close, fallback to Close.
    """
    if df is None or df.empty:
        return None
    cols = list(df.columns)
    if "Adj Close" in cols:
        return df["Adj Close"].astype(float)
    if "Close" in cols:
        return df["Close"].astype(float)
    return None


def compute_log_return(price: pd.Series) -> pd.Series:
    r = np.log(price / price.shift(1))
    r = r.replace([np.inf, -np.inf], np.nan)
    return r


def vol_zscore_rolling_20(vol: pd.Series) -> pd.Series:
    mu = vol.rolling(20, min_periods=20).mean()
    sd = vol.rolling(20, min_periods=20).std()
    z = (vol - mu) / (sd + 1e-8)
    return z


# -------------------------
# Main builder
# -------------------------
def main():
    parser = argparse.ArgumentParser(description="Build train/val/test .pt packs from yearly CSV storage.")
    parser.add_argument("--base_dir", type=str, default="../data/data_revise")
    parser.add_argument("--out_dir", type=str, default="../data/data_pt")

    parser.add_argument("--etfs", type=str, default="GLD,SPY,TLT,UUP,XLE")
    parser.add_argument(
        "--macro_keys",
        type=str,
        default="cpi,dgs1,dgs2,dgs5,dgs10,dgs30,dxy,fedfunds,gdp,hyspread,unrate",
    )
    parser.add_argument("--include_vix", action="store_true", help="If set, add 'vix' from index/vix.csv")
    parser.set_defaults(include_vix=True)
    parser.add_argument("--master_ticker", type=str, default="SPY", help="Use this ETF's calendar as master index.")

    parser.add_argument("--train", type=str, default="2020-2024")
    parser.add_argument("--val", type=str, default="2025-2025")
    parser.add_argument("--test", type=str, default="2026-2026")

    parser.add_argument("--drop_rolling", type=int, default=20, help="Drop first N rows due to rolling features.")
    args = parser.parse_args()

    base_dir = args.base_dir
    out_dir = args.out_dir or os.path.join(base_dir, "pt_dict")
    os.makedirs(out_dir, exist_ok=True)

    etfs = [x.strip() for x in args.etfs.split(",") if x.strip()]
    macro_keys = [x.strip() for x in args.macro_keys.split(",") if x.strip()]
    index_names = macro_keys + (["vix"] if args.include_vix else [])

    train_y0, train_y1 = parse_year_range(args.train)
    val_y0, val_y1 = parse_year_range(args.val)
    test_y0, test_y1 = parse_year_range(args.test)

    split_years: Dict[str, List[int]] = {
        "train": years_in_range(train_y0, train_y1),
        "val": years_in_range(val_y0, val_y1),
        "test": years_in_range(test_y0, test_y1),
    }
    total_years = sorted(set(split_years["train"] + split_years["val"] + split_years["test"]))

    # -------------------------
    # Load all series (concatenate across years)
    # -------------------------
    merged: Dict[str, pd.DataFrame] = {}

    # ETFs
    for t in etfs:
        paths = find_existing_year_files(base_dir, total_years, "etf", t)
        if not paths:
            print(f"[WARN] Missing ETF files for {t} in years={total_years}.")
            continue
        dfs = [load_csv(p) for p in paths]
        df = pd.concat(dfs).sort_index()
        df = df[~df.index.duplicated(keep="first")]
        merged[t] = df

    # Index/macro
    for n in index_names:
        paths = find_existing_year_files(base_dir, total_years, "index", n)
        if not paths:
            print(f"[WARN] Missing index files for {n} in years={total_years}.")
            continue
        dfs = [load_csv(p) for p in paths]
        df = pd.concat(dfs).sort_index()
        df = df[~df.index.duplicated(keep="first")]
        merged[n] = df

    # -------------------------
    # Master calendar
    # -------------------------
    master = args.master_ticker
    if master not in merged or merged[master].empty:
        raise ValueError(f"Missing master_ticker={master} data. Cannot create master calendar.")
    master_idx = merged[master].index

    # -------------------------
    # Build X and R
    # -------------------------
    X_df = pd.DataFrame(index=master_idx)
    R_df = pd.DataFrame(index=master_idx)

    etf_feature_cols: List[str] = []

    # ETF features (NO forward fill)
    for t in etfs:
        if t not in merged or merged[t].empty:
            print(f"[WARN] Skip ETF {t}: no data.")
            continue
        df = merged[t].reindex(master_idx)

        price = pick_price_series(df)
        if price is None:
            print(f"[WARN] {t}: missing Adj Close/Close; skip returns.")
        else:
            r = compute_log_return(price)
            R_df[t] = r
            col_ret = f"{t}_ret"
            X_df[col_ret] = r
            etf_feature_cols.append(col_ret)

        if "Volume" in df.columns:
            vol = pd.to_numeric(df["Volume"], errors="coerce")
            col_vol = f"{t}_vol"
            X_df[col_vol] = vol_zscore_rolling_20(vol)
            etf_feature_cols.append(col_vol)
        else:
            print(f"[WARN] {t}: missing Volume; skip volume feature.")

    # Macro/index features (pre-cleaned already; light safety fill only)
    for n in index_names:
        if n not in merged or merged[n].empty:
            print(f"[WARN] Skip index {n}: no data.")
            continue

        df = merged[n].reindex(master_idx)

        # Light safety fill for any remaining gaps after pre-clean
        df = df.bfill().ffill()

        if n == "vix":
            price = pick_price_series(df)
            if price is None:
                X_df[n] = pd.to_numeric(df.iloc[:, 0], errors="coerce")
            else:
                X_df[n] = price.astype(float)
        else:
            X_df[n] = pd.to_numeric(df.iloc[:, 0], errors="coerce")

    if X_df.shape[1] == 0:
        raise ValueError("X_df has no feature columns. Check your inputs and folder structure.")

    # Drop rolling warmup rows
    drop_n = int(args.drop_rolling)
    if drop_n > 0 and len(X_df) > drop_n:
        X_df = X_df.iloc[drop_n:].copy()
        R_df = R_df.loc[X_df.index].copy()

    # Replace inf with NaN first
    X_df = X_df.replace([np.inf, -np.inf], np.nan)
    R_df = R_df.replace([np.inf, -np.inf], np.nan)

    # Drop rows with missing ETF values (returns / ETF-derived features)
    required_etf_cols = [c for c in etf_feature_cols if c in X_df.columns]
    required_r_cols = [c for c in etfs if c in R_df.columns]

    if len(required_r_cols) == 0:
        raise ValueError("R_df has no ETF return columns. Check ETF inputs and price columns.")

    row_mask = pd.Series(True, index=X_df.index)

    if required_etf_cols:
        row_mask &= X_df[required_etf_cols].notna().all(axis=1)

    row_mask &= R_df[required_r_cols].notna().all(axis=1)

    X_df = X_df.loc[row_mask].copy()
    R_df = R_df.loc[row_mask].copy()

    # Optional safety for any unexpected remaining NaN in features
    # (should be rare after pre-clean + light safety fill)
    remaining_nan_rows = X_df.isna().any(axis=1)
    if remaining_nan_rows.any():
        print(f"[WARN] Dropping {int(remaining_nan_rows.sum())} rows with remaining NaN in X.")
        X_df = X_df.loc[~remaining_nan_rows].copy()
        R_df = R_df.loc[X_df.index].copy()

    if X_df.empty or R_df.empty:
        raise ValueError("After NaN filtering, X_df or R_df is empty. Check missing-data handling and date coverage.")

    feature_names = list(X_df.columns)
    etf_order = [t for t in etfs if t in R_df.columns]

    # -------------------------
    # Save packs by split years
    # -------------------------
    for split, years in split_years.items():
        mask = X_df.index.year.isin(years)
        X_split = X_df.loc[mask]
        R_split = R_df.loc[mask]

        if X_split.empty:
            print(f"[WARN] split={split} empty for years={years}. Skip.")
            continue

        start = X_split.index.min().strftime("%Y-%m-%d")
        end = X_split.index.max().strftime("%Y-%m-%d")

        X_t = torch.tensor(X_split.values, dtype=torch.float32)
        R_t = torch.tensor(R_split[etf_order].values, dtype=torch.float32)

        pack = {
            "meta": {
                "split": split,
                "years": years,
                "start_date": start,
                "end_date": end,
                "freq": f"{master} trading calendar",
                "master_ticker": master,
                "etfs": etf_order,
                "macro_keys": macro_keys,
                "vix_included": bool(args.include_vix),
                "feature_names": feature_names,
                "feature_dim": int(X_t.shape[1]),
                "return_def": "raw log return from Adj Close (or Close fallback); rows with missing ETF data are dropped",
                "note": "Input CSVs are pre-cleaned. No global normalization. Use window normalization in Dataset if needed.",
            },
            "dates": [d.strftime("%Y-%m-%d") for d in X_split.index],
            "X": X_t,
            "R": R_t,
        }

        base = f"market_{split}_{start}_to_{end}"
        pt_path = os.path.join(out_dir, f"{base}.pt")
        torch.save(pack, pt_path)

        # previews
        preview_rows = pd.concat([X_split.head(10), X_split.tail(10)])
        preview_rows.to_csv(os.path.join(out_dir, f"{base}__X_preview.csv"), float_format="%.6f")
        preview_ret = pd.concat([R_split[etf_order].head(10), R_split[etf_order].tail(10)])
        preview_ret.to_csv(os.path.join(out_dir, f"{base}__R_preview.csv"), float_format="%.6f")

        # manifest
        with open(os.path.join(out_dir, f"{base}__manifest.json"), "w", encoding="utf-8") as f:
            json.dump(pack["meta"], f, ensure_ascii=False, indent=2)

        print(f"[SAVE] {split}: X={tuple(X_t.shape)}, R={tuple(R_t.shape)} -> {pt_path}")

    print("\nDONE: Saved dict-style .pt packs + previews + manifest.")
    print(f"Out dir: {os.path.abspath(out_dir)}")


if __name__ == "__main__":
    main()
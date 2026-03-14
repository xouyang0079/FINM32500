# -*- coding: utf-8 -*-
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import yfinance as yf
from pandas_datareader import data as pdr


# Match 1_train defaults (5 ETFs + 11 FRED + VIX = 12 index inputs total incl vix)
DEFAULT_ETFS = ["GLD", "SPY", "TLT", "UUP", "XLE"]
DEFAULT_FRED_MAP: Dict[str, str] = {
    "cpi": "CPIAUCSL",
    "dgs1": "DGS1",
    "dgs2": "DGS2",
    "dgs5": "DGS5",
    "dgs10": "DGS10",
    "dgs30": "DGS30",
    "dxy": "DTWEXBGS",
    "fedfunds": "FEDFUNDS",
    "gdp": "GDP",
    "hyspread": "BAMLH0A0HYM2",
    "unrate": "UNRATE",
}
DEFAULT_VIX_TICKER = "^VIX"


@dataclass
class PackWindow:
    # keep field names used by live_runner for compatibility
    pack_path: str
    dates: List[str]
    X: torch.Tensor
    R: np.ndarray
    etf_symbols: List[str]
    feature_dim: int
    signal_date: str
    t_idx: int
    x_window_20xF: torch.Tensor


# -----------------------------------------------------------------------------
# IO helpers
# -----------------------------------------------------------------------------
def _download_yf(ticker: str, start: str, end_exclusive: str) -> pd.DataFrame:
    df = yf.download(
        ticker,
        start=start,
        end=end_exclusive,
        progress=False,
        threads=False,
        auto_adjust=False,
    )
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="first")]
    return df


import time
import pandas as pd
from pandas_datareader import data as pdr


def _download_fred_series(
    fred_id: str,
    start,
    end_inclusive,
    max_retries: int = 4,
    sleep_seconds: float = 1.5,
):
    """
    Download one FRED series with retries.
    Returns a DataFrame indexed by date with a single column named fred_id.
    """
    last_err = None

    for attempt in range(1, max_retries + 1):
        try:
            df = pdr.get_data_fred(fred_id, start=start, end=end_inclusive)
            if df is None:
                raise ValueError(f"FRED {fred_id}: returned None")
            if not isinstance(df, pd.DataFrame):
                df = pd.DataFrame(df)
            if df.empty:
                # Keep empty result behavior explicit; caller decides whether to fail or fallback.
                return df

            df = df.sort_index()
            # Ensure the column name is stable
            if len(df.columns) == 1:
                df.columns = [fred_id]
            return df

        except Exception as e:
            last_err = e
            if attempt == max_retries:
                break
            time.sleep(sleep_seconds * attempt)  # light backoff

    raise RuntimeError(f"FRED download failed after {max_retries} attempts for {fred_id}: {last_err}")


# -----------------------------------------------------------------------------
# Feature engineering (aligned with 1_train/03_process_data.py)
# -----------------------------------------------------------------------------
def _pick_price_series(df: pd.DataFrame) -> Optional[pd.Series]:
    if df is None or df.empty:
        return None
    if "Adj Close" in df.columns:
        return pd.to_numeric(df["Adj Close"], errors="coerce")
    if "Close" in df.columns:
        return pd.to_numeric(df["Close"], errors="coerce")
    return None


def _compute_log_return(price: pd.Series) -> pd.Series:
    r = np.log(price / price.shift(1))
    return r.replace([np.inf, -np.inf], np.nan)


def _vol_zscore_rolling_20(vol: pd.Series) -> pd.Series:
    mu = vol.rolling(20, min_periods=20).mean()
    sd = vol.rolling(20, min_periods=20).std()
    z = (vol - mu) / (sd + 1e-8)
    return z


def _bfill_then_ffill(df: pd.DataFrame) -> pd.DataFrame:
    # User explicitly requested "fill in with next day data" => bfill first.
    # We keep ffill as a trailing-edge fallback if the newest row is missing.
    return df.bfill().ffill()


def _ensure_required_columns(df: pd.DataFrame, name: str) -> None:
    cols = set(df.columns.astype(str))
    if ("Adj Close" not in cols) and ("Close" not in cols):
        raise ValueError(f"{name}: missing Adj Close/Close")
    if "Volume" not in cols:
        raise ValueError(f"{name}: missing Volume")


def build_live_window_online(
    signal_date: str,
    seq_len: int = 20,
    etfs: Optional[List[str]] = None,
    fred_map: Optional[Dict[str, str]] = None,
    vix_ticker: str = DEFAULT_VIX_TICKER,
    raw_calendar_lookback_days: int = 120,
) -> PackWindow:
    """
    Download fresh data online at runtime, build features like 1_train/03_process_data,
    and return the signal-date aligned 20xF model input window.

    Important behavior:
      - NO local .pt pack loading
      - fills missing values using next-day data first (bfill), then ffill fallback
      - uses SPY trading calendar as master index (same as training process default)
    """
    etfs = list(etfs or DEFAULT_ETFS)
    fred_map = dict(fred_map or DEFAULT_FRED_MAP)

    sig = pd.Timestamp(signal_date).normalize()
    signal_date_str = sig.strftime("%Y-%m-%d")
    # Need extra raw history to compute rolling 20 volume z-score and obtain 20 valid rows.
    start = (sig - pd.Timedelta(days=int(raw_calendar_lookback_days))).strftime("%Y-%m-%d")
    end_inclusive = signal_date
    end_exclusive = (sig + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    merged: Dict[str, pd.DataFrame] = {}

    # 1) ETFs from yfinance
    for t in etfs:
        df = _download_yf(t, start=start, end_exclusive=end_exclusive)
        if df is None or df.empty:
            raise ValueError(f"ETF {t}: downloaded empty dataframe")
        _ensure_required_columns(df, t)
        # Fill missing raw market fields with next-day data as requested
        df = _bfill_then_ffill(df)
        merged[t] = df

    # 2) Macro/index from FRED
    for key, fred_id in fred_map.items():
        df = _download_fred_series(fred_id, start=start, end_inclusive=end_inclusive)
        if df is None or df.empty:
            raise ValueError(f"FRED index {key} ({fred_id}): downloaded empty dataframe")
        df = _bfill_then_ffill(df)
        merged[key] = df

    # 3) VIX from yfinance
    vix = _download_yf(vix_ticker, start=start, end_exclusive=end_exclusive)
    if vix is None or vix.empty:
        raise ValueError("VIX: downloaded empty dataframe")
    # User requested fill missing with next-day data
    vix = _bfill_then_ffill(vix)
    merged["vix"] = vix

    # -------------------------
    # Build master calendar from SPY and clamp to <= signal_date
    # -------------------------
    if "SPY" not in merged:
        raise ValueError("SPY data missing; cannot build master trading calendar")
    master_idx = merged["SPY"].index
    master_idx = master_idx[master_idx <= sig]
    master_idx = master_idx.sort_values().unique()
    if len(master_idx) == 0:
        raise ValueError("Master calendar is empty after filtering to signal_date")

    X_df = pd.DataFrame(index=master_idx)
    R_df = pd.DataFrame(index=master_idx)
    etf_feature_cols: List[str] = []

    # ETF features + returns
    for t in etfs:
        df = merged[t].reindex(master_idx)
        # Fill reindex gaps using next-day data first (user request)
        df = _bfill_then_ffill(df)

        price = _pick_price_series(df)
        if price is None:
            raise ValueError(f"{t}: missing price series after reindex")
        ret = _compute_log_return(price)
        R_df[t] = ret
        X_df[f"{t}_ret"] = ret
        etf_feature_cols.append(f"{t}_ret")

        vol = pd.to_numeric(df["Volume"], errors="coerce")
        X_df[f"{t}_vol"] = _vol_zscore_rolling_20(vol)
        etf_feature_cols.append(f"{t}_vol")

    # Macro / index features (11 FRED + vix)
    for k in list(fred_map.keys()):
        df = merged[k].reindex(master_idx)
        df = _bfill_then_ffill(df)  # next-day fill first
        X_df[k] = pd.to_numeric(df.iloc[:, 0], errors="coerce")

    df = merged["vix"].reindex(master_idx)
    df = _bfill_then_ffill(df)  # next-day fill first
    vix_price = _pick_price_series(df)
    X_df["vix"] = pd.to_numeric(vix_price if vix_price is not None else df.iloc[:, 0], errors="coerce")

    # Safety cleanup
    X_df = X_df.replace([np.inf, -np.inf], np.nan)
    R_df = R_df.replace([np.inf, -np.inf], np.nan)

    required_r_cols = [t for t in etfs if t in R_df.columns]
    row_mask = pd.Series(True, index=X_df.index)
    row_mask &= X_df[etf_feature_cols].notna().all(axis=1)
    row_mask &= R_df[required_r_cols].notna().all(axis=1)

    # For index/macro columns, fill again using next-day data if any gaps remain after reindex.
    idx_cols = [c for c in X_df.columns if c not in etf_feature_cols]
    if idx_cols:
        X_df[idx_cols] = X_df[idx_cols].bfill().ffill()

    # If any feature still NaN, perform requested next-day fill globally, then fallback.
    if X_df.isna().any().any():
        X_df = X_df.bfill().ffill()

    # Final row mask after fills (ETF returns/vol still must be valid)
    row_mask = pd.Series(True, index=X_df.index)
    row_mask &= X_df[etf_feature_cols].notna().all(axis=1)
    row_mask &= R_df[required_r_cols].notna().all(axis=1)
    X_df = X_df.loc[row_mask].copy()
    R_df = R_df.loc[row_mask].copy()

    if signal_date_str not in set(X_df.index.strftime("%Y-%m-%d")):
        # if SPY calendar includes signal_date but row got dropped due to warmup/NaNs, fail loudly
        raise ValueError(
            f"signal_date={signal_date} not available in final live feature table. "
            f"Available tail dates={list(X_df.index.strftime('%Y-%m-%d')[-5:])}"
        )

    if len(X_df) < seq_len:
        raise ValueError(f"Not enough valid rows after processing: have {len(X_df)}, need {seq_len}")

    dates = list(X_df.index.strftime("%Y-%m-%d"))
    t_idx = dates.index(signal_date_str)
    if t_idx < seq_len - 1:
        raise ValueError(f"signal_date index {t_idx} has < {seq_len} rows of history")

    # Keep full X/R history up to signal_date for covariance estimation (still downloaded online this run)
    X_np = X_df.to_numpy(dtype=np.float32)
    R_np = R_df[etfs].to_numpy(dtype=np.float32)
    X_t = torch.from_numpy(X_np)
    xw = X_t[t_idx - seq_len + 1 : t_idx + 1]

    if xw.shape[0] != seq_len:
        raise ValueError(f"Bad x_window shape {tuple(xw.shape)}")

    return PackWindow(
        pack_path="ONLINE_RUNTIME_NO_LOCAL_PACK",
        dates=dates,
        X=X_t,
        R=R_np,
        etf_symbols=etfs,
        feature_dim=int(X_t.shape[1]),
        signal_date=signal_date_str,
        t_idx=int(t_idx),
        x_window_20xF=xw,
    )


def save_market_snapshot_csv(path: str, rows: List[Dict[str, object]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)

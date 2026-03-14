# -*- coding: utf-8 -*-
"""
Infer/Eval (FULL): Return full model -> weights via optimizer -> backtest
- Load dict pack (X,R) for split
- For each day t:
    input: X[idx:idx+20] (past19+today), window-normalized
    pred:  r_hat_{t+1} from FULL model ckpt (one pth)
    w_pred_t: max-sharpe(mu=r_hat, Sigma=cov from past K realized returns up to t)
              + turnover penalty to reduce churn (optional)
    w_oracle_t: oracle teacher max-sharpe weights using realized returns over 21 days (past19+today+tomorrow)
    realized portfolio return uses true r_{t+1}
- Metrics: SAME set for both PredReturn->Opt and OracleTeacher
"""

import os
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PT_DIR = os.path.abspath(os.path.join(THIS_DIR, "..", "data", "data_pt"))
MODEL_DIR = os.path.abspath(os.path.join(THIS_DIR, "..", "model"))

import glob
import math
import argparse
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import minimize


np.set_printoptions(suppress=True, precision=6)
torch.set_printoptions(precision=6, sci_mode=False)


# -----------------------------
# Utils
# -----------------------------
def find_pack(pt_dir: str, split: str) -> str:
    cand = sorted(glob.glob(os.path.join(pt_dir, f"market_{split}_*.pt")))
    if not cand:
        raise FileNotFoundError(f"Cannot find pack for split='{split}' under {pt_dir}")
    return cand[0]


def window_normalize(w: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    mu = w.mean(dim=0, keepdim=True)
    std = w.std(dim=0, keepdim=True, unbiased=False).add(eps)
    return (w - mu) / std


# -----------------------------
# FULL model (must match your TRAIN for state_dict keys)
# Note: This class matches your current infer baseline (no tanh shrink).
# If your TRAIN forward includes tau*tanh, you must update forward accordingly.
# -----------------------------
class ReturnFullModel(nn.Module):
    def __init__(
        self,
        num_feat: int = 22,
        seq_len: int = 20,
        patch_size: int = 5,
        embed_dim: int = 256,
        enc_layers: int = 6,
        dropout: float = 0.2,
        out_dim: int = 5,
    ):
        super().__init__()
        if seq_len % patch_size != 0:
            raise ValueError("seq_len must be divisible by patch_size")

        self.num_feat = num_feat
        self.seq_len = seq_len
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        patch_dim = patch_size * num_feat
        num_patches = seq_len // patch_size

        feat_mask = torch.zeros(num_feat)
        feat_mask[:10] = 1.0
        if num_feat >= 22:
            feat_mask[21] = 1.0
        self.register_buffer("feat_mask", feat_mask)
        self.register_buffer("patch_mask", feat_mask.repeat(patch_size))
        self.patch_mask_token = nn.Parameter(torch.zeros(1, 1, patch_dim))

        self.patch_proj = nn.Linear(patch_dim, embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))

        enc_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=8,
            dim_feedforward=1024,
            batch_first=True,
            dropout=dropout,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=enc_layers)

        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 128),
            nn.GELU(),
            nn.Linear(128, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, F = x.shape
        if T != self.seq_len or F != self.num_feat:
            raise ValueError(
                f"Input shape mismatch: got {tuple(x.shape)} expected [B,{self.seq_len},{self.num_feat}]"
            )

        P = self.seq_len // self.patch_size
        patch_dim = self.patch_size * self.num_feat
        x_patches = x.reshape(B, P, patch_dim)
        tokens = self.patch_proj(x_patches) + self.pos_embed
        latent = self.encoder(tokens)
        emb = latent.mean(dim=1)
        return self.head(emb)  # [B,5]


# -----------------------------
# Optimizer with turnover penalty
# -----------------------------
def max_sharpe_long_only(
    mu: np.ndarray,
    Sigma: np.ndarray,
    w_prev: Optional[np.ndarray] = None,
    gamma_turn: float = 0.0,
    turnover_l1: bool = False,
    eps: float = 1e-8,
) -> np.ndarray:
    """
    Solve:
        maximize  Sharpe(w; mu, Sigma) - gamma_turn * penalty(w, w_prev)
    subject to:
        w>=0, sum(w)=1

    penalty:
        if turnover_l1: ||w - w_prev||_1
        else:          ||w - w_prev||_2^2   (default, smoother & easier)
    """
    mu = np.asarray(mu, dtype=np.float64)
    Sigma = np.asarray(Sigma, dtype=np.float64)
    N = mu.shape[0]
    Sigma = Sigma + np.eye(N) * 1e-6

    w0 = np.ones(N, dtype=np.float64) / N

    if w_prev is not None:
        w_prev = np.asarray(w_prev, dtype=np.float64).clip(0.0, 1.0)
        s = w_prev.sum()
        w_prev = (w_prev / s) if s > 0 else w0
        w0 = w_prev.copy()

    def neg_objective(w):
        num = float(np.dot(w, mu))
        den = float(np.sqrt(np.dot(w, Sigma @ w) + eps))
        sharpe = num / den

        if (w_prev is None) or (gamma_turn <= 0.0):
            pen = 0.0
        else:
            if turnover_l1:
                pen = gamma_turn * float(np.sum(np.abs(w - w_prev)))
            else:
                d = (w - w_prev)
                pen = gamma_turn * float(np.dot(d, d))  # L2^2

        return -(sharpe - pen)

    cons = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0},)
    bounds = [(0.0, 1.0)] * N

    res = minimize(
        neg_objective,
        w0,
        method="SLSQP",
        bounds=bounds,
        constraints=cons,
        options={"maxiter": 200, "ftol": 1e-9},
    )
    if (not res.success) or np.any(np.isnan(res.x)):
        return w0

    w = res.x.clip(0.0, 1.0)
    s = w.sum()
    return w0 if s <= 0 else (w / s)


def oracle_teacher_weights(returns_21xN: np.ndarray) -> np.ndarray:
    R = np.asarray(returns_21xN, dtype=np.float64)
    mu = R.mean(axis=0)
    Sigma = np.cov(R, rowvar=False)
    return max_sharpe_long_only(mu, Sigma)


def estimate_cov_pastK(returns_TxN: np.ndarray, end_idx_inclusive: int, K: int) -> np.ndarray:
    start = max(0, end_idx_inclusive - K + 1)
    win = returns_TxN[start:end_idx_inclusive + 1]
    N = returns_TxN.shape[1]
    if win.shape[0] < 2:
        return np.eye(N) * 1e-3
    Sigma = np.cov(win, rowvar=False)
    return Sigma + np.eye(N) * 1e-6


# -----------------------------
# Metrics (same set)
# -----------------------------
def compute_metrics(port_ret: np.ndarray, weights: np.ndarray, ann_factor: int = 252) -> Dict[str, float]:
    port_ret = np.asarray(port_ret, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)

    equity = np.cumprod(1.0 + port_ret)
    peak = np.maximum.accumulate(equity)
    drawdown = equity / peak - 1.0
    max_dd = float(drawdown.min())

    mean_d = float(port_ret.mean())
    vol_d = float(port_ret.std(ddof=0))
    ann_ret = float((equity[-1] ** (ann_factor / max(1, len(port_ret))) - 1.0))
    ann_vol = float(vol_d * math.sqrt(ann_factor))
    sharpe = float((mean_d / (vol_d + 1e-12)) * math.sqrt(ann_factor))
    calmar = float(ann_ret / (abs(max_dd) + 1e-12))

    if len(weights) <= 1:
        avg_turn = 0.0
    else:
        dw = np.abs(weights[1:] - weights[:-1]).sum(axis=1) * 0.5
        avg_turn = float(dw.mean())

    return {
        "AnnRet": ann_ret,
        "AnnVol": ann_vol,
        "Sharpe": sharpe,
        "MaxDD": max_dd,
        "Calmar": calmar,
        "AvgTurnover": avg_turn,
    }


def print_metrics_table(name_to_metrics: Dict[str, Dict[str, float]]):
    keys = ["AnnRet", "AnnVol", "Sharpe", "MaxDD", "Calmar", "AvgTurnover"]
    header = f"{'Model':<18} " + " ".join([f"{k:>12}" for k in keys])
    print("\n" + header)
    print("-" * len(header))
    for name, m in name_to_metrics.items():
        row = f"{name:<18} " + " ".join([f"{m[k]:12.6f}" for k in keys])
        print(row)


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pt_dir", type=str, default=PT_DIR)
    ap.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    ap.add_argument(
    "--ckpt",
    type=str,
    default=os.path.join(MODEL_DIR, "downstream_return_full.pth"),
    help="FULL model checkpoint (one pth)",)
    ap.add_argument("--cov_window", type=int, default=60, help="past K days for covariance estimate")
    ap.add_argument("--device", type=str, default="cuda")

    # New knobs for turnover control
    ap.add_argument("--gamma_turn", type=float, default=0.0, help="turnover penalty strength (0=baseline)")
    ap.add_argument("--turnover_l1", action="store_true", help="use L1 penalty instead of L2^2")
    ap.add_argument("--mu_ema", type=float, default=0.0, help="EMA alpha for predicted mu (0=off, typical 0.5~0.9)")

    args = ap.parse_args()

    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")

    pack_path = find_pack(args.pt_dir, args.split)
    pack = torch.load(pack_path, map_location="cpu")
    X = pack["X"].float()  # [T,22]
    R = pack["R"].float()  # [T,5]
    meta = pack.get("meta", {})

    print(f"[PACK] {pack_path}")
    print(
        f"       split={meta.get('split')} X={tuple(X.shape)} R={tuple(R.shape)} "
        f"date_range={meta.get('start_date')}..{meta.get('end_date')}"
    )
    print(f"[KNOBS] gamma_turn={args.gamma_turn} turnover_l1={args.turnover_l1} mu_ema={args.mu_ema}")

    # Build FULL model and load ONE ckpt
    model = ReturnFullModel(
        num_feat=22, seq_len=20, patch_size=5, embed_dim=256, enc_layers=6, dropout=0.2, out_dim=5
    )
    sd = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(sd, strict=True)
    model.to(device).eval()

    T = X.shape[0]
    max_idx = T - 21
    if max_idx < 0:
        raise ValueError(f"Not enough data: T={T} need at least 21")

    R_np = R.numpy()

    w_pred_list, w_oracle_list = [], []
    port_pred, port_oracle = [], []

    w_prev = None
    mu_ema_vec = None

    for idx in range(max_idx + 1):
        xw = window_normalize(X[idx:idx + 20]).unsqueeze(0).to(device)  # [1,20,22]

        with torch.no_grad():
            r_hat = model(xw).squeeze(0).cpu().numpy()  # [5]

        # Optional EMA smoothing on predicted mu
        if args.mu_ema > 0.0:
            a = float(args.mu_ema)
            if mu_ema_vec is None:
                mu_ema_vec = r_hat
            else:
                mu_ema_vec = a * mu_ema_vec + (1.0 - a) * r_hat
            mu_use = mu_ema_vec
        else:
            mu_use = r_hat

        t_idx = idx + 19
        Sigma = estimate_cov_pastK(R_np, end_idx_inclusive=t_idx, K=args.cov_window)

        # Pred strategy: with optional turnover penalty
        w_pred = max_sharpe_long_only(
            mu=mu_use,
            Sigma=Sigma,
            w_prev=w_prev,
            gamma_turn=args.gamma_turn,
            turnover_l1=args.turnover_l1,
        )
        w_prev = w_pred

        # Oracle teacher
        w_oracle = oracle_teacher_weights(R_np[idx:idx + 21, :])

        r_real = R_np[idx + 20, :]
        port_pred.append(float(np.dot(w_pred, r_real)))
        port_oracle.append(float(np.dot(w_oracle, r_real)))

        w_pred_list.append(w_pred)
        w_oracle_list.append(w_oracle)

        print(f"[{idx:04d}] t={t_idx}  "
              f"mu_hat={np.round(mu_use, 6)}  "
              f"w_pred={np.round(w_pred, 4)}  "
              f"w_oracle={np.round(w_oracle, 4)}  "
              f"r_real={np.round(r_real, 6)}  "
              f"port_pred={float(np.dot(w_pred, r_real)):.6f}  "
              f"port_oracle={float(np.dot(w_oracle, r_real)):.6f}")

    w_pred_arr = np.vstack(w_pred_list)
    w_oracle_arr = np.vstack(w_oracle_list)
    port_pred = np.array(port_pred, dtype=np.float64)
    port_oracle = np.array(port_oracle, dtype=np.float64)

    m_pred = compute_metrics(port_pred, w_pred_arr)
    m_oracle = compute_metrics(port_oracle, w_oracle_arr)

    print_metrics_table({
        "PredReturn->Opt": m_pred,
        "OracleTeacher": m_oracle,
    })


if __name__ == "__main__":
    main()
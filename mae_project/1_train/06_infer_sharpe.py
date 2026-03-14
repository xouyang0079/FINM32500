# -*- coding: utf-8 -*-
"""
Infer/Eval (End-to-End): Model directly outputs portfolio weights
- Load dict pack (X,R) for split
- For each day t:
    input: X[idx:idx+20] (past19+today), window-normalized
    pred:  w_pred_t directly from FULL model ckpt (Softmax output)
    w_oracle_t: oracle teacher max-sharpe weights using realized returns over 21 days (past19+today+tomorrow)
    realized portfolio return uses true r_{t+1}
- Metrics: SAME set for both PredWeights and OracleTeacher
"""

import os
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
# FULL model (Matches Train exactly)
# -----------------------------
class ReturnFullModel(nn.Module):
    def __init__(
        self,
        num_feat: int = 22,
        seq_len: int = 20,
        patch_size: int = 5,
        embed_dim: int = 256,
        enc_layers: int = 6,
        dropout: float = 0.1,
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
            nn.Softmax(dim=-1) # NEW: Directly output portfolio weights
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
# Oracle Optimizer (Retained only for the Oracle Teacher Baseline)
# -----------------------------
def max_sharpe_long_only(
    mu: np.ndarray,
    Sigma: np.ndarray,
    eps: float = 1e-8,
) -> np.ndarray:
    mu = np.asarray(mu, dtype=np.float64)
    Sigma = np.asarray(Sigma, dtype=np.float64)
    N = mu.shape[0]
    Sigma = Sigma + np.eye(N) * 1e-6

    w0 = np.ones(N, dtype=np.float64) / N

    def neg_objective(w):
        num = float(np.dot(w, mu))
        den = float(np.sqrt(np.dot(w, Sigma @ w) + eps))
        return -(num / den)

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
    ap.add_argument("--pt_dir", type=str, default="../data/data_pt")
    ap.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    ap.add_argument("--ckpt", type=str, default="../model/downstream_e2e_sharpe.pth", help="FULL model checkpoint")
    ap.add_argument("--device", type=str, default="cuda")

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

    # Build FULL model and load ONE ckpt
    model = ReturnFullModel(
        num_feat=22, seq_len=20, patch_size=5, embed_dim=256, enc_layers=6, dropout=0.1, out_dim=5
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

    for idx in range(max_idx + 1):
        xw = window_normalize(X[idx:idx + 20]).unsqueeze(0).to(device)  # [1,20,22]

        # 直接预测仓位权重！没有协方差预测，也没有二次规划求解了
        with torch.no_grad():
            w_pred = model(xw).squeeze(0).cpu().numpy()  # [5]

        t_idx = idx + 19

        # Oracle teacher (用于参照系比对)
        w_oracle = oracle_teacher_weights(R_np[idx:idx + 21, :])

        r_real = R_np[idx + 20, :]
        port_pred.append(float(np.dot(w_pred, r_real)))
        port_oracle.append(float(np.dot(w_oracle, r_real)))

        w_pred_list.append(w_pred)
        w_oracle_list.append(w_oracle)

        print(f"[{idx:04d}] t={t_idx}  "
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
        "E2E_PredWeights": m_pred,
        "OracleTeacher": m_oracle,
    })


if __name__ == "__main__":
    main()
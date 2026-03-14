# -*- coding: utf-8 -*-
"""
Strategy Class:
- Predict next-day ETF returns using trained model
- Estimate covariance from past K realized returns
- Solve long-only max-Sharpe weights
- Convert target weights into rebalance orders
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import minimize


np.set_printoptions(suppress=True, precision=6)
torch.set_printoptions(precision=6, sci_mode=False)


def window_normalize(w: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    mu = w.mean(dim=0, keepdim=True)
    std = w.std(dim=0, keepdim=True, unbiased=False).add(eps)
    return (w - mu) / std


class ReturnFullModel(nn.Module):
    """
    Must match train/infer model architecture for checkpoint compatibility.
    This version intentionally keeps the same extra buffers/params as the user's old infer code:
      - feat_mask
      - patch_mask
      - patch_mask_token
    even if they are not used in forward().
    """

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

        # Keep these to match the user's checkpoint keys exactly
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
        bsz, t_len, n_feat = x.shape
        if t_len != self.seq_len or n_feat != self.num_feat:
            raise ValueError(
                f"Input shape mismatch: got {tuple(x.shape)} expected [B,{self.seq_len},{self.num_feat}]"
            )

        num_patches = self.seq_len // self.patch_size
        patch_dim = self.patch_size * self.num_feat
        x_patches = x.reshape(bsz, num_patches, patch_dim)
        tokens = self.patch_proj(x_patches) + self.pos_embed
        latent = self.encoder(tokens)
        emb = latent.mean(dim=1)
        return self.head(emb)


def max_sharpe_long_only(
    mu: np.ndarray,
    Sigma: np.ndarray,
    w_prev: Optional[np.ndarray] = None,
    gamma_turn: float = 0.0,
    turnover_l1: bool = False,
    eps: float = 1e-8,
) -> np.ndarray:
    """
    Same core logic as user's infer code.

    maximize  Sharpe(w; mu, Sigma) - gamma_turn * penalty(w, w_prev)
    s.t.      w >= 0, sum(w) = 1
    """
    mu = np.asarray(mu, dtype=np.float64)
    Sigma = np.asarray(Sigma, dtype=np.float64)
    n = mu.shape[0]
    Sigma = Sigma + np.eye(n) * 1e-6

    w0 = np.ones(n, dtype=np.float64) / n

    if w_prev is not None:
        w_prev = np.asarray(w_prev, dtype=np.float64).clip(0.0, 1.0)
        s = w_prev.sum()
        w_prev = (w_prev / s) if s > 0 else w0
        w0 = w_prev.copy()

    def neg_objective(w: np.ndarray) -> float:
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
                pen = gamma_turn * float(np.dot(d, d))
        return -(sharpe - pen)

    cons = ({"type": "eq", "fun": lambda w: np.sum(w) - 1.0},)
    bounds = [(0.0, 1.0)] * n

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


def estimate_cov_pastK(returns_TxN: np.ndarray, end_idx_inclusive: int, K: int) -> np.ndarray:
    start = max(0, end_idx_inclusive - K + 1)
    win = returns_TxN[start : end_idx_inclusive + 1]
    n = returns_TxN.shape[1]
    if win.shape[0] < 2:
        return np.eye(n) * 1e-3
    Sigma = np.cov(win, rowvar=False)
    return Sigma + np.eye(n) * 1e-6


@dataclass
class RebalanceOrder:
    symbol: str
    side: str            # "buy" or "sell"
    qty: float
    order_type: str = "market"
    tif: str = "day"
    target_weight: Optional[float] = None
    current_weight: Optional[float] = None
    delta_notional: Optional[float] = None


class MLReturnToWeightStrategy:
    """
    Strategy wrapper around the user's existing infer logic.
    Generates target weights and buy/sell rebalance orders.
    """

    def __init__(
        self,
        ckpt_path: str,
        etf_symbols: List[str],
        num_feat: int = 22,
        seq_len: int = 20,
        cov_window: int = 60,
        gamma_turn: float = 0.5,
        turnover_l1: bool = False,
        mu_ema: float = 0.0,
        min_trade_notional: float = 10.0,
        lot_size: float = 1e-6,
        device: str = "cpu",
    ):
        self.etf_symbols = etf_symbols
        self.seq_len = seq_len
        self.cov_window = int(cov_window)
        self.gamma_turn = float(gamma_turn)
        self.turnover_l1 = bool(turnover_l1)
        self.mu_ema_alpha = float(mu_ema)
        self.min_trade_notional = float(min_trade_notional)
        self.lot_size = float(lot_size)

        self.device = torch.device(device if (device == "cpu" or torch.cuda.is_available()) else "cpu")

        self.model = ReturnFullModel(
            num_feat=num_feat,
            seq_len=seq_len,
            patch_size=5,
            embed_dim=256,
            enc_layers=6,
            dropout=0.2,
            out_dim=len(etf_symbols),
        )
        sd = torch.load(ckpt_path, map_location="cpu")
        self.model.load_state_dict(sd, strict=True)
        self.model.to(self.device).eval()

        self.w_prev: Optional[np.ndarray] = None
        self.mu_ema_vec: Optional[np.ndarray] = None

    def reset(self) -> None:
        self.w_prev = None
        self.mu_ema_vec = None

    def predict_mu(self, x_window_20xF: torch.Tensor) -> np.ndarray:
        xw = window_normalize(x_window_20xF).unsqueeze(0).to(self.device)
        with torch.no_grad():
            r_hat = self.model(xw).squeeze(0).cpu().numpy()

        if self.mu_ema_alpha > 0.0:
            a = self.mu_ema_alpha
            if self.mu_ema_vec is None:
                self.mu_ema_vec = r_hat
            else:
                self.mu_ema_vec = a * self.mu_ema_vec + (1.0 - a) * r_hat
            return self.mu_ema_vec
        return r_hat

    def generate_target_weights(
        self,
        x_window_20xF: torch.Tensor,
        all_returns_TxN: np.ndarray,
        t_idx_inclusive: int,
    ) -> np.ndarray:
        mu = self.predict_mu(x_window_20xF)
        Sigma = estimate_cov_pastK(all_returns_TxN, end_idx_inclusive=t_idx_inclusive, K=self.cov_window)
        w = max_sharpe_long_only(
            mu=mu,
            Sigma=Sigma,
            w_prev=self.w_prev,
            gamma_turn=self.gamma_turn,
            turnover_l1=self.turnover_l1,
        )
        self.w_prev = w
        return w

    def generate_rebalance_orders(
        self,
        target_weights: np.ndarray,
        prices: Dict[str, float],
        current_positions: Dict[str, float],
        cash: float,
        total_equity: float,
    ) -> List[RebalanceOrder]:
        """
        Convert target weights into market rebalance orders.
        The caller can send sell orders first to free cash.
        """
        _ = cash  # Kept for API symmetry; validation is handled in order_manager.
        orders: List[RebalanceOrder] = []

        current_notional: Dict[str, float] = {}
        for sym in self.etf_symbols:
            px = float(prices[sym])
            qty = float(current_positions.get(sym, 0.0))
            current_notional[sym] = qty * px

        for i, sym in enumerate(self.etf_symbols):
            px = float(prices[sym])
            cur_val = current_notional[sym]
            target_val = float(target_weights[i]) * float(total_equity)
            delta_val = target_val - cur_val

            if abs(delta_val) < self.min_trade_notional:
                continue

            qty = abs(delta_val) / max(px, 1e-12)

            if self.lot_size > 0:
                qty = math.floor(qty / self.lot_size) * self.lot_size

            if qty <= 0:
                continue

            cur_w = 0.0 if total_equity <= 0 else (cur_val / total_equity)
            side = "buy" if delta_val > 0 else "sell"

            orders.append(
                RebalanceOrder(
                    symbol=sym,
                    side=side,
                    qty=float(qty),
                    order_type="market",
                    tif="day",
                    target_weight=float(target_weights[i]),
                    current_weight=float(cur_w),
                    delta_notional=float(delta_val),
                )
            )

        orders.sort(key=lambda o: 0 if o.side == "sell" else 1)
        return orders
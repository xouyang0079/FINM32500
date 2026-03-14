# -*- coding: utf-8 -*-
"""
Downstream Task: Predict next-day ETF returns r_{t+1} (5-dim)

Modified to reduce "mean-collapse" (mu_hat nearly constant):
- smaller Huber beta for daily-return scale
- volatility-weighted Huber loss (per asset)
- lighter regularization
- collapse diagnostics (pred std vs target std)

Scheme B added:
- cross-sectional correlation auxiliary loss (per-sample over 5 assets)
"""

import os
import glob
import math
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import LambdaLR


# -------------------------
# Utils
# -------------------------
def find_pack(pt_dir: str, split: str) -> str:
    cand = sorted(glob.glob(os.path.join(pt_dir, f"market_{split}_*.pt")))
    if not cand:
        raise FileNotFoundError(f"Cannot find pack for split='{split}' under {pt_dir}")
    return cand[0]


def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def window_normalize(w: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    mu = w.mean(dim=0, keepdim=True)
    std = w.std(dim=0, keepdim=True, unbiased=False).add(eps)
    return (w - mu) / std


# -------------------------
# MAE backbone skeleton (must match MAE checkpoint keys)
# -------------------------
class MacroConditionedMAE(nn.Module):
    def __init__(
        self,
        num_feat: int = 22,
        seq_len: int = 20,
        patch_size: int = 5,
        embed_dim: int = 256,
        enc_layers: int = 6,
        dec_layers: int = 3,
        dropout: float = 0.1,   # changed from 0.2
    ):
        super().__init__()
        if seq_len % patch_size != 0:
            raise ValueError("seq_len must be divisible by patch_size")

        self.num_feat = num_feat
        self.seq_len = seq_len
        self.patch_size = patch_size
        self.num_patches = seq_len // patch_size
        patch_dim = patch_size * num_feat

        # required to match MAE checkpoint keys
        feat_mask = torch.zeros(num_feat)
        feat_mask[:10] = 1.0
        if num_feat >= 22:
            feat_mask[21] = 1.0
        self.register_buffer("feat_mask", feat_mask)  # [F]

        patch_mask = feat_mask.repeat(patch_size)  # [patch_dim]
        self.register_buffer("patch_mask", patch_mask)

        self.patch_mask_token = nn.Parameter(torch.zeros(1, 1, patch_dim))

        self.patch_proj = nn.Linear(patch_dim, embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))

        enc_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=8,
            dim_feedforward=1024,
            batch_first=True,
            dropout=dropout,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=enc_layers)

        # keep decoder parts so strict=True works for MAE checkpoint loading
        self.decoder_proj = nn.Linear(embed_dim, 128)
        self.dec_pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, 128))
        dec_layer = nn.TransformerEncoderLayer(
            d_model=128,
            nhead=4,
            dim_feedforward=512,
            batch_first=True,
            dropout=dropout,
        )
        self.decoder = nn.TransformerEncoder(dec_layer, num_layers=dec_layers)
        self.head = nn.Linear(128, patch_dim)


# -------------------------
# Downstream FULL model (encoder + head in one module)
# -------------------------
class ReturnFullModel(nn.Module):
    def __init__(
        self,
        num_feat: int = 22,
        seq_len: int = 20,
        patch_size: int = 5,
        embed_dim: int = 256,
        enc_layers: int = 6,
        dropout: float = 0.1,   # changed from 0.2
        out_dim: int = 5,
    ):
        super().__init__()
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
        # x: [B,20,22]
        B, T, F = x.shape
        if T != self.seq_len or F != self.num_feat:
            raise ValueError(
                f"Input shape mismatch: got {tuple(x.shape)} expected [B,{self.seq_len},{self.num_feat}]"
            )

        P = self.seq_len // self.patch_size
        patch_dim = self.patch_size * self.num_feat

        x_patches = x.reshape(B, P, patch_dim)
        tokens = self.patch_proj(x_patches) + self.pos_embed
        latent = self.encoder(tokens)  # [B,P,embed_dim]
        emb = latent.mean(dim=1)       # [B,embed_dim]
        return self.head(emb)          # [B,5]

    @torch.no_grad()
    def init_from_mae(self, mae_ckpt_path: str):
        """
        Load MAE checkpoint and copy encoder-related weights into this full model.
        Initialization only. Then finetune all params.
        """
        mae = MacroConditionedMAE(
            num_feat=self.num_feat,
            seq_len=self.seq_len,
            patch_size=self.patch_size,
            embed_dim=self.embed_dim,
            enc_layers=len(self.encoder.layers),
            dec_layers=3,
            dropout=0.1,
        )
        sd = torch.load(mae_ckpt_path, map_location="cpu")
        mae.load_state_dict(sd, strict=True)

        # copy encoder weights
        self.patch_proj.load_state_dict(mae.patch_proj.state_dict(), strict=True)
        self.pos_embed.copy_(mae.pos_embed)
        self.encoder.load_state_dict(mae.encoder.state_dict(), strict=True)

        # copy compatibility params
        self.patch_mask_token.copy_(mae.patch_mask_token)
        self.feat_mask.copy_(mae.feat_mask)
        self.patch_mask.copy_(mae.patch_mask)


# -------------------------
# Dataset
# -------------------------
class ReturnPredDataset(Dataset):
    """
    Each sample:
      input  : X[idx:idx+20] (20,22) window-normalized
      target : R[idx+20]     (5,)   next-day raw log returns
    """
    def __init__(self, pt_pack: str, window_size: int = 20):
        pack = torch.load(pt_pack, map_location="cpu")
        if not isinstance(pack, dict) or "X" not in pack or "R" not in pack:
            raise TypeError(f"Bad pack format: {pt_pack}")

        self.meta = pack.get("meta", {})
        X = pack["X"]
        R = pack["R"]

        self.X = torch.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0).float()
        self.R = torch.nan_to_num(R, nan=0.0, posinf=0.0, neginf=0.0).float()

        self.window_size = int(window_size)
        if self.window_size != 20:
            raise ValueError("This script assumes window_size=20 (past19+today).")
        if len(self.X) != len(self.R):
            raise ValueError("X and R must have same length.")
        if len(self.X) < self.window_size + 1:
            raise ValueError(f"T too short: T={len(self.X)} need at least {self.window_size+1}")

        self.max_idx = len(self.X) - (self.window_size + 1)
        print(
            f"[DATA] {self.meta.get('split')} X={tuple(self.X.shape)} "
            f"R={tuple(self.R.shape)} samples={len(self)}"
        )

    def __len__(self):
        return max(0, self.max_idx + 1)

    def __getitem__(self, idx: int):
        w = self.X[idx: idx + self.window_size]  # [20,22]
        x = window_normalize(w)                  # [20,22]
        y = self.R[idx + self.window_size]       # [5]
        return x, y


# -------------------------
# Loss: volatility-weighted Huber
# -------------------------
class WeightedHuberLoss(nn.Module):
    """
    Per-asset weighted SmoothL1 (Huber).
    weights: shape [5], usually inverse-vol normalized.
    """
    def __init__(self, beta: float = 0.01, weights: torch.Tensor = None):
        super().__init__()
        self.beta = float(beta)
        if weights is None:
            self.register_buffer("weights", torch.ones(1, 5))
        else:
            self.register_buffer("weights", weights.float().view(1, -1))

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff = pred - target
        abs_diff = diff.abs()
        beta = self.beta

        loss = torch.where(
            abs_diff < beta,
            0.5 * (diff ** 2) / beta,
            abs_diff - 0.5 * beta
        )
        loss = loss * self.weights
        return loss.mean()


# -------------------------
# Scheme B: cross-sectional correlation auxiliary loss
# -------------------------
class BatchCrossSectionCorrLoss(nn.Module):
    """
    For each sample in batch, compute Pearson corr(pred[i,:], target[i,:])
    across asset dimension (D=5), then average over batch.

    Returns: 1 - mean_corr (lower is better)
    """
    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # pred, target: [B, D]
        pred_c = pred - pred.mean(dim=1, keepdim=True)
        targ_c = target - target.mean(dim=1, keepdim=True)

        num = (pred_c * targ_c).sum(dim=1)  # [B]
        den = torch.sqrt(
            (pred_c ** 2).sum(dim=1) * (targ_c ** 2).sum(dim=1) + self.eps
        )  # [B]

        corr = num / (den + self.eps)
        corr = torch.clamp(corr, -1.0, 1.0)
        return 1.0 - corr.mean()


# -------------------------
# Eval
# -------------------------
@torch.no_grad()
def eval_loss(model: nn.Module, loader: DataLoader, device: torch.device, loss_fn: nn.Module) -> float:
    model.eval()
    s = 0.0
    n = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        pred = model(x)
        loss = loss_fn(pred, y)
        s += float(loss.item()) * x.size(0)
        n += x.size(0)
    return s / max(1, n)


@torch.no_grad()
def eval_corr_loss(model: nn.Module, loader: DataLoader, device: torch.device, corr_loss_fn: nn.Module) -> float:
    model.eval()
    s = 0.0
    n = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        pred = model(x)
        loss_corr = corr_loss_fn(pred, y)
        s += float(loss_corr.item()) * x.size(0)
        n += x.size(0)
    return s / max(1, n)


@torch.no_grad()
def eval_pred_stats(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return per-asset std of predictions and targets on a loader.
    Useful to detect collapse (pred std too small).
    """
    model.eval()
    pred_list = []
    tgt_list = []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        pred = model(x).cpu()
        pred_list.append(pred)
        tgt_list.append(y)

    P = torch.cat(pred_list, dim=0).numpy()
    Y = torch.cat(tgt_list, dim=0).numpy()
    return P.std(axis=0), Y.std(axis=0)


# -------------------------
# Train
# -------------------------
def train():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data
    pt_dir = "../data/data_pt"
    train_pack = find_pack(pt_dir, "train")
    val_pack = find_pack(pt_dir, "val")

    train_ds = ReturnPredDataset(train_pack, window_size=20)
    val_ds = ReturnPredDataset(val_pack, window_size=20)

    # smaller batch often helps avoid over-smoothing in noisy financial prediction
    BATCH_SIZE = 32

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        pin_memory=torch.cuda.is_available(),
    )
    # for fair train eval & std diagnostics
    train_eval_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        drop_last=False,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        drop_last=False,
        pin_memory=torch.cuda.is_available(),
    )

    # build per-asset inverse-vol weights from train targets
    # dataset targets are R[idx+20], so all possible targets correspond to R[20:]
    R_train_targets = train_ds.R[20:]  # [N,5]
    target_std = R_train_targets.std(dim=0, unbiased=False).clamp_min(1e-6)
    inv_vol = (1.0 / target_std)
    inv_vol = inv_vol / inv_vol.mean()  # normalize around 1
    print(f"[LOSS] target_std={target_std.numpy()} inv_vol_w={inv_vol.numpy()}")

    # Model + MAE init
    mae_init_ckpt = "../model/market_encoder_best_ema.pth"  # init only
    model = ReturnFullModel(
        num_feat=22,
        seq_len=20,
        patch_size=5,
        embed_dim=256,
        enc_layers=6,
        dropout=0.1,
        out_dim=5,
    )
    model.init_from_mae(mae_init_ckpt)
    model = model.to(device)

    # train ALL params
    params = [p for p in model.parameters() if p.requires_grad]
    print(f"[INFO] Trainable params: {sum(p.numel() for p in params):,}")

    EPOCHS = 100
    LR = 2e-4
    WEIGHT_DECAY = 0.01  # reduced from 0.05

    optimizer = optim.AdamW(params, lr=LR, weight_decay=WEIGHT_DECAY)

    # key change: smaller beta for daily return scale
    loss_fn = WeightedHuberLoss(beta=0.01, weights=inv_vol).to(device)

    # Scheme B auxiliary loss
    corr_loss_fn = BatchCrossSectionCorrLoss().to(device)
    LAMBDA_CORR = 0.2  # try 0.1~0.3; start from 0.2

    # Scheduler: warmup + cosine decay (per-step)
    steps_per_epoch = len(train_loader)
    total_steps = EPOCHS * steps_per_epoch
    warmup_steps = int(0.08 * total_steps)
    min_lr_ratio = 0.01

    def lr_lambda(current_step: int):
        if current_step < warmup_steps:
            # avoid exact 0 lr at first step
            return max(1e-3, float(current_step) / float(max(1, warmup_steps)))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

    best_val = float("inf")
    save_path = "../model/downstream_return_full.pth"

    print(f"\n--- Train Return FULL model on {device} | save={save_path} ---")
    print(f"[AUX] Using corr loss: total = huber + {LAMBDA_CORR:.3f} * corr_loss")

    for ep in range(1, EPOCHS + 1):
        model.train()
        s = 0.0
        s_huber = 0.0
        s_corr = 0.0
        n = 0

        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            pred = model(x)

            loss_huber = loss_fn(pred, y)
            loss_corr = corr_loss_fn(pred, y)
            loss = loss_huber + LAMBDA_CORR * loss_corr

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            s += float(loss.item()) * x.size(0)
            s_huber += float(loss_huber.item()) * x.size(0)
            s_corr += float(loss_corr.item()) * x.size(0)
            n += x.size(0)

        tr_running = s / max(1, n)
        tr_running_huber = s_huber / max(1, n)
        tr_running_corr = s_corr / max(1, n)

        # fair comparison: eval() huber loss on train and val
        tr_eval = eval_loss(model, train_eval_loader, device, loss_fn)
        va = eval_loss(model, val_loader, device, loss_fn)

        # auxiliary metric (not used for model selection by default)
        va_corr = eval_corr_loss(model, val_loader, device, corr_loss_fn)

        if va < best_val:
            best_val = va
            torch.save(model.state_dict(), save_path)

        if ep == 1 or ep % 5 == 0:
            cur_lr = optimizer.param_groups[0]["lr"]

            # collapse diagnostics
            pred_std_tr, tgt_std_tr = eval_pred_stats(model, train_eval_loader, device)
            pred_std_va, tgt_std_va = eval_pred_stats(model, val_loader, device)

            print(
                f"Epoch {ep:03d} | lr={cur_lr:.6e} | "
                f"train_total={tr_running:.6f} | train_huber={tr_running_huber:.6f} | train_corr={tr_running_corr:.6f} | "
                f"train_eval={tr_eval:.6f} | val={va:.6f} | val_corr={va_corr:.6f} | best_val={best_val:.6f}"
            )
            print(
                f"   [STD train] pred={np.round(pred_std_tr, 6)} "
                f"tgt={np.round(tgt_std_tr, 6)} "
                f"| ratio={np.round(pred_std_tr / (tgt_std_tr + 1e-12), 3)}"
            )
            print(
                f"   [STD val  ] pred={np.round(pred_std_va, 6)} "
                f"tgt={np.round(tgt_std_va, 6)} "
                f"| ratio={np.round(pred_std_va / (tgt_std_va + 1e-12), 3)}"
            )

    print(f"\nDone. Best val={best_val:.6f} saved to {save_path}")
    print("Infer should load ONLY this .pth (no mae_ckpt needed).")


if __name__ == "__main__":
    train()
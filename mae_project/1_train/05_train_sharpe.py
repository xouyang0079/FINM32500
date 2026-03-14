# -*- coding: utf-8 -*-
"""
Downstream Task 1 (FULL finetune): Predict portfolio weights directly.

End-to-End Portfolio Optimization:
- Dataset outputs a trajectory of length `seq_len`.
- Model outputs allocation weights (Softmax).
- Loss = negative Sharpe Ratio computed on *net* portfolio returns over trajectories
         + turnover penalty (in return units) to reduce churn.

Notes:
- Uses ONE checkpoint for downstream model.
- MAE ckpt is used ONLY for initialization (optional).

Gradual Unfreeze Plan (80 epochs total):
- Always train: head
- Encoder has 6 layers. Unfreeze from last -> first, every 10 epochs:
    ep 01-10: 0 encoder layers
    ep 11-20: last 1 layer
    ep 21-30: last 2 layers
    ep 31-40: last 3 layers
    ep 41-50: last 4 layers
    ep 51-60: last 5 layers
    ep 61-70: last 6 layers (all)
- patch_proj + pos_embed are unfrozen after encoder reaches 3 layers (from ep 31).
- IMPORTANT: optimizer param_groups are created ONCE (no add_param_group) to avoid scheduler crash.
"""

import os
import glob
import math
from typing import Tuple, Dict

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
        dropout: float = 0.1,
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
            nn.Softmax(dim=-1),  # output weights sum to 1
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
        emb = latent.mean(dim=1)  # [B,embed_dim]
        return self.head(emb)  # [B,5]

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

        self.patch_proj.load_state_dict(mae.patch_proj.state_dict(), strict=True)
        self.pos_embed.copy_(mae.pos_embed)
        self.encoder.load_state_dict(mae.encoder.state_dict(), strict=True)

        self.patch_mask_token.copy_(mae.patch_mask_token)
        self.feat_mask.copy_(mae.feat_mask)
        self.patch_mask.copy_(mae.patch_mask)


# -------------------------
# Trajectory Dataset (Block Sampling)
# -------------------------
class PortfolioTrajectoryDataset(Dataset):
    """
    Each sample outputs a continuous temporal block (trajectory):
      input  : X_seq [seq_len, window_size, 22] (normalized rolling windows)
      target : R_seq [seq_len, 5] (corresponding next-day log returns)
    """

    def __init__(self, pt_pack: str, window_size: int = 20, seq_len: int = 20):
        pack = torch.load(pt_pack, map_location="cpu")
        if not isinstance(pack, dict) or "X" not in pack or "R" not in pack:
            raise TypeError(f"Bad pack format: {pt_pack}")

        self.meta = pack.get("meta", {})
        X = pack["X"]
        R = pack["R"]

        self.X = torch.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0).float()
        self.R = torch.nan_to_num(R, nan=0.0, posinf=0.0, neginf=0.0).float()

        self.window_size = int(window_size)
        self.seq_len = int(seq_len)

        if len(self.X) != len(self.R):
            raise ValueError("X and R must have same length.")

        # Need enough data to form at least one full sequence of window_size + seq_len
        self.max_start_idx = len(self.X) - self.window_size - self.seq_len
        if self.max_start_idx < 0:
            raise ValueError(f"T too short: T={len(self.X)} need at least {self.window_size + self.seq_len}")

        print(
            f"[DATA] {self.meta.get('split')} X={tuple(self.X.shape)} "
            f"seq_len={self.seq_len} samples={len(self)}"
        )

    def __len__(self):
        return max(0, self.max_start_idx + 1)

    def __getitem__(self, idx: int):
        X_seq = torch.zeros((self.seq_len, self.window_size, self.X.shape[1]), dtype=torch.float32)
        R_seq = torch.zeros((self.seq_len, self.R.shape[1]), dtype=torch.float32)

        for i in range(self.seq_len):
            w = self.X[idx + i : idx + i + self.window_size]  # [20, 22]
            X_seq[i] = window_normalize(w)
            R_seq[i] = self.R[idx + i + self.window_size]  # [5]  (next-day return)

        return X_seq, R_seq


# -------------------------
# End-to-End Sharpe Loss (net returns + one-way turnover; stable Sharpe)
# -------------------------
class EndToEndSharpeLoss(nn.Module):
    """
    weights:        [B, S, N]
    future_returns: [B, S, N]  (next-day returns aligned with weights)

    Compute net portfolio return:
      r_p[t] = w_t · r_{t+1} - cost_rate * one_way_turnover_t
    where:
      one_way_turnover_t = 0.5 * sum_i |w_t(i) - w_{t-1}(i)|

    Sharpe stability:
      compute Sharpe on (B*S) pooled samples instead of per-trajectory ratio-then-avg.
    """

    def __init__(
        self,
        cost_rate: float = 0.0,  # transaction-cost proxy in return units
        ann_factor: int = 252,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.cost_rate = float(cost_rate)
        self.ann_factor = int(ann_factor)
        self.eps = float(eps)

    def forward(self, weights: torch.Tensor, future_returns: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        if weights.shape != future_returns.shape:
            raise ValueError(f"Shape mismatch: weights={tuple(weights.shape)} returns={tuple(future_returns.shape)}")

        # gross daily portfolio return: [B, S]
        port_gross = torch.sum(weights * future_returns, dim=-1)

        # one-way turnover per step: [B, S]
        if weights.shape[1] > 1:
            dw = (weights[:, 1:, :] - weights[:, :-1, :]).abs().sum(dim=-1) * 0.5  # [B, S-1]
            turnover = torch.cat([torch.zeros_like(dw[:, :1]), dw], dim=1)  # pad t=0 with 0
        else:
            turnover = torch.zeros_like(port_gross)

        # net return: [B, S]
        port_net = port_gross - self.cost_rate * turnover

        # pooled Sharpe across all (B*S) samples for stability
        flat = port_net.reshape(-1)
        mu = flat.mean()
        sigma = flat.std(unbiased=False) + self.eps
        sharpe = (mu / sigma) * math.sqrt(self.ann_factor)

        loss = -sharpe

        stats = {
            "sharpe": float(sharpe.detach().cpu().item()),
            "turnover": float(turnover.mean().detach().cpu().item()),
            "ann_ret": float((mu.detach().cpu().item()) * self.ann_factor),
            "gross_mu_d": float(port_gross.mean().detach().cpu().item()),
            "net_mu_d": float(mu.detach().cpu().item()),
            "net_vol_d": float(sigma.detach().cpu().item()),
        }
        return loss, stats


# -------------------------
# Eval
# -------------------------
@torch.no_grad()
def eval_sharpe_loss(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    loss_fn: EndToEndSharpeLoss,
) -> Tuple[float, float, float]:
    model.eval()
    total_loss, total_sharpe, total_turnover = 0.0, 0.0, 0.0
    n = 0

    for x_seq, r_seq in loader:
        x_seq = x_seq.to(device, non_blocking=True)  # [B,S,20,22]
        r_seq = r_seq.to(device, non_blocking=True)  # [B,S,5]

        B, S, T, F = x_seq.shape
        x_flat = x_seq.view(B * S, T, F)

        weights_flat = model(x_flat)  # [B*S,5]
        weights_seq = weights_flat.view(B, S, -1)  # [B,S,5]

        loss, stats = loss_fn(weights_seq, r_seq)

        total_loss += float(loss.item()) * B
        total_sharpe += float(stats["sharpe"]) * B
        total_turnover += float(stats["turnover"]) * B
        n += B

    return total_loss / max(1, n), total_sharpe / max(1, n), total_turnover / max(1, n)


# -------------------------
# Gradual unfreeze helper
# -------------------------
def apply_gradual_unfreeze(
    model: ReturnFullModel,
    ep: int,
    unfreeze_every: int,
    unfreeze_patch_pos_after_k: int = 3,
) -> Dict[str, int]:
    """
    Unfreeze encoder layers from last to first over time.
    Also optionally unfreeze patch_proj and pos_embed after k layers are unfrozen.

    Returns a small dict for logging.
    """
    L = len(model.encoder.layers)  # expected 6

    # k=0 for ep 1..unfreeze_every
    # k=1 for next block, etc.
    k = (ep - 1) // unfreeze_every
    num_layers_to_unfreeze = int(min(max(k, 0), L))

    # First freeze all encoder layers every epoch (idempotent)
    for p in model.encoder.parameters():
        p.requires_grad = False

    # Unfreeze last num_layers_to_unfreeze layers
    if num_layers_to_unfreeze > 0:
        start = L - num_layers_to_unfreeze
        for li in range(start, L):
            for p in model.encoder.layers[li].parameters():
                p.requires_grad = True

    # patch_proj + pos_embed schedule
    patch_pos_on = num_layers_to_unfreeze >= int(unfreeze_patch_pos_after_k)
    for p in model.patch_proj.parameters():
        p.requires_grad = patch_pos_on
    model.pos_embed.requires_grad = patch_pos_on

    return {
        "unfrozen_encoder_layers": num_layers_to_unfreeze,
        "patch_pos_on": int(patch_pos_on),
    }


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

    # CONFIG
    SEQ_LEN = 64
    WINDOW_SIZE = 20
    BATCH_SIZE = 32

    # Transaction-cost proxy:
    # - cost_rate multiplies one-way turnover (0..1) and subtracts from daily return.
    # - Example: cost_rate=0.0005 means 5 bps cost per 100% one-way turnover.
    COST_RATE = 0.0005

    # Gradual unfreeze schedule
    EPOCHS = 80
    UNFREEZE_EVERY = 10               # every 10 epochs unfreeze 1 more encoder layer (from last -> first)
    UNFREEZE_PATCH_POS_AFTER_K = 3    # unfreeze patch_proj + pos_embed after >=3 encoder layers are unfrozen

    # LR
    LR_HEAD = 5e-4
    LR_ENCODER = 5e-5
    WEIGHT_DECAY = 0.01

    train_ds = PortfolioTrajectoryDataset(train_pack, window_size=WINDOW_SIZE, seq_len=SEQ_LEN)
    val_ds = PortfolioTrajectoryDataset(val_pack, window_size=WINDOW_SIZE, seq_len=SEQ_LEN)

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, pin_memory=torch.cuda.is_available()
    )
    train_eval_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=False, drop_last=False, pin_memory=torch.cuda.is_available()
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False, drop_last=False, pin_memory=torch.cuda.is_available()
    )

    # Model + MAE init
    mae_init_ckpt = "../model/market_encoder_best_ema.pth"  # init only
    model = ReturnFullModel(
        num_feat=22,
        seq_len=WINDOW_SIZE,
        patch_size=5,
        embed_dim=256,
        enc_layers=6,
        dropout=0.1,
        out_dim=5,
    )
    if os.path.isfile(mae_init_ckpt):
        model.init_from_mae(mae_init_ckpt)
    else:
        print(f"[WARN] MAE ckpt not found: {mae_init_ckpt} (skip init)")

    model = model.to(device)

    # Build optimizer param groups ONCE (do not add groups later).
    # We include all encoder + patch_proj params in the optimizer from the start, even if frozen.
    encoder_params = list(model.encoder.parameters()) + list(model.patch_proj.parameters()) + [model.pos_embed]
    head_params = []
    for name, p in model.named_parameters():
        if name.startswith("encoder.") or name.startswith("patch_proj.") or name == "pos_embed":
            continue
        head_params.append(p)

    optimizer = optim.AdamW(
        [
            {"params": head_params, "lr": LR_HEAD, "weight_decay": WEIGHT_DECAY},
            {"params": encoder_params, "lr": LR_ENCODER, "weight_decay": WEIGHT_DECAY},
        ],
    )

    # Loss
    loss_fn = EndToEndSharpeLoss(cost_rate=COST_RATE, ann_factor=252, eps=1e-6).to(device)

    # Scheduler: warmup + cosine decay (works because param_groups count is constant)
    steps_per_epoch = len(train_loader)
    total_steps = EPOCHS * steps_per_epoch
    warmup_steps = int(0.08 * total_steps)
    min_lr_ratio = 0.01

    def lr_lambda(current_step: int):
        if current_step < warmup_steps:
            return max(1e-3, float(current_step) / float(max(1, warmup_steps)))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

    best_val_loss = float("inf")
    save_path = "../model/downstream_e2e_sharpe.pth"

    print(f"\n--- Train E2E Portfolio Model on {device} | save={save_path} ---")
    print(
        f"[CONFIG] EPOCHS={EPOCHS} | Traj SEQ_LEN={SEQ_LEN} | COST_RATE={COST_RATE} | "
        f"UNFREEZE_EVERY={UNFREEZE_EVERY} | PATCH_POS_AFTER_K={UNFREEZE_PATCH_POS_AFTER_K} | "
        f"LR_HEAD={LR_HEAD} | LR_ENCODER={LR_ENCODER}"
    )

    for ep in range(1, EPOCHS + 1):
        # Apply gradual unfreeze schedule (idempotent; safe to call every epoch)
        uinfo = apply_gradual_unfreeze(
            model,
            ep=ep,
            unfreeze_every=UNFREEZE_EVERY,
            unfreeze_patch_pos_after_k=UNFREEZE_PATCH_POS_AFTER_K,
        )

        if ep == 1 or ep % UNFREEZE_EVERY == 1:
            print(
                f"[UNFREEZE_PLAN] ep={ep:03d} unfrozen_encoder_layers={uinfo['unfrozen_encoder_layers']} "
                f"patch+pos={'ON' if uinfo['patch_pos_on'] else 'OFF'}"
            )

        model.train()
        s_loss, s_sharpe, s_turn = 0.0, 0.0, 0.0
        n = 0

        for x_seq, r_seq in train_loader:
            x_seq = x_seq.to(device, non_blocking=True)
            r_seq = r_seq.to(device, non_blocking=True)

            B, S, T, F = x_seq.shape
            x_flat = x_seq.view(B * S, T, F)

            optimizer.zero_grad(set_to_none=True)

            weights_flat = model(x_flat)
            weights_seq = weights_flat.view(B, S, -1)

            loss, stats = loss_fn(weights_seq, r_seq)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            s_loss += float(loss.item()) * B
            s_sharpe += float(stats["sharpe"]) * B
            s_turn += float(stats["turnover"]) * B
            n += B

        tr_loss = s_loss / max(1, n)
        tr_sharpe = s_sharpe / max(1, n)
        tr_turn = s_turn / max(1, n)

        # Evaluate
        va_loss, va_sharpe, va_turn = eval_sharpe_loss(model, val_loader, device, loss_fn)

        if va_loss < best_val_loss:
            best_val_loss = va_loss
            torch.save(model.state_dict(), save_path)

        if ep == 1 or ep % 5 == 0:
            # show each group's current lr for debugging
            lrs = [g["lr"] for g in optimizer.param_groups]
            te_loss, te_sharpe, te_turn = eval_sharpe_loss(model, train_eval_loader, device, loss_fn)
            print(
                f"Epoch {ep:03d} | lrs={','.join([f'{x:.2e}' for x in lrs])} | "
                f"Train: loss={tr_loss:.4f}, sharpe={tr_sharpe:.4f}, turn={tr_turn:.4f} | "
                f"TrainEval: loss={te_loss:.4f}, sharpe={te_sharpe:.4f}, turn={te_turn:.4f} | "
                f"Val: loss={va_loss:.4f}, sharpe={va_sharpe:.4f}, turn={va_turn:.4f} | "
                f"BestValLoss={best_val_loss:.4f}"
            )

    print(f"\nDone. Best val loss={best_val_loss:.6f} saved to {save_path}")


if __name__ == "__main__":
    train()
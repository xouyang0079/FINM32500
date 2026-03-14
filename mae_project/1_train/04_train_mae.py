# -*- coding: utf-8 -*-
import os
import math
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# ==============================================================================
# 1) DATASET (load dict pack)
# ==============================================================================
class MarketMAEDataset(Dataset):
    def __init__(self, pt_pack: str, window_size: int = 20, return_stats: bool = False, return_meta: bool = False):
        if not os.path.exists(pt_pack):
            raise FileNotFoundError(f"PT pack not found: {pt_pack}")

        pack = torch.load(pt_pack, map_location="cpu")
        if not isinstance(pack, dict) or "X" not in pack:
            raise TypeError(f"Expected dict with key 'X' in {pt_pack}, got keys={list(pack.keys()) if isinstance(pack, dict) else type(pack)}")

        X = pack["X"]
        if not isinstance(X, torch.Tensor):
            raise TypeError(f"Expected pack['X'] torch.Tensor, got {type(X)}")
        if X.ndim != 2:
            raise ValueError(f"Expected X shape [T,F], got {tuple(X.shape)}")

        self.meta = pack.get("meta", {})
        self.dates = pack.get("dates", None)

        self.data = torch.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0).float()
        self.window_size = int(window_size)
        self.return_stats = bool(return_stats)
        self.return_meta = bool(return_meta)

        if self.window_size <= 0:
            raise ValueError("window_size must be positive")
        if len(self.data) < self.window_size:
            raise ValueError(f"Data too short: T={len(self.data)} < window_size={self.window_size}")

        # Optional sanity prints
        if self.meta:
            print(f"[DATA] split={self.meta.get('split')} X={tuple(self.data.shape)} range={self.meta.get('start_date')}..{self.meta.get('end_date')}")

    def __len__(self):
        return max(0, len(self.data) - self.window_size + 1)

    def __getitem__(self, idx: int):
        w = self.data[idx: idx + self.window_size]  # [W,F]

        # Window normalization (your preference)
        mu = w.mean(dim=0, keepdim=True)
        std = w.std(dim=0, keepdim=True, unbiased=False).add(1e-5)
        x = (w - mu) / std

        if self.return_stats or self.return_meta:
            out = [x]
            if self.return_stats:
                out += [mu, std]
            if self.return_meta:
                # Provide date range for this window if available
                if isinstance(self.dates, list) and len(self.dates) == len(self.data):
                    start = self.dates[idx]
                    end = self.dates[idx + self.window_size - 1]
                else:
                    start, end = None, None
                out += [start, end]
            return tuple(out)

        return x


# ==============================================================================
# 2) EMA
# ==============================================================================
class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = float(decay)
        self.ema_model = copy.deepcopy(model).eval()
        for p in self.ema_model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module):
        d = self.decay
        msd = model.state_dict()
        esd = self.ema_model.state_dict()
        for k in esd.keys():
            esd[k].mul_(d).add_(msd[k], alpha=1.0 - d)
        self.ema_model.load_state_dict(esd, strict=True)


# ==============================================================================
# 3) LR schedule + mask schedule
# ==============================================================================
def lr_warmup_cosine(step: int, total_steps: int, warmup_steps: int, max_lr: float, min_lr: float) -> float:
    if step < warmup_steps:
        return max_lr * (step + 1) / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, (total_steps - warmup_steps))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + (max_lr - min_lr) * cosine


def mask_ratio_schedule(epoch: int, total_epochs: int, start: float = 0.15, end: float = 0.50) -> float:
    t = epoch / max(1, total_epochs - 1)
    c = 0.5 * (1.0 - math.cos(math.pi * t))  # 0->1
    return start + (end - start) * c


# ==============================================================================
# 4) MODEL: Macro-conditioned MAE v2 (unchanged)
# ==============================================================================
class MacroConditionedMAE(nn.Module):
    def __init__(
        self,
        num_feat: int = 22,
        seq_len: int = 20,
        patch_size: int = 5,
        embed_dim: int = 256,
        enc_layers: int = 6,
        dec_layers: int = 3,
        dropout: float = 0.2,
    ):
        super().__init__()
        if seq_len % patch_size != 0:
            raise ValueError("seq_len must be divisible by patch_size")

        self.num_feat = num_feat
        self.seq_len = seq_len
        self.patch_size = patch_size
        self.num_patches = seq_len // patch_size
        patch_dim = patch_size * num_feat

        feat_mask = torch.zeros(num_feat)
        feat_mask[:10] = 1.0
        if num_feat >= 22:
            feat_mask[21] = 1.0
        self.register_buffer("feat_mask", feat_mask)  # [F]

        patch_mask = feat_mask.repeat(patch_size)      # [patch_dim]
        self.register_buffer("patch_mask", patch_mask)

        self.patch_mask_token = nn.Parameter(torch.zeros(1, 1, patch_dim))

        self.patch_proj = nn.Linear(patch_dim, embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        enc_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=8, dim_feedforward=1024, batch_first=True, dropout=dropout
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=enc_layers)

        self.decoder_proj = nn.Linear(embed_dim, 128)
        self.dec_pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, 128))
        dec_layer = nn.TransformerEncoderLayer(
            d_model=128, nhead=4, dim_feedforward=512, batch_first=True, dropout=dropout
        )
        self.decoder = nn.TransformerEncoder(dec_layer, num_layers=dec_layers)
        self.head = nn.Linear(128, patch_dim)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        nn.init.normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.dec_pos_embed, std=0.02)
        nn.init.normal_(self.patch_mask_token, std=0.02)
        with torch.no_grad():
            self.patch_mask_token.mul_(self.patch_mask.view(1, 1, -1))

    def _make_patch_mask(self, B: int, device: torch.device, mask_ratio: float, mask_seed: int | None):
        P = self.num_patches
        if (not self.training) and (mask_seed is not None):
            g = torch.Generator(device=device)
            g.manual_seed(int(mask_seed))
            noise = torch.rand(B, P, device=device, generator=g)
        else:
            noise = torch.rand(B, P, device=device)

        len_keep = int(P * (1.0 - mask_ratio))
        len_keep = max(1, min(len_keep, P - 1))

        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        mask = torch.ones([B, P], device=device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)
        return mask

    def forward(
        self,
        x: torch.Tensor,
        mask_ratio: float = 0.20,
        mask_seed: int | None = None,
        noise_std: float = 0.02,
        visible_loss_weight: float = 0.10,
    ):
        B, T, F = x.shape
        if T != self.seq_len or F != self.num_feat:
            raise ValueError(f"Input shape mismatch: got {tuple(x.shape)} expected [B,{self.seq_len},{self.num_feat}]")

        P = self.num_patches
        ps = self.patch_size
        patch_dim = ps * F

        x_patches = x.reshape(B, P, patch_dim)

        patch_mask = self.patch_mask.view(1, 1, -1)
        mask_token = self.patch_mask_token * patch_mask

        mask_patches = self._make_patch_mask(B, x.device, mask_ratio, mask_seed)
        mask_p = mask_patches.view(B, P, 1)

        x_masked = x_patches * (1.0 - patch_mask) + mask_token * patch_mask
        x_in = x_patches * (1.0 - mask_p) + x_masked * mask_p

        if self.training and noise_std > 0:
            noise = torch.randn_like(x_in) * noise_std
            x_in = x_in + noise * patch_mask

        tokens = self.patch_proj(x_in) + self.pos_embed
        latent = self.encoder(tokens)
        dec_in = self.decoder_proj(latent) + self.dec_pos_embed
        dec_out = self.decoder(dec_in)
        preds = self.head(dec_out)

        loss_fn = nn.HuberLoss(reduction="none", delta=1.0)
        diff = loss_fn(preds, x_patches) * patch_mask

        masked_diff = diff * mask_p
        masked_denom = (mask_p.sum() * patch_mask.sum()).clamp_min(1.0)
        masked_loss = masked_diff.sum() / masked_denom

        visible_mask = (1.0 - mask_p)
        visible_diff = diff * visible_mask
        visible_denom = (visible_mask.sum() * patch_mask.sum()).clamp_min(1.0)
        visible_loss = visible_diff.sum() / visible_denom

        return masked_loss + visible_loss_weight * visible_loss


# ==============================================================================
# 5) TRAIN
# ==============================================================================
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # NOTE: now we load dict packs
    data_dir = "../data/data_pt"
    train_pack = os.path.join(data_dir, "market_train_2020-01-31_to_2024-12-31.pt")
    val_pack = os.path.join(data_dir, "market_val_2025-01-02_to_2025-12-31.pt")

    EPOCHS = 300
    BATCH_SIZE = 64
    WINDOW_SIZE = 20

    MAX_LR = 5.0e-4
    MIN_LR = 1.0e-6
    WEIGHT_DECAY = 0.10
    DROPOUT = 0.20
    EMA_DECAY = 0.999

    TRAIN_MASK_START = 0.15
    TRAIN_MASK_END = 0.30

    VAL_MASK_RATIO = 0.20
    VAL_MASK_SEED = 1234

    model = MacroConditionedMAE(
        num_feat=22, seq_len=WINDOW_SIZE, patch_size=5,
        embed_dim=256, enc_layers=6, dec_layers=3, dropout=DROPOUT
    ).to(device)
    ema = EMA(model, decay=EMA_DECAY)

    train_ds = MarketMAEDataset(train_pack, window_size=WINDOW_SIZE)
    val_ds = MarketMAEDataset(val_pack, window_size=WINDOW_SIZE)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, drop_last=True, pin_memory=torch.cuda.is_available())

    optimizer = optim.AdamW(model.parameters(), lr=MAX_LR, weight_decay=WEIGHT_DECAY)
    total_steps = EPOCHS * len(train_loader)
    warmup_steps = int(0.05 * total_steps)
    global_step = 0

    model_root = "../model"
    os.makedirs(model_root, exist_ok=True)

    best_fixed_val = float("inf")
    best_raw_path = os.path.join(model_root, "market_encoder_best_raw.pth")
    best_ema_path = os.path.join(model_root, "market_encoder_best_ema.pth")

    print(f"--- Training MacroConditionedMAE on {device} ---")

    for epoch in range(EPOCHS):
        train_mr = mask_ratio_schedule(epoch, EPOCHS, start=TRAIN_MASK_START, end=TRAIN_MASK_END)

        model.train()
        train_sum = 0.0
        for batch in train_loader:
            batch = batch.to(device, non_blocking=True)

            lr = lr_warmup_cosine(global_step, total_steps, warmup_steps, MAX_LR, MIN_LR)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            optimizer.zero_grad(set_to_none=True)
            loss = model(batch, mask_ratio=train_mr, mask_seed=None, noise_std=0.02, visible_loss_weight=0.10)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

            ema.update(model)
            train_sum += float(loss.item())
            global_step += 1

        avg_train = train_sum / max(1, len(train_loader))

        ema_model = ema.ema_model.to(device)
        ema_model.eval()

        fixed_val_sum = 0.0
        robust_val_sum = 0.0

        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device, non_blocking=True)

                v_fixed = ema_model(batch, mask_ratio=VAL_MASK_RATIO, mask_seed=VAL_MASK_SEED, noise_std=0.0, visible_loss_weight=0.0)
                fixed_val_sum += float(v_fixed.item())

                v_robust = ema_model(batch, mask_ratio=train_mr, mask_seed=VAL_MASK_SEED, noise_std=0.0, visible_loss_weight=0.0)
                robust_val_sum += float(v_robust.item())

        fixed_val = fixed_val_sum / max(1, len(val_loader))
        robust_val = robust_val_sum / max(1, len(val_loader))

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(
                f"Epoch {epoch+1:03d} | train_mr={train_mr:.2f} | "
                f"Train: {avg_train:.4f} | ValFixed(EMA,mr={VAL_MASK_RATIO:.2f}): {fixed_val:.4f} | "
                f"ValRobust(EMA,mr={train_mr:.2f}): {robust_val:.4f} | LR: {optimizer.param_groups[0]['lr']:.2e}"
            )

        if fixed_val < best_fixed_val:
            best_fixed_val = fixed_val
            torch.save(model.state_dict(), best_raw_path)
            torch.save(ema.ema_model.state_dict(), best_ema_path)

    print(f"Done. Best ValFixed(EMA)={best_fixed_val:.6f}")
    print(f"Saved best raw: {best_raw_path}")
    print(f"Saved best ema: {best_ema_path}")


if __name__ == "__main__":
    train()
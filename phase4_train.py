"""
Phase 4: Lightweight CNN for No-Reference Video Quality Assessment (MOS Prediction)
Backbone: EfficientNet-B4 (via timm) - pretrained on ImageNet
Strategy: Sample N frames per video -> CNN features -> Temporal mean pool -> MOS regressor
Dataset: YouTube-UGC (primary) + KoNViD-150k-B + LIVE-VQC (if available)
"""

import os
import sys
import json
import time
import random
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import timm
# decord is imported lazily inside sample_frames to avoid fork/spawn issues
from scipy import stats
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────
CFG = {
    'seed': 42,
    'num_frames': 8,           # 8 frames — fast decode, still good temporal coverage
    'img_size': 224,
    'backbone': 'efficientnet_b4',
    'pretrained': True,
    'dropout': 0.3,
    'batch_size': 8,           # physical batch — 8×8=64 frames fit easily in 32GB
    'grad_accum': 4,           # effective batch = 8×4 = 32
    'num_workers': 4,          # 4 workers — less contention than 8
    'epochs': 30,
    'lr': 1e-4,
    'weight_decay': 1e-4,
    'scheduler': 'cosine',
    'warmup_epochs': 2,
    'clip_grad': 1.0,
    'mos_min': 1.0,
    'mos_max': 5.0,
    # Paths
    'ugc_dir': '/workspace/datasets/youtube_ugc',
    'ugc_mos': '/workspace/datasets/youtube_ugc/MOS_for_YouTube_UGC_dataset.xlsx',
    'konvid_dir': '/workspace/datasets/konvid_150k_b',
    'konvid_scores': '/workspace/datasets/konvid_150k_b/k150kb_scores.csv',
    'livevqc_dir': '/workspace/datasets/live_vqc',
    'checkpoint_dir': '/workspace/checkpoints',
    'log_dir': '/workspace/logs',
}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Reduce memory fragmentation on large VRAM GPUs
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'


# ─────────────────────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────────────────────
def load_ugc_annotations(mos_path, video_dir):
    """Load YouTube-UGC MOS scores from the 'MOS' sheet and match to video files."""
    print(f"  Loading YouTube-UGC from {mos_path}")
    # YouTube-UGC xlsx has 2 sheets: 'readme' and 'MOS'
    df = pd.read_excel(mos_path, sheet_name='MOS', header=0, engine='openpyxl')
    print(f"  Columns: {df.columns.tolist()}")
    print(f"  Shape: {df.shape}")

    # Known column names from YouTube-UGC MOS sheet
    vid_col = 'vid'
    mos_col = 'MOS full'

    # Fallback: auto-detect if column names differ
    if vid_col not in df.columns or mos_col not in df.columns:
        print(f"  WARNING: Expected columns not found, auto-detecting...")
        vid_col, mos_col = None, None
        for c in df.columns:
            cl = str(c).lower()
            if vid_col is None and ('vid' in cl or 'name' in cl or 'clip' in cl):
                vid_col = c
            if mos_col is None and 'mos' in cl and 'full' in cl:
                mos_col = c
        if mos_col is None:
            for c in df.columns:
                if 'mos' in str(c).lower():
                    mos_col = c
                    break
        if vid_col is None or mos_col is None:
            vid_col, mos_col = df.columns[0], df.columns[4]

    print(f"  Using video_col='{vid_col}', mos_col='{mos_col}'")

    records = []
    video_dir = Path(video_dir)
    for _, row in df.iterrows():
        try:
            vid_name = str(row[vid_col]).strip()
            mos_raw = float(row[mos_col])
        except (ValueError, TypeError):
            continue  # skip malformed rows

        if pd.isna(mos_raw) or vid_name in ('nan', ''):
            continue

        # Normalize MOS [1,5] -> [0,1]
        mos_norm = float(np.clip((mos_raw - CFG['mos_min']) / (CFG['mos_max'] - CFG['mos_min']), 0.0, 1.0))

        # UGC vid ID is a prefix of the filename, e.g.
        # vid='Animation_1080P-05f8' -> 'Animation_1080P-05f8_crf_10_ss_00_t_20.0.mp4'
        candidates = list(video_dir.glob(f'{vid_name}*'))
        if not candidates:
            candidates = list(video_dir.glob(f'*{vid_name}*'))

        if candidates:
            path = candidates[0]
            # ── Skip 4K (2160P) UGC videos — they take 6-16s to decode
            if '2160P' in path.name:
                continue
            records.append({'video': str(path), 'mos': mos_norm, 'mos_raw': mos_raw, 'source': 'ugc'})

    print(f"  Matched {len(records)}/{len(df)} videos")
    return records


def load_konvid_annotations(scores_path, video_dir):
    """Load KoNViD-150k-B annotations."""
    scores_path = Path(scores_path)
    if not scores_path.exists():
        print("  KoNViD scores CSV not found, skipping")
        return []

    print(f"  Loading KoNViD-150k-B from {scores_path}")
    df = pd.read_csv(scores_path)
    print(f"  Columns: {df.columns.tolist()} | Shape: {df.shape}")

    # KoNViD-150k-B scores CSV has: video_name, mos, video_score
    vid_col = 'video_name' if 'video_name' in df.columns else df.columns[0]
    mos_col = 'mos' if 'mos' in df.columns else df.columns[1]
    print(f"  Using video_col='{vid_col}', mos_col='{mos_col}'")

    video_dir = Path(video_dir)
    if not any(video_dir.iterdir()):
        print(f"  KoNViD video dir is empty (zip not extracted yet), skipping")
        return []

    records = []
    for _, row in df.iterrows():
        try:
            vid_name = str(row[vid_col]).strip()
            mos_raw = float(row[mos_col])
        except (ValueError, TypeError):
            continue

        if pd.isna(mos_raw):
            continue

        # KoNViD MOS is 1-5 scale
        mos_norm = float(np.clip((mos_raw - 1.0) / 4.0, 0.0, 1.0))

        candidates = list(video_dir.glob(f'{vid_name}*'))
        if not candidates:
            candidates = list(video_dir.glob(f'*{vid_name}*'))
        if candidates:
            records.append({'video': str(candidates[0]), 'mos': mos_norm, 'mos_raw': mos_raw, 'source': 'konvid'})

    print(f"  Matched {len(records)} videos")
    return records


def sample_frames(video_path, num_frames):
    """Sample N evenly-spaced frames using decord — imported lazily to avoid fork deadlock."""
    try:
        import decord as _decord  # lazy import: safe in forked/spawned workers
        vr = _decord.VideoReader(str(video_path), num_threads=2, ctx=_decord.cpu(0))
        total = len(vr)
        if total <= 0:
            return None
        indices = np.linspace(0, total - 1, num_frames, dtype=int)
        frames_np = vr.get_batch(indices).asnumpy()  # [N, H, W, 3] uint8
        return [frames_np[i] for i in range(len(frames_np))]
    except Exception:
        return None


class VideoQualityDataset(Dataset):
    def __init__(self, records, num_frames=16, img_size=224, augment=False):
        self.records = records
        self.num_frames = num_frames
        self.augment = augment

        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        if augment:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
                transforms.ToTensor(),
                normalize,
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                normalize,
            ])

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        frames = sample_frames(rec['video'], self.num_frames)

        if frames is None:
            # Return zeros if video is unreadable
            dummy = torch.zeros(self.num_frames, 3, CFG['img_size'], CFG['img_size'])
            return dummy, torch.tensor(rec['mos'], dtype=torch.float32)

        frame_tensors = torch.stack([self.transform(f) for f in frames])
        mos = torch.tensor(rec['mos'], dtype=torch.float32)
        return frame_tensors, mos


# ─────────────────────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────────────────────
class VideoQualityModel(nn.Module):
    """
    EfficientNet-B4 backbone with temporal mean pooling for video MOS prediction.
    Input: (B, T, C, H, W)  -- B=batch, T=frames
    Output: (B,) -- predicted MOS in [0, 1]
    """
    def __init__(self, backbone='efficientnet_b4', pretrained=True, dropout=0.3):
        super().__init__()
        self.backbone = timm.create_model(backbone, pretrained=pretrained, num_classes=0)
        feat_dim = self.backbone.num_features
        print(f"  Backbone: {backbone} | Feature dim: {feat_dim}")

        self.regressor = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feat_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(256, 1),
            nn.Sigmoid()   # output in [0, 1]
        )

    def forward(self, x):
        # x: (B, T, C, H, W)
        B, T, C, H, W = x.shape
        # Merge batch and time dims
        x = x.view(B * T, C, H, W)
        feats = self.backbone(x)          # (B*T, feat_dim)
        feats = feats.view(B, T, -1)      # (B, T, feat_dim)
        feats = feats.mean(dim=1)         # Temporal mean pool -> (B, feat_dim)
        out = self.regressor(feats)       # (B, 1)
        return out.squeeze(1)             # (B,)


# ─────────────────────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────────────────────
def compute_metrics(preds, targets):
    """Compute PLCC, SRCC, RMSE on raw (unnormalized) MOS."""
    preds = np.array(preds)
    targets = np.array(targets)
    # Denormalize
    preds_raw = preds * (CFG['mos_max'] - CFG['mos_min']) + CFG['mos_min']
    targets_raw = targets * (CFG['mos_max'] - CFG['mos_min']) + CFG['mos_min']

    plcc, _ = stats.pearsonr(preds_raw, targets_raw)
    srcc, _ = stats.spearmanr(preds_raw, targets_raw)
    rmse = np.sqrt(np.mean((preds_raw - targets_raw) ** 2))
    return {'PLCC': plcc, 'SRCC': srcc, 'RMSE': rmse}


# ─────────────────────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────────────────────
def train_epoch(model, loader, optimizer, criterion, scaler, device, scheduler=None, grad_accum=1):
    model.train()
    total_loss = 0
    all_preds, all_targets = [], []
    optimizer.zero_grad()

    for batch_idx, (frames, mos) in enumerate(loader):
        frames = frames.to(device, non_blocking=True)
        mos = mos.to(device, non_blocking=True)

        with torch.amp.autocast('cuda'):
            preds = model(frames)
            loss = criterion(preds, mos) / grad_accum  # scale loss

        scaler.scale(loss).backward()

        # Step optimizer every grad_accum batches
        if (batch_idx + 1) % grad_accum == 0 or (batch_idx + 1) == len(loader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), CFG['clip_grad'])
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        total_loss += loss.item() * grad_accum  # undo scaling for logging
        all_preds.extend(preds.detach().cpu().numpy())
        all_targets.extend(mos.cpu().numpy())

        if batch_idx % 20 == 0:
            print(f"    Batch {batch_idx}/{len(loader)} | Loss: {loss.item() * grad_accum:.4f}")

    if scheduler:
        scheduler.step()

    metrics = compute_metrics(all_preds, all_targets)
    return total_loss / len(loader), metrics


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds, all_targets = [], []

    for frames, mos in loader:
        frames = frames.to(device, non_blocking=True)
        mos = mos.to(device, non_blocking=True)

        with torch.amp.autocast('cuda'):
            preds = model(frames)
            loss = criterion(preds, mos)

        total_loss += loss.item()
        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(mos.cpu().numpy())

    metrics = compute_metrics(all_preds, all_targets)
    return total_loss / len(loader), metrics


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=CFG['epochs'])
    parser.add_argument('--batch_size', type=int, default=CFG['batch_size'])
    parser.add_argument('--grad_accum', type=int, default=CFG['grad_accum'])
    parser.add_argument('--lr', type=float, default=CFG['lr'])
    parser.add_argument('--num_frames', type=int, default=CFG['num_frames'])
    parser.add_argument('--backbone', type=str, default=CFG['backbone'])
    parser.add_argument('--dry_run', action='store_true', help='Use 50 samples for quick test')
    args = parser.parse_args()

    CFG.update(vars(args))
    set_seed(CFG['seed'])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*60}")
    print(f"  Drishya Phase 4 — MOS Prediction Training")
    print(f"  Device: {device} | GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
    print(f"  Backbone: {CFG['backbone']} | Frames: {CFG['num_frames']}")
    print(f"  Batch: {CFG['batch_size']} × GradAccum: {CFG['grad_accum']} = Effective: {CFG['batch_size']*CFG['grad_accum']}")
    print(f"{'='*60}\n")

    # ── Load annotations ──────────────────────────────────────
    print("[1] Loading dataset annotations...")
    all_records = []

    ugc_recs = load_ugc_annotations(CFG['ugc_mos'], CFG['ugc_dir'])
    all_records.extend(ugc_recs)

    konvid_recs = load_konvid_annotations(CFG['konvid_scores'], CFG['konvid_dir'])
    all_records.extend(konvid_recs)

    print(f"\n  Total records: {len(all_records)}")
    print(f"    YouTube-UGC: {len(ugc_recs)}")
    print(f"    KoNViD-150k-B: {len(konvid_recs)}")

    if not all_records:
        print("ERROR: No records loaded. Check dataset paths!")
        sys.exit(1)

    if CFG.get('dry_run'):
        all_records = all_records[:50]
        print(f"  DRY RUN: using {len(all_records)} samples")

    # ── Train/Val split ───────────────────────────────────────
    train_recs, val_recs = train_test_split(
        all_records, test_size=0.15, random_state=CFG['seed']
    )
    print(f"\n  Train: {len(train_recs)} | Val: {len(val_recs)}")

    # ── Datasets & Loaders ────────────────────────────────────
    print("\n[2] Building dataloaders...")
    train_ds = VideoQualityDataset(train_recs, CFG['num_frames'], CFG['img_size'], augment=True)
    val_ds = VideoQualityDataset(val_recs, CFG['num_frames'], CFG['img_size'], augment=False)

    nw = CFG['num_workers']
    train_loader = DataLoader(train_ds, batch_size=CFG['batch_size'], shuffle=True,
                              num_workers=nw, pin_memory=(nw > 0), drop_last=True,
                              persistent_workers=(nw > 0),
                              multiprocessing_context='spawn' if nw > 0 else None)
    val_loader   = DataLoader(val_ds,   batch_size=CFG['batch_size'], shuffle=False,
                              num_workers=nw, pin_memory=(nw > 0),
                              persistent_workers=(nw > 0),
                              multiprocessing_context='spawn' if nw > 0 else None)

    # ── Model ─────────────────────────────────────────────────
    print(f"\n[3] Building model ({CFG['backbone']})...")
    model = VideoQualityModel(CFG['backbone'], CFG['pretrained'], CFG['dropout']).to(device)
    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  Total parameters: {total_params:.1f}M")

    # ── Training setup ────────────────────────────────────────
    criterion = nn.SmoothL1Loss()
    optimizer = optim.AdamW(model.parameters(), lr=CFG['lr'], weight_decay=CFG['weight_decay'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CFG['epochs'])
    scaler = torch.amp.GradScaler('cuda')

    # ── Logging ───────────────────────────────────────────────
    os.makedirs(CFG['checkpoint_dir'], exist_ok=True)
    os.makedirs(CFG['log_dir'], exist_ok=True)
    run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = Path(CFG['log_dir']) / f'train_{run_id}.json'
    best_srcc = -1
    history = []

    print(f"\n[4] Training for {CFG['epochs']} epochs...")
    print(f"  Log: {log_file}")
    print(f"  Checkpoints: {CFG['checkpoint_dir']}\n")

    for epoch in range(1, CFG['epochs'] + 1):
        print(f"\n── Epoch {epoch}/{CFG['epochs']} ──")
        t0 = time.time()

        train_loss, train_m = train_epoch(model, train_loader, optimizer, criterion, scaler, device, scheduler,
                                           grad_accum=CFG['grad_accum'])
        val_loss, val_m = eval_epoch(model, val_loader, criterion, device)

        elapsed = time.time() - t0
        lr_now = optimizer.param_groups[0]['lr']

        print(f"  Train | Loss={train_loss:.4f} PLCC={train_m['PLCC']:.4f} SRCC={train_m['SRCC']:.4f} RMSE={train_m['RMSE']:.4f}")
        print(f"  Val   | Loss={val_loss:.4f}  PLCC={val_m['PLCC']:.4f} SRCC={val_m['SRCC']:.4f} RMSE={val_m['RMSE']:.4f}")
        print(f"  LR={lr_now:.2e} | Time={elapsed:.1f}s")

        row = {
            'epoch': int(epoch),
            'train_loss': float(train_loss), 'val_loss': float(val_loss),
            'train_plcc': float(train_m['PLCC']), 'val_plcc': float(val_m['PLCC']),
            'train_srcc': float(train_m['SRCC']), 'val_srcc': float(val_m['SRCC']),
            'train_rmse': float(train_m['RMSE']), 'val_rmse': float(val_m['RMSE']),
            'lr': float(lr_now)
        }
        history.append(row)
        with open(log_file, 'w') as f:
            json.dump(history, f, indent=2)

        # Save best
        if val_m['SRCC'] > best_srcc:
            best_srcc = val_m['SRCC']
            ckpt_path = Path(CFG['checkpoint_dir']) / f'best_{run_id}.pt'
            torch.save({
                'epoch': epoch, 'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'val_srcc': best_srcc, 'val_plcc': val_m['PLCC'],
                'val_rmse': val_m['RMSE'], 'cfg': CFG
            }, ckpt_path)
            print(f"  ✅ New best SRCC={best_srcc:.4f} → saved {ckpt_path.name}")

    print(f"\n{'='*60}")
    print(f"  Training complete! Best Val SRCC: {best_srcc:.4f}")
    print(f"  Log: {log_file}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()

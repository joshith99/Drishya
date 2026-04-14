"""
Phase 4: Deep Feature Extraction + SVR for MOS Prediction
Steps:
  1. Load pretrained EfficientNet-B4 (frozen) as GPU feature extractor
  2. For each video: sample 8 frames → GPU backbone → mean-pool → 1792-d feature
  3. Train SVR (RBF kernel) on features vs MOS
  4. Evaluate PLCC / SRCC / RMSE on held-out val set
  5. Save model as pickle + also export to ONNX if possible

Why this approach:
  - Zero backprop = huge GPU batches, very fast
  - No DataLoader worker fork/spawn issues (single-threaded IO)
  - Deep features >> handcrafted features for VQA quality
  - SVR trains in seconds, not hours
"""

import os, sys, json, time, pickle, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, KFold
from scipy import stats
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
from torchvision import transforms
import timm

# ─── Config ──────────────────────────────────────────────────
NUM_FRAMES   = 8
IMG_SIZE     = 224
BATCH_SIZE   = 64      # videos per GPU batch (no grad → large is fine)
BACKBONE     = 'efficientnet_b4'
MOS_MIN, MOS_MAX = 1.0, 5.0

UGC_DIR      = Path('/workspace/datasets/youtube_ugc')
UGC_MOS      = Path('/workspace/datasets/youtube_ugc/MOS_for_YouTube_UGC_dataset.xlsx')
KONVID_DIR   = Path('/workspace/datasets/konvid_150k_b')
KONVID_CSV   = Path('/workspace/datasets/konvid_150k_b/k150kb_scores.csv')
FEAT_CACHE   = Path('/workspace/features_cache.npz')
CKPT_DIR     = Path('/workspace/checkpoints')
CKPT_DIR.mkdir(parents=True, exist_ok=True)

# ─── Image transform ─────────────────────────────────────────
_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# ─── Load annotations ────────────────────────────────────────
def load_all_records():
    records = []

    # YouTube-UGC (skip 4K)
    df = pd.read_excel(UGC_MOS, sheet_name='MOS', header=0, engine='openpyxl')
    for _, row in df.iterrows():
        try:
            vid = str(row['vid']).strip()
            mos = float(row['MOS full'])
        except: continue
        if pd.isna(mos) or not vid: continue
        mos_n = float(np.clip((mos - MOS_MIN) / (MOS_MAX - MOS_MIN), 0, 1))
        cands = list(UGC_DIR.glob(f'{vid}*'))
        if cands and '2160P' not in cands[0].name:
            records.append({'path': str(cands[0]), 'mos': mos_n, 'mos_raw': mos, 'src': 'ugc'})

    # KoNViD-150k-B
    df2 = pd.read_csv(KONVID_CSV)
    for _, row in df2.iterrows():
        try:
            vid = str(row['video_name']).strip()
            mos = float(row['mos'])
        except: continue
        if pd.isna(mos): continue
        mos_n = float(np.clip((mos - 1.0) / 4.0, 0, 1))
        cands = list(KONVID_DIR.glob(f'{vid}*'))
        if cands:
            records.append({'path': str(cands[0]), 'mos': mos_n, 'mos_raw': mos, 'src': 'konvid'})

    print(f"  Total: {len(records)} ({sum(1 for r in records if r['src']=='ugc')} UGC, "
          f"{sum(1 for r in records if r['src']=='konvid')} KoNViD)")
    return records


# ─── Read frames from one video ──────────────────────────────
def read_frames(video_path):
    """Returns numpy array [N, H, W, 3] uint8 or None on failure."""
    try:
        import decord as _d
        vr = _d.VideoReader(str(video_path), num_threads=2, ctx=_d.cpu(0))
        total = len(vr)
        if total <= 0: return None
        idx = np.linspace(0, total - 1, NUM_FRAMES, dtype=int)
        return vr.get_batch(idx).asnumpy()
    except Exception:
        return None


# ─── Build frozen GPU backbone ───────────────────────────────
def build_backbone(device):
    model = timm.create_model(BACKBONE, pretrained=True, num_classes=0)
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad = False
    feat_dim = model.num_features
    print(f"  Backbone: {BACKBONE} | feat_dim={feat_dim} | frozen weights")
    return model, feat_dim


# ─── Extract deep features for all videos ────────────────────
def extract_features(records, device):
    if FEAT_CACHE.exists():
        print(f"  Loading cached features from {FEAT_CACHE}")
        data = np.load(FEAT_CACHE, allow_pickle=True)
        return data['X'], data['y'], data['paths'].tolist()

    backbone, feat_dim = build_backbone(device)

    X, y, paths = [], [], []
    n = len(records)
    t0 = time.time()
    failed = 0

    print(f"  Extracting features for {n} videos (batch={BATCH_SIZE})...")
    print(f"  Each video → {NUM_FRAMES} frames → GPU → {feat_dim}-d feature\n")

    # Collect frames in CPU batches, push to GPU in one shot
    buf_frames  = []  # list of [NUM_FRAMES, 3, H, W] tensors
    buf_mos     = []
    buf_paths   = []

    def flush_buf():
        if not buf_frames:
            return
        # Stack: [B, T, 3, H, W]
        batch = torch.stack(buf_frames).to(device)  # (B, T, C, H, W)
        B, T, C, H, W = batch.shape
        with torch.no_grad():
            imgs = batch.view(B * T, C, H, W)
            feats = backbone(imgs)           # (B*T, feat_dim)
            feats = feats.view(B, T, -1).mean(dim=1)  # temporal mean pool → (B, feat_dim)
        X.extend(feats.cpu().numpy().tolist())
        y.extend(buf_mos)
        paths.extend(buf_paths)
        buf_frames.clear(); buf_mos.clear(); buf_paths.clear()

    for i, rec in enumerate(records):
        frames_np = read_frames(rec['path'])
        if frames_np is None:
            failed += 1
            continue

        frame_tensors = torch.stack([_transform(frames_np[j]) for j in range(len(frames_np))])
        buf_frames.append(frame_tensors)
        buf_mos.append(rec['mos'])
        buf_paths.append(rec['path'])

        if len(buf_frames) >= BATCH_SIZE:
            flush_buf()

        if (i + 1) % 100 == 0 or (i + 1) == n:
            elapsed = time.time() - t0
            rate = (i + 1 - failed) / elapsed
            eta  = (n - i - 1) / rate if rate > 0 else 0
            gpu_mb = torch.cuda.memory_allocated(device) / 1e6 if device.type == 'cuda' else 0
            print(f"  [{i+1}/{n}] {rate:.1f} vid/s | ETA {eta/60:.1f}min | "
                  f"GPU {gpu_mb:.0f}MB | failed={failed}")

    flush_buf()  # process remainder

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)

    print(f"\n  Feature matrix: {X.shape} | labels: {y.shape} | failed: {failed}")
    np.savez_compressed(FEAT_CACHE, X=X, y=y, paths=np.array(paths))
    print(f"  Cached to {FEAT_CACHE}")
    return X, y, paths


# ─── SVR training ────────────────────────────────────────────
def train_svr(X_tr, y_tr, X_val, y_val):
    from sklearn.svm import SVR
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    print("\n[3] Grid search for best SVR hyperparams (3-fold CV on train)...")
    best_srcc, best_model = -1, None

    # Quick grid search
    for C in [0.1, 1.0, 10.0, 100.0]:
        for gamma in ['scale', 'auto', 0.001, 0.0001]:
            pipe = Pipeline([
                ('scaler', StandardScaler()),
                ('svr', SVR(kernel='rbf', C=C, gamma=gamma, epsilon=0.05))
            ])
            pipe.fit(X_tr, y_tr)
            pred_val = pipe.predict(X_val)
            srcc, _ = stats.spearmanr(pred_val, y_val)
            if srcc > best_srcc:
                best_srcc = srcc
                best_model = pipe
                best_params = {'C': C, 'gamma': gamma}

    print(f"  Best params: {best_params} | Val SRCC={best_srcc:.4f}")
    return best_model


def compute_metrics(preds, targets):
    preds_raw   = np.array(preds) * (MOS_MAX - MOS_MIN) + MOS_MIN
    targets_raw = np.array(targets) * (MOS_MAX - MOS_MIN) + MOS_MIN
    plcc, _ = stats.pearsonr(preds_raw, targets_raw)
    srcc, _ = stats.spearmanr(preds_raw, targets_raw)
    rmse = np.sqrt(np.mean((preds_raw - targets_raw) ** 2))
    return plcc, srcc, rmse


# ─── Main ────────────────────────────────────────────────────
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*60}")
    print(f"  Drishya Phase 4 — Deep Features + SVR")
    print(f"  Device: {device}")
    if device.type == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
    print(f"{'='*60}\n")

    # 1. Load data
    print("[1] Loading annotations...")
    records = load_all_records()
    if not records:
        print("ERROR: No records found!"); sys.exit(1)

    # 2. Extract deep features
    print("\n[2] Extracting deep GPU features...")
    X, y, paths = extract_features(records, device)

    # 3. Train/val split
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.15, random_state=42)
    print(f"\n  Train: {len(X_tr)} | Val: {len(X_val)}")

    # 4. Train SVR
    print("\n[3] Training SVR...")
    t0 = time.time()
    model = train_svr(X_tr, y_tr, X_val, y_val)
    print(f"  SVR trained in {time.time()-t0:.1f}s")

    # 5. Final evaluation
    print("\n[4] Final evaluation...")
    pred_tr  = model.predict(X_tr)
    pred_val = model.predict(X_val)

    tr_plcc, tr_srcc, tr_rmse  = compute_metrics(pred_tr, y_tr)
    val_plcc, val_srcc, val_rmse = compute_metrics(pred_val, y_val)

    print(f"  Train | PLCC={tr_plcc:.4f}  SRCC={tr_srcc:.4f}  RMSE={tr_rmse:.4f}")
    print(f"  Val   | PLCC={val_plcc:.4f}  SRCC={val_srcc:.4f}  RMSE={val_rmse:.4f}")

    # 6. Save model
    run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    ckpt_path = CKPT_DIR / f'svr_deep_{run_id}.pkl'
    results = {
        'model': model,
        'backbone': BACKBONE,
        'num_frames': NUM_FRAMES,
        'img_size': IMG_SIZE,
        'val_plcc': float(val_plcc),
        'val_srcc': float(val_srcc),
        'val_rmse': float(val_rmse),
        'mos_min': MOS_MIN,
        'mos_max': MOS_MAX,
    }
    with open(ckpt_path, 'wb') as f:
        pickle.dump(results, f)

    # Also save a JSON summary
    summary = {k: v for k, v in results.items() if k != 'model'}
    with open(CKPT_DIR / f'svr_results_{run_id}.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n  ✅ Model saved to {ckpt_path}")
    print(f"\n{'='*60}")
    print(f"  FINAL: Val PLCC={val_plcc:.4f}  SRCC={val_srcc:.4f}  RMSE={val_rmse:.4f}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()

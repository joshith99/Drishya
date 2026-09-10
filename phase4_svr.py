"""
Phase 4: Deep Feature Extraction + SVR for MOS Prediction
Steps:
  1. Load pretrained EfficientNet-B4 (frozen) as feature extractor
  2. For each video: sample 8 frames → backbone → mean-pool → 1792-d feature
  3. Grid-search SVR (RBF kernel) via k-fold CV on the training fold
  4. Evaluate PLCC / SRCC / RMSE on a held-out test set (never touched during tuning)
  5. Save model (SVR pipeline + frozen backbone weights) as pickle

Why this approach:
  - Zero backprop = huge batches, very fast (runs on CPU too)
  - No DataLoader worker fork/spawn issues (single-threaded IO)
  - Deep features >> handcrafted features for VQA quality
  - SVR trains in seconds, not hours
  - Hyperparameters are selected by cross-validation only, so the final
    reported metrics come from a genuinely held-out split.

All paths are configurable via CLI args / environment variables, so this runs on
any machine (no hardcoded /workspace paths). See `python phase4_svr.py --help`.
"""

import os, sys, json, time, pickle, warnings, argparse
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
from torchvision import transforms
import timm

# ─── Config ──────────────────────────────────────────────────
MOS_MIN, MOS_MAX = 1.0, 5.0


def parse_args():
    def _env(name, default):
        return os.environ.get(name, default)

    p = argparse.ArgumentParser(description="Phase 4 — deep features + SVR for MOS prediction")
    p.add_argument('--data-root', default=_env('DRISHYA_DATA_ROOT', 'datasets'),
                   help='Root folder holding datasets/ (default: datasets)')
    p.add_argument('--ckpt-dir', default=_env('DRISHYA_CKPT_DIR', 'checkpoints'),
                   help='Where to save model checkpoints (default: checkpoints)')
    p.add_argument('--feat-cache', default=_env('DRISHYA_FEAT_CACHE', 'features_cache.npz'),
                   help='Path to the feature cache (default: features_cache.npz)')
    p.add_argument('--backbone', default='efficientnet_b4', help='timm backbone name')
    p.add_argument('--num-frames', type=int, default=8, help='frames sampled per video')
    p.add_argument('--img-size', type=int, default=224, help='resize side length')
    p.add_argument('--batch-size', type=int, default=64, help='videos per feature batch')
    p.add_argument('--limit', type=int, default=None,
                   help='Max records to use (deterministic sample, for quick runs)')
    p.add_argument('--test-size', type=float, default=0.15, help='held-out test fraction')
    p.add_argument('--seed', type=int, default=42)
    return p.parse_args()


CFG = None  # populated by main()


# ─── Image transform ─────────────────────────────────────────
def build_transform(cfg):
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((cfg['img_size'], cfg['img_size'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


# ─── Load annotations ────────────────────────────────────────
def load_ugc_records(cfg):
    """YouTube-UGC (optional). Returns [] if the xlsx is absent."""
    ugc_dir = cfg['data_root'] / 'youtube_ugc'
    ugc_mos = ugc_dir / 'MOS_for_YouTube_UGC_dataset.xlsx'
    if not ugc_mos.exists():
        return []
    df = pd.read_excel(ugc_mos, sheet_name='MOS', header=0, engine='openpyxl')
    records = []
    for _, row in df.iterrows():
        try:
            vid = str(row['vid']).strip()
            mos = float(row['MOS full'])
        except Exception:
            continue
        if pd.isna(mos) or not vid:
            continue
        mos_n = float(np.clip((mos - MOS_MIN) / (MOS_MAX - MOS_MIN), 0, 1))
        cands = list(ugc_dir.glob(f'{vid}*'))
        if cands and '2160P' not in cands[0].name:
            records.append({'path': str(cands[0]), 'mos': mos_n, 'mos_raw': mos, 'src': 'ugc'})
    return records


def load_konvid_records(cfg):
    """KoNViD (1k or 150k-B). Auto-detects the metadata CSV and column names."""
    konvid_dir = cfg['data_root'] / 'konvid1k'
    csv_path = None
    if konvid_dir.exists():
        for cand in ('KoNViD-1k_metadata.csv', 'koNViD_1k_metadata.csv',
                     'k150kb_scores.csv', 'scores.csv'):
            if (konvid_dir / cand).exists():
                csv_path = konvid_dir / cand
                break
        else:
            existing = sorted(konvid_dir.glob('*.csv'))
            if existing:
                csv_path = existing[0]
    if csv_path is None or not csv_path.exists():
        return []

    df = pd.read_csv(csv_path)
    mos_col = next((c for c in df.columns if 'mos' in str(c).lower()), None)
    if mos_col is None:
        return []
    name_col = next((c for c in df.columns if str(c).lower() in
                     ('file_name', 'video_name', 'name', 'clip', 'flickr_id')), None)
    if name_col is None:
        name_col = df.columns[0]

    records = []
    for _, row in df.iterrows():
        try:
            vid = str(row[name_col]).strip()
            mos = float(row[mos_col])
        except Exception:
            continue
        if pd.isna(mos) or not vid or vid.lower() in ('nan', 'none'):
            continue
        mos_n = float(np.clip((mos - 1.0) / 4.0, 0, 1))
        cands = list(konvid_dir.glob(f'{vid}*')) or list(konvid_dir.glob(f'*{vid}*'))
        if cands:
            records.append({'path': str(cands[0]), 'mos': mos_n, 'mos_raw': mos, 'src': 'konvid'})
    return records


def load_all_records(cfg):
    records = []
    records += load_ugc_records(cfg)
    records += load_konvid_records(cfg)

    if cfg['limit'] and len(records) > cfg['limit']:
        rng = np.random.default_rng(cfg['seed'])
        idx = rng.choice(len(records), cfg['limit'], replace=False)
        records = [records[i] for i in sorted(idx)]

    src_counts = {}
    for r in records:
        src_counts[r['src']] = src_counts.get(r['src'], 0) + 1
    print(f"  Total: {len(records)} ({', '.join(f'{k}={v}' for k, v in src_counts.items())})")
    return records


# ─── Read frames from one video ──────────────────────────────
def read_frames(video_path, num_frames):
    """Returns numpy array [N, H, W, 3] uint8 or None on failure."""
    try:
        import decord as _d
        vr = _d.VideoReader(str(video_path), num_threads=2, ctx=_d.cpu(0))
        total = len(vr)
        if total <= 0:
            return None
        idx = np.linspace(0, total - 1, num_frames, dtype=int)
        return vr.get_batch(idx).asnumpy()
    except Exception:
        return None


# ─── Build frozen backbone ───────────────────────────────────
def build_backbone(cfg, device):
    model = timm.create_model(cfg['backbone'], pretrained=True, num_classes=0)
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad = False
    print(f"  Backbone: {cfg['backbone']} | feat_dim={model.num_features} | frozen weights")
    return model, model.num_features


# ─── Extract deep features for all videos ────────────────────
def extract_features(records, cfg, device, transform):
    feat_cache = cfg['feat_cache']
    if feat_cache.exists():
        data = np.load(feat_cache, allow_pickle=True)
        cached_paths = data['paths'].tolist()
        if len(cached_paths) == len(records):
            print(f"  Loading cached features from {feat_cache}")
            return data['X'], data['y'], cached_paths
        print(f"  Cache miss ({len(cached_paths)} cached vs {len(records)} records); recomputing...")

    backbone, feat_dim = build_backbone(cfg, device)
    batch_size = cfg['batch_size']

    X, y, paths = [], [], []
    n = len(records)
    t0 = time.time()
    failed = 0

    print(f"  Extracting features for {n} videos (batch={batch_size})...")
    print(f"  Each video → {cfg['num_frames']} frames → {'GPU' if device.type=='cuda' else 'CPU'} → {feat_dim}-d feature\n")

    buf_frames, buf_mos, buf_paths = [], [], []

    def flush_buf():
        if not buf_frames:
            return
        batch = torch.stack(buf_frames).to(device)          # (B, T, C, H, W)
        B, T, C, H, W = batch.shape
        with torch.no_grad():
            imgs = batch.view(B * T, C, H, W)
            feats = backbone(imgs)
            feats = feats.view(B, T, -1).mean(dim=1)
        X.extend(feats.cpu().numpy().tolist())
        y.extend(buf_mos)
        paths.extend(buf_paths)
        buf_frames.clear(); buf_mos.clear(); buf_paths.clear()

    for i, rec in enumerate(records):
        frames_np = read_frames(rec['path'], cfg['num_frames'])
        if frames_np is None:
            failed += 1
            continue
        frame_tensors = torch.stack([transform(frames_np[j]) for j in range(len(frames_np))])
        buf_frames.append(frame_tensors)
        buf_mos.append(rec['mos'])
        buf_paths.append(rec['path'])

        if len(buf_frames) >= batch_size:
            flush_buf()

        if (i + 1) % 25 == 0 or (i + 1) == n:
            elapsed = time.time() - t0
            rate = (i + 1 - failed) / elapsed if elapsed > 0 else 0
            eta = (n - i - 1) / rate if rate > 0 else 0
            print(f"  [{i+1}/{n}] {rate:.2f} vid/s | ETA {eta/60:.1f} min | failed={failed}")

    flush_buf()

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    print(f"\n  Feature matrix: {X.shape} | labels: {y.shape} | failed: {failed}")
    feat_cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(feat_cache, X=X, y=y, paths=np.array(paths))
    print(f"  Cached to {feat_cache}")
    return X, y, paths


# ─── SVR training ────────────────────────────────────────────
def train_svr(X_tr, y_tr, n_splits=3, seed=42):
    """
    Grid-search SVR hyperparameters using k-fold CV on the training fold only.

    This keeps the held-out test set completely untouched during model
    selection, avoiding optimistic/overfit performance estimates.
    """
    print(f"\n[3] Grid search for best SVR hyperparams ({n_splits}-fold CV on train)...")
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    best_srcc, best_model, best_params = -1, None, None

    for C in [0.1, 1.0, 10.0, 100.0]:
        for gamma in ['scale', 'auto', 0.001, 0.0001]:
            pipe = Pipeline([
                ('scaler', StandardScaler()),
                ('svr', SVR(kernel='rbf', C=C, gamma=gamma, epsilon=0.05))
            ])
            cv_srcc = []
            for tr_idx, va_idx in kf.split(X_tr):
                pipe.fit(X_tr[tr_idx], y_tr[tr_idx])
                pred = pipe.predict(X_tr[va_idx])
                srcc, _ = stats.spearmanr(pred, y_tr[va_idx])
                cv_srcc.append(srcc)
            mean_srcc = float(np.mean(cv_srcc))
            if mean_srcc > best_srcc:
                best_srcc = mean_srcc
                best_model = pipe
                best_params = {'C': C, 'gamma': gamma}

    print(f"  Best params: {best_params} | CV SRCC={best_srcc:.4f}")
    best_model.fit(X_tr, y_tr)  # retrain on the full training fold
    return best_model, best_params


def compute_metrics(preds, targets):
    preds_raw = np.array(preds) * (MOS_MAX - MOS_MIN) + MOS_MIN
    targets_raw = np.array(targets) * (MOS_MAX - MOS_MIN) + MOS_MIN
    plcc, _ = stats.pearsonr(preds_raw, targets_raw)
    srcc, _ = stats.spearmanr(preds_raw, targets_raw)
    rmse = np.sqrt(np.mean((preds_raw - targets_raw) ** 2))
    return plcc, srcc, rmse


# ─── Main ────────────────────────────────────────────────────
def main():
    global CFG
    args = parse_args()
    CFG = {
        'data_root': Path(args.data_root),
        'ckpt_dir': Path(args.ckpt_dir),
        'feat_cache': Path(args.feat_cache),
        'backbone': args.backbone,
        'num_frames': args.num_frames,
        'img_size': args.img_size,
        'batch_size': args.batch_size,
        'limit': args.limit,
        'test_size': args.test_size,
        'seed': args.seed,
    }
    CFG['ckpt_dir'].mkdir(parents=True, exist_ok=True)

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
    records = load_all_records(CFG)
    if not records:
        print("ERROR: No records found! Check --data-root and dataset layout.")
        print("  Expected: <data-root>/konvid1k/{videos, KoNViD-1k_metadata.csv}")
        sys.exit(1)

    # 2. Extract deep features
    print("\n[2] Extracting deep features...")
    transform = build_transform(CFG)
    X, y, paths = extract_features(records, CFG, device, transform)

    # 3. Split: hold out a test set; hyperparams tuned via CV on train only
    X_tr, X_test, y_tr, y_test = train_test_split(
        X, y, test_size=CFG['test_size'], random_state=CFG['seed'])
    print(f"\n  Train: {len(X_tr)} | Test (held out): {len(X_test)}")

    # 4. Train SVR (hyperparams chosen by k-fold CV on the training fold)
    print("\n[3] Training SVR...")
    t0 = time.time()
    model, best_params = train_svr(X_tr, y_tr, seed=CFG['seed'])
    print(f"  SVR trained in {time.time()-t0:.1f}s")

    # 5. Final evaluation on the held-out test set
    print("\n[4] Final evaluation...")
    pred_tr = model.predict(X_tr)
    pred_test = model.predict(X_test)

    tr_plcc, tr_srcc, tr_rmse = compute_metrics(pred_tr, y_tr)
    te_plcc, te_srcc, te_rmse = compute_metrics(pred_test, y_test)

    print(f"  Train | PLCC={tr_plcc:.4f}  SRCC={tr_srcc:.4f}  RMSE={tr_rmse:.4f}")
    print(f"  Test  | PLCC={te_plcc:.4f}  SRCC={te_srcc:.4f}  RMSE={te_rmse:.4f}")

    # 6. Save model (SVR pipeline + frozen backbone weights)
    print("\n  Capturing frozen backbone weights for offline inference...")
    backbone, feat_dim = build_backbone(CFG, device)
    backbone_state = {k: v.detach().cpu() for k, v in backbone.state_dict().items()}

    run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    ckpt_path = CFG['ckpt_dir'] / f'svr_deep_{run_id}.pkl'
    results = {
        'model': model,
        'backbone': CFG['backbone'],
        'backbone_state_dict': backbone_state,
        'num_frames': CFG['num_frames'],
        'img_size': CFG['img_size'],
        'test_plcc': float(te_plcc),
        'test_srcc': float(te_srcc),
        'test_rmse': float(te_rmse),
        'best_params': best_params,
        'n_train': int(len(X_tr)),
        'n_test': int(len(X_test)),
        'seed': CFG['seed'],
        'mos_min': MOS_MIN,
        'mos_max': MOS_MAX,
    }
    with open(ckpt_path, 'wb') as f:
        pickle.dump(results, f)

    summary = {k: v for k, v in results.items() if k not in ('model', 'backbone_state_dict')}
    with open(CFG['ckpt_dir'] / f'svr_results_{run_id}.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n  ✅ Model saved to {ckpt_path}")
    print(f"\n{'='*60}")
    print(f"  FINAL: Test PLCC={te_plcc:.4f}  SRCC={te_srcc:.4f}  RMSE={te_rmse:.4f}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
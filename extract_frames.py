"""
One-time frame extractor: reads all training videos and saves N evenly-spaced
JPEG frames to disk. After this, training uses a simple image DataLoader —
no video decoding overhead, epochs run in minutes not hours.

Usage:
    python3 extract_frames.py
    
Output tree:
    /workspace/datasets/frames/
        ugc/{video_stem}/frame_{:04d}.jpg
        konvid/{video_stem}/frame_{:04d}.jpg
    /workspace/datasets/frames/train_manifest.csv
    /workspace/datasets/frames/val_manifest.csv
"""

import os
import sys
import csv
import time
import signal
import numpy as np
import pandas as pd
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

# ── Config ────────────────────────────────────────────────────
NUM_FRAMES    = 16
JPEG_QUALITY  = 90
NUM_WORKERS   = 8     # parallel video readers
FRAME_DIR     = Path('/workspace/datasets/frames')
UGC_DIR       = Path('/workspace/datasets/youtube_ugc')
UGC_MOS       = Path('/workspace/datasets/youtube_ugc/MOS_for_YouTube_UGC_dataset.xlsx')
KONVID_DIR    = Path('/workspace/datasets/konvid_150k_b')
KONVID_SCORES = Path('/workspace/datasets/konvid_150k_b/k150kb_scores.csv')
MOS_MIN, MOS_MAX = 1.0, 5.0

# ── Load annotations ──────────────────────────────────────────
def load_annotations():
    records = []

    # YouTube-UGC
    df = pd.read_excel(UGC_MOS, sheet_name='MOS', header=0, engine='openpyxl')
    for _, row in df.iterrows():
        try:
            vid_name = str(row['vid']).strip()
            mos_raw  = float(row['MOS full'])
        except (ValueError, TypeError):
            continue
        if pd.isna(mos_raw) or vid_name in ('nan', ''):
            continue
        mos_norm = float(np.clip((mos_raw - MOS_MIN) / (MOS_MAX - MOS_MIN), 0.0, 1.0))
        candidates = list(UGC_DIR.glob(f'{vid_name}*'))
        if candidates:
            records.append({'video': str(candidates[0]), 'mos': mos_norm,
                            'mos_raw': mos_raw, 'source': 'ugc', 'stem': vid_name})

    # KoNViD-150k-B
    df2 = pd.read_csv(KONVID_SCORES)
    for _, row in df2.iterrows():
        try:
            vid_name = str(row['video_name']).strip()
            mos_raw  = float(row['mos'])
        except (ValueError, TypeError):
            continue
        if pd.isna(mos_raw):
            continue
        mos_norm = float(np.clip((mos_raw - 1.0) / 4.0, 0.0, 1.0))
        candidates = list(KONVID_DIR.glob(f'{vid_name}*'))
        if candidates:
            records.append({'video': str(candidates[0]), 'mos': mos_norm,
                            'mos_raw': mos_raw, 'source': 'konvid', 'stem': vid_name})

    print(f"Total records: {len(records)}")
    return records


# ── Worker: extract frames for one video ──────────────────────
def extract_one(args):
    video_path, stem, source, mos, mos_raw = args
    import decord
    from PIL import Image
    import warnings
    warnings.filterwarnings('ignore')

    out_dir = FRAME_DIR / source / stem
    # Check if already extracted
    existing = list(out_dir.glob('frame_*.jpg'))
    if len(existing) >= NUM_FRAMES:
        frame_paths = sorted(existing)
        return (video_path, source, stem, mos, mos_raw,
                [str(p) for p in frame_paths[:NUM_FRAMES]], 'cached')

    try:
        t0 = time.time()
        vr = decord.VideoReader(str(video_path), num_threads=2, ctx=decord.cpu(0))
        total = len(vr)
        if total <= 0:
            return (video_path, source, stem, mos, mos_raw, [], 'empty')

        indices = np.linspace(0, total - 1, NUM_FRAMES, dtype=int)
        frames_np = vr.get_batch(indices).asnumpy()  # [N, H, W, 3]

        out_dir.mkdir(parents=True, exist_ok=True)
        frame_paths = []
        for i, frame in enumerate(frames_np):
            jpg_path = out_dir / f'frame_{i:04d}.jpg'
            img = Image.fromarray(frame)
            img.save(str(jpg_path), 'JPEG', quality=JPEG_QUALITY, optimize=True)
            frame_paths.append(str(jpg_path))

        elapsed = time.time() - t0
        return (video_path, source, stem, mos, mos_raw, frame_paths,
                f'ok:{elapsed:.1f}s')

    except Exception as e:
        return (video_path, source, stem, mos, mos_raw, [], f'error:{e}')


# ── Main ──────────────────────────────────────────────────────
def main():
    FRAME_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Frame output dir: {FRAME_DIR}")
    print(f"Workers: {NUM_WORKERS}, Frames per video: {NUM_FRAMES}\n")

    records = load_annotations()
    total = len(records)

    args_list = [(r['video'], r['stem'], r['source'], r['mos'], r['mos_raw'])
                 for r in records]

    done_records = []
    t_start = time.time()
    ok_count, error_count, cached_count = 0, 0, 0

    print(f"Extracting frames for {total} videos using {NUM_WORKERS} workers...\n")

    with ProcessPoolExecutor(max_workers=NUM_WORKERS,
                             mp_context=mp.get_context('spawn')) as executor:
        futures = {executor.submit(extract_one, a): a for a in args_list}

        for i, fut in enumerate(as_completed(futures)):
            vid_path, source, stem, mos, mos_raw, frame_paths, status = fut.result()

            if 'ok' in status or 'cached' in status:
                if 'cached' in status:
                    cached_count += 1
                else:
                    ok_count += 1
                done_records.append({
                    'stem': stem, 'source': source, 'mos': mos,
                    'mos_raw': mos_raw,
                    'frames': ','.join(frame_paths)
                })
            else:
                error_count += 1

            if (i + 1) % 50 == 0 or (i + 1) == total:
                elapsed = time.time() - t_start
                rate = (i + 1) / elapsed
                eta = (total - i - 1) / rate if rate > 0 else 0
                print(f"  [{i+1}/{total}] OK={ok_count} Cached={cached_count} "
                      f"Err={error_count} | {rate:.1f} vid/s | ETA={eta/60:.1f}min | "
                      f"Last: {status}")

    # Write manifest
    manifest_path = FRAME_DIR / 'manifest.csv'
    with open(manifest_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['stem', 'source', 'mos', 'mos_raw', 'frames'])
        writer.writeheader()
        writer.writerows(done_records)

    elapsed_total = time.time() - t_start
    print(f"\n✅ Extraction complete in {elapsed_total/60:.1f} min")
    print(f"   OK: {ok_count}, Cached: {cached_count}, Errors: {error_count}")
    print(f"   Manifest: {manifest_path}")


if __name__ == '__main__':
    main()

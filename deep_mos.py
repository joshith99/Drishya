"""
Standalone deep MOS inference (Phase 4 SVR + EfficientNet-B4).

This is a terminal-friendly extraction of the inference logic that lives inside
`phase5_app.py`. That file runs Streamlit at import time (st.set_page_config),
so its functions cannot be imported headlessly. Keeping a standalone copy here
means the working UI is never touched or imported.

Usage:
    python deep_mos.py testing_videos/video_small.mp4
    python deep_mos.py testing_videos/video_small.mp4 -v
    python deep_mos.py testing_videos/*.mp4 --save-frames out_frames
"""

import argparse
import os
import pickle
import sys
import time
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings('ignore')

CKPT_DIR_DEFAULT = 'checkpoints'

# ImageNet normalization used by the training/inference transform.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def _find_latest_ckpt(ckpt_dir):
    ckpt_dir = Path(ckpt_dir)
    pkls = sorted(ckpt_dir.glob('svr_deep_*.pkl'), reverse=True)
    return pkls[0] if pkls else None


def _rating(mos):
    if mos is None:
        return "N/A"
    if mos >= 4.0:
        return "Excellent"
    if mos >= 3.0:
        return "Good"
    if mos >= 2.0:
        return "Fair"
    return "Poor"


def load_deep_model(ckpt_dir=CKPT_DIR_DEFAULT):
    """
    Load the newest Phase 4 SVR checkpoint + EfficientNet-B4 backbone.

    Returns (svr, backbone, transform, cfg, meta) where cfg = (device,
    num_frames, mos_min, mos_max) and meta describes the checkpoint/backbone.
    Returns (None, None, None, None, None) if no checkpoint exists.
    """
    import torch
    import timm
    from torchvision import transforms

    ckpt_path = _find_latest_ckpt(ckpt_dir)
    if ckpt_path is None:
        return None, None, None, None, None

    with open(ckpt_path, 'rb') as f:
        saved = pickle.load(f)

    svr = saved['model']
    num_frames = saved.get('num_frames', 8)
    img_size = saved.get('img_size', 224)
    mos_min = saved.get('mos_min', 1.0)
    mos_max = saved.get('mos_max', 5.0)
    backbone_name = saved.get('backbone', 'efficientnet_b4')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    backbone = timm.create_model(backbone_name, pretrained=True, num_classes=0)
    if saved.get('backbone_state_dict'):
        backbone.load_state_dict(saved['backbone_state_dict'])
        weights_source = "saved state_dict"
    else:
        weights_source = "ImageNet fallback"
    backbone = backbone.to(device).eval()
    for p in backbone.parameters():
        p.requires_grad = False

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    meta = {
        'ckpt_path': str(ckpt_path),
        'backbone': backbone_name,
        'weights_source': weights_source,
        'img_size': img_size,
        'num_frames': num_frames,
        'mos_min': mos_min,
        'mos_max': mos_max,
        'feat_dim': int(backbone.num_features),
    }
    return svr, backbone, transform, (device, num_frames, mos_min, mos_max), meta


def predict_mos(video_path, svr, backbone, transform, cfg, num_frames=None):
    """
    Run deep feature extraction + SVR prediction on one video.

    Returns a details dict on success, or None on failure. Keys:
        mos, mos_norm, rating, frames_np, preprocessed, total_frames, fps,
        width, height, duration, frame_indices, timestamps, feat_dim,
        feat_mean, feat_std, feat_norm, per_frame_norms, elapsed
    """
    import torch

    device, cfg_frames, mos_min, mos_max = cfg
    n_frames = int(num_frames) if num_frames else cfg_frames
    try:
        import decord as _d
        vr = _d.VideoReader(str(video_path), num_threads=2, ctx=_d.cpu(0))
        total = len(vr)
        if total == 0:
            return None
        fps = float(vr.get_avg_fps() or 0.0)
        idx = np.linspace(0, total - 1, n_frames, dtype=int)
        frames_np = vr.get_batch(idx).asnumpy()             # (T, H, W, 3) uint8

        t0 = time.time()
        tensors = torch.stack([transform(frames_np[i]) for i in range(len(frames_np))])
        preprocessed = tensors.detach().cpu()               # (T, C, H, W), normalized
        tensors = tensors.unsqueeze(0).to(device)           # (1, T, C, H, W)
        B, T, C, H, W = tensors.shape
        with torch.no_grad():
            feats = backbone(tensors.view(B * T, C, H, W))  # (T, feat_dim)
            per_frame_norms = feats.norm(dim=1).cpu().numpy()
            pooled = feats.mean(dim=0, keepdim=True).cpu().numpy()  # (1, feat_dim)
        elapsed = time.time() - t0

        mos_norm = float(np.clip(svr.predict(pooled)[0], 0, 1))
        mos_raw = mos_norm * (mos_max - mos_min) + mos_min

        return {
            'mos': round(mos_raw, 2),
            'mos_norm': round(mos_norm, 4),
            'rating': _rating(mos_raw),
            'frames_np': frames_np,
            'preprocessed': preprocessed,
            'total_frames': int(total),
            'fps': fps,
            'width': int(frames_np.shape[2]),
            'height': int(frames_np.shape[1]),
            'duration': (total / fps) if fps > 0 else None,
            'frame_indices': [int(i) for i in idx],
            'timestamps': [(int(i) / fps) if fps > 0 else None for i in idx],
            'feat_dim': int(pooled.shape[1]),
            'feat_mean': float(pooled.mean()),
            'feat_std': float(pooled.std()),
            'feat_norm': float(np.linalg.norm(pooled)),
            'per_frame_norms': [float(x) for x in per_frame_norms],
            'elapsed': elapsed,
        }
    except Exception:
        return None


def _denormalize_to_pil(pre_frame):
    """Invert the ImageNet normalization so a preprocessed frame is viewable."""
    import torch
    from torchvision.transforms.functional import to_pil_image
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    x = (pre_frame.cpu() * std + mean).clamp(0.0, 1.0)
    return to_pil_image(x)


def _make_montage(raw_frames, pre_tensor, out_path, cell=224):
    """Top row = raw sampled frames; bottom row = preprocessed (denormalized)."""
    from PIL import Image
    n = len(raw_frames)
    canvas = Image.new('RGB', (cell * n, cell * 2), (0, 0, 0))
    for i in range(n):
        raw = Image.fromarray(raw_frames[i]).resize((cell, cell))
        canvas.paste(raw, (i * cell, 0))
        pre = _denormalize_to_pil(pre_tensor[i]).resize((cell, cell))
        canvas.paste(pre, (i * cell, cell))
    canvas.save(out_path)
    return out_path


def _print_report(path, res, verbose, save_dir, save_pre_dir=None, show_pre=False):
    name = Path(path).name
    print(f"\n{name}")
    if res is None:
        print("  error      : could not decode / run inference")
        return
    dur = f"{res['duration']:.2f} s" if res['duration'] is not None else "n/a"
    print(f"  video      : {res['width']}x{res['height']} | {res['fps']:.2f} fps | "
          f"{res['total_frames']} frames | {dur}")
    idx_str = ", ".join(str(i) for i in res['frame_indices'])
    print(f"  sampled    : {len(res['frame_indices'])} frames @ idx [{idx_str}]")

    pre = res.get('preprocessed')
    if pre is not None:
        mean_s = ", ".join(f"{m:.3f}" for m in IMAGENET_MEAN)
        std_s = ", ".join(f"{s:.3f}" for s in IMAGENET_STD)
        print(f"  preprocessed: tensor {tuple(pre.shape)} float32 | "
              f"normalize mean=[{mean_s}] std=[{std_s}]")
        print(f"                 {'ch':<3}{'min':>9}{'max':>9}{'mean':>9}{'std':>9}")
        for c, label in enumerate('RGB'):
            ch = pre[:, c, :, :]
            print(f"                 {label:<3}{float(ch.min()):>9.3f}"
                  f"{float(ch.max()):>9.3f}{float(ch.mean()):>9.3f}"
                  f"{float(ch.std()):>9.3f}")

    print(f"  features   : dim={res['feat_dim']} | mean={res['feat_mean']:+.3f} "
          f"std={res['feat_std']:.3f} | L2={res['feat_norm']:.3f}")
    print(f"  MOS        : {res['mos']:.2f} / 5.00  (norm {res['mos_norm']:.3f})  "
          f"[{res['rating']}]")
    print(f"  time       : {res['elapsed']:.2f} s")

    if verbose:
        print("  per-frame  :")
        for i, (fi, ts, nrm) in enumerate(zip(res['frame_indices'],
                                              res['timestamps'],
                                              res['per_frame_norms'])):
            ts_s = f"{ts:.2f}s" if ts is not None else "n/a"
            print(f"      frame {i:2d}  idx={fi:>6d}  t={ts_s:>8}  feat_L2={nrm:.3f}")

    if save_dir is not None and res.get('frames_np') is not None:
        from PIL import Image
        out = Path(save_dir) / Path(path).stem
        out.mkdir(parents=True, exist_ok=True)
        for i, frame in enumerate(res['frames_np']):
            Image.fromarray(frame).save(out / f"frame_{i:02d}.jpg", 'JPEG', quality=90)
        print(f"  saved raw  : {len(res['frames_np'])} frames -> {out}")

    if save_pre_dir is not None and pre is not None:
        out = Path(save_pre_dir) / Path(path).stem
        out.mkdir(parents=True, exist_ok=True)
        for i in range(pre.shape[0]):
            _denormalize_to_pil(pre[i]).save(out / f"pre_{i:02d}.jpg", 'JPEG', quality=90)
        print(f"  saved pre  : {pre.shape[0]} frames -> {out}")

    if show_pre and pre is not None and res.get('frames_np') is not None:
        montage_path = Path.cwd() / f"preprocess_preview_{Path(path).stem}.png"
        _make_montage(res['frames_np'], pre, montage_path)
        print(f"  montage    : {montage_path} (raw top / preprocessed bottom)")
        try:
            os.startfile(str(montage_path))  # Windows
        except Exception:
            pass


def main():
    parser = argparse.ArgumentParser(description="Deep MOS prediction (Phase 4 SVR)")
    parser.add_argument('video_paths', nargs='+', help="Path(s) to video file(s)")
    parser.add_argument('--ckpt-dir', default=CKPT_DIR_DEFAULT,
                        help="Folder holding svr_deep_*.pkl (default: checkpoints)")
    parser.add_argument('--num-frames', type=int, default=None,
                        help="Frames to sample (default: value stored in checkpoint)")
    parser.add_argument('-v', '--verbose', action='store_true',
                        help="Also print per-frame feature L2 norms and timestamps")
    parser.add_argument('--save-frames', default=None, metavar='DIR',
                        help="Write the sampled frames as JPEGs under DIR/<video_stem>/")
    parser.add_argument('--save-preprocessed', default=None, metavar='DIR',
                        help="Write the preprocessed (denormalized 224x224) frames under DIR/<video_stem>/")
    parser.add_argument('--show-preprocessed', action='store_true',
                        help="Save and open a montage: raw frames (top) vs preprocessed (bottom)")
    args = parser.parse_args()

    for p in args.video_paths:
        if not Path(p).exists():
            print(f"Error: video not found: {p}")
            return 1

    svr, backbone, transform, cfg, meta = load_deep_model(args.ckpt_dir)
    if svr is None:
        print(f"Error: no svr_deep_*.pkl found in '{args.ckpt_dir}'")
        return 1

    device, cfg_frames, mos_min, mos_max = cfg
    n_frames = args.num_frames or cfg_frames
    print("\nDeep MOS - Phase 4")
    print(f"  checkpoint : {meta['ckpt_path']}")
    print(f"  backbone   : {meta['backbone']}  ({meta['weights_source']}; "
          f"feat_dim={meta['feat_dim']})")
    print(f"  device     : {device} | frames: {n_frames} | "
          f"img: {meta['img_size']}x{meta['img_size']} | "
          f"MOS range: [{meta['mos_min']}, {meta['mos_max']}]")
    print("=" * 70)

    for p in args.video_paths:
        res = predict_mos(p, svr, backbone, transform, cfg, args.num_frames)
        _print_report(p, res, args.verbose, args.save_frames,
                      args.save_preprocessed, args.show_preprocessed)

    print("\n" + "=" * 70 + "\n")
    return 0


if __name__ == '__main__':
    sys.exit(main())

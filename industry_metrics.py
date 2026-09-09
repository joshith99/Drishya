"""
IndustryMetrics - Integration of industry-standard classical video quality metrics.

This module provides access to peer-reviewed, well-established no-reference
quality assessment algorithms:
  - BRISQUE (Blind/Referenceless Image Spatial Quality Evaluator)
  - NIQE   (Natural Image Quality Evaluator, implemented via the `piq` package)

Note: this file is intentionally NOT named `brisque.py` so it never shadows the
`brisque` PyPI package. Scripts that were previously named `brisque.py`/
`compare_brisque.py` have been removed; run `python NQS.py` or the
`industry_metrics.py` CLI instead.
"""

import os
import sys
import argparse

import cv2
import numpy as np

# ── BRISQUE (pip package: `brisque`) ─────────────────────────
try:
    from brisque import BRISQUE
    BRISQUE_AVAILABLE = True
except ImportError:
    BRISQUE = None
    BRISQUE_AVAILABLE = False


# The `brisque` package can return a 2-D feature array where a 1-D array is
# expected, producing a wrong (36x36) scaling matrix. Guard every element so the
# standard per-element scaling formula is always applied to a flat 1-D vector.
def _fixed_scale_features(self, features):
    min_ = np.array(self.scale_params['min_'], dtype=np.float64)
    max_ = np.array(self.scale_params['max_'], dtype=np.float64)

    def _make_scalar(x):
        if isinstance(x, np.ndarray):
            return float(np.mean(x))
        try:
            return float(x)
        except Exception:
            return float(np.mean(np.asarray(x)))

    features_flat = np.array([_make_scalar(f) for f in features], dtype=np.float64)
    return -1 + (2.0 / (max_ - min_) * (features_flat - min_))


if BRISQUE_AVAILABLE:
    BRISQUE.scale_features = _fixed_scale_features


# ── NIQE (self-contained, scipy) ─────────────────────────────
# The `piq` and `brisque` packages no longer ship NIQE (piq dropped it; the
# `brisque` package has BRISQUE only), so we implement NIQE directly from the
# algorithm in Mittal et al., "Making a 'Completely Blind' Image Quality
# Analyzer" (IEEE SPL 2013), using the official natural-scene model parameters
# shipped in `niqe_pris_params.npz` (same parameters used by BasicSR/EDVR).
import math
from pathlib import Path

import scipy
from scipy.ndimage import convolve
from scipy.special import gamma as _gamma

_NIQE_PARAMS_PATH = Path(__file__).resolve().parent / 'niqe_pris_params.npz'

try:
    _niqe_params = np.load(str(_NIQE_PARAMS_PATH))
    _MU_PRIS = np.asarray(_niqe_params['mu_pris_param'], dtype=np.float64)
    _COV_PRIS = np.asarray(_niqe_params['cov_pris_param'], dtype=np.float64)
    _GAUSS_WIN = np.asarray(_niqe_params['gaussian_window'], dtype=np.float64)
    NIQE_AVAILABLE = True
except (IOError, OSError, KeyError):
    _MU_PRIS = _COV_PRIS = _GAUSS_WIN = None
    NIQE_AVAILABLE = False

# GGD/AGGD shape-parameter lookup table (gamma values, 0.2 .. 10.0)
_GAM = np.arange(0.2, 10.001, 0.001)
_GAM_RECIP = np.reciprocal(_GAM)
_R_GAM = np.square(_gamma(_GAM_RECIP * 2)) / (
    _gamma(_GAM_RECIP) * _gamma(_GAM_RECIP * 3))


def _estimate_aggd_param(block):
    """Estimate AGGD (Asymmetric Generalized Gaussian Distribution) parameters."""
    block = block.flatten()
    left_std = np.sqrt(np.mean(block[block < 0] ** 2))
    right_std = np.sqrt(np.mean(block[block > 0] ** 2))
    gammahat = left_std / right_std
    rhat = (np.mean(np.abs(block))) ** 2 / np.mean(block ** 2)
    rhatnorm = (rhat * (gammahat ** 3 + 1) * (gammahat + 1)) / ((gammahat ** 2 + 1) ** 2)
    alpha = _GAM[np.argmin((_R_GAM - rhatnorm) ** 2)]
    beta_l = left_std * np.sqrt(_gamma(1 / alpha) / _gamma(3 / alpha))
    beta_r = right_std * np.sqrt(_gamma(1 / alpha) / _gamma(3 / alpha))
    return alpha, beta_l, beta_r


def _compute_feature(block):
    """Compute the 18 NIQE features for one block (Eq. 7-9 of the paper)."""
    feat = []
    alpha, beta_l, beta_r = _estimate_aggd_param(block)
    feat.extend([alpha, (beta_l + beta_r) / 2])

    shifts = [[0, 1], [1, 0], [1, 1], [1, -1]]
    for shift in shifts:
        shifted = np.roll(block, shift, axis=(0, 1))
        alpha, beta_l, beta_r = _estimate_aggd_param(block * shifted)
        mean = (beta_r - beta_l) * (_gamma(2 / alpha) / _gamma(1 / alpha))
        feat.extend([alpha, mean, beta_l, beta_r])
    return feat


def _niqe_score(gray, mu_pris, cov_pris, gaussian_window, block_size=96):
    """Full NIQE on a grayscale image [0,255] (adapted from BasicSR/EDVR)."""
    h, w = gray.shape
    if h < block_size or w < block_size:
        return None
    num_block_h = math.floor(h / block_size)
    num_block_w = math.floor(w / block_size)
    img = gray[0:num_block_h * block_size, 0:num_block_w * block_size]

    distparam = []
    for scale in (1, 2):
        mu = convolve(img, gaussian_window, mode='nearest')
        sigma = np.sqrt(np.abs(convolve(np.square(img), gaussian_window, mode='nearest') - np.square(mu)))
        img_normalized = (img - mu) / (sigma + 1)

        feat = []
        for idx_w in range(num_block_w):
            for idx_h in range(num_block_h):
                block = img_normalized[
                    idx_h * block_size // scale:(idx_h + 1) * block_size // scale,
                    idx_w * block_size // scale:(idx_w + 1) * block_size // scale,
                ]
                feat.append(_compute_feature(block))
        distparam.append(np.array(feat))

        if scale == 1:
            h, w = img.shape
            img = cv2.resize(img / 255.0, (w // 2, h // 2),
                             interpolation=cv2.INTER_LINEAR) * 255.0

    distparam = np.concatenate(distparam, axis=1)
    mu_dist = np.nanmean(distparam, axis=0)
    dist_no_nan = distparam[~np.isnan(distparam).any(axis=1)]
    cov_dist = np.cov(dist_no_nan, rowvar=False)

    invcov = np.linalg.pinv((cov_pris + cov_dist) / 2)
    quality = np.sqrt(np.matmul(
        np.matmul((mu_pris - mu_dist), invcov), np.transpose((mu_pris - mu_dist))))
    return float(np.squeeze(quality))


class NIQECalculator:
    """
    Natural Image Quality Evaluator (NIQE) - self-contained implementation.

    Opinion-unaware metric built from Natural Scene Statistics of local patches
    compared against a multivariate Gaussian model of natural images.
    Lower scores indicate better quality (typical natural images ~3-10).
    """

    def compute_niqe_score(self, frame):
        """
        Compute the NIQE score for a frame.

        Args:
            frame: Input image (BGR or grayscale, numpy array [0-255])

        Returns:
            float: NIQE score (lower is better), or None if unavailable/error.
        """
        if not NIQE_AVAILABLE:
            return None
        try:
            if len(frame.shape) == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                gray = frame
            img = np.round(gray.astype(np.float64))
            return _niqe_score(img, _MU_PRIS, _COV_PRIS, _GAUSS_WIN)
        except Exception as e:
            print(f"NIQE computation error: {e}")
            return None


class IndustryMetrics:
    """
    Unified interface for industry-standard no-reference video quality metrics.

    Supports:
    - BRISQUE: Blind/Referenceless Image Spatial Quality Evaluator
    - NIQE: Natural Image Quality Evaluator

    Usage:
        metrics = IndustryMetrics()
        frame_scores = metrics.score_frame(frame)
        video_scores = metrics.score_video(video_path, num_samples=10)
    """

    def __init__(self):
        """Initialize industry metrics with available evaluators."""
        self.brisque = BRISQUE(url=False) if BRISQUE_AVAILABLE else None
        self.niqe = NIQECalculator() if NIQE_AVAILABLE else None

    def score_frame(self, frame):
        """
        Score a single frame using available industry metrics.

        Args:
            frame: Input frame (BGR format, numpy array)

        Returns:
            dict: {brisque_score, niqe_score, available_metrics}
        """
        scores = {
            'brisque_score': None,
            'niqe_score': None,
            'available_metrics': []
        }

        # Convert to RGB for metric libraries
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            rgb_frame = frame

        # BRISQUE scoring
        if self.brisque:
            try:
                scores['brisque_score'] = self.brisque.score(rgb_frame)
                scores['available_metrics'].append('brisque')
            except Exception as e:
                print(f"BRISQUE score failed: {e}")

        # NIQE scoring (original BGR frame — cvtColor to gray expects BGR)
        if self.niqe:
            try:
                niqe_val = self.niqe.compute_niqe_score(frame)
                if niqe_val is not None:
                    scores['niqe_score'] = niqe_val
                    scores['available_metrics'].append('niqe')
            except Exception as e:
                print(f"NIQE score failed: {e}")

        return scores

    def score_video(self, video_path, num_samples=10, sample_method='random', seed=None):
        """
        Score video frames using industry metrics.

        Args:
            video_path: Path to video file
            num_samples: Number of frames to sample
            sample_method: 'random' or 'uniform' frame sampling
            seed: Optional RNG seed for reproducible random sampling

        Returns:
            dict: Video-level statistics for each metric
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        if total_frames == 0:
            cap.release()
            raise ValueError("Video has no frames")

        if seed is not None:
            np.random.seed(seed)

        # Select frames
        n = min(num_samples, total_frames)
        if sample_method == 'random':
            frame_indices = sorted(np.random.choice(total_frames, n, replace=False))
        elif sample_method == 'uniform':
            frame_indices = np.linspace(0, total_frames - 1, n, dtype=int)
        else:
            cap.release()
            raise ValueError("sample_method must be 'random' or 'uniform'")

        frame_scores = []

        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                continue

            scores = self.score_frame(frame)
            scores['frame_idx'] = frame_idx
            scores['timestamp'] = frame_idx / fps
            frame_scores.append(scores)

        cap.release()

        # Aggregate statistics
        video_stats = {
            'video_path': video_path,
            'total_frames_analyzed': len(frame_scores),
            'frame_scores': frame_scores
        }

        # Per-metric statistics
        if frame_scores:
            avail = set()
            for s in frame_scores:
                avail.update(s['available_metrics'])

            if 'brisque' in avail:
                brisque_scores = [s['brisque_score'] for s in frame_scores if s['brisque_score'] is not None]
                if brisque_scores:
                    video_stats['brisque_avg'] = np.mean(brisque_scores)
                    video_stats['brisque_std'] = np.std(brisque_scores)
                    video_stats['brisque_min'] = np.min(brisque_scores)
                    video_stats['brisque_max'] = np.max(brisque_scores)

            if 'niqe' in avail:
                niqe_scores = [s['niqe_score'] for s in frame_scores if s['niqe_score'] is not None]
                if niqe_scores:
                    video_stats['niqe_avg'] = np.mean(niqe_scores)
                    video_stats['niqe_std'] = np.std(niqe_scores)
                    video_stats['niqe_min'] = np.min(niqe_scores)
                    video_stats['niqe_max'] = np.max(niqe_scores)

        return video_stats

    def print_scores(self, frame_scores, metric='brisque'):
        """Pretty-print scores for a list of frames."""
        if not frame_scores:
            print("No frame scores to display")
            return

        print(f"\n{'Frame':>6} | {'Time':>7} | {metric.upper():>8}")
        print("-" * 30)
        for score in frame_scores:
            if metric == 'brisque':
                val = score.get('brisque_score', 'N/A')
            else:
                val = score.get('niqe_score', 'N/A')

            if val != 'N/A':
                print(f"{score['frame_idx']:6d} | {score['timestamp']:7.2f} | {val:8.2f}")
            else:
                print(f"{score['frame_idx']:6d} | {score['timestamp']:7.2f} | {val:>8}")


def main():
    parser = argparse.ArgumentParser(
        description="Industry Metrics - BRISQUE and NIQE video quality analyzer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze single video
  python industry_metrics.py video.mp4

  # Compare two videos
  python industry_metrics.py video1.mp4 video2.mp4

  # Analyze with more samples
  python industry_metrics.py video.mp4 --samples 20 --method uniform
        """
    )

    parser.add_argument("video_paths", nargs='+', help="Path(s) to video file(s)")
    parser.add_argument("--samples", "-s", type=int, default=10,
                        help="Number of frames to sample (default: 10)")
    parser.add_argument("--method", "-m", choices=['random', 'uniform'], default='random',
                        help="Frame sampling method (default: random)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducible sampling (default: None)")

    args = parser.parse_args()

    # Validate inputs
    for video_path in args.video_paths:
        if not os.path.exists(video_path):
            print(f"Error: Video file not found: {video_path}")
            sys.exit(1)

    # Initialize metrics
    metrics = IndustryMetrics()

    if len(args.video_paths) == 1:
        # Single video analysis
        print(f"\nAnalyzing: {args.video_paths[0]}")
        stats = metrics.score_video(args.video_paths[0], args.samples, args.method, seed=args.seed)

        print("\n" + "=" * 80)
        print(f"INDUSTRY METRICS SUMMARY: {os.path.basename(args.video_paths[0])}")
        print("=" * 80)

        if 'brisque_avg' in stats:
            print(f"BRISQUE Score (lower is better):")
            print(f"  Average: {stats['brisque_avg']:.2f}")
            print(f"  Range: {stats['brisque_min']:.2f} - {stats['brisque_max']:.2f}")
            print(f"  Std Dev: {stats['brisque_std']:.2f}")

        if 'niqe_avg' in stats:
            print(f"NIQE Score (lower is better):")
            print(f"  Average: {stats['niqe_avg']:.2f}")
            print(f"  Range: {stats['niqe_min']:.2f} - {stats['niqe_max']:.2f}")
            print(f"  Std Dev: {stats['niqe_std']:.2f}")

        if not ('brisque_avg' in stats or 'niqe_avg' in stats):
            print("No industry metrics available for this video.")

        print("=" * 80)

    elif len(args.video_paths) == 2:
        # Comparison
        print(f"\nComparing:")
        print(f"  Video A: {args.video_paths[0]}")
        print(f"  Video B: {args.video_paths[1]}")

        stats_a = metrics.score_video(args.video_paths[0], args.samples, args.method, seed=args.seed)
        stats_b = metrics.score_video(args.video_paths[1], args.samples, args.method, seed=args.seed)

        print("\n" + "=" * 80)
        print("INDUSTRY METRICS COMPARISON")
        print("=" * 80)

        if 'brisque_avg' in stats_a and 'brisque_avg' in stats_b:
            print(f"\nBRISQUE Score (lower is better):")
            print(f"  Video A: {stats_a['brisque_avg']:.2f} (range {stats_a['brisque_min']:.2f}-{stats_a['brisque_max']:.2f})")
            print(f"  Video B: {stats_b['brisque_avg']:.2f} (range {stats_b['brisque_min']:.2f}-{stats_b['brisque_max']:.2f})")
            diff = stats_b['brisque_avg'] - stats_a['brisque_avg']
            better = "A" if diff > 0 else ("B" if diff < 0 else "Tie")
            print(f"  Difference (B - A): {diff:+.2f} → Better: {better}")

        if 'niqe_avg' in stats_a and 'niqe_avg' in stats_b:
            print(f"\nNIQE Score (lower is better):")
            print(f"  Video A: {stats_a['niqe_avg']:.2f} (range {stats_a['niqe_min']:.2f}-{stats_a['niqe_max']:.2f})")
            print(f"  Video B: {stats_b['niqe_avg']:.2f} (range {stats_b['niqe_min']:.2f}-{stats_b['niqe_max']:.2f})")
            diff = stats_b['niqe_avg'] - stats_a['niqe_avg']
            better = "A" if diff > 0 else ("B" if diff < 0 else "Tie")
            print(f"  Difference (B - A): {diff:+.2f} → Better: {better}")

        print("=" * 80)

    else:
        print("Error: Please provide 1 or 2 video paths")
        sys.exit(1)


if __name__ == "__main__":
    main()

"""
NQS - No-Reference Quality Scorer
A comprehensive video quality assessment tool that combines multiple NR metrics.
"""

import cv2
import numpy as np
import argparse
import os
import sys
from pathlib import Path
from nr_features import NoReferenceFeatures
from industry_metrics import IndustryMetrics

# Ensure current directory is in path for local imports
_current_dir = os.path.dirname(os.path.abspath(__file__))
if _current_dir not in sys.path:
    sys.path.insert(0, _current_dir)

from temporal_features import TemporalFeatures

# Import BRISQUE with safe path handling
BRISQUE_AVAILABLE = False
BRISQUE = None

try:
    # Handle the import shadowing issue
    current_dir = os.path.dirname(os.path.abspath(__file__))
    original_path = sys.path[:]
    
    # Remove current directory and relative paths to avoid shadowing
    if current_dir in sys.path:
        sys.path.remove(current_dir)
    if '' in sys.path:
        sys.path.remove('')
    if os.path.abspath('.') in sys.path:
        sys.path.remove(os.path.abspath('.'))

    from brisque import BRISQUE
    BRISQUE_AVAILABLE = True

    # Monkey-patch BRISQUE.scale_features to handle non-scalar features safely
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

        # Use same scale operation as original method.
        return -1 + (2.0 / (max_ - min_) * (features_flat - min_))

    BRISQUE.scale_features = _fixed_scale_features

    # Restore path
    sys.path = original_path
except ImportError as e:
    print(f"Warning: BRISQUE package not available. Install with: pip install brisque ({e})")



class NoReferenceQualityScorer:
    """
    Comprehensive No-Reference Quality Scorer that combines multiple metrics
    into a unified quality assessment framework.
    """

    def __init__(self):
        self.nr_features = NoReferenceFeatures()
        self.industry_metrics = IndustryMetrics()
        self.temporal_features = TemporalFeatures()

    def score_frame(self, frame):
        """
        Calculate comprehensive quality score for a single frame.

        Args:
            frame: Input frame (BGR format)

        Returns:
            dict: Quality metrics and composite score
        """
        # Convert to grayscale if needed
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame

        # Calculate NR features
        blur_score = self.nr_features.laplacian_variance(frame)
        block8_score = self.nr_features.blockiness(frame, block_size=8)
        block16_score = self.nr_features.blockiness(frame, block_size=16)
        noise_score = self.nr_features.noise_estimation(frame)

        # Get industry metrics (BRISQUE, NIQE, etc.)
        industry_scores = self.industry_metrics.score_frame(frame)
        brisque_score = industry_scores.get('brisque_score')
        niqe_score = industry_scores.get('niqe_score')

        # Calculate composite quality score
        composite_score = self._calculate_composite_score(
            blur_score, block8_score, block16_score, noise_score, brisque_score, niqe_score
        )

        return {
            'blur_score': blur_score,
            'block8_score': block8_score,
            'block16_score': block16_score,
            'noise_score': noise_score,
            'brisque_score': brisque_score,
            'niqe_score': niqe_score,
            'composite_quality_score': composite_score,
            'quality_rating': self._get_quality_rating(composite_score)
        }

    def _calculate_composite_score(self, blur, block8, block16, noise, brisque=None, niqe=None, temporal=None):
        """
        Calculate a composite quality score from individual metrics.

        Higher composite score = better quality
        """
        # Normalize each metric to 0-1 scale (1 = best quality, 0 = worst)
        # Blur: higher is better, normalize to 0-1
        blur_norm = min(blur / 500.0, 1.0)  # Assume 500 is good blur threshold

        # Blockiness: lower is better, invert and normalize
        block8_norm = max(0, 1.0 - block8 / 20.0)  # Assume 20 is high blockiness
        block16_norm = max(0, 1.0 - block16 / 30.0)  # Assume 30 is high blockiness

        # Noise: lower is better, invert and normalize
        noise_norm = max(0, 1.0 - noise / 50.0)  # Assume 50 is high noise

        # BRISQUE: lower is better (typically 0-100), invert and normalize
        brisque_norm = 1.0
        if brisque is not None:
            brisque_norm = max(0, 1.0 - brisque / 100.0)

        # NIQE: lower is better (typically 0-100), invert and normalize
        niqe_norm = 1.0
        if niqe is not None:
            niqe_norm = max(0, 1.0 - niqe / 100.0)

        temporal_norm = temporal if temporal is not None else 1.0

        # Weighted combination (renormalized over available metrics).
        weighted_metrics = [
            ('blur', blur_norm, 0.25),
            ('block8', block8_norm, 0.20),
            ('block16', block16_norm, 0.15),
            ('noise', noise_norm, 0.20),
        ]

        if brisque is not None:
            weighted_metrics.append(('brisque', brisque_norm, 0.15))
        if niqe is not None:
            weighted_metrics.append(('niqe', niqe_norm, 0.05))
        if temporal is not None:
            weighted_metrics.append(('temporal', temporal_norm, 0.20))

        weighted_sum = sum(val * weight for _, val, weight in weighted_metrics)
        total_weight = sum(weight for _, _, weight in weighted_metrics)
        return weighted_sum / total_weight if total_weight > 0 else 0.0

    def _get_quality_rating(self, composite_score):
        """
        Convert composite score to qualitative rating.
        """
        if composite_score >= 0.8:
            return "Excellent"
        elif composite_score >= 0.6:
            return "Good"
        elif composite_score >= 0.4:
            return "Fair"
        elif composite_score >= 0.2:
            return "Poor"
        else:
            return "Very Poor"

    def analyze_video(self, video_path, num_samples=10, sample_method='random', temporal=False, temporal_stride=1):
        """
        Analyze video quality by sampling frames.

        Args:
            video_path: Path to video file
            num_samples: Number of frames to sample
            sample_method: 'random' or 'uniform'

        Returns:
            dict: Video-level quality statistics
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        if total_frames == 0:
            raise ValueError("Video has no frames")

        # Select frames to analyze
        if sample_method == 'random':
            frame_indices = sorted(np.random.choice(total_frames, min(num_samples, total_frames), replace=False))
        elif sample_method == 'uniform':
            frame_indices = np.linspace(0, total_frames-1, min(num_samples, total_frames), dtype=int)
        else:
            raise ValueError("sample_method must be 'random' or 'uniform'")

        print(f"Analyzing {len(frame_indices)} frames from '{os.path.basename(video_path)}'...")
        print("-" * 90)
        print(f"{'Frame':>6} | {'Time':>7} | {'Blur':>6} | {'Block8':>7} | {'Noise':>6} | {'BRISQUE':>7} | {'Quality':>8}")
        print("-" * 90)

        frame_scores = []
        temporal_frames = []

        for idx, frame_idx in enumerate(frame_indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()

            if not ret:
                continue

            timestamp = frame_idx / fps
            scores = self.score_frame(frame)

            if temporal and (idx % max(1, temporal_stride) == 0):
                temporal_frames.append(frame.copy())

            print(f"{frame_idx:6d} | {timestamp:7.2f} | {scores['blur_score']:6.2f} | "
                  f"{scores['block8_score']:7.2f} | {scores['noise_score']:6.2f} | "
                  f"{str(scores['brisque_score']) if scores['brisque_score'] is not None else 'N/A':>7} | "
                  f"{scores['composite_quality_score']:8.2f}")

            frame_scores.append({
                'frame_idx': frame_idx,
                'timestamp': timestamp,
                **scores
            })

        cap.release()

        temporal_summary = self.temporal_features.score_sequence(temporal_frames) if temporal else None

        if temporal_summary and temporal_summary.get('temporal_quality_score') is not None:
            temporal_quality = temporal_summary['temporal_quality_score']
            for s in frame_scores:
                s['temporal_quality_score'] = temporal_quality
                s['composite_quality_score'] = self._calculate_composite_score(
                    s['blur_score'],
                    s['block8_score'],
                    s['block16_score'],
                    s['noise_score'],
                    s.get('brisque_score'),
                    s.get('niqe_score'),
                    temporal=temporal_quality,
                )
                s['quality_rating'] = self._get_quality_rating(s['composite_quality_score'])

        # Calculate video-level statistics
        if frame_scores:
            composite_scores = [s['composite_quality_score'] for s in frame_scores]
            video_stats = {
                'video_path': video_path,
                'total_frames_analyzed': len(frame_scores),
                'avg_quality_score': np.mean(composite_scores),
                'std_quality_score': np.std(composite_scores),
                'min_quality_score': np.min(composite_scores),
                'max_quality_score': np.max(composite_scores),
                'quality_rating': self._get_quality_rating(np.mean(composite_scores)),
                'frame_scores': frame_scores
            }

            if temporal_summary:
                video_stats['temporal_metrics'] = temporal_summary

            print("\n" + "="*90)
            print(f"VIDEO QUALITY SUMMARY: {os.path.basename(video_path)}")
            print("="*90)
            print(f"Average Quality Score: {video_stats['avg_quality_score']:.3f} ({video_stats['quality_rating']})")
            print(f"Score Range: {video_stats['min_quality_score']:.3f} - {video_stats['max_quality_score']:.3f}")
            print(f"Score Std Dev: {video_stats['std_quality_score']:.3f}")
            if temporal_summary and temporal_summary.get('temporal_quality_score') is not None:
                print("\nTemporal Metrics:")
                print(f"  Temporal Quality Score: {temporal_summary['temporal_quality_score']:.3f} (higher is better)")
                print(f"  Avg SSIM: {temporal_summary['avg_ssim']:.4f}")
                print(f"  Freeze Ratio: {temporal_summary['freeze_ratio']:.3f}")
                print(f"  Freeze Events: {temporal_summary['freeze_events']}")
                print(f"  Avg Jitter Index: {temporal_summary['avg_jitter_index']:.3f}")
            print("="*90)

            return video_stats

        return None

    def compare_videos(self, video_a_path, video_b_path, num_samples=10, sample_method='random', temporal=False, temporal_stride=1):
        """
        Compare quality between two videos.

        Returns:
            dict: Comparison results
        """
        print("Analyzing Video A...")
        stats_a = self.analyze_video(video_a_path, num_samples, sample_method, temporal, temporal_stride)

        print("\nAnalyzing Video B...")
        stats_b = self.analyze_video(video_b_path, num_samples, sample_method, temporal, temporal_stride)

        if not stats_a or not stats_b:
            return None

        comparison = {
            'video_a': stats_a,
            'video_b': stats_b,
            'quality_difference': stats_a['avg_quality_score'] - stats_b['avg_quality_score'],
            'better_video': 'A' if stats_a['avg_quality_score'] > stats_b['avg_quality_score'] else 'B'
        }

        print("\n" + "="*90)
        print("VIDEO COMPARISON RESULTS")
        print("="*90)
        print(f"Video A ({os.path.basename(video_a_path)}): {stats_a['avg_quality_score']:.3f} ({stats_a['quality_rating']})")
        print(f"Video B ({os.path.basename(video_b_path)}): {stats_b['avg_quality_score']:.3f} ({stats_b['quality_rating']})")
        print(f"Quality Difference (A - B): {comparison['quality_difference']:+.3f}")
        print(f"Better Video: {comparison['better_video']}")
        print("="*90)

        return comparison


def main():
    parser = argparse.ArgumentParser(
        description="NQS - No-Reference Quality Scorer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze single video
  python NQS.py video.mp4

  # Compare two videos
  python NQS.py video1.mp4 video2.mp4

  # Analyze with more samples
  python NQS.py video.mp4 --samples 20 --method uniform
        """
    )

    parser.add_argument("video_paths", nargs='+', help="Path(s) to video file(s)")
    parser.add_argument("--samples", "-s", type=int, default=10,
                        help="Number of frames to sample (default: 10)")
    parser.add_argument("--method", "-m", choices=['random', 'uniform'], default='random',
                        help="Frame sampling method (default: random)")
    parser.add_argument("--temporal", action='store_true',
                        help="Enable temporal quality metrics (SSIM/flow/freeze)")
    parser.add_argument("--temporal_stride", type=int, default=1,
                        help="Use every Nth sampled frame for temporal metrics (default: 1)")

    args = parser.parse_args()

    # Validate inputs
    for video_path in args.video_paths:
        if not os.path.exists(video_path):
            print(f"Error: Video file not found: {video_path}")
            return

    # Initialize scorer
    scorer = NoReferenceQualityScorer()

    if len(args.video_paths) == 1:
        # Single video analysis
        scorer.analyze_video(
            args.video_paths[0],
            args.samples,
            args.method,
            temporal=args.temporal,
            temporal_stride=args.temporal_stride,
        )
    elif len(args.video_paths) == 2:
        # Video comparison
        scorer.compare_videos(
            args.video_paths[0],
            args.video_paths[1],
            args.samples,
            args.method,
            temporal=args.temporal,
            temporal_stride=args.temporal_stride,
        )
    else:
        print("Error: Please provide 1 or 2 video paths")


if __name__ == "__main__":
    main()

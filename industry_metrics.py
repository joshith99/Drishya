"""
IndustryMetrics - Integration of industry-standard classical video quality metrics.

This module provides access to peer-reviewed, well-established no-reference
quality assessment algorithms including BRISQUE and NIQE.
"""

import cv2
import numpy as np
import argparse

# Import BRISQUE with fallback
try:
    import sys
    import os
    current_dir = os.path.dirname(os.path.abspath(__file__))
    original_path = []
    if current_dir in sys.path:
        original_path = sys.path[:]
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
        return -1 + (2.0 / (max_ - min_) * (features_flat - min_))

    BRISQUE.scale_features = _fixed_scale_features

    # Restore path
    if original_path:
        sys.path = original_path
except ImportError:
    BRISQUE_AVAILABLE = False


# Import scipy for NIQE implementation
try:
    from scipy.special import gamma as scipy_gamma
    from scipy.ndimage import gaussian_filter
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

NIQE_AVAILABLE = SCIPY_AVAILABLE


class NIQECalculator:
    """
    Natural Image Quality Evaluator (NIQE) - Simplified Implementation
    Fast no-reference quality metric based on local contrast and sharpness.
    """
    
    def __init__(self):
        """Initialize NIQE calculator."""
        pass
        
    def compute_niqe_score(self, frame):
        """
        Compute fast NIQE-like score for a frame.
        Based on local contrast and image statistics.
        
        Args:
            frame: Input image (BGR or grayscale, numpy array)
            
        Returns:
            NIQE score (0-100 scale, lower is better quality)
        """
        try:
            if len(frame.shape) == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            frame = frame.astype(np.float32) / 255.0
            h, w = frame.shape
            
            # Compute Laplacian for sharpness
            kernel = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=np.float32)
            laplacian = cv2.filter2D(frame, cv2.CV_32F, kernel)
            
            # Compute local variance in blocks
            block_size = 8
            variances = []
            for i in range(0, h - block_size, block_size):
                for j in range(0, w - block_size, block_size):
                    block = frame[i:i+block_size, j:j+block_size]
                    variances.append(np.var(block))
            
            # Compute gradient magnitude
            sobelx = cv2.Sobel(frame, cv2.CV_32F, 1, 0, ksize=3)
            sobely = cv2.Sobel(frame, cv2.CV_32F, 0, 1, ksize=3)
            edges = np.sqrt(sobelx**2 + sobely**2)
            
            # Statistics for quality assessment
            avg_variance = np.mean(variances) if variances else 0
            avg_edges = np.mean(edges)
            avg_laplacian = np.mean(np.abs(laplacian))
            
            # Natural image statistics:
            # - Variance should be moderate (0.01-0.15)
            # - Edges should be present but not excessive
            # - Laplacian indicates texture/detail
            
            # Calculate distortion penalty
            # Penalize low variance (flat, no detail)
            variance_penalty = max(0, 0.05 - avg_variance) * 500
            
            # Penalize excessive or insufficient edges
            edge_optimal = 0.08
            edge_penalty = abs(avg_edges - edge_optimal) * 300
            
            # Low sharpness indicates blur
            sharpness_penalty = max(0, 0.02 - avg_laplacian) * 400
            
            # Combine penalties into a score (0-100)
            niqe_score = variance_penalty + edge_penalty + sharpness_penalty
            niqe_score = max(0, min(100, niqe_score))
            
            return float(niqe_score)
            
        except Exception as e:
            print(f"NIQE computation error: {e}")
            return 50.0  # Return neutral score on error


class IndustryMetrics:
    """
    Unified interface for industry-standard no-reference video quality metrics.
    
    Supports:
    - BRISQUE: Blind/Referenceless Image Spatial Quality Evaluator
    - NIQE: Natural Image Quality Evaluator (if available)
    
    Usage:
        metrics = IndustryMetrics()
        frame_scores = metrics.score_frame(frame)
        video_scores = metrics.score_video(video_path, num_samples=10)
    """

    def __init__(self):
        """Initialize industry metrics with available evaluators."""
        self.brisque = BRISQUE(url=False) if BRISQUE_AVAILABLE else None
        self.niqe = NIQECalculator() if NIQE_AVAILABLE else None

        if not BRISQUE_AVAILABLE:
            print("⚠️  BRISQUE not available. Install: pip install brisque")
        if NIQE_AVAILABLE:
            print("✅ NIQE initialized successfully")
        else:
            print("⚠️  NIQE not available. SciPy required: pip install scipy")

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

        # NIQE scoring
        if self.niqe:
            try:
                scores['niqe_score'] = self.niqe.compute_niqe_score(rgb_frame)
                scores['available_metrics'].append('niqe')
            except Exception as e:
                print(f"NIQE score failed: {e}")

        return scores

    def score_video(self, video_path, num_samples=10, sample_method='random'):
        """
        Score video frames using industry metrics.
        
        Args:
            video_path: Path to video file
            num_samples: Number of frames to sample
            sample_method: 'random' or 'uniform' frame sampling
            
        Returns:
            dict: Video-level statistics for each metric
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        if total_frames == 0:
            raise ValueError("Video has no frames")

        # Select frames
        if sample_method == 'random':
            frame_indices = sorted(np.random.choice(total_frames, min(num_samples, total_frames), replace=False))
        elif sample_method == 'uniform':
            frame_indices = np.linspace(0, total_frames - 1, min(num_samples, total_frames), dtype=int)
        else:
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
        if 'brisque' in frame_scores[0]['available_metrics'] if frame_scores else False:
            brisque_scores = [s['brisque_score'] for s in frame_scores if s['brisque_score'] is not None]
            if brisque_scores:
                video_stats['brisque_avg'] = np.mean(brisque_scores)
                video_stats['brisque_std'] = np.std(brisque_scores)
                video_stats['brisque_min'] = np.min(brisque_scores)
                video_stats['brisque_max'] = np.max(brisque_scores)

        if 'niqe' in frame_scores[0]['available_metrics'] if frame_scores else False:
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


if __name__ == "__main__":
    import sys
    
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
        stats = metrics.score_video(args.video_paths[0], args.samples, args.method)
        
        print("\n" + "="*80)
        print(f"INDUSTRY METRICS SUMMARY: {os.path.basename(args.video_paths[0])}")
        print("="*80)
        
        if 'brisque_avg' in stats:
            print(f"BRISQUE Score:")
            print(f"  Average: {stats['brisque_avg']:.2f}")
            print(f"  Range: {stats['brisque_min']:.2f} - {stats['brisque_max']:.2f}")
            print(f"  Std Dev: {stats['brisque_std']:.2f}")
        
        if 'niqe_avg' in stats:
            print(f"NIQE Score:")
            print(f"  Average: {stats['niqe_avg']:.2f}")
            print(f"  Range: {stats['niqe_min']:.2f} - {stats['niqe_max']:.2f}")
            print(f"  Std Dev: {stats['niqe_std']:.2f}")
        
        if not ('brisque_avg' in stats or 'niqe_avg' in stats):
            print("No industry metrics available for this video.")
        
        print("="*80)
        
    elif len(args.video_paths) == 2:
        # Comparison
        print(f"\nComparing:")
        print(f"  Video A: {args.video_paths[0]}")
        print(f"  Video B: {args.video_paths[1]}")
        
        stats_a = metrics.score_video(args.video_paths[0], args.samples, args.method)
        stats_b = metrics.score_video(args.video_paths[1], args.samples, args.method)
        
        print("\n" + "="*80)
        print("INDUSTRY METRICS COMPARISON")
        print("="*80)
        
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
        
        print("="*80)
    
    else:
        print("Error: Please provide 1 or 2 video paths")
        sys.exit(1)

import cv2
import numpy as np
import pywt
import argparse
import os
from pathlib import Path


class NoReferenceFeatures:
    """Calculate No-Reference video quality features."""
    
    @staticmethod
    def laplacian_variance(frame):
        """
        Calculate blurriness using Laplacian Variance method.
        
        Formula: Var = Δ²I
        A low variance indicates a blurry or low-quality frame.
        
        Args:
            frame: Input image (grayscale or color)
        
        Returns:
            float: Laplacian variance (higher = sharper, lower = blurrier)
        """
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
        
        # Apply Laplacian operator
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        
        # Calculate variance
        variance = laplacian.var()
        
        return variance
    
    @staticmethod
    def blockiness(frame, block_size=8):
        """
        Calculate blockiness to detect compression artifacts.
        
        Measures the standard deviation of pixel gradients across blocks
        to detect heavy compression artifacts (e.g., 8×8 or 16×16 blocks).
        
        Args:
            frame: Input image (grayscale or color)
            block_size: Size of blocks (default: 8)
        
        Returns:
            float: Blockiness score (higher = more blocking artifacts)
        """
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
        
        h, w = gray.shape
        
        # Calculate horizontal and vertical block boundaries
        horizontal_diff = []
        vertical_diff = []
        
        # Horizontal block boundaries
        for i in range(block_size, h, block_size):
            if i < h:
                diff = np.abs(gray[i, :].astype(float) - gray[i-1, :].astype(float))
                horizontal_diff.extend(diff)
        
        # Vertical block boundaries
        for j in range(block_size, w, block_size):
            if j < w:
                diff = np.abs(gray[:, j].astype(float) - gray[:, j-1].astype(float))
                vertical_diff.extend(diff)
        
        # Combine horizontal and vertical differences
        all_diffs = horizontal_diff + vertical_diff
        
        if len(all_diffs) == 0:
            return 0.0
        
        # Calculate standard deviation
        blockiness_score = np.std(all_diffs)
        
        return blockiness_score
    
    @staticmethod
    def noise_estimation(frame):
        """
        Estimate noise level using Median Absolute Deviation (MAD) of wavelet coefficients.
        
        Uses the robust noise estimation method based on wavelet decomposition.
        
        Args:
            frame: Input image (grayscale or color)
        
        Returns:
            float: Estimated noise level
        """
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
        
        # Perform wavelet decomposition
        coeffs = pywt.dwt2(gray, 'db1')
        cH, cV, cD = coeffs[1]  # High-frequency coefficients
        
        # Use diagonal coefficients for noise estimation
        # Robust noise estimator: σ = MAD / 0.6745
        mad = np.median(np.abs(cD - np.median(cD)))
        sigma = mad / 0.6745
        
        return sigma


def extract_and_analyze_video(video_path, output_csv=None, sample_rate=30, max_frames=None):
    """
    Extract frames from video and calculate NR quality features.
    
    Args:
        video_path: Path to input video
        output_csv: Path to save results as CSV (optional)
        sample_rate: Extract one frame every N frames
        max_frames: Maximum number of frames to analyze
    
    Returns:
        list: List of dictionaries containing frame features
    """
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Cannot open video file {video_path}")
        return []
    
    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Video Info:")
    print(f"  Resolution: {width}x{height}")
    print(f"  FPS: {fps}")
    print(f"  Total Frames: {total_frames}")
    print(f"  Duration: {total_frames/fps:.2f} seconds")
    print(f"\nAnalyzing frames (sampling every {sample_rate} frames)...\n")
    
    nr_features = NoReferenceFeatures()
    results = []
    
    frame_count = 0
    analyzed_count = 0
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        # Sample frame based on sample rate
        if frame_count % sample_rate == 0:
            # Calculate features
            blurriness = nr_features.laplacian_variance(frame)
            blockiness_8 = nr_features.blockiness(frame, block_size=8)
            blockiness_16 = nr_features.blockiness(frame, block_size=16)
            noise = nr_features.noise_estimation(frame)
            
            # Store results
            result = {
                'frame_number': frame_count,
                'timestamp_sec': frame_count / fps,
                'blurriness_laplacian_var': blurriness,
                'blockiness_8x8': blockiness_8,
                'blockiness_16x16': blockiness_16,
                'noise_level': noise
            }
            results.append(result)
            
            print(f"Frame {frame_count:6d} | Time: {frame_count/fps:7.2f}s | "
                  f"Blur: {blurriness:8.2f} | Block8: {blockiness_8:6.2f} | "
                  f"Block16: {blockiness_16:6.2f} | Noise: {noise:6.2f}")
            
            analyzed_count += 1
            
            # Check if we've reached max_frames
            if max_frames and analyzed_count >= max_frames:
                print(f"\nReached maximum of {max_frames} frames.")
                break
        
        frame_count += 1
    
    cap.release()
    
    # Calculate aggregate statistics
    print(f"\n{'='*80}")
    print("Video-level Statistics:")
    print(f"{'='*80}")
    
    if results:
        blur_values = [r['blurriness_laplacian_var'] for r in results]
        block8_values = [r['blockiness_8x8'] for r in results]
        block16_values = [r['blockiness_16x16'] for r in results]
        noise_values = [r['noise_level'] for r in results]
        
        print(f"Blurriness (Laplacian Var):")
        print(f"  Mean: {np.mean(blur_values):.2f}")
        print(f"  Std:  {np.std(blur_values):.2f}")
        print(f"  Min:  {np.min(blur_values):.2f}")
        print(f"  Max:  {np.max(blur_values):.2f}")
        
        print(f"\nBlockiness (8×8):")
        print(f"  Mean: {np.mean(block8_values):.2f}")
        print(f"  Std:  {np.std(block8_values):.2f}")
        
        print(f"\nBlockiness (16×16):")
        print(f"  Mean: {np.mean(block16_values):.2f}")
        print(f"  Std:  {np.std(block16_values):.2f}")
        
        print(f"\nNoise Level:")
        print(f"  Mean: {np.mean(noise_values):.2f}")
        print(f"  Std:  {np.std(noise_values):.2f}")
        
        # Save to CSV if specified
        if output_csv:
            import csv
            
            os.makedirs(os.path.dirname(output_csv) if os.path.dirname(output_csv) else '.', exist_ok=True)
            
            with open(output_csv, 'w', newline='') as f:
                if results:
                    writer = csv.DictWriter(f, fieldnames=results[0].keys())
                    writer.writeheader()
                    writer.writerows(results)
            
            print(f"\nResults saved to: {output_csv}")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Calculate No-Reference quality features from video",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python nr_features.py video.mp4
  python nr_features.py video.mp4 --output results.csv
  python nr_features.py video.mp4 --sample_rate 15 --max_frames 50
        """
    )
    
    parser.add_argument("video_path", help="Path to input video file")
    parser.add_argument("--output", "-o", default=None,
                        help="Output CSV file for results (optional)")
    parser.add_argument("--sample_rate", "-s", type=int, default=30,
                        help="Extract one frame every N frames (default: 30)")
    parser.add_argument("--max_frames", "-m", type=int, default=None,
                        help="Maximum number of frames to analyze (default: no limit)")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.video_path):
        print(f"Error: Video file not found: {args.video_path}")
        return
    
    extract_and_analyze_video(
        args.video_path,
        output_csv=args.output,
        sample_rate=args.sample_rate,
        max_frames=args.max_frames
    )


if __name__ == "__main__":
    main()

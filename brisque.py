import sys
import os

# --- FIX FOR IMPORT SHADOWING ---
# Because this script is named `brisque.py`, Python will try to import itself
# when we do `from brisque import BRISQUE`, causing a ModuleNotFoundError.
# To bypass this, we temporarily hide the current directory from sys.path 
# so Python forcefully looks for the pip installed 'brisque' package instead.
current_dir = os.path.dirname(os.path.abspath(__file__))
original_path = sys.path[:]

if current_dir in sys.path:
    sys.path.remove(current_dir)
if '' in sys.path:
    sys.path.remove('')
if os.path.abspath('.') in sys.path:
    sys.path.remove(os.path.abspath('.'))

try:
    from brisque import BRISQUE
finally:
    # Restore the path so we can import our local modules like nr_features.py
    sys.path = original_path
# --------------------------------

import cv2
import numpy as np
import random
import argparse
from nr_features import NoReferenceFeatures


def analyze_video_brisque(video_path, num_samples=10):
    """
    Take random frames from a video and calculate BRISQUE along with other metrics.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}")
        return
        
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    if total_frames == 0:
        print("Error: Video has 0 frames or cannot be read.")
        return
        
    # Pick random frames to analyze
    samples_to_take = min(num_samples, total_frames)
    frame_indices = sorted(random.sample(range(total_frames), samples_to_take))
    
    print(f"\nAnalyzing {samples_to_take} random frames from '{os.path.basename(video_path)}'...")
    print("-" * 85)
    print(f"{'Frame':>6} | {'Time(s)':>7} | {'Blur':>8} | {'Block8':>6} | {'Block16':>7} | {'Noise':>6} | {'BRISQUE':>7}")
    print("-" * 85)
    
    nr_features = NoReferenceFeatures()
    brisq_obj = BRISQUE(url=False)
    
    results = []
    
    for target_frame in frame_indices:
        # Jump directly to the random frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
        ret, frame = cap.read()
        
        if not ret:
            print(f"Warning: Could not read frame {target_frame}")
            continue
            
        timestamp = target_frame / fps
        
        # 1. Existing Features
        blur = nr_features.laplacian_variance(frame)
        block8 = nr_features.blockiness(frame, block_size=8)
        block16 = nr_features.blockiness(frame, block_size=16)
        noise = nr_features.noise_estimation(frame)
        
        # 2. BRISQUE Score
        # The brisque package expects a PIL Image or Ndarray
        from PIL import Image
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_frame)
        brisque_score = brisq_obj.score(pil_img)
        
        results.append({
            'frame': target_frame,
            'timestamp': timestamp,
            'blur': blur,
            'block8': block8,
            'block16': block16,
            'noise': noise,
            'brisque': brisque_score
        })
        
        print(f"{target_frame:6d} | {timestamp:7.2f} | {blur:8.2f} | "
              f"{block8:6.2f} | {block16:7.2f} | {noise:6.2f} | {brisque_score:7.2f}")

    cap.release()
    
    # Calculate Averages for the Video
    if results:
        avg_brisque = np.mean([r['brisque'] for r in results])
        avg_blur = np.mean([r['blur'] for r in results])
        avg_block8 = np.mean([r['block8'] for r in results])
        avg_block16 = np.mean([r['block16'] for r in results])
        avg_noise = np.mean([r['noise'] for r in results])
        
        print("=" * 85)
        print(f"VIDEO SUMMARY: {os.path.basename(video_path)}")
        print("=" * 85)
        print(f"Overall Average BRISQUE Score : {avg_brisque:.2f} (Lower is better perceptual quality)")
        print(f"Overall Average Blur          : {avg_blur:.2f} (Higher is sharper)")
        print(f"Overall Average Blockiness(8) : {avg_block8:.2f} (Lower means less compression)")
        print(f"Overall Average Blockiness(16): {avg_block16:.2f} (Lower means less compression)")
        print(f"Overall Average Noise         : {avg_noise:.2f} (Lower is cleaner)")
        print("=" * 85 + "\n")

    return results


def main():
    parser = argparse.ArgumentParser(description="Calculate BRISQUE and NR features for random frames in a video.")
    parser.add_argument("video_path", help="Path to the input video")
    parser.add_argument("--samples", "-s", type=int, default=10, help="Number of random frames to sample (default: 10)")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.video_path):
        print(f"Error: Video file not found: {args.video_path}")
        return
        
    analyze_video_brisque(args.video_path, args.samples)


if __name__ == "__main__":
    main()

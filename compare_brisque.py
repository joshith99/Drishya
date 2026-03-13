import sys
import os

# Bypass local brisque.py shadowing issue
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
    sys.path = original_path

import cv2
import numpy as np
import random
import argparse
import matplotlib.pyplot as plt
from PIL import Image
from nr_features import NoReferenceFeatures

def analyze_video(video_path, num_samples, brisq_obj, nr_features):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}")
        return None
        
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    if total_frames == 0:
        return None
        
    samples_to_take = min(num_samples, total_frames)
    frame_indices = sorted(random.sample(range(total_frames), samples_to_take))
    
    results = []
    
    for target_frame in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
        ret, frame = cap.read()
        if not ret: continue
        
        blur = nr_features.laplacian_variance(frame)
        block8 = nr_features.blockiness(frame, block_size=8)
        block16 = nr_features.blockiness(frame, block_size=16)
        noise = nr_features.noise_estimation(frame)
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_frame)
        brisque_score = brisq_obj.score(pil_img)
        
        results.append({
            'blur': blur, 'block8': block8, 'block16': block16,
            'noise': noise, 'brisque': brisque_score
        })

    cap.release()
    
    if not results: return None
    
    return {
        'brisque': np.mean([r['brisque'] for r in results]),
        'blur': np.mean([r['blur'] for r in results]),
        'block8': np.mean([r['block8'] for r in results]),
        'block16': np.mean([r['block16'] for r in results]),
        'noise': np.mean([r['noise'] for r in results])
    }

def main():
    parser = argparse.ArgumentParser(description="Compare two videos using BRISQUE and other No-Reference features.")
    parser.add_argument("video_a", help="Path to first video")
    parser.add_argument("video_b", help="Path to second video")
    parser.add_argument("--samples", "-s", type=int, default=10, help="Number of random frames to sample")
    
    args = parser.parse_args()
    
    brisq_obj = BRISQUE(url=False)
    nr_features = NoReferenceFeatures()
    
    print(f"Analyzing '{os.path.basename(args.video_a)}'...")
    res_a = analyze_video(args.video_a, args.samples, brisq_obj, nr_features)
    
    print(f"Analyzing '{os.path.basename(args.video_b)}'...")
    res_b = analyze_video(args.video_b, args.samples, brisq_obj, nr_features)
    
    if not res_a or not res_b:
        print("Analysis failed for one or both videos.")
        return
        
    print("\n" + "="*70)
    print(f"COMPARISON RESULTS ({args.samples} frames sampled)")
    print("="*70)
    print(f"{'Metric':<30} | {'Video A':<15} | {'Video B':<15}")
    print("-" * 70)
    print(f"{'BRISQUE (Lower is Better)':<30} | {res_a['brisque']:<15.2f} | {res_b['brisque']:<15.2f}")
    print(f"{'Blur (Higher is Better)':<30} | {res_a['blur']:<15.2f} | {res_b['blur']:<15.2f}")
    print(f"{'Blockiness 8x8 (Lower = Better)':<30} | {res_a['block8']:<15.2f} | {res_b['block8']:<15.2f}")
    print(f"{'Noise (Lower = Better)':<30} | {res_a['noise']:<15.2f} | {res_b['noise']:<15.2f}")
    
    print("\n" + "="*70)
    if res_a['brisque'] < res_b['brisque']:
        print(f"🏆 CONCLUSION: '{os.path.basename(args.video_a)}' has BETTER overall quality.")
    elif res_b['brisque'] < res_a['brisque']:
        print(f"🏆 CONCLUSION: '{os.path.basename(args.video_b)}' has BETTER overall quality.")
    else:
        print("⚖️ CONCLUSION: Both videos have identical average BRISQUE quality.")
    print("="*70)
    
    # Visualizing the comparison
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'Video Quality Comparison ({args.samples} frames averaged)', fontsize=16, fontweight='bold')
    
    vid_a_name = os.path.basename(args.video_a)
    vid_b_name = os.path.basename(args.video_b)
    
    def plot_bar(ax, title, val_a, val_b, ylabel, lower_is_better=True):
        bars = ax.bar(['Video A\n' + vid_a_name, 'Video B\n' + vid_b_name], [val_a, val_b], color=['#3498db', '#e74c3c'])
        ax.set_title(title, fontweight='bold')
        ax.set_ylabel(ylabel)
        
        # Highlight the better one with a green border
        if val_a != val_b:
            if lower_is_better:
                better_idx = 0 if val_a < val_b else 1
            else:
                better_idx = 0 if val_a > val_b else 1
            bars[better_idx].set_edgecolor('#27ae60')
            bars[better_idx].set_linewidth(4)
            
        # Add labels on top of bars
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')
                        
        # add padding above max for text
        ax.set_ylim(0, max(val_a, val_b) * 1.2)

    plot_bar(axs[0, 0], 'BRISQUE Score\n(Lower Score = Higher Perceptual Quality)', res_a['brisque'], res_b['brisque'], 'Score', lower_is_better=True)
    plot_bar(axs[0, 1], 'Sharpness / Blur (Laplacian)\n(Higher Score = Sharper Video)', res_a['blur'], res_b['blur'], 'Score', lower_is_better=False)
    plot_bar(axs[1, 0], 'Compression (Blockiness 8x8)\n(Lower Score = Less Artifacts)', res_a['block8'], res_b['block8'], 'Std Dev', lower_is_better=True)
    plot_bar(axs[1, 1], 'Noise Level\n(Lower Score = Cleaner Signal)', res_a['noise'], res_b['noise'], 'Noise σ', lower_is_better=True)

    plt.tight_layout()
    output_png = "comparison_brisque.png"
    plt.savefig(output_png, dpi=300)
    print(f"\nSaved graphical comparison report to: {output_png}")

if __name__ == "__main__":
    main()

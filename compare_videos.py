import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from nr_features import NoReferenceFeatures


def analyze_video_for_comparison(video_path, sample_rate=30, max_frames=None):
    """
    Analyze video and extract NR quality metrics.
    
    Args:
        video_path: Path to input video
        sample_rate: Extract one frame every N frames
        max_frames: Maximum number of frames to analyze
    
    Returns:
        tuple: (results_dict, video_info_dict)
    """
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Cannot open video file {video_path}")
        return None, None
    
    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Get file size in MB
    file_size = os.path.getsize(video_path) / (1024 * 1024)
    
    video_info = {
        'path': video_path,
        'file_size_mb': file_size,
        'width': width,
        'height': height,
        'fps': fps,
        'total_frames': total_frames,
        'duration_sec': total_frames / fps
    }
    
    print(f"\nAnalyzing: {os.path.basename(video_path)}")
    print(f"  File Size: {file_size:.2f} MB")
    print(f"  Resolution: {width}x{height}")
    print(f"  Duration: {total_frames/fps:.2f}s")
    
    nr_features = NoReferenceFeatures()
    results = {
        'frame_numbers': [],
        'timestamps': [],
        'blurriness': [],
        'blockiness_8x8': [],
        'blockiness_16x16': [],
        'noise': []
    }
    
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
            
            results['frame_numbers'].append(frame_count)
            results['timestamps'].append(frame_count / fps)
            results['blurriness'].append(blurriness)
            results['blockiness_8x8'].append(blockiness_8)
            results['blockiness_16x16'].append(blockiness_16)
            results['noise'].append(noise)
            
            analyzed_count += 1
            
            # Check if we've reached max_frames
            if max_frames and analyzed_count >= max_frames:
                break
        
        frame_count += 1
    
    cap.release()
    
    # Convert to numpy arrays
    for key in results:
        if key not in ['frame_numbers']:
            results[key] = np.array(results[key])
    
    print(f"  Analyzed {analyzed_count} frames")
    
    return results, video_info


def create_comparison_visualization(video_a_path, video_b_path, output_image=None, sample_rate=30):
    """
    Create a comprehensive side-by-side comparison visualization.
    
    Args:
        video_a_path: Path to first video
        video_b_path: Path to second video
        output_image: Path to save visualization (optional)
        sample_rate: Frame sampling rate
    """
    print("="*80)
    print("VIDEO QUALITY COMPARISON ANALYSIS")
    print("="*80)
    
    # Analyze both videos
    results_a, info_a = analyze_video_for_comparison(video_a_path, sample_rate=sample_rate)
    results_b, info_b = analyze_video_for_comparison(video_b_path, sample_rate=sample_rate)
    
    if results_a is None or results_b is None:
        print("Error: Could not analyze videos")
        return
    
    # Create comprehensive figure
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle('No-Reference Video Quality Assessment Comparison', fontsize=16, fontweight='bold')
    
    # Color scheme
    color_a = '#E74C3C'  # Red for Video A
    color_b = '#27AE60'  # Green for Video B
    
    # ==================== Blurriness Comparison ====================
    ax1 = plt.subplot(3, 2, 1)
    ax1.plot(results_a['timestamps'], results_a['blurriness'], 
             label='Video A', color=color_a, linewidth=2, alpha=0.7)
    ax1.plot(results_b['timestamps'], results_b['blurriness'], 
             label='Video B', color=color_b, linewidth=2, alpha=0.7)
    ax1.fill_between(results_a['timestamps'], results_a['blurriness'], alpha=0.2, color=color_a)
    ax1.fill_between(results_b['timestamps'], results_b['blurriness'], alpha=0.2, color=color_b)
    ax1.set_xlabel('Time (seconds)')
    ax1.set_ylabel('Laplacian Variance')
    ax1.set_title('Blurriness Metric (Higher = Sharper)', fontweight='bold')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    
    # Add mean lines
    mean_a = np.mean(results_a['blurriness'])
    mean_b = np.mean(results_b['blurriness'])
    ax1.axhline(y=mean_a, color=color_a, linestyle='--', alpha=0.5, linewidth=1.5)
    ax1.axhline(y=mean_b, color=color_b, linestyle='--', alpha=0.5, linewidth=1.5)
    
    # ==================== Blockiness (8x8) Comparison ====================
    ax2 = plt.subplot(3, 2, 2)
    ax2.plot(results_a['timestamps'], results_a['blockiness_8x8'], 
             label='Video A', color=color_a, linewidth=2, alpha=0.7)
    ax2.plot(results_b['timestamps'], results_b['blockiness_8x8'], 
             label='Video B', color=color_b, linewidth=2, alpha=0.7)
    ax2.fill_between(results_a['timestamps'], results_a['blockiness_8x8'], alpha=0.2, color=color_a)
    ax2.fill_between(results_b['timestamps'], results_b['blockiness_8x8'], alpha=0.2, color=color_b)
    ax2.set_xlabel('Time (seconds)')
    ax2.set_ylabel('Standard Deviation')
    ax2.set_title('Blockiness (8×8) - Compression Artifacts', fontweight='bold')
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)
    
    # ==================== Blockiness (16x16) Comparison ====================
    ax3 = plt.subplot(3, 2, 3)
    ax3.plot(results_a['timestamps'], results_a['blockiness_16x16'], 
             label='Video A', color=color_a, linewidth=2, alpha=0.7)
    ax3.plot(results_b['timestamps'], results_b['blockiness_16x16'], 
             label='Video B', color=color_b, linewidth=2, alpha=0.7)
    ax3.fill_between(results_a['timestamps'], results_a['blockiness_16x16'], alpha=0.2, color=color_a)
    ax3.fill_between(results_b['timestamps'], results_b['blockiness_16x16'], alpha=0.2, color=color_b)
    ax3.set_xlabel('Time (seconds)')
    ax3.set_ylabel('Standard Deviation')
    ax3.set_title('Blockiness (16×16) - Compression Artifacts', fontweight='bold')
    ax3.legend(loc='best')
    ax3.grid(True, alpha=0.3)
    
    # ==================== Noise Level Comparison ====================
    ax4 = plt.subplot(3, 2, 4)
    ax4.plot(results_a['timestamps'], results_a['noise'], 
             label='Video A', color=color_a, linewidth=2, alpha=0.7)
    ax4.plot(results_b['timestamps'], results_b['noise'], 
             label='Video B', color=color_b, linewidth=2, alpha=0.7)
    ax4.fill_between(results_a['timestamps'], results_a['noise'], alpha=0.2, color=color_a)
    ax4.fill_between(results_b['timestamps'], results_b['noise'], alpha=0.2, color=color_b)
    ax4.set_xlabel('Time (seconds)')
    ax4.set_ylabel('Noise Level (σ)')
    ax4.set_title('Noise Level (Lower = Cleaner)', fontweight='bold')
    ax4.legend(loc='best')
    ax4.grid(True, alpha=0.3)
    
    # ==================== Summary Statistics Comparison ====================
    ax5 = plt.subplot(3, 2, 5)
    ax5.axis('off')
    
    # Prepare statistics table
    stats_text = "QUALITY METRICS SUMMARY\n" + "="*50 + "\n\n"
    
    stats_text += f"{'VIDEO A (Large File)':<30} VIDEO B (Small File)\n"
    stats_text += "-"*60 + "\n"
    
    stats_text += f"File Size: {info_a['file_size_mb']:.2f} MB{' '*16} {info_b['file_size_mb']:.2f} MB\n"
    stats_text += f"Resolution: {info_a['width']}x{info_a['height']}{' '*18} {info_b['width']}x{info_b['height']}\n"
    stats_text += f"Duration: {info_a['duration_sec']:.2f}s{' '*21} {info_b['duration_sec']:.2f}s\n\n"
    
    stats_text += "Blurriness (Laplacian Var) - Higher is Better\n"
    mean_a = np.mean(results_a['blurriness'])
    mean_b = np.mean(results_b['blurriness'])
    stats_text += f"  Mean: {mean_a:.2f}{' '*26} {mean_b:.2f}\n"
    improvement = ((mean_b - mean_a) / mean_a * 100) if mean_a != 0 else 0
    stats_text += f"  Improvement: {improvement:+.1f}%\n\n"
    
    stats_text += "Blockiness (8×8) - Lower is Better\n"
    mean_a = np.mean(results_a['blockiness_8x8'])
    mean_b = np.mean(results_b['blockiness_8x8'])
    stats_text += f"  Mean: {mean_a:.2f}{' '*26} {mean_b:.2f}\n"
    improvement = ((mean_a - mean_b) / mean_a * 100) if mean_a != 0 else 0
    stats_text += f"  Improvement: {improvement:+.1f}%\n\n"
    
    stats_text += "Noise Level - Lower is Better\n"
    mean_a = np.mean(results_a['noise'])
    mean_b = np.mean(results_b['noise'])
    stats_text += f"  Mean: {mean_a:.2f}{' '*26} {mean_b:.2f}\n"
    improvement = ((mean_a - mean_b) / mean_a * 100) if mean_a != 0 else 0
    stats_text += f"  Improvement: {improvement:+.1f}%\n"
    
    ax5.text(0.05, 0.95, stats_text, transform=ax5.transAxes, 
             fontfamily='monospace', fontsize=9, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    # ==================== Quality Score Radar ====================
    ax6 = plt.subplot(3, 2, 6)
    
    # Normalize metrics to 0-10 scale for comparison
    blur_a = np.mean(results_a['blurriness']) / np.max([np.mean(results_a['blurriness']), 
                                                         np.mean(results_b['blurriness'])]) * 10
    blur_b = np.mean(results_b['blurriness']) / np.max([np.mean(results_a['blurriness']), 
                                                         np.mean(results_b['blurriness'])]) * 10
    
    block_a = 10 - (np.mean(results_a['blockiness_8x8']) / 
                    np.max([np.mean(results_a['blockiness_8x8']), 
                           np.mean(results_b['blockiness_8x8'])]) * 10)
    block_b = 10 - (np.mean(results_b['blockiness_8x8']) / 
                    np.max([np.mean(results_a['blockiness_8x8']), 
                           np.mean(results_b['blockiness_8x8'])]) * 10)
    
    noise_a = 10 - (np.mean(results_a['noise']) / 
                   np.max([np.mean(results_a['noise']), np.mean(results_b['noise'])]) * 10)
    noise_b = 10 - (np.mean(results_b['noise']) / 
                   np.max([np.mean(results_a['noise']), np.mean(results_b['noise'])]) * 10)
    
    metrics = ['Sharpness', 'Compression\nQuality', 'Noise\nCleanness']
    x = np.arange(len(metrics))
    width = 0.35
    
    bars1 = ax6.bar(x - width/2, [blur_a, block_a, noise_a], width, 
                    label='Video A', color=color_a, alpha=0.8)
    bars2 = ax6.bar(x + width/2, [blur_b, block_b, noise_b], width, 
                    label='Video B', color=color_b, alpha=0.8)
    
    ax6.set_ylabel('Quality Score (0-10)')
    ax6.set_title('Normalized Quality Metrics', fontweight='bold')
    ax6.set_xticks(x)
    ax6.set_xticklabels(metrics)
    ax6.legend()
    ax6.set_ylim([0, 11])
    ax6.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax6.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    
    # Save figure
    if output_image:
        plt.savefig(output_image, dpi=300, bbox_inches='tight')
        print(f"\nVisualization saved to: {output_image}")
    
    # Display
    plt.show()
    
    # Print summary
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"\nVideo A (Large): {info_a['file_size_mb']:.2f} MB")
    print(f"Video B (Small): {info_b['file_size_mb']:.2f} MB")
    print(f"\nFile size ratio: {info_a['file_size_mb']/info_b['file_size_mb']:.2f}x larger")


def main():
    parser = argparse.ArgumentParser(
        description="Compare quality metrics between two videos",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python compare_videos.py video_large.mp4 video_small.mp4
  python compare_videos.py video_a.mp4 video_b.mp4 --output comparison.png
  python compare_videos.py video_a.mp4 video_b.mp4 --sample_rate 15
        """
    )
    
    parser.add_argument("video_a", help="Path to first video (typically larger file)")
    parser.add_argument("video_b", help="Path to second video (typically smaller file)")
    parser.add_argument("--output", "-o", default=None,
                        help="Output image file for visualization (optional)")
    parser.add_argument("--sample_rate", "-s", type=int, default=30,
                        help="Extract one frame every N frames (default: 30)")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.video_a):
        print(f"Error: Video A not found: {args.video_a}")
        return
    
    if not os.path.exists(args.video_b):
        print(f"Error: Video B not found: {args.video_b}")
        return
    
    create_comparison_visualization(
        args.video_a,
        args.video_b,
        output_image=args.output,
        sample_rate=args.sample_rate
    )


if __name__ == "__main__":
    main()

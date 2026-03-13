import cv2
import os
import argparse

def sample_frames(video_path, output_dir, sample_rate=30, max_frames=None):
    """
    Sample frames from a video at regular intervals.
    
    Args:
        video_path: Path to input video file
        output_dir: Directory to save extracted frames
        sample_rate: Extract one frame every N frames (default: 30)
        max_frames: Maximum number of frames to extract (default: None - extract all)
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Open video file
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Cannot open video file {video_path}")
        return
    
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
    print(f"\nSampling every {sample_rate} frames...")
    
    frame_count = 0
    saved_count = 0
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        # Sample frame based on sample rate
        if frame_count % sample_rate == 0:
            frame_filename = os.path.join(output_dir, f"frame_{frame_count:06d}.jpg")
            cv2.imwrite(frame_filename, frame)
            saved_count += 1
            print(f"Saved: {frame_filename}")
            
            # Check if we've reached max_frames
            if max_frames and saved_count >= max_frames:
                print(f"\nReached maximum of {max_frames} frames.")
                break
        
        frame_count += 1
    
    cap.release()
    print(f"\nExtraction complete!")
    print(f"Total frames extracted: {saved_count}")
    print(f"Saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Sample frames from a video file")
    parser.add_argument("video_path", help="Path to input video file")
    parser.add_argument("--output_dir", "-o", default="frames", 
                        help="Output directory for frames (default: frames)")
    parser.add_argument("--sample_rate", "-s", type=int, default=30,
                        help="Extract one frame every N frames (default: 30)")
    parser.add_argument("--max_frames", "-m", type=int, default=None,
                        help="Maximum number of frames to extract (default: no limit)")
    
    args = parser.parse_args()
    
    sample_frames(args.video_path, args.output_dir, args.sample_rate, args.max_frames)


if __name__ == "__main__":
    main()

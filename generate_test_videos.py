import cv2
import numpy as np
import argparse
import os
import sys

def apply_blur(frame, kernel_size=(15, 15)):
    """Apply Gaussian blur to the frame."""
    return cv2.GaussianBlur(frame, kernel_size, 0)

def apply_noise(frame, std_dev=25):
    """Add Gaussian noise to the frame."""
    noise = np.random.normal(0, std_dev, frame.shape).astype(np.float32)
    noisy_frame = cv2.add(frame.astype(np.float32), noise)
    return np.clip(noisy_frame, 0, 255).astype(np.uint8)

def apply_compression(frame, quality=10):
    """Apply severe JPEG compression to simulate blockiness artifacts."""
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    result, encimg = cv2.imencode('.jpg', frame, encode_param)
    if result:
        return cv2.imdecode(encimg, 1)
    return frame

def process_video(input_path, output_path, degradation_type, intensity=None):
    """Process the video and apply the appropriate degradation to every frame."""
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: Could not open {input_path}")
        return False
        
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Define codec and create VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    print(f"Generating '{output_path}' ({degradation_type} effect)...")
    
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        if degradation_type == "blur":
            k_size = intensity if intensity else 15
            # Ensure kernel size is odd
            k_size = k_size if k_size % 2 == 1 else k_size + 1
            processed = apply_blur(frame, (k_size, k_size))
            
        elif degradation_type == "noise":
            std_dev = intensity if intensity else 25
            processed = apply_noise(frame, std_dev)
            
        elif degradation_type == "compress":
            quality = intensity if intensity else 10
            processed = apply_compression(frame, quality)
            
        else:
            processed = frame
            
        out.write(processed)
        
        frame_idx += 1
        if frame_idx % 30 == 0:
            sys.stdout.write(f"\rProgress: {frame_idx}/{total_frames} frames ({(frame_idx/total_frames)*100:.1f}%)")
            sys.stdout.flush()
            
    print(f"\rProgress: {total_frames}/{total_frames} frames (100.0%) - Done!")
    
    cap.release()
    out.release()
    return True

def main():
    parser = argparse.ArgumentParser(description="Generate degraded video datasets for No-Reference Video Quality Assessment testing.")
    parser.add_argument("input_video", help="Path to the high-quality input video")
    parser.add_argument("--blur", action="store_true", help="Generate a blurry version")
    parser.add_argument("--noise", action="store_true", help="Generate a noisy version")
    parser.add_argument("--compress", action="store_true", help="Generate a highly compressed (blocky) version")
    parser.add_argument("--all", action="store_true", help="Generate all three degraded versions")
    parser.add_argument("--output_dir", default=".", help="Directory to save the generated videos")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_video):
        print(f"Error: Input video '{args.input_video}' not found.")
        return
        
    os.makedirs(args.output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(args.input_video))[0]
    
    modes = []
    if args.all or args.blur: modes.append("blur")
    if args.all or args.noise: modes.append("noise")
    if args.all or args.compress: modes.append("compress")
    
    if not modes:
        print("Please specify at least one degradation type (--blur, --noise, --compress, or --all).")
        return
        
    print(f"Starting test video generation for '{args.input_video}'...")
    
    for mode in modes:
        output_path = os.path.join(args.output_dir, f"{base_name}_{mode}.mp4")
        process_video(args.input_video, output_path, mode)

if __name__ == "__main__":
    main()

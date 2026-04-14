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


def apply_jitter(frame, max_shift_px=4):
    """Apply small random x/y translation to simulate temporal jitter."""
    h, w = frame.shape[:2]
    dx = np.random.randint(-max_shift_px, max_shift_px + 1)
    dy = np.random.randint(-max_shift_px, max_shift_px + 1)
    matrix = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(frame, matrix, (w, h), borderMode=cv2.BORDER_REFLECT)

def process_video(
    input_path,
    output_path,
    degradation_type,
    intensity=None,
    freeze_every=60,
    freeze_duration=10,
    jitter_px=4,
):
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
    freeze_ref = None
    freeze_active_until = -1

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

        elif degradation_type == "jitter":
            max_shift = intensity if intensity else jitter_px
            processed = apply_jitter(frame, max_shift)

        elif degradation_type == "freeze":
            if freeze_every <= 0:
                freeze_every = 60
            if freeze_duration <= 0:
                freeze_duration = 10

            if frame_idx % freeze_every == 0:
                freeze_ref = frame.copy()
                freeze_active_until = frame_idx + freeze_duration

            if freeze_ref is not None and frame_idx < freeze_active_until:
                processed = freeze_ref.copy()
            else:
                processed = frame
            
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
    parser.add_argument("--freeze", action="store_true", help="Generate a freeze-artifact version")
    parser.add_argument("--jitter", action="store_true", help="Generate a jittery motion version")
    parser.add_argument("--all", action="store_true", help="Generate all three degraded versions")
    parser.add_argument("--output_dir", default=".", help="Directory to save the generated videos")
    parser.add_argument("--freeze_every", type=int, default=60,
                        help="Start a freeze every N frames (default: 60)")
    parser.add_argument("--freeze_duration", type=int, default=10,
                        help="Freeze length in frames (default: 10)")
    parser.add_argument("--jitter_px", type=int, default=4,
                        help="Max pixel shift for jitter mode (default: 4)")
    
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
    if args.all or args.freeze: modes.append("freeze")
    if args.all or args.jitter: modes.append("jitter")
    
    if not modes:
        print("Please specify at least one degradation type (--blur, --noise, --compress, or --all).")
        return
        
    print(f"Starting test video generation for '{args.input_video}'...")
    
    for mode in modes:
        output_path = os.path.join(args.output_dir, f"{base_name}_{mode}.mp4")
        process_video(
            args.input_video,
            output_path,
            mode,
            freeze_every=args.freeze_every,
            freeze_duration=args.freeze_duration,
            jitter_px=args.jitter_px,
        )

if __name__ == "__main__":
    main()

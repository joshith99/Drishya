# No-Reference Video Quality Assessment Toolkit

## Overview
A practical toolkit for evaluating video quality without requiring reference videos. This project implements classical computer vision methods to detect and quantify common video distortions including blur, compression artifacts (blockiness), and noise. Ideal for quick video quality analysis, comparison studies, and educational purposes in computer vision and image processing courses.

## What is No-Reference Quality Assessment?
No-Reference Video Quality Assessment (NR-VQA), also known as blind quality assessment, evaluates video quality without comparing to a pristine reference. This toolkit uses proven mathematical techniques to measure perceptual quality indicators.

### Implemented Quality Metrics

#### 1. **Blurriness Detection (Laplacian Variance)**
- Uses the Laplacian operator to measure image sharpness
- Higher variance indicates sharper, clearer frames
- Formula: `Var(∇²I)` where ∇² is the Laplacian operator
- Useful for detecting out-of-focus content, motion blur, or low-quality encoding

#### 2. **Blockiness Detection (Compression Artifacts)**
- Detects JPEG/H.264/H.265 compression artifacts at block boundaries
- Analyzes both 8×8 and 16×16 block structures
- Measures standard deviation of pixel differences across block edges
- Higher values indicate more visible compression artifacts

#### 3. **Noise Estimation (Wavelet-based MAD)**
- Estimates noise level using Median Absolute Deviation of wavelet coefficients
- Formula: `σ = MAD(cD) / 0.6745` using Daubechies wavelet
- Lower values indicate cleaner, less noisy video
- Robust against content variations

## Features
✅ Three standalone Python tools (no deep learning required)  
✅ Frame extraction from video files  
✅ Comprehensive NR quality metrics calculation  
✅ Side-by-side video comparison with visualizations  
✅ CSV export for further analysis  
✅ Fast processing with configurable sampling rates  
✅ Command-line interface for all tools  

## Project Structure
```
Review-1/
├── sample_frames.py      # Extract frames from videos
├── nr_features.py        # Calculate NR quality metrics
├── compare_videos.py     # Compare two videos side-by-side
├── requirements.txt      # Python dependencies
├── CHANGELOG.md          # Version history
└── README.md             # This file
```

## Installation

### Prerequisites
- Python 3.8 or higher
- 4GB+ RAM

### Quick Setup
```bash
# Navigate to project directory
cd Review-1

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Dependencies
- **OpenCV** (opencv-python >= 4.8.0) - Video processing
- **NumPy** (>= 1.24.0) - Numerical computations
- **PyWavelets** (>= 1.4.1) - Wavelet-based noise estimation
- **Matplotlib** (>= 3.7.0) - Visualization

## Usage

### 1. Extract Frames from Video
```bash
# Basic usage - extract every 30th frame
python sample_frames.py video.mp4

# Custom output directory
python sample_frames.py video.mp4 --output_dir my_frames

# Extract every 15th frame, max 50 frames
python sample_frames.py video.mp4 --sample_rate 15 --max_frames 50
```

**Options:**
- `--output_dir, -o`: Output directory (default: `frames`)
- `--sample_rate, -s`: Extract one frame every N frames (default: 30)
- `--max_frames, -m`: Maximum number of frames to extract

### 2. Calculate Quality Metrics
```bash
# Basic analysis
python nr_features.py video.mp4

# Save results to CSV
python nr_features.py video.mp4 --output results.csv

# Custom sampling
python nr_features.py video.mp4 --sample_rate 15 --max_frames 100
```

**Output Metrics:**
- Frame number and timestamp
- Blurriness (Laplacian Variance) - Higher is better
- Blockiness 8×8 (Std Dev) - Lower is better
- Blockiness 16×16 (Std Dev) - Lower is better
- Noise Level (σ) - Lower is better

### 3. Compare Two Videos
```bash
# Compare two videos with visualization
python compare_videos.py video1.mp4 video2.mp4

# Save comparison plot
python compare_videos.py video1.mp4 video2.mp4 --output comparison.png

# Custom sampling for faster processing
python compare_videos.py video1.mp4 video2.mp4 --sample_rate 60
```

**Output:**
- 6-panel comparison visualization
- Temporal quality graphs for all metrics
- Statistical summary (means, improvements)
- Normalized quality radar chart

## Example Workflow

### Scenario: Comparing Original vs. Compressed Video
```bash
# 1. Analyze original video
python nr_features.py original.mp4 --output original_metrics.csv

# 2. Analyze compressed video
python nr_features.py compressed.mp4 --output compressed_metrics.csv

# 3. Visual comparison
python compare_videos.py original.mp4 compressed.mp4 --output comparison.png
```

### Interpreting Results

**Blurriness (Laplacian Variance):**
- `> 500`: Sharp, high-quality video
- `100-500`: Moderate quality
- `< 100`: Blurry or low-quality video

**Blockiness (8×8 and 16×16):**
- `< 5`: Minimal compression artifacts
- `5-15`: Moderate compression
- `> 15`: Heavy compression artifacts visible

**Noise Level:**
- `< 5`: Very clean video
- `5-15`: Acceptable noise level
- `> 15`: Noisy, potentially degraded video

## Applications
- **Video Compression Testing**: Evaluate codec quality and bitrate settings
- **Quality Control**: Monitor webcam, surveillance, or streaming video quality
- **Educational Projects**: Learn computer vision and image processing concepts
- **Preprocessing Analysis**: Assess video quality before machine learning pipelines
- **Content Comparison**: Compare different video sources or encoding settings

## Limitations
- Classical methods cannot capture all perceptual quality aspects
- No temporal distortion detection (frame freezing, jitter)
- Content-dependent - natural scenes vs. synthetic content may vary
- Not a substitute for subjective human evaluation
- Single frame analysis - doesn't model temporal dependencies

## Future Enhancements
- [ ] Temporal distortion detection (frame freezing, jitter)
- [ ] Color quality metrics (saturation, dynamic range)
- [ ] Perceptual quality scoring using deep learning
- [ ] Real-time video stream analysis
- [ ] GUI interface for easier usage
- [ ] Batch processing for multiple videos

## Technical References
- **Laplacian Variance**: Pech-Pacheco et al. (2000) - "Diatom autofocusing in brightfield microscopy"
- **Blockiness Detection**: Wang et al. (2000) - "A universal image quality index"
- **Wavelet Noise Estimation**: Donoho & Johnstone (1994) - "Ideal spatial adaptation by wavelet shrinkage"

## Contributing
Contributions welcome! Areas for improvement:
- Additional quality metrics (BRISQUE, NIQE, etc.)
- Performance optimizations
- Better visualization options
- Documentation improvements

## License
MIT License - Free for academic and commercial use

## Academic Context
**Course**: Machine Vision  
**Project**: Review-1 - No-Reference Video Quality Assessment  
**Date**: February 2026

---
**Note**: This is an educational project demonstrating classical computer vision techniques for video quality assessment. For production systems, consider combining these metrics with perceptual models or deep learning approaches.

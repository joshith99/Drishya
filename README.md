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
├── brisque.py            # Calculate BRISQUE and classical NR metrics (Requires internet for model download on first run)
├── compare_brisque.py    # Compare two videos using BRISQUE & classical metrics
├── generate_test_videos.py # Tool to generate degraded videos (blur, noise, compression)
├── nrvqa_roadmap.md      # Strategic roadmap for future enhancements
├── archive/              # Unused or deprecated scripts (e.g., nr_features.py, compare_videos.py, sample_frames.py)
├── testing_videos/       # Directory for input and generated test video files
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

### 1. Generate Test Videos
Generate controlled degradations (blur, noise, compression) from a high-quality source video:
```bash
python generate_test_videos.py testing_videos/original.mp4 --all --output_dir testing_videos/
```

### 2. Calculate Quality Metrics
Run BRISQUE and classical metrics on random frames:
```bash
python brisque.py testing_videos/video.mp4 --samples 10
```

### 3. Compare Two Videos
Compare original vs. degraded video with visualization:
```bash
python compare_brisque.py testing_videos/original.mp4 testing_videos/degraded.mp4 --samples 15
```

## Example Workflow

### Scenario: Comparing Original vs. Compressed Video
```bash
# 1. Generate heavily compressed video
python generate_test_videos.py testing_videos/original.mp4 --compress --output_dir testing_videos/

# 2. Visual comparison
python compare_brisque.py testing_videos/original.mp4 testing_videos/original_compress.mp4
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

# Drishya: No-Reference Video Quality Assessment Toolkit

## Overview
A comprehensive toolkit for evaluating video quality without requiring reference videos. This project implements both classical computer vision methods and industry-standard metrics to detect and quantify common video distortions including blur, compression artifacts (blockiness), and noise. Ideal for quick video quality analysis, comparison studies, and educational purposes in computer vision and image processing courses.

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

#### 4. **BRISQUE (Blind/Referenceless Image Spatial Quality Evaluator)**
- Industry-standard perceptual quality metric
- Trained on human subjective quality scores
- Lower scores indicate better perceptual quality (typically 0-100)
- Accounts for natural scene statistics

#### 5. **NIQE (Natural Image Quality Evaluator)**
- Fast no-reference quality metric based on local contrast and sharpness
- Simplified implementation using image statistics
- Lower scores indicate better quality

## Features
✅ Multiple NR quality metrics (classical + industry-standard)  
✅ Frame extraction from video files with configurable sampling  
✅ Comprehensive quality scoring with composite metrics  
✅ Side-by-side video comparison with visualizations  
✅ CSV export for further analysis  
✅ Command-line interface for all tools  
✅ Test video generation for validation  
✅ Fast processing with minimal dependencies  

## Project Structure and Files

### Core Analysis Tools
- **`NQS.py`** - No-Reference Quality Scorer (Main tool)
  - Combines all metrics into a unified quality assessment
  - Provides composite quality scores and ratings
  - Command: `python NQS.py video.mp4 [--samples 10] [--method random]`
  - Command: `python NQS.py video1.mp4 video2.mp4` (comparison)

- **`industry_metrics.py`** - Industry-standard metrics (BRISQUE, NIQE)
  - Implements BRISQUE and NIQE quality assessment
  - Provides per-frame and video-level statistics
  - Command: `python industry_metrics.py video.mp4 [--samples 10] [--method random]`
  - Command: `python industry_metrics.py video1.mp4 video2.mp4` (comparison)

- **`nr_features.py`** - Custom no-reference features
  - Calculates Laplacian blur, blockiness, and wavelet noise
  - Extracts features from videos with configurable sampling
  - Command: `python nr_features.py video.mp4 [--output results.csv] [--sample_rate 30] [--max_frames 50]`

### Comparison and Testing Tools
- **`compare_brisque.py`** - Video comparison with BRISQUE
  - Compares two videos using BRISQUE and custom metrics
  - Generates graphical comparison charts
  - Command: `python compare_brisque.py video1.mp4 video2.mp4 [--samples 10]`

- **`generate_test_videos.py`** - Test video generation
  - Creates degraded videos (blur, noise, compression) for testing
  - Useful for validating quality assessment algorithms
  - Command: `python generate_test_videos.py input.mp4 --blur --noise --compress [--output_dir ./testing_videos]`

### Supporting Files
- **`requirements.txt`** - Python dependencies
  - Core: opencv-python, numpy, PyWavelets, matplotlib, Pillow
  - Optional: brisque (for BRISQUE metric)
  - Install: `pip install -r requirements.txt`

- **`CHANGELOG.md`** - Version history and updates

- **`nrvqa_roadmap.md`** - Strategic roadmap for future enhancements
  - Phase 1: ✅ Core metrics + testing framework
  - Phase 2: ✅ Industry-standard metrics (BRISQUE + NIQE)
  - Phase 3: 🔄 Temporal quality assessment
  - Phase 4: 🎯 Deep learning-based quality prediction
  - Phase 5: 🎨 Production UI and batch processing

### Archive and Testing
- **`archive/`** - Deprecated or unused scripts
  - `compare_videos.py` - Original comparison tool (superseded)
  - `sample_frames.py` - Basic frame extraction utility

- **`testing_videos/`** - Directory for test video files
  - Store high-quality reference videos here
  - Generated degraded videos are saved here

- **`__pycache__/`** - Python bytecode cache (auto-generated)

### Temporary/Debug Files
- **`brisque.py`** - BRISQUE implementation (may be replaced by package)
- **`tmp_*.py`** - Temporary debug and testing scripts

## Installation

### Prerequisites
- Python 3.8 or higher
- 4GB+ RAM

### Quick Setup
```bash
# Navigate to project directory
cd Drishya

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

### 2. Analyze Single Video Quality
Run comprehensive quality assessment on a video:
```bash
python NQS.py testing_videos/video.mp4 --samples 10
```

### 3. Compare Two Videos
Compare original vs. degraded video with detailed analysis:
```bash
python NQS.py testing_videos/original.mp4 testing_videos/degraded.mp4 --samples 15
```

### 4. Industry Metrics Only
Run BRISQUE and NIQE metrics specifically:
```bash
python industry_metrics.py testing_videos/video.mp4 --samples 10
```

### 5. Extract Raw Features
Calculate individual quality features and export to CSV:
```bash
python nr_features.py testing_videos/video.mp4 --output results.csv --sample_rate 30
```

### 6. Visual Comparison with Charts
Generate graphical comparison between two videos:
```bash
python compare_brisque.py testing_videos/original.mp4 testing_videos/degraded.mp4 --samples 15
```

## Example Workflow

### Scenario: Comparing Original vs. Compressed Video
```bash
# 1. Generate heavily compressed video
python generate_test_videos.py testing_videos/original.mp4 --compress --output_dir testing_videos/

# 2. Comprehensive quality comparison
python NQS.py testing_videos/original.mp4 testing_videos/original_compress.mp4
```

### Scenario: Batch Analysis of Multiple Videos
```bash
# Analyze multiple videos and export features
python nr_features.py video1.mp4 --output video1_features.csv
python nr_features.py video2.mp4 --output video2_features.csv

# Compare with industry metrics
python industry_metrics.py video1.mp4 video2.mp4
```

### Interpreting Results

**Composite Quality Score (NQS):**
- `0.8-1.0`: Excellent quality
- `0.6-0.8`: Good quality
- `0.4-0.6`: Fair quality
- `0.2-0.4`: Poor quality
- `0.0-0.2`: Very poor quality

**BRISQUE Score:**
- `0-30`: Excellent perceptual quality
- `30-50`: Good quality
- `50-70`: Fair quality
- `70-100`: Poor quality

**NIQE Score:**
- `0-30`: Excellent quality
- `30-50`: Good quality
- `50-70`: Fair quality
- `70-100`: Poor quality

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
Based on the strategic roadmap in `nrvqa_roadmap.md`:

**Phase 3: Temporal Quality Assessment** 🔄 *In Planning*
- Jitter and freeze detection using frame similarity (SSIM)
- Optical flow analysis for motion smoothness
- Temporal consistency metrics

**Phase 4: Perceptual Machine Learning** 🎯 *Future*
- Deep learning-based quality prediction (MOS scores)
- SVR models trained on classical features
- Lightweight CNN for real-time assessment

**Phase 5: Production Pipeline** 🎨 *Future*
- Streamlit web UI for drag-and-drop analysis
- PyQt native desktop application
- Batch processing for thousands of videos
- Parallel processing optimization

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
**Project**: Drishya - No-Reference Video Quality Assessment  
**Date**: April 2026

---
**Note**: This is an educational project demonstrating classical computer vision techniques for video quality assessment. For production systems, consider combining these metrics with perceptual models or deep learning approaches.

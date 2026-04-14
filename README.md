# 🎬 Drishya — No-Reference Video Quality Assessment

A comprehensive, production-ready NR-VQA system that combines classical signal processing metrics with deep learning-based MOS prediction.

---

## 🚀 Phases Overview

| Phase | Description | Status |
|-------|-------------|--------|
| **1** | Core NR metrics (blur, noise, blockiness) + test harness | ✅ Complete |
| **2** | Industry-standard BRISQUE integration | ✅ Complete |
| **3** | Temporal quality (SSIM, freeze detection, jitter) | ✅ Complete |
| **4** | Deep learning MOS prediction (EfficientNet-B4 → SVR) | ✅ Complete |
| **5** | Production Streamlit UI + batch processing | ✅ Complete |

---

## 🖥️ Quick Start — Phase 5 UI

```bash
pip install -r requirements.txt
streamlit run phase5_app.py
```

Open **http://localhost:8501** to access the dashboard.

### UI Features
- 🔍 **Analyze Video** — Upload a video, get MOS prediction + quality radar chart + per-frame breakdown
- ⚖️ **Compare Videos** — Side-by-side quality comparison with overlay radar chart
- 📦 **Batch Process** — Point to a folder, get a full CSV report

---

## 📊 Phase 4 — Deep Learning MOS Prediction

EfficientNet-B4 (frozen, ImageNet-pretrained) is used as a GPU feature extractor. 8 evenly-spaced frames are sampled per video, passed through the backbone, and temporally mean-pooled into a 1792-d descriptor. An SVR with RBF kernel is then trained on this representation against human MOS scores.

### Training Data
- **YouTube-UGC:** 907 matched videos (1080P and below)
- **KoNViD-150k-B:** 1,576 videos (540p, 5s clips)
- **Total:** 2,483 videos with human MOS labels

### Results (validation set, 15%)

| Metric | Score |
|--------|-------|
| PLCC | 0.526 |
| SRCC | 0.512 |
| RMSE | 0.393 |

> Baseline result from frozen features. Fine-tuning the backbone end-to-end would push SRCC to ~0.80+.

### Re-run feature extraction + SVR training

```bash
# Requires datasets on disk (see Datasets section)
python phase4_svr.py
```

---

## 🎛️ CLI Usage — NQS Analyzer

```bash
# Analyze a single video
python NQS.py video.mp4

# Analyze with temporal metrics
python NQS.py video.mp4 --temporal

# Compare two videos
python NQS.py video_a.mp4 video_b.mp4

# More samples, uniform spacing
python NQS.py video.mp4 --samples 20 --method uniform
```

---

## 📐 Metrics Computed

| Metric | Method | Phase |
|--------|--------|-------|
| Sharpness | Laplacian variance | 1 |
| Blockiness | DCT block boundary analysis | 1 |
| Noise | Wavelet high-frequency estimation | 1 |
| BRISQUE | Blind Image Spatial Quality Evaluator | 2 |
| Temporal smoothness | SSIM across consecutive frames | 3 |
| Freeze detection | SSIM = 1.0 event counting | 3 |
| Jitter | Optical flow variance | 3 |
| MOS (predicted) | EfficientNet-B4 features + SVR | 4 |

---

## 📁 Project Structure

```
Drishya/
├── phase5_app.py          # 🖥️  Streamlit production UI
├── phase4_svr.py          # 🧠  Deep feature extraction + SVR training
├── phase4_train.py        # 🔬  CNN fine-tuning pipeline (future use)
├── NQS.py                 # 🎛️  CLI analyzer (Phases 1-3)
├── nr_features.py         # Classical NR feature extractors
├── industry_metrics.py    # BRISQUE / NIQE wrappers
├── temporal_features.py   # Temporal quality metrics
├── brisque.py             # BRISQUE implementation
├── generate_test_videos.py# Synthetic degradation test data generator
├── extract_frames.py      # Batch frame extraction utility
├── requirements.txt       # All dependencies
└── checkpoints/           # Saved models (gitignored)
```

---

## 🗂️ Datasets

To re-train Phase 4, you need:

| Dataset | URL | Notes |
|---------|-----|-------|
| YouTube-UGC | [media.withyoutube.com](https://media.withyoutube.com/) | Download videos + MOS xlsx |
| KoNViD-150k-B | [qualinet.eu](http://database.mmsp-kn.de/) | ~1.9 GB ZIP |
| LIVE-VQC | [utexas.edu](https://live.ece.utexas.edu/research/LIVEVQC/) | Access request required |

Place extracted datasets under `datasets/` (gitignored).

---

## 🛠️ Development

```bash
# Create venv
python -m venv .venv
.venv\Scripts\activate       # Windows
source .venv/bin/activate    # Linux/Mac

pip install -r requirements.txt
```

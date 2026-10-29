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

### Results (held-out test set, 15%)

| Metric | Score |
|--------|-------|
| PLCC | 0.526 |
| SRCC | 0.512 |
| RMSE | 0.393 |

> Baseline result from frozen features (older run, tuned on a validation split).
> The current pipeline now selects SVR hyperparameters via k-fold CV on the
> training fold only and reports metrics on a truly held-out test set, and saves
> the frozen backbone weights with the checkpoint so the UI never re-downloads
> ImageNet weights. Fine-tuning the backbone end-to-end would push SRCC to ~0.80+.

### Trained model (Phase 5 deep MOS)

The trained model ships in `checkpoints/svr_deep_*.pkl` (gitignored). When present,
the Phase 5 UI loads it and shows real MOS predictions; verified on the bundled
test clips (baseline MOS 3.48 vs 3.44–3.47 for blur/noise/compress). If your
checkpoint was produced by an older run without saved backbone weights, the UI
falls back to ImageNet-pretrained weights (identical to training-time features).

### Re-run feature extraction + SVR training

Paths are configurable via CLI args or `DRISHYA_DATA_ROOT` / `DRISHYA_CKPT_DIR` /
`DRISHYA_FEAT_CACHE` env vars (no hardcoded `/workspace` paths).

```bash
# KoNViD-1k layout (see Datasets section)
python phase4_svr.py --data-root datasets --limit 250   # quick CPU run
python phase4_svr.py --data-root datasets               # full run
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

# Reproducible random sampling
python NQS.py video.mp4 --samples 20 --seed 42
```

---

## 🧪 Terminal Testing (no UI)

Run both pipelines headlessly on the bundled `testing_videos/` clips.

**Both classical + deep MOS (recommended):**
```bash
python cli_test.py testing_videos/video_small.mp4 --temporal
python cli_test.py testing_videos/video_small_blur.mp4 testing_videos/video_small.mp4 --temporal
python cli_test.py testing_videos/*.mp4 --json report.json
```

**Classical only (Phases 1-3):**
```bash
python cli_test.py testing_videos/video_small.mp4 --no-deep --temporal
```

**Deep MOS only (Phase 4):**
```bash
python cli_test.py testing_videos/video_small.mp4 --no-classical
# or call the module directly
python deep_mos.py testing_videos/video_small.mp4
python deep_mos.py testing_videos/video_small.mp4 -v            # per-frame feature norms
python deep_mos.py testing_videos/video_small.mp4 --save-frames out_frames
python deep_mos.py testing_videos/video_small.mp4 --save-preprocessed out_pre
python deep_mos.py testing_videos/video_small.mp4 --show-preprocessed
```

`deep_mos.py` prints the checkpoint/backbone used, video resolution/fps/duration,
sampled frame indices, the preprocessed input tensor (shape + per-channel
min/max/mean/std + normalization constants), pooled-feature stats, and MOS
(raw + normalized + rating). `-v` adds per-frame feature L2 norms;
`--save-frames DIR` dumps the raw sampled frames to `DIR/<video_stem>/`;
`--save-preprocessed DIR` dumps the denormalized 224x224 model input;
`--show-preprocessed` saves and opens a raw-vs-preprocessed montage.

`cli_test.py` reuses `NQS.py` for classical metrics and `deep_mos.py` for the
frozen EfficientNet-B4 + SVR MOS prediction. `deep_mos.py` is a standalone copy
of the UI's inference logic so the Streamlit app is never imported headlessly.

---

## 📐 Metrics Computed

| Metric | Method | Phase |
|--------|--------|-------|
| Sharpness | Laplacian variance | 1 |
| Blockiness | DCT block boundary analysis | 1 |
| Noise | Wavelet high-frequency estimation | 1 |
| BRISQUE | Blind Image Spatial Quality Evaluator | 2 |
| NIQE | Natural Image Quality Evaluator (self-contained, scipy) | 2 |
| Temporal smoothness | SSIM across consecutive frames | 3 |
| Freeze detection | SSIM = 1.0 event counting | 3 |
| Jitter | Optical flow variance | 3 |
| MOS (predicted) | EfficientNet-B4 features + SVR | 4 |

---

## 📁 Project Structure

```
Drishya/
├── phase5_app.py          # 🖥️  Streamlit production UI
├── cli_test.py            # 🧪  Terminal runner (classical + deep MOS, no UI)
├── deep_mos.py            # 🔮  Standalone deep MOS inference (Phase 4)
├── phase4_svr.py          # 🧠  Deep feature extraction + SVR training
├── phase4_train.py        # 🔬  CNN fine-tuning pipeline (future use)
├── NQS.py                 # 🎛️  CLI analyzer (Phases 1-3)
├── nr_features.py         # Classical NR feature extractors
├── industry_metrics.py    # BRISQUE / NIQE wrappers (self-contained NIQE)
├── niqe_pris_params.npz   # Official NIQE natural-scene model parameters
├── temporal_features.py   # Temporal quality metrics
├── generate_test_videos.py# Synthetic degradation test data generator
├── extract_frames.py      # Batch frame extraction utility
├── requirements.txt       # All dependencies
└── checkpoints/           # Saved models (gitignored)
```

> Removed in the 2026-09 cleanup: `brisque.py` / `compare_brisque.py` (they shadowed
> the `brisque` pip package and duplicated `NQS.py` / `industry_metrics.py`
> functionality), the `archive/` scripts, and the superseded `report1.md`.

---

## 🗂️ Datasets

To re-train Phase 4, you need:

| Dataset | URL | Notes |
|---------|-----|-------|
| YouTube-UGC | [media.withyoutube.com](https://media.withyoutube.com/) | Download videos + MOS xlsx |
| KoNViD-1k | [database.mmsp-kn.de](http://database.mmsp-kn.de/konvid-1k-database.html) | 2.3 GB zip + metadata (MOS 1–5, public) |
| KoNViD-150k-B | [qualinet.eu](http://database.mmsp-kn.de/) | ~1.9 GB ZIP |
| LIVE-VQC | [utexas.edu](https://live.ece.utexas.edu/research/LIVEVQC/) | Access request required |

Place extracted datasets under `datasets/` (gitignored). Layout expected by
`phase4_svr.py`: `datasets/konvid1k/{videos, KoNViD-1k_metadata.csv}`
(videos + metadata csv) and/or `datasets/youtube_ugc/` (videos + MOS xlsx).

---

## 🛠️ Development

```bash
# Create venv
python -m venv .venv
.venv\Scripts\activate       # Windows
source .venv/bin/activate    # Linux/Mac

pip install -r requirements.txt
```

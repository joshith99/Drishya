# Changelog

All notable changes to Drishya are documented here.

## [Unreleased]

### Added
- [2026-09-09] **Maintenance & correctness pass** — see Removed/Changed below
- [2026-08-06] **Review 1 report: base paper switched to RQ-VQA + 2024–2025 literature**
  - `report_phase1_review1.md`: adopted **RQ-VQA** (Sun et al., arXiv:2405.08745, NTIRE 2024 winner) as the base paper; VSFA retained as a cited model
  - Rewrote §3 base-paper comparison (RQ-VQA vs Drishya) incl. new generalization row
  - Added recent-literature rows to §2.1 (RQ-VQA, Q-Align, FineVQ) and a 2024–2025 frontier-shift trend bullet in §2.2
  - References renumbered 1–20 by first appearance; citations updated throughout
  - Downloaded 7 new open-access papers into `papers/` (12_rqvqa, 13_qalign, 14_kvq, 15_ais2024, 16_lmmvqa, 17_vqa_survey, 18_finevq)

### Removed
- [2026-09-09] `brisque.py`, `compare_brisque.py` — shadowed the `brisque` pip package; functionality covered by `NQS.py`/`industry_metrics.py`
- [2026-09-09] `archive/` (`compare_videos.py`, `sample_frames.py`) — superseded by `NQS.py`/`extract_frames.py`
- [2026-09-09] `report1.md` — superseded by `report_phase1_review1.md`

### Changed
- [2026-09-09] `industry_metrics.py`: real NIQE via `piq` (previously an ad-hoc heuristic); removed duplicated `brisque` import-shadowing hacks
- [2026-09-09] `NQS.py`: temporal metrics computed over a contiguous window of adjacent frames (were over sparse, far-apart samples); new `--seed` flag; removed dead BRISQUE import code
- [2026-09-09] `phase4_svr.py`: SVR hyperparams via k-fold CV on train only; metrics on a held-out test set; frozen backbone weights saved with checkpoint; dropped unused `optuna`
- [2026-09-09] `phase5_app.py`: loads saved backbone weights instead of re-downloading ImageNet pretrained
- [2026-09-09] `phase4_train.py`: drops undecodable videos at dataset build time (previously trained on zero-frame tensors with real MOS labels)
- [2026-09-09] `calibrate_temporal.py`: no longer crashes when a metric is missing
- [2026-09-09] `requirements.txt`: added PyWavelets, matplotlib, requests; `piq` now used for NIQE
- [2026-09-09] `.gitignore`: ignore `papers/` (copyrighted publisher PDFs)
- [2026-09-10] `industry_metrics.py`: NIQE is now a **self-contained scipy implementation** (`piq` ships no NIQE in 0.7/0.8, so the earlier `piq`-based NIQE silently disabled itself); uses the official model parameters in a new tracked `niqe_pris_params.npz`
- [2026-09-10] `requirements.txt`: dropped `piq` (no longer needed); `README.md`: NIQE row/structure updated
- [2026-09-10] `phase4_svr.py`: made fully portable — configurable data/checkpoint/cache paths (CLI args or `DRISHYA_*` env vars), KoNViD-1k metadata auto-loader, `--limit` for quick CPU runs
- [2026-09-10] `phase4_train.py`: dataset/checkpoint/log paths configurable via env vars
- [2026-09-10] **Trained Phase-4 checkpoint restored** (`checkpoints/svr_deep_20260414_192956.pkl`, PLCC 0.526 / SRCC 0.512 / RMSE 0.393) — verified end-to-end: real MOS predictions on test clips (baseline 3.48 vs 3.44–3.47 degraded) and the Phase-5 UI loads it successfully

## [Phase 5] — 2026-04-15
### Added
- `phase5_app.py`: Full Streamlit production UI with three pages:
  - **Analyze Video**: upload → MOS gauge + quality radar + per-frame table + sampled frames
  - **Compare Videos**: side-by-side MOS gauges + radar overlay chart
  - **Batch Process**: folder scan → results CSV download + MOS histogram
- Dark-theme premium UI with Inter font, animated hover effects, Plotly charts

## [Phase 4] — 2026-04-15
### Added
- `phase4_svr.py`: GPU feature extraction pipeline
  - EfficientNet-B4 (frozen, pretrained) as backbone; 8-frame temporal sampling
  - SVR (RBF kernel) trained on 1792-d features from 2,483 videos
  - Val PLCC=0.526, SRCC=0.512, RMSE=0.393
- `phase4_train.py`: End-to-end CNN fine-tuning pipeline (AdamW + AMP + grad accumulation)
- Integrated KoNViD-150k-B dataset (1,576 videos)
- `extract_frames.py`: parallel JPEG frame pre-extraction utility
### Changed
- `requirements.txt`: fully updated with all Phase 4–5 dependencies
- `.gitignore`: added `*.pkl`, `*.npz`, `datasets/`
- `README.md`: complete rewrite documenting all 5 phases

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.0.0] - 2026-04-14

### Added
- [2026-04-14] **Phase 3: Temporal Quality Assessment MVP**
  - Created `temporal_features.py` - Temporal no-reference quality metrics
    - Frame-to-frame SSIM similarity scoring
    - Optical flow analysis for motion and jitter detection
    - Freeze detection using frame similarity and motion thresholds
    - Aggregate temporal_quality_score (0-1, higher is better)
  - Extended `generate_test_videos.py` with temporal degradation modes
    - `--freeze` mode: Generates synthetic freeze artifacts (periodic frame duplication)
    - `--jitter` mode: Generates synthetic jitter (random pixel translation)
    - Tunable parameters: `--freeze_every`, `--freeze_duration`, `--jitter_px`
  - Added `calibrate_temporal.py` - Calibration and validation framework
  - Added `PHASE3_COMPLETION_REPORT.md` - Detailed Phase 3 implementation report

### Changed
- [2026-04-14] Updated `NQS.py` to integrate temporal metrics
  - New CLI flags: `--temporal` (enable temporal analysis) and `--temporal_stride` (sample stride)
  - Temporal metrics now weighted at 20% in composite quality score
  - New output section with frame and video-level temporal metrics
  - Fixed sys.path handling to resolve import conflicts
- [2026-04-14] Updated `README.md` with temporal quality assessment documentation
  - Added `temporal_features.py` module documentation
  - Added temporal usage examples and interpretation guide
  - Updated Phase 3 status from "In Planning" to "MVP In Progress"
  - Added temporal metric interpretation thresholds
- [2026-04-14] Updated `nrvqa_roadmap.md`
  - Marked Phase 3 as "MVP IMPLEMENTED (April 2026)"
  - Added calibration results and recommended defaults

### Fixed
- [2026-04-14] Fixed import path conflicts between local modules and pip packages
  - Improved sys.path handling in NQS.py to avoid BRISQUE import shadowing
  - Ensured temporal_features.py imports resolve correctly across all entry points

## [1.1.0] - 2026-03-13

### Added
- [2026-03-13] Cleaned repository structure (moved original scripts to `archive/`)
- [2026-03-13] Added `testing_videos/` directory to store high-quality and degraded video samples
- [2026-03-13] Added `nrvqa_roadmap.md` detailing future plans including AI and Temporal integration
- [2026-03-13] Added `generate_test_videos.py` to create synthetic noise, blur, and compression datasets
- [2026-03-13] Added `brisque.py` and `compare_brisque.py` for executing perception-based NR-VQA

## [1.0.0] - 2026-02-09

### Added
- Initial project setup with three core tools
- [2026-02-09] Created `sample_frames.py` - Frame extraction utility with configurable sampling rates
- [2026-02-09] Created `nr_features.py` - No-Reference quality metrics calculator
  - Laplacian Variance for blurriness detection
  - Blockiness detection (8×8 and 16×16 blocks) for compression artifact analysis
  - Wavelet-based noise estimation using MAD method
  - CSV export functionality for analysis results
- [2026-02-09] Created `compare_videos.py` - Side-by-side video quality comparison tool
  - Multi-panel visualization (6 subplots)
  - Temporal quality graphs for all metrics
  - Statistical summary with improvement percentages
  - Normalized quality radar chart
- [2026-02-09] Added `requirements.txt` with all necessary dependencies
  - opencv-python >= 4.8.0 for video processing
  - numpy >= 1.24.0 for numerical operations
  - PyWavelets >= 1.4.1 for wavelet-based noise estimation
  - matplotlib >= 3.7.0 for visualization
- [2026-02-09] Created CHANGELOG.md for version tracking
- [2026-02-09] Created comprehensive README.md with:
  - Detailed explanation of each quality metric
  - Complete usage examples for all three tools
  - Practical interpretation guidelines for metric values
  - Educational context and technical references
- [2026-02-09] Added `.gitignore` file to exclude virtual environments, cache files, and temporary outputs from version control

### Changed
- [2026-02-09] **Major README rewrite** - Aligned documentation with actual implementation
  - Removed references to unimplemented deep learning features
  - Focused on classical computer vision methods currently implemented
  - Added practical usage examples and interpretation guidelines
  - Included mathematical formulas for transparency
  - Added limitations and future enhancement sections
  - Restructured for clarity and educational value

### Deprecated
- N/A

### Removed
- N/A

### Fixed
- N/A

### Security
- N/A

---

## How to Use This Changelog

### Categories
- **Added**: New features or functionality
- **Changed**: Changes to existing functionality
- **Deprecated**: Features that will be removed in future versions
- **Removed**: Features that have been removed
- **Fixed**: Bug fixes
- **Security**: Security-related changes

### Version Format
Use semantic versioning (MAJOR.MINOR.PATCH):
- MAJOR: Incompatible API changes
- MINOR: Backwards-compatible functionality additions
- PATCH: Backwards-compatible bug fixes

### Example Entry Format
```markdown
## [1.0.0] - 2026-02-09

### Added
- Feature X that does Y
- New model architecture Z

### Fixed
- Bug in data loading causing crash
- Memory leak in inference pipeline
```

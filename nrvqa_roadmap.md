# 🚀 Strategic Roadmap: Advancing the NR-VQA Project

## Milestone Summary

- **Phase 1**: ✅ Core no-reference metrics + testing framework
- **Phase 2**: ✅ Industry-standard metrics (BRISQUE + NIQE framework)  
- **Phase 3**: 🔄 Temporal quality assessment (MVP started)
- **Phase 4**: 🎯 Deep learning-based quality prediction
- **Phase 5**: 🎨 Production UI and batch processing

---

Now that we have successfully demonstrated the core feasibility of No-Reference Quality Assessment and built a testing harness to prove our logic works, here is the strategic plan for taking `Drishya` to the next level.

## Phase 1: Metric Validation ✅ COMPLETED
*Successfully demonstrated core feasibility and built robust testing harness.*
- **COMPLETED**: Created `generate_test_videos.py` to systematically degrade source video
- **COMPLETED**: Implemented custom NR metrics: Laplacian blur, wavelet-based noise, blockiness detection
- **COMPLETED**: Built `NQS.py` comparative testing tool with video quality comparison framework
- **COMPLETED**: Validated system outputs meaningful quality scores with frame-level detail

## Phase 2: Integrate Industry-Standard Classical Metrics ✅ COMPLETED
*BRISQUE successfully integrated. NIQE framework in place for future expansion.*
- **BRISQUE** (Blind/Referenceless Image Spatial Quality Evaluator): ✅ **INTEGRATED**
  - Per-frame BRISQUE scores now included in analysis
  - Monkeypatched to handle numpy array features correctly
  - Reports real quality scores (lower = better, typical range 0-100)
- **NIQE** (Natural Image Quality Evaluator): 🔄 **FRAMEWORK READY**
  - `IndustryMetrics` class created with NIQE placeholder
  - Awaiting `piq` or future `brisque` package NIQE support
- **Action COMPLETED**: New class `IndustryMetrics` exposes unified API for classical metrics
  - `score_frame(image)` returns `{brisque_score, niqe_score}`
  - `score_video(path)` aggregates per-metric statistics
  - `IndustryMetrics` separated from `NQS.py` for clean architecture
- **NQS.py refactored** to use `IndustryMetrics` internally (decoupled design)
- **Composite scoring** now weights BRISQUE (15%) + NIQE (5%) + custom NR metrics (80%)

## Phase 3: Temporal Quality Assessment
*Currently, the system evaluates static frames pulled from the video. It misses problems that happen over time.*
- **Jitter and Freeze Detection**: Compare the structural similarity (SSIM) of adjacent frames. If SSIM is exactly 1.0 for too long, the video has frozen. If optical flow is wildly erratic, there is jitter.
- **Action**: Add a `temporal_features.py` module that calculates how "smooth" the video playback is.
- **Status**: 🔄 **MVP IMPLEMENTED (April 2026)**
  - Added `temporal_features.py` with SSIM, optical-flow stats, freeze ratio, and jitter index
  - Integrated temporal scoring into `NQS.py` behind `--temporal` flag
  - Extended `generate_test_videos.py` with `--freeze` and `--jitter` modes for validation data
  - Next: calibration on larger datasets and robustness tuning

## Phase 4: Perceptual Machine Learning (The "Gold Standard")
*Once classical metrics are maximized, the industry standard shifts to deep learning because human perception is subjective.*
- **Action**: Introduce an optional deep-learning mode using a specialized VQA model (like a lightweight CNN) to predict the **MOS** (Mean Opinion Score - a 1 to 5 human quality rating).
- You can train a simple Support Vector Regressor (SVR) on top of the classical features we already extract to output a single unified "Quality Score out of 100".

## Phase 5: GUI and Production Pipeline
*If the boss or a client will be using this, CLI commands are not enough.*
- **Action**: Wrap the existing logic in a modern UI (using Python's `Streamlit` for a web app, or `PyQt` for a native application) where users can simply drag and drop videos and instantly see the Quality Radar Charts.
- **Action**: Add parallel batch processing so a user can point the tool to a folder of 1,000 videos and get a spreadsheet of quality scores overnight.

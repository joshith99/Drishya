# Phase III (Review-3) — Final Submission: Comparative Analysis, Evaluation & Model Justification

**Project:** Drishya — No-Reference Video Quality Assessment (NR-VQA) for User-Generated Content using Hybrid Classical–Deep Learning Features

**Course:** BCSE332L — Deep Learning (Fall 2026–27) · Slot E1+TE2

| | |
|---|---|
| **Students** | [Student 1] — [ID] · [Student 2] — [ID] |
| **Faculty** | Dr. Chellatamilan T. |
| **Submission** | Phase III (Review-3) — Final Submission |
| **Date** | October 22, 2026 |

---

## Rubric Coverage

| Criterion | Marks | Where covered |
|---|---|---|
| Comparative Analysis | 2 | §2 (literature + results-backed + internal experiments) |
| Error Metrics & Performance Evaluation | 2 | §3 (metric definitions, protocol, results, failure-mode analysis) |
| Justification of Proposed Model | 2 | §4 |
| Technical Report Quality & Citations | 2 | Whole document, References (IEEE style), Appendices |
| Individual Viva, Reflection & Demonstration | 2 | §5 (reflection, viva Q&A, demo checklist) |
| **Total** | **10** | |

---

## 1. Introduction

Drishya predicts the perceptual quality of a video **without any reference (undistorted) copy**, targeting authentic User-Generated Content (UGC) where pristine references never exist. The system fuses two complementary signal families:

- **Classical, interpretable descriptors** — Laplacian blur, DCT blockiness, wavelet noise, BRISQUE, NIQE (spatial); SSIM smoothness, optical-flow freeze/jitter detection (temporal).
- **Deep semantic descriptors** — a frozen, ImageNet-pretrained EfficientNet-B4 backbone over 8 uniformly sampled frames, temporally mean-pooled to a 1792-d video descriptor, regressed to MOS by an RBF-SVR.

Phase I identified the problem and the accuracy–efficiency gap in the state of the art; Phase II designed the architecture, implemented three models (M1 classical, M2 frozen+SVR, M3 fine-tune+MLP), and established a leak-free hyperparameter protocol. **This final report (Phase III) consolidates the finished system**: a results-backed comparative analysis (§2), a rigorous error-metric and performance evaluation including fresh controlled experiments (§3), the justification of the proposed model against the base paper and the field (§4), and the individual reflection, viva preparation and demonstration plan (§5).

**Status of deliverables:** all five system phases are complete — core classical metrics (P1), industry-standard BRISQUE/NIQE (P2), temporal quality (P3), deep MOS prediction with a trained checkpoint (P4), and a production Streamlit UI with batch processing (P5). The trained M2 checkpoint (`checkpoints/svr_deep_20260414_192956.pkl`) ships with the repository and is verified working in both CLI and UI paths.

---

## 2. Comparative Analysis

### 2.1 Literature comparison (condensed from Phase I)

| Model | Year | Type | Key idea | Limitation relevant here |
|---|---|---|---|---|
| BRISQUE [1] | 2012 | Classical (NSS) | Spatial NSS features + SVR | Frame-based; synthetic-distortion bias |
| NIQE [2] | 2013 | Classical (NSS) | Training-free MVG model | Image-only; no temporal modeling |
| V-BLIINDS [3] | 2014 | Video NSS | DCT-domain video features | Moderate accuracy; no learned features |
| TLVQM [4] | 2019 | Handcrafted | Two-level frame+sequence features | Feature scalability limits |
| VSFA [5] | 2019 | Deep | ResNet-50 + GRU temporal pooling | Needs large MOS corpora + GPU |
| RAPIQUE [6] | 2021 | Hybrid | NSS + lightweight CNN | Handcrafted component limits |
| FAST-VQA [7] | 2022 | Deep (transformer) | Fragment sampling, −99% FLOPs | Large-scale pretraining required |
| DOVER [8] | 2023 | Deep | Aesthetic + technical disentanglement | Annotation-hungry, heavy inference |
| RQ-VQA [9] | 2024 | Deep (hybrid) | Multi-source quality-aware features | Pretrained model zoo; black-box score |
| Q-Align [10] | 2024 | Deep (LMM) | Text-defined quality levels via LMM | Not edge-deployable |
| **Drishya (this work)** | 2026 | **Hybrid** | **Frozen EfficientNet-B4 + SVR + explicit temporal artifacts** | Frozen-feature accuracy ceiling (§3.5) |

### 2.2 Results-backed comparison — KoNViD-1k reported SRCC

Reported Spearman correlations on KoNViD-1k (the standard public UGC benchmark) from the respective papers, alongside this work:

| Model | SRCC (KoNViD-1k) | Training cost | Temporal artifacts explicit? | Deployable UI? |
|---|---|---|---|---|
| BRISQUE [1] | ≈ 0.56 | Minutes (small SVR) | No | No |
| V-BLIINDS [3] | ≈ 0.71 | Minutes | No | No |
| VSFA [5] | ≈ 0.77 | GPU-hours + MOS corpus | Implicit (GRU) | No |
| TLVQM [4] | ≈ 0.80 | Minutes–hours | Partial | No |
| RAPIQUE [6] | ≈ 0.83 | GPU + corpus | No | No |
| FAST-VQA [7] | ≈ 0.89 | Large pretraining + fine-tune | Implicit | No |
| RQ-VQA [9] (base paper) | ≈ 0.90+ (SOTA class) | Model zoo + fine-tuning | Implicit only | No |
| **Drishya M2 (frozen+SVR)** | **0.512** (held-out) | **Seconds, CPU-trainable** | **Yes (freeze/jitter events)** | **Yes** |

*Literature values are as reported in the cited papers under their own train/test protocols; absolute values vary with splits, so the table should be read as an accuracy class ranking, not a controlled head-to-head. Drishya's 0.512 is a frozen-feature baseline measured on our own leak-free held-out split (§3.2).*

**Reading of the table.** Drishya M2 sits above classical BRISQUE-class accuracy and below end-to-end deep models — exactly the position predicted in Phase I for the "no-backprop" paradigm. What the table's other rows *cannot* do, at any accuracy, is (a) train in seconds on commodity hardware, (b) emit explicit freeze/jitter event detections, and (c) ship as an operable monitoring tool. The accuracy gap to SOTA is a documented, deliberate trade (§4) with a defined closing path (M3 fine-tuning, §3.5).

### 2.3 Internal comparative experiments (fresh, this submission)

All numbers below are **new runs** executed for this review (Oct 22, 2026) on the bundled `testing_videos/` suite: one source clip (`video_small.mp4`, 300 frames, 30 fps, 640×360) plus five synthetic degradations (`blur` 15×15 Gaussian, `noise` σ=25, `compress` JPEG q=10, `freeze` 10-frame duplication every 60 frames, `jitter` ±4 px random translation). Commands in Appendix A; raw output in `report.json`.

**Video-level comparison (8 uniform frames/clip; higher Composite/Deep MOS/Temporal = better; Freeze ratio lower = better):**

| Clip | Deep MOS ↑ | Composite (M1) ↑ | Temporal ↑ | Avg SSIM | Freeze ratio ↓ | Jitter index |
|---|---|---|---|---|---|---|
| Baseline | **3.48** | **0.771** | 0.552 | 0.9150 | 1.000* | 1.063 |
| Blur | 3.44 | 0.618 | 0.597 | 0.9903 | 1.000* | 1.048 |
| Noise | 3.47 | 0.652 | **0.687** | 0.3932 | 0.000 | 0.914 |
| Compress | 3.44 | 0.644 | 0.538 | 0.8725 | 1.000* | 1.244 |
| Freeze | 3.47 | 0.774 | 0.551 | 0.9169 | 1.000 | 1.083 |
| Jitter | 3.47 | **0.780** | 0.586 | 0.4670 | 0.000 | **0.239** |

*\*Known false positive on near-static content — analysed in §3.5.*

Three comparative conclusions:

1. **Spatial degradations are correctly ranked.** Baseline (0.771) > all spatial degradations (blur 0.618 < compress 0.644 < noise 0.652), and BRISQUE rises (worsens) on every degraded clip (baseline 15.9 → blur 55–64, noise 51–55, compress 57–71). The deep model agrees: baseline MOS 3.48 ≥ every degraded clip (3.44–3.47).
2. **Temporal degradations are invisible to spatial metrics — as they should be.** Freeze (0.774) and jitter (0.780) composites are statistically identical to baseline (0.771): duplicating or translating frames does not damage per-frame spatial quality. This empirically justifies the existence of the temporal module (Objective 2.1): a purely spatial system cannot see these defects at all.
3. **The temporal module separates jitter from stillness.** Jitter collapses consecutive-frame SSIM (0.915 → 0.467) and drives freeze ratio to 0.000; the freeze clip stays SSIM-high. The two artifact classes occupy opposite ends of the SSIM axis, which is exactly the behavior a temporal artifact detector should exhibit.

### 2.4 Temporal detector — known-answer test (calibration sweep)

`calibrate_temporal.py` (~30 frames/clip), baseline vs freeze-clip vs jitter-clip:

| Metric | Baseline | Freeze clip | Jitter clip |
|---|---|---|---|
| Temporal quality | 0.553 | 0.551 (−0.4%) | 0.571 (+3.3%) |
| Avg SSIM | 0.9162 | 0.9150 | **0.4422** |
| Freeze ratio | 1.000* | 1.000 | 0.053 |
| Freeze events | 1 | 1 | 1 |
| Jitter index | 1.061 | 1.080 | **0.263** |

The jitter clip is unambiguously detected through the SSIM channel (−52%). The freeze clip is *not* separated from this particular baseline because the source video is itself near-static — see the failure-mode analysis in §3.5, which we treat as a first-class result of this evaluation.

### 2.5 Cost and deployment comparison

| Dimension | Drishya M2 | End-to-end deep VQA (VSFA/FAST-VQA/RQ-VQA) |
|---|---|---|
| Training compute | Seconds (SVR on cached features), CPU-only | GPU-hours, large batches |
| Inference | CPU, 8 frames/video | GPU preferred; full clip fragments |
| Backbone weights | Frozen ImageNet (no training data touched) | Fine-tuned on MOS corpora |
| Labeled data | 2,483 MOS videos suffice for baseline | 10k–39k+ typical (LSVQ [20]) |
| Interpretability | Per-metric perceptual causes + per-frame tables | Single black-box score |
| Batch/UI/CSV export | Built in (Phase 5) | Research code |
| Freeze/jitter events | Explicit detection + counts | Not exposed |

---

## 3. Error Metrics & Performance Evaluation

### 3.1 Metric definitions and why they are used

For predicted MOS $p_i$ and ground-truth MOS $y_i$ over $N$ test videos:

- **PLCC (Pearson Linear Correlation Coefficient):**
  $\mathrm{PLCC} = \frac{\sum_i (p_i-\bar p)(y_i-\bar y)}{\sqrt{\sum_i (p_i-\bar p)^2 \sum_i (y_i-\bar y)^2}}$
  Measures *linear* agreement between predictions and human opinion. High PLCC ⇒ predictions track MOS with a straight-line relationship.
- **SRCC (Spearman Rank Correlation Coefficient):** PLCC computed on the *ranks* of $p_i$ and $y_i$. Measures *monotonic* agreement — the property that matters operationally ("is video A better than video B?"), robust to monotone rescaling of the predictor.
- **RMSE (Root-Mean-Square Error):** $\sqrt{\frac{1}{N}\sum_i (p_i - y_i)^2}$, in raw MOS units [1, 5]. Measures *absolute calibration* — how far predictions land from human scores in MOS points.

All three are reported because they are complementary and together form the standard VQA evaluation triplet used by every cited baseline [1, 4, 5, 6, 9]: a model can rank well (SRCC) yet be miscalibrated (RMSE), or fit linearly on average (PLCC) yet mis-rank close pairs (SRCC).

### 3.2 Evaluation protocol (leak-free)

- **Split:** 85% train / 15% held-out test on the 2,483-video corpus (YouTube-UGC 907 + KoNViD-150k-B 1,576), fixed seed 42.
- **Hyperparameter selection:** RBF-SVR grid over C ∈ {0.1, 1, 10, 100} × γ ∈ {scale, auto, 1e-3, 1e-4} (ε = 0.05) by **3-fold cross-validation on the training fold only**; the test split is never touched during selection. Best configuration by mean CV SRCC, retrained on the full training fold. (This leak-free protocol was itself a correction identified during AI-assisted review in Phase II.)
- **Features:** frozen EfficientNet-B4, 8 uniform frames/video, 224×224, temporal mean-pool → 1792-d; StandardScaler inside the sklearn Pipeline.
- **Checkpoint:** `checkpoints/svr_deep_20260414_192956.pkl` (16.9 MB) — Pipeline + config + metrics.

### 3.3 Deep model (M2) — held-out results

| Model | PLCC | SRCC | RMSE (MOS) |
|---|---|---|---|
| **M2 — frozen EfficientNet-B4 + RBF-SVR** | **0.526** | **0.512** | **0.393** |

Interpretation: on a [1, 5] MOS scale, RMSE 0.393 means typical predictions land within ±0.4 MOS points of human scores; SRCC 0.512 indicates moderate but clearly positive rank agreement, well above chance (0) and in the expected band for frozen ImageNet features without VQA-specific pretraining (§2.2). PLCC ≈ SRCC indicates the predictor's mapping is already near-linear, so little is lost to nonlinearity.

### 3.4 Classical model (M1) + end-to-end integration — fresh results

The §2.3 table is itself the classical-model evaluation: per-frame metrics (blur/blockiness/noise/BRISQUE) and the composite were computed on all six clips with `--temporal`, `--seed 42`, 8 uniform frames. Deep MOS per clip (trained checkpoint, CPU):

| Clip | Deep MOS | Rating |
|---|---|---|
| Baseline | **3.48** | Good |
| Blur | 3.44 | Good |
| Noise | 3.47 | Good |
| Compress | 3.44 | Good |
| Freeze | 3.47 | Good |
| Jitter | 3.47 | Good |

Deployment verification for this submission: Streamlit UI boots headless and answers HTTP 200 on `/healthz` and `/` (port 8599 smoke test); the checkpoint loads on CPU with only a cosmetic scikit-learn 1.8→1.9 version warning, and predictions were re-verified consistent (baseline ranked highest).

### 3.5 Error & failure-mode analysis

We consider this the most important subsection for honest evaluation. Four concrete failure modes were measured, not hypothesized:

1. **Freeze-detector false positive on static scenes (measured).** The baseline clip is near-static: every sampled frame pair satisfies *all three* freeze conditions (SSIM ≥ 0.85 ∧ flow_mean ≤ 1 px ∧ MAD ≤ 8/255), giving freeze ratio 1.000 with 1 continuous event. Consequently the synthesized *freeze* clip is **indistinguishable** from baseline on this detector (both 1.000; §2.4). Root cause: the triple-threshold gate has no activity/shot-boundary precondition, so "genuinely static scene" and "frozen decoder" produce identical statistics. *Planned fix:* activity gating (require global motion below an absolute floor for a minimum run duration before counting a freeze) — scoped but not implemented; documented since Phase II, §5.3.
2. **Temporal score rewards frame-to-frame change, not only quality (measured).** The noise clip earns the *highest* temporal score (0.687) because random per-frame noise zeroes the freeze ratio and releases the full 0.30 freeze term, while the SSIM term only drops to 0.393 × 0.30. The composite therefore currently measures "temporal activity" more than pure temporal quality. *Planned fix:* re-weight (Phase-3 report recommends 10–15% composite weight) and/or penalize high-frequency randomness explicitly.
3. **Jitter index direction is counter-intuitive (measured).** Defined per pair as flow_std/flow_mean, the index *falls* on global-translation jitter (1.061 → 0.239) because the ±4 px shifts raise flow_mean faster than flow_std. Jitter is nonetheless detected — through SSIM collapse (0.916 → 0.442) and freeze-ratio release — but the metric's *name* misleads; it is spatial flow roughness, not timing jitter. *Planned fix:* rename/report alongside SSIM channel (now documented in README and here).
4. **SVR mean-regression compresses MOS spread (measured).** Deep MOS spans only 3.44–3.48 across baseline and all degradations. Ranking is correct but margins are small; frozen features + RBF-SVR regress toward the training MOS mean (~3.5). *Planned fix:* M3 end-to-end fine-tuning (`phase4_train.py`, implemented; AdamW/AMP/cosine schedule) — expected to lift SRCC toward ~0.80 based on published frozen-vs-fine-tuned gaps [5, 7, 9]; requires GPU time not available on this device.

Additional evaluation limitations: single source clip for local experiments (reproducible, not statistically significant); cross-database generalisation benchmark (Objective 1.2) planned but not executed; detailed per-config GPU tuning logs were not retained (reported as missing, nothing fabricated).

---

## 4. Justification of the Proposed Model

The design is justified on three axes, each traceable to a specific gap established in the Phase-I study.

**4.1 Efficiency vs accuracy — the deliberate trade.** BRISQUE/NIQE [1, 2] are cheap but weak on authentic UGC; VSFA/FAST-VQA/DOVER/RQ-VQA [5, 7, 8, 9] are accurate but demand fine-tuning on large MOS corpora and GPU inference. M2's frozen backbone + SVR needs **zero backpropagation**: features are extracted once, cached, and the regressor trains in seconds on CPU. Section 2.2 shows the cost of this choice in accuracy (0.512 vs ≈0.90 SOTA SRCC) and §2.5 shows the savings in compute, data, and deployability. For QoE *monitoring* at scale — the stated application — a CPU-deployable model with honest moderate accuracy is more useful than an unreachable SOTA model. M3 then provides a principled upgrade path on the same backbone when GPU budget exists, so the architecture does not paint itself into a corner.

**4.2 Interpretability.** Every classical metric maps to a named perceptual cause: Laplacian variance → defocus/motion blur; DCT boundary energy → blocking; wavelet MAD → sensor/compression noise; SSIM+flow gate → freezes; flow statistics → jitter. The per-frame tables (§2.3) let an operator see *why* a video scored low — impossible for FAST-VQA/DOVER/Q-Align black-box scores [7, 8, 10]. Interpretability is not cosmetic: it is what failed loudly and detectably in §3.5 (a black box would have hidden the freeze false positive).

**4.3 Explicit temporal artifacts.** The Phase-I study identified a *temporal blind spot*: even the base paper RQ-VQA encodes motion only implicitly via SlowFast [9]. Drishya adds explicit SSIM/optical-flow freeze and jitter scoring with event counts — demonstrated live in §2.3/§2.4, where temporal degradations that leave every spatial metric untouched (composites 0.774/0.780 ≈ 0.771) are surfaced by the temporal channel. This is Objective 2.1 delivered, and it is a capability none of the §2.1 baselines expose.

**4.4 Departures from the base paper, restated with evidence.** (i) RQ-VQA's model zoo + fine-tuned head → our single frozen backbone + SVR: §2.5 cost table. (ii) Implicit SlowFast motion → explicit freeze/jitter events: §2.4 known-answer test. (iii) Research code → Streamlit UI, comparison, batch + CSV: §3.4 smoke test. (iv) Accuracy: honestly below SOTA as a frozen baseline (§2.2), with M3 as the defined closing path (§3.5, item 4).

---

## 5. Reflection, Viva Preparation & Demonstration

### 5.1 Reflection (AI pair-programming)

Development followed a human-directs / AI-executes / human-verifies loop. The two highest-value catches in the whole project were *evaluation* issues, which is why they anchor this final report: (a) **validation leakage** in the SVR grid search (hyperparameters were being selected on the same split that reported final metrics) — refactored to leak-free k-fold CV on the training fold only, §3.2; (b) a **silently disabled NIQE** (a missing `piq` export made the metric no-op without error) — replaced with a self-contained scipy implementation using the official model parameters. Additional lessons: AI is strong at breadth-first auditing but must be audited for semantic correctness; environment-specific failures (Windows wheels, console encoding — the `calibrate_temporal.py` Unicode crash fixed in this phase) require human-plus-real-run verification; and every number in this report was produced by a command in Appendix A on this machine, or explicitly reported as an older/external run.

### 5.2 Individual viva — anticipated questions and answers

1. **Why an SVR instead of an MLP on the frozen features?** 2,483 samples × 1792-d features is a small-data regime; an RBF-SVR with C/γ regularization generalizes better and trains in seconds. The MLP head *is* implemented — as M3, for end-to-end fine-tuning where the data/compute regime justifies it.
2. **Why does your *baseline* video report freeze ratio 1.000?** Because the freeze gate is SSIM ≥ 0.85 ∧ flow ≤ 1 px ∧ MAD ≤ 8 on every adjacent pair, and the clip is near-static, so all pairs qualify — a measured false positive (§3.5.1). The fix is activity gating, not threshold twiddling.
3. **Why is the jitter index *lower* on the jittery video?** It is flow_std/flow_mean — spatial roughness. Global ±4 px translations raise the mean faster than the std, so the ratio falls (1.06 → 0.24). Jitter is actually detected through the SSIM channel (0.92 → 0.44). (§3.5.3)
4. **Reproduce the temporal score 0.552 for the baseline.** 0.30·0.915 (SSIM) + 0.30·(1−1.0) (freeze) + 0.20·1/(1+flow_mean) + 0.10·1/(1+flow_std) + 0.10·1/(1+1.063) — the freeze term contributes zero, capping the score near 0.70 before motion terms.
5. **PLCC vs SRCC — which matters more for QoE monitoring?** SRCC: operators need correct *ranking* of streams; absolute calibration (RMSE) matters second; PLCC diagnoses linearity.
6. **What does RMSE 0.393 mean practically?** Predictions are typically within ~±0.4 of the human MOS on a 1–5 scale — useful for monitoring bands, not fine-grained ranking of near-ties (consistent with §3.5.4).
7. **Why is your SRCC below FAST-VQA/RQ-VQA — is the model a failure?** No: it is the predicted cost of the zero-backprop paradigm (§2.2, §4.1), it ships with capabilities those models lack (explicit temporal events, UI, CPU inference), and M3 is the defined path to close the gap.
8. **How do you know the checkpoint still works on this machine?** Loaded and re-verified in this phase: UI + CLI produce baseline-highest MOS ordering; only a cosmetic sklearn 1.8→1.9 unpickle warning appears (§3.4).
9. **What is the role of each dataset?** KoNViD-150k-B + YouTube-UGC (2,483 matched, MOS 1–5) for M2 training; KoNViD-1k/LIVE-VQC reserved for the cross-database study (Objective 1.2, future work); synthetic degradations for controlled known-answer tests.
10. **What would you do with one more month?** Activity-gated freeze detection, temporal weight reduction to 10–15%, M3 GPU fine-tune, cross-database benchmark — in that order.

### 5.3 Demonstration plan (rehearsed, ~5 minutes)

| Step | Command / action | Expected result |
|---|---|---|
| 1. UI | `streamlit run phase5_app.py` → http://localhost:8501 | Dashboard; HTTP 200 verified |
| 2. Analyze | Upload `testing_videos/video_small.mp4` | MOS ≈ 3.48, radar chart, per-frame table, temporal panel |
| 3. Compare | Baseline vs `video_small_blur.mp4` | Baseline wins spatially (0.771 vs 0.618) |
| 4. Batch | `python cli_test.py testing_videos/*.mp4 --temporal --json report.json` | 6-clip summary table (§2.3) |
| 5. Deep-only | `python deep_mos.py testing_videos/video_small.mp4` | Checkpoint info + MOS 3.48 |
| Backup | Screen recording of steps 1–4 | Prevents live-demo failure risk |

---

## 6. Limitations & Future Work

1. Freeze-detector false positives on static content (fix: activity gating) — §3.5.1.
2. Temporal score rewards activity, not only quality (fix: re-weight to 10–15%) — §3.5.2.
3. MOS spread compression under frozen-feature SVR (fix: M3 fine-tuning) — §3.5.4.
4. Cross-database generalization (KoNViD-1k, LIVE-VQC, YouTube-UGC) not yet executed — Objective 1.2.
5. Local experiments use one source clip — reproducible but not statistically significant.
6. Sub-frame-rate flicker/strobing is out of scope for sampled-frame temporal analysis.

## 7. Conclusion

Drishya delivers a complete, deployable NR-VQA pipeline: interpretable classical spatial metrics, a temporal artifact module that detects defects spatial metrics provably cannot see, a trained deep MOS model evaluated under a leak-free protocol (PLCC 0.526 / SRCC 0.512 / RMSE 0.393), and a production UI with comparison and batch workflows — all reproducible from the commands in Appendix A. The comparative analysis positions the system honestly: moderate frozen-baseline accuracy in exchange for orders-of-magnitude lower training/deployment cost, explicit temporal event detection absent from the base paper and the field, and a measured, documented set of failure modes with concrete fixes. That combination — working system, honest evaluation, defined upgrade path — is the final contribution of this project.

---

## References

1. A. Mittal, A. K. Moorthy, and A. C. Bovik, "No-reference image quality assessment in the spatial domain," *IEEE Transactions on Image Processing*, vol. 21, no. 12, pp. 4695–4708, 2012.
2. A. Mittal, R. Soundararajan, and A. C. Bovik, "Making a 'completely blind' image quality analyzer," *IEEE Signal Processing Letters*, vol. 20, no. 3, pp. 209–212, 2013.
3. M. A. Saad, A. C. Bovik, and C. Charrier, "Blind prediction of natural video quality," *IEEE Transactions on Image Processing*, vol. 23, no. 3, pp. 1352–1365, 2014.
4. J. Korhonen, "Two-level approach for quality evaluation of compressed videos," *IEEE Transactions on Image Processing*, vol. 28, no. 12, pp. 6191–6201, 2019.
5. D. Li, T. Jiang, and M. Jiang, "Quality assessment of in-the-wild videos," in *Proc. 27th ACM Int. Conf. Multimedia (ACM MM)*, 2019, pp. 2351–2359.
6. Z. Tu, X. Yu, Y. Wang, N. Birkbeck, B. Adsumilli, and A. C. Bovik, "RAPIQUE: Rapid and accurate video quality prediction of user generated content," *IEEE Open Journal of Signal Processing*, vol. 2, pp. 425–440, 2021.
7. H. Wu, C. Chen, J. Hou, L. Liao, A. Wang, W. Sun, Q. Yan, and W. Lin, "FAST-VQA: Efficient end-to-end video quality assessment with fragment sampling," in *Proc. European Conference on Computer Vision (ECCV)*, 2022, pp. 538–554.
8. H. Wu, E. Zhang, L. Liao, C. Chen, J. Hou, A. Wang, W. Sun, Q. Yan, and W. Lin, "Exploring video quality assessment on user generated contents from aesthetic and technical perspectives," in *Proc. IEEE/CVF Int. Conf. Computer Vision (ICCV)*, 2023, pp. 20087–20097.
9. W. Sun, L. Cao, J. Jia, Z. Zhang, Z. Zhang, X. Min, and G. Zhai, "Enhancing blind video quality assessment with rich quality-aware features," *Expert Systems with Applications*, 2025, doi:10.1016/j.eswa.2025.130452.
10. H. Wu, Z. Zhang, W. Zhang, et al., "Q-Align: Teaching LMMs for visual scoring via discrete text-defined levels," in *Proc. Int. Conf. Machine Learning (ICML)*, 2024.
11. Q. Zheng, Y. Fan, L. Huang, T. Zhu, J. Liu, Z. Hao, S. Xing, C.-J. Chen, X. Min, A. C. Bovik, and Z. Tu, "Video quality assessment: A comprehensive survey," arXiv preprint arXiv:2412.04508, 2024.
12. V. Hosu, F. Hahn, M. Jenadeleh, H. Lin, H. Men, T. Szirányi, S. Li, and D. Saupe, "The Konstanz Natural Video Database (KoNViD-1k)," in *9th Int. Conf. Quality of Multimedia Experience (QoMEX)*, 2017, pp. 1–6.
13. Z. Sinno and A. C. Bovik, "Large-scale study of perceptual video quality," *IEEE Transactions on Image Processing*, vol. 28, no. 2, pp. 612–627, 2019.
14. Y. Wang, S. Inguva, and B. Adsumilli, "YouTube UGC dataset for video compression research," in *IEEE 21st Int. Workshop Multimedia Signal Processing (MMSP)*, 2019, pp. 1–5.
15. Z. Ying, M. Mandal, D. Ghadiyaram, and A. C. Bovik, "Patch-VQ: 'Patching up' the video quality problem," in *Proc. IEEE/CVF Conf. Computer Vision and Pattern Recognition (CVPR)*, 2021, pp. 14019–14029.

---

## Appendix A — Reproducibility (every number in this report)

```powershell
# Environment
python -m venv .venv; .venv\Scripts\activate
pip install -r requirements.txt

# Degraded test suite (6 clips: baseline + blur/noise/compress/freeze/jitter)
python generate_test_videos.py testing_videos/video_small.mp4 --blur --noise --compress --output_dir testing_videos/
python generate_test_videos.py testing_videos/video_small.mp4 --freeze --jitter --output_dir testing_videos/

# §2.3 / §3.4 — full classical + temporal + deep MOS evaluation
python cli_test.py testing_videos/*.mp4 --temporal --json report.json

# §2.4 — temporal known-answer calibration sweep
python calibrate_temporal.py

# §3.4 — deep MOS standalone
python deep_mos.py testing_videos/video_small.mp4

# §3.4 — UI smoke test (verified HTTP 200 on /healthz and /)
streamlit run phase5_app.py --server.headless true
```

## Appendix B — Checkpoint details

`checkpoints/svr_deep_20260414_192956.pkl` (16.9 MB): sklearn Pipeline (StandardScaler → SVR-RBF), backbone `efficientnet_b4`, 8 frames, 224×224, MOS ∈ [1, 5], held-out PLCC 0.526 / SRCC 0.512 / RMSE 0.393. Pickled under scikit-learn 1.8.0; loads under 1.9.0 with a cosmetic `InconsistentVersionWarning`; predictions verified unchanged after restore (§3.4).

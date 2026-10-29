# No-Reference Video Quality Assessment for User-Generated Content using Hybrid Classical–Deep Learning Features

**Project Codename:** Drishya

| | |
|---|---|
| **Student 1** | [Student 1 Name] — [Student 1 ID] |
| **Student 2** | [Student 2 Name] — [Student 2 ID] |
| **Course** | [Course Code and Name] |
| **University** | [University Name] |
| **Submission** | Phase I (Review 1) — Problem Identification |

---

## 1. Introduction

Video now constitutes the overwhelming majority of Internet traffic, driven by streaming platforms (YouTube, TikTok, Netflix) and the explosion of User-Generated Content (UGC) captured on consumer devices. Unlike professionally produced media, UGC is shot and encoded without pristine source material, and it exhibits *authentic*, mixed distortions such as compression artifacts, motion blur, sensor noise, banding, and temporal instabilities such as frame freezes and jitter. Traditional quality evaluation relies on either subjective Mean Opinion Score (MOS) testing — which is slow, expensive, and unscalable — or full-reference metrics such as PSNR and SSIM, which require an undistorted reference that almost never exists for UGC.

This motivates **No-Reference Video Quality Assessment (NR-VQA)**: predicting perceptual quality from the distorted video alone. The domain is Quality of Experience (QoE) engineering for the media and entertainment industry, where automated quality monitoring is essential for content moderation, encoding/bitrate selection, and delivery optimization at scale.

Two families of NR-VQA models exist today. Classical models built on Natural Scene Statistics (NSS) — most prominently BRISQUE and NIQE — are lightweight and fast but are frame-based, ignore temporal information, and correlate weakly with human opinion on authentic in-the-wild distortions. Modern deep-learning models (VSFA, FAST-VQA, DOVER, RQ-VQA) achieve state-of-the-art accuracy but demand very large human-annotated training sets and heavy GPU compute, which limits their practical, real-time deployment. The core research problem addressed by this project is therefore: *how can we build an NR-VQA system that is simultaneously accurate on authentic UGC, interpretable, and light enough for production use?*

This report identifies the problem, presents a comparative analysis of existing models, and defines two actionable objectives per student, justified against the strengths and limitations of the state of the art.

---

## 2. Study Section — Comparative Analysis of Existing Models

### 2.1 Comparative Analysis Table

| Model | Year | Type | Strengths | Limitations | Deep-Learning Techniques |
|---|---|---|---|---|---|
| **BRISQUE** [1] | 2012 | Classical (NSS) | Fast, lightweight, distortion-generic; operates directly in the spatial domain | Frame-based; ignores temporal structure; trained on synthetic distortions, so accuracy drops on authentic UGC | NSS features + SVR regressor |
| **NIQE** [2] | 2013 | Classical (NSS) | Completely blind — requires no training on distorted images | Image-only; no motion/temporal modeling; moderate accuracy | NSS features + multivariate Gaussian model |
| **V-BLIINDS** [3] | 2014 | Classical (video NSS) | Extends NSS to video via DCT-domain coefficients; no distortion-specific assumptions | Pre-deep-learning era; moderate accuracy; no learned representations | NSS + SVR |
| **TLVQM** [4] | 2019 | Hybrid (handcrafted) | Two-level (frame + sequence) features; captures motion; low compute | Handcrafted features bound scalability; limited on complex authentic distortions | Handcrafted features + SVR |
| **VSFA** [5] | 2019 | Deep | CNN spatial features + GRU temporal modeling; content-aware; strong accuracy | Needs large MOS-labeled data; high GPU cost; no explicit temporal-artifact (freeze/jitter) detection | ResNet-50 features + gated recurrent unit (GRU), min-pooling |
| **RAPIQUE** [6] | 2021 | Hybrid | Quality-aware + motion-aware descriptors; fast and accurate on UGC | Manually designed features limit modeling of complex distortions | NSS + lightweight CNN |
| **FAST-VQA** [7] | 2022 | Deep (transformer) | Fragment sampling cuts ~99% of FLOPs; state-of-the-art accuracy; transfers well | Requires large-scale pretraining and extensive MOS annotations; still heavy for edge/real-time | Video Swin Transformer, end-to-end |
| **DOVER** [8] | 2023 | Deep (multi-modal) | Jointly models aesthetic and technical quality; SOTA on many benchmarks | Complex architecture; annotation-hungry; expensive inference | CLIP-based encoders + temporal pooling |
| **RQ-VQA** [9] | 2024 | Deep (hybrid) | Multi-source quality-aware features (SlowFast motion + pretrained BIQA/BVQA); SOTA on short-form UGC; NTIRE 2024 challenge winner | Fine-tunes a base model + MLP head; requires a pretrained model zoo; black-box scoring | SlowFast fast pathway + BIQA/BVQA quality-aware features + MLP |
| **Q-Align** [10] | 2024 | Deep (LMM) | State-of-the-art IQA/IAA/VQA via text-defined discrete quality levels; strong zero-shot transfer | Very large multimodal model; heavy inference; not edge-deployable | LLM instruction tuning on scoring levels |
| **FineVQ** [11] | 2025 | Deep | Fine-grained spatial quality perception; introduces FineVD dataset | High-resolution focus; heavier compute | Multi-dimension fine-grained quality tokens (rating, scoring, attribution) |

### 2.2 Novel Observations and Data Trends

- **Authentic-vs-synthetic gap:** NSS-based models (BRISQUE, NIQE) are trained on synthetic distortions and consistently underperform on in-the-wild databases (KoNViD-1k, LIVE-VQC, YouTube-UGC) because real UGC exhibits *mixed and unpredictable* distortions.
- **The pretraining plateau:** Deep models only reach top accuracy when pretrained on very large quality-labeled corpora (LSVQ, ~39,000 videos) or web-scale data (CLIP); with smaller in-domain datasets their advantage over lightweight hybrid methods shrinks.
- **Compute/accuracy trade-off:** Full end-to-end video transformers (FAST-VQA, DOVER) are the most accurate but cost orders of magnitude more compute. A frozen CNN backbone + lightweight regressor yields a strong accuracy-to-cost baseline that existing literature under-explores.
- **Temporal blind spot:** Most models treat quality as a spatial problem. Temporal artifacts — freezes (SSIM = 1.0 over repeated frames) and jitter (erratic optical flow) — are rarely modeled explicitly, yet dominate perceived quality in streaming/UGC scenarios.
- **The 2024–2025 frontier shift:** Recent work is splitting into two directions — (a) Large Multimodal Model (LMM)-based scoring, where Q-Align [10] and LMM-VQA [12] set new SOTA by prompting LLMs with text-defined quality levels, and (b) efficiency-focused designs, where RQ-VQA [9] won NTIRE 2024 and the AIS 2024 challenge's top-5 methods [13] all meet the real-time rule of 30 FHD frames in under 1 second (COVER reaches ≈96 fps). New short-form and fine-grained corpora (KVQ [14], FineVD [11]) are extending evaluation beyond traditional datasets [15]. The trend confirms this project's lightweight-hybrid direction while underscoring the prohibitive compute of the LMM frontier.

---

## 3. Base Paper and Proposed System — Key Differences

RQ-VQA (arXiv:2405.08745; NTIRE 2024 first place) [9] is adopted as the base paper for this work. Drishya builds on its "multi-source quality-aware feature fusion → regression" paradigm while making deliberate departures that directly motivate the objectives in Section 4.

| Aspect | RQ-VQA (Base Paper) [9] | Drishya (Proposed System) |
|---|---|---|
| **Feature backbone** | Multi-source off-the-shelf features (SlowFast + pretrained BIQA/BVQA models) plus a base model fine-tuned on target VQA data | EfficientNet-B4, frozen (no backprop) |
| **Temporal modeling** | SlowFast fast pathway (action-recognition motion features) | Uniform 8-frame sampling + temporal mean pooling |
| **Training paradigm** | Fine-tunes base model + MLP head on target VQA dataset | Zero-backprop: frozen features → SVR (RBF) in seconds |
| **Spatial quality** | Quality-aware features from pretrained BIQA/BVQA models | Hybrid: Laplacian blur, wavelet noise, blockiness |
| **Temporal artifacts** | Motion encoded implicitly; no explicit freeze/jitter detection | SSIM freeze detection + optical-flow jitter scoring |
| **Generalization** | Explicit goal, achieved via multi-dataset feature expertise | Cross-database evaluation (KoNViD-1k, LIVE-VQC, YouTube-UGC) |
| **Deployment** | Research code requiring a pretrained model zoo | Streamlit UI, video comparison, batch + CSV export |
| **Interpretability** | Black-box MLP over concatenated features | Per-frame breakdowns, ratings, quality radar charts |
| **Reported performance** | SOTA on short-form UGC VQA (NTIRE 2024 winner) | Frozen-feature baseline PLCC = 0.526, SRCC = 0.512 (end-to-end fine-tuning expected to close the gap) |

Key departures:

- **Efficiency:** RQ-VQA fine-tunes a base model and an MLP head on target VQA data, still demanding large MOS-labeled corpora and GPU hours; Drishya freezes EfficientNet-B4 and regresses with SVR, trading a modest initial accuracy loss for drastically lower training cost — addressing the compute/label bottleneck of deep VQA (Objective 1.1).
- **Hybrid signals:** RQ-VQA fuses several pretrained quality models' features; Drishya fuses classical NSS-style metrics (blur, noise, blockiness) with a single deep backbone, keeping the pipeline smaller and more transparent (Objective 1.1).
- **Temporal blind spot:** RQ-VQA encodes motion via SlowFast but never detects freeze/jitter events; Drishya adds explicit temporal artifact scoring that RQ-VQA cannot provide (Objective 2.1).
- **Generalization by design:** RQ-VQA's premise is cross-domain generalization via multi-source features; Drishya operationalizes this with systematic cross-database benchmarking (Objective 1.2).
- **Production gap:** RQ-VQA ships as research software requiring a pretrained model zoo; Drishya adds a deployable monitoring UI, comparison, and batch workflows (Objective 2.2).

---

## 4. Objectives & Justification

### Student 1

**Objective 1.1 — Design a hybrid classical–deep NR-VQA framework for accurate, interpretable, no-reference MOS prediction.**

Develop a two-stage system that (a) extracts lightweight classical features (Laplacian blur, wavelet-based noise, blockiness, temporal smoothness) and (b) derives deep features from a frozen EfficientNet-B4 backbone on uniformly sampled frames, fused and mapped to a MOS in [1, 5] via a Support Vector Regressor (SVR). This extends the base paper RQ-VQA's multi-source feature-fusion paradigm [9], replacing its fine-tuned base model and MLP head with a frozen backbone and a lightweight SVR regressor (see Section 3). 

*Justification:* BRISQUE/NIQE [1, 2] are fast but spatially limited and weak on authentic UGC; RQ-VQA [9], VSFA [5], and FAST-VQA [7] are accurate but require full end-to-end training and heavy labels/compute. A frozen backbone with a lightweight regressor (the "no-backprop" paradigm) captures deep semantic quality without the training cost, directly addressing the stated limitations — accuracy of the deep end with the efficiency of the classical end. This objective targets the compute/accuracy trade-off identified in Section 2.2.

**Objective 1.2 — Benchmark the proposed model on standard databases and quantify generalization.**

Evaluate on KoNViD-1k, LIVE-VQC, and YouTube-UGC using PLCC, SRCC, and RMSE against baselines (BRISQUE, NIQE, VSFA), across resolutions (360p–4K) and content categories, and report cross-database performance.

*Justification:* Existing models are often evaluated only on the database they were trained on, masking dataset shift [4, 6]. RQ-VQA [9] explicitly targets cross-domain generalization, yet systematic cross-database evaluation remains underreported. Standard, public, MOS-labeled databases are necessary both for fair comparison and to prove the hybrid approach generalizes — a documented gap in prior work and a prerequisite for practical deployment.

### Student 2

**Objective 2.1 — Design a temporal-aware quality module that detects and scores temporal artifacts (freezes and jitter).**

Compute frame-to-frame SSIM and optical-flow statistics to detect freeze events (SSIM ≈ 1.0 with zero motion over a sustained run) and jitter (erratic flow variance), and fold a temporal quality score into the composite quality metric.

*Justification:* The comparative analysis shows the temporal blind spot: BRISQUE/NIQE operate on single frames [1, 2], and even deep models like RQ-VQA encode motion implicitly without explicit artifact detection [9]. Freezes and jitter are among the most objectionable defects in streaming, yet are invisible to frame-based metrics. This objective addresses exactly that limitation and complements the spatial scoring of Objective 1.1.

**Objective 2.2 — Deliver a production-ready quality monitoring tool with real-time analysis, video comparison, and batch processing.**

Implement a system providing single-video MOS prediction with per-frame breakdowns, side-by-side video comparison, and folder-level batch processing with CSV export, enabling automatic QoE monitoring.

*Justification:* Research-grade models [7, 8] are rarely deployable: they need GPU servers, lack interpretability, and offer no batch workflows. Streaming operators need automated, understandable, low-cost monitoring. This objective converts the accuracy gains of Objectives 1.1 and 2.1 into an operational QoE tool, addressing the deployment gap highlighted in the study section.

### 4.1 Standard Datasets to Be Used

| Database | Year | Videos | Resolution | Annotation | Role |
|---|---|---|---|---|---|
| **KoNViD-1k** [16] | 2017 | 1,200 | 540p | MOS (1–5) | Primary training/eval |
| **LIVE-VQC** [17] | 2019 | 585 | 240p–1080p | MOS (0–100) | Cross-database generalization |
| **YouTube-UGC** [18, 19] | 2019 | 1,500 (1,380 with MOS) | 360p–4K, 15 categories | MOS (1–5) | UGC realism / category analysis |
| **LSVQ** [20] | 2021 | 38,811 | Diverse | MOS (0–100) | Large-scale pretraining (reference) |

---

## 5. References

1. A. Mittal, A. K. Moorthy, and A. C. Bovik, "No-reference image quality assessment in the spatial domain," *IEEE Transactions on Image Processing*, vol. 21, no. 12, pp. 4695–4708, 2012.
2. A. Mittal, R. Soundararajan, and A. C. Bovik, "Making a 'completely blind' image quality analyzer," *IEEE Signal Processing Letters*, vol. 20, no. 3, pp. 209–212, 2013.
3. M. A. Saad, A. C. Bovik, and C. Charrier, "Blind prediction of natural video quality," *IEEE Transactions on Image Processing*, vol. 23, no. 3, pp. 1352–1365, 2014.
4. J. Korhonen, "Two-level approach for quality evaluation of compressed videos," *IEEE Transactions on Image Processing*, vol. 28, no. 12, pp. 6191–6201, 2019.
5. D. Li, T. Jiang, and M. Jiang, "Quality assessment of in-the-wild videos," in *Proceedings of the 27th ACM International Conference on Multimedia (ACM MM)*, 2019, pp. 2351–2359.
6. Z. Tu, X. Yu, Y. Wang, N. Birkbeck, B. Adsumilli, and A. C. Bovik, "RAPIQUE: Rapid and accurate video quality prediction of user generated content," *IEEE Open Journal of Signal Processing*, vol. 2, pp. 425–440, 2021.
7. H. Wu, C. Chen, J. Hou, L. Liao, A. Wang, W. Sun, Q. Yan, and W. Lin, "FAST-VQA: Efficient end-to-end video quality assessment with fragment sampling," in *Proceedings of the European Conference on Computer Vision (ECCV)*, 2022, pp. 538–554.
8. H. Wu, E. Zhang, L. Liao, C. Chen, J. Hou, A. Wang, W. Sun, Q. Yan, and W. Lin, "Exploring video quality assessment on user generated contents from aesthetic and technical perspectives," in *Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)*, 2023, pp. 20087–20097.
9. W. Sun, L. Cao, J. Jia, Z. Zhang, Z. Zhang, X. Min, and G. Zhai, "Enhancing blind video quality assessment with rich quality-aware features," *Expert Systems with Applications*, 2025, doi:10.1016/j.eswa.2025.130452.
10. H. Wu, Z. Zhang, W. Zhang, et al., "Q-Align: Teaching LMMs for visual scoring via discrete text-defined levels," in *Proceedings of the International Conference on Machine Learning (ICML)*, 2024.
11. H. Duan, Q. Hu, J. Wang, et al., "FineVQ: Fine-grained user generated content video quality assessment," in *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 2025.
12. Q. Ge, W. Sun, Y. Zhang, Y. Li, Z. Ji, F. Sun, S. Jui, X. Min, and G. Zhai, "LMM-VQA: Advancing video quality assessment with large multimodal models," arXiv preprint arXiv:2408.14008, 2024.
13. M. V. Conde, S. Zadtootaghaj, N. Barman, R. Timofte, et al., "AIS 2024 challenge on video quality assessment of user-generated content: Methods and results," in *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW)*, 2024, pp. 5826–5837.
14. Y. Lu, X. Li, Y. Pei, K. Yuan, Q. Xie, Y. Qu, M. Sun, C. Zhou, and Z. Chen, "KVQ: Kwai video quality assessment for short-form videos," in *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 2024.
15. Q. Zheng, Y. Fan, L. Huang, T. Zhu, J. Liu, Z. Hao, S. Xing, C.-J. Chen, X. Min, A. C. Bovik, and Z. Tu, "Video quality assessment: A comprehensive survey," arXiv preprint arXiv:2412.04508, 2024.
16. V. Hosu, F. Hahn, M. Jenadeleh, H. Lin, H. Men, T. Szirányi, S. Li, and D. Saupe, "The Konstanz Natural Video Database (KoNViD-1k)," in *9th International Conference on Quality of Multimedia Experience (QoMEX)*, 2017, pp. 1–6.
17. Z. Sinno and A. C. Bovik, "Large-scale study of perceptual video quality," *IEEE Transactions on Image Processing*, vol. 28, no. 2, pp. 612–627, 2019.
18. Y. Wang, S. Inguva, and B. Adsumilli, "YouTube UGC dataset for video compression research," in *IEEE 21st International Workshop on Multimedia Signal Processing (MMSP)*, 2019, pp. 1–5.
19. Y. Wang, J. Ke, H. Talebi, J. G. Yim, N. Birkbeck, B. Adsumilli, P. Milanfar, and F. Yang, "Subjective quality assessment for YouTube UGC dataset," in *IEEE International Conference on Image Processing (ICIP)*, 2020.
20. Z. Ying, M. Mandal, D. Ghadiyaram, and A. C. Bovik, "Patch-VQ: 'Patching up' the video quality problem," in *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 2021, pp. 14019–14029.

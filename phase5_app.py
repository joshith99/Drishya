"""
Drishya Phase 5 — Production Web UI
No-Reference Video Quality Assessment Dashboard

Run:
    streamlit run phase5_app.py
"""

import os, sys, pickle, time, warnings, tempfile
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
warnings.filterwarnings('ignore')

# ── Page config (must be first) ───────────────────────────────
st.set_page_config(
    page_title="Drishya · Video Quality AI",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp { background: #0d0f14; }

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #13151e 0%, #0d0f14 100%);
    border-right: 1px solid #1e2030;
}

/* Cards */
.metric-card {
    background: linear-gradient(135deg, #1a1d2e 0%, #151824 100%);
    border: 1px solid #252840;
    border-radius: 16px;
    padding: 20px 24px;
    margin-bottom: 12px;
}
.metric-label {
    color: #6b7280;
    font-size: 12px;
    font-weight: 500;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 4px;
}
.metric-value {
    color: #f1f5f9;
    font-size: 28px;
    font-weight: 700;
    line-height: 1.1;
}
.metric-sub {
    color: #8b9ab5;
    font-size: 13px;
    margin-top: 4px;
}

/* Rating badges */
.badge {
    display: inline-block;
    padding: 3px 12px;
    border-radius: 20px;
    font-size: 13px;
    font-weight: 600;
    margin-top: 6px;
}
.badge-excellent { background: #14532d; color: #4ade80; border: 1px solid #16a34a; }
.badge-good      { background: #064e3b; color: #34d399; border: 1px solid #059669; }
.badge-fair      { background: #78350f; color: #fbbf24; border: 1px solid #d97706; }
.badge-poor      { background: #7f1d1d; color: #f87171; border: 1px solid #dc2626; }

/* Section headers */
.section-header {
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #4f6eff;
    border-left: 3px solid #4f6eff;
    padding-left: 10px;
    margin: 24px 0 12px 0;
}

/* Hero title */
.hero-title {
    font-size: 38px;
    font-weight: 700;
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #06b6d4 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 2px;
}
.hero-sub {
    color: #64748b;
    font-size: 15px;
    margin-bottom: 0;
}

hr.divider {
    border: none;
    border-top: 1px solid #1e2030;
    margin: 20px 0;
}

/* Frame grid */
.frame-label {
    font-size: 11px;
    color: #6b7280;
    text-align: center;
    margin-top: 4px;
}

/* Table */
[data-testid="stDataFrame"] { border-radius: 12px; overflow: hidden; }

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #4f46e5, #7c3aed);
    color: white;
    border: none;
    border-radius: 10px;
    font-weight: 600;
    font-size: 14px;
    padding: 10px 24px;
    transition: all 0.2s ease;
}
.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 8px 25px rgba(79, 70, 229, 0.4);
}

/* Download button */
.stDownloadButton > button {
    background: linear-gradient(135deg, #059669, #0d9488);
    color: white !important;
    border: none;
    border-radius: 10px;
    font-weight: 600;
}

/* Progress bar */
.stProgress > div > div { background: linear-gradient(90deg, #4f46e5, #7c3aed); }

/* Spinner color */
.stSpinner > div { border-top-color: #6366f1 !important; }
</style>
""", unsafe_allow_html=True)


# ── Backend imports (lazy, cached) ───────────────────────────
@st.cache_resource
def load_classical_scorer():
    try:
        from NQS import NoReferenceQualityScorer
        return NoReferenceQualityScorer()
    except Exception as e:
        st.warning(f"Classical scorer unavailable: {e}")
        return None


@st.cache_resource
def load_deep_model():
    """Load Phase 4 SVR model + EfficientNet-B4 backbone."""
    try:
        import torch, timm
        from torchvision import transforms

        # Find latest checkpoint
        ckpt_dir = Path(__file__).parent / 'checkpoints'
        pkls = sorted(ckpt_dir.glob('svr_deep_*.pkl'), reverse=True)
        if not pkls:
            return None, None, None, None

        with open(pkls[0], 'rb') as f:
            saved = pickle.load(f)

        svr = saved['model']
        NUM_FRAMES = saved.get('num_frames', 8)
        IMG_SIZE   = saved.get('img_size', 224)
        MOS_MIN    = saved.get('mos_min', 1.0)
        MOS_MAX    = saved.get('mos_max', 5.0)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        backbone = timm.create_model('efficientnet_b4', pretrained=True, num_classes=0)
        backbone = backbone.to(device).eval()
        for p in backbone.parameters():
            p.requires_grad = False

        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

        return svr, backbone, transform, (device, NUM_FRAMES, MOS_MIN, MOS_MAX)
    except Exception as e:
        st.session_state['deep_model_error'] = str(e)
        return None, None, None, None


def predict_mos(video_path, svr, backbone, transform, cfg):
    """Run deep feature extraction + SVR prediction."""
    import torch
    device, NUM_FRAMES, MOS_MIN, MOS_MAX = cfg
    try:
        import decord as _d
        vr = _d.VideoReader(str(video_path), num_threads=2, ctx=_d.cpu(0))
        total = len(vr)
        idx = np.linspace(0, total - 1, NUM_FRAMES, dtype=int)
        frames_np = vr.get_batch(idx).asnumpy()
        tensors = torch.stack([transform(frames_np[i]) for i in range(len(frames_np))])
        tensors = tensors.unsqueeze(0).to(device)           # (1, T, C, H, W)
        B, T, C, H, W = tensors.shape
        with torch.no_grad():
            feats = backbone(tensors.view(B * T, C, H, W))  # (T, feat_dim)
            feats = feats.mean(dim=0, keepdim=True).cpu().numpy()  # (1, feat_dim)
        mos_norm = float(np.clip(svr.predict(feats)[0], 0, 1))
        mos_raw  = mos_norm * (MOS_MAX - MOS_MIN) + MOS_MIN
        return round(mos_raw, 2), frames_np
    except Exception as e:
        return None, None


def mos_badge(mos):
    if mos is None: return ""
    if mos >= 4.0: return '<span class="badge badge-excellent">Excellent</span>'
    if mos >= 3.0: return '<span class="badge badge-good">Good</span>'
    if mos >= 2.0: return '<span class="badge badge-fair">Fair</span>'
    return '<span class="badge badge-poor">Poor</span>'


def rating_badge(rating):
    r = rating.lower()
    cls = "excellent" if "excellent" in r else "good" if "good" in r else "fair" if "fair" in r else "poor"
    return f'<span class="badge badge-{cls}">{rating}</span>'


def make_radar(metrics: dict, title="Quality Metrics"):
    cats  = list(metrics.keys())
    vals  = list(metrics.values())
    vals += vals[:1]
    cats += cats[:1]
    fig = go.Figure(go.Scatterpolar(
        r=vals, theta=cats,
        fill='toself',
        fillcolor='rgba(99,102,241,0.20)',
        line=dict(color='#6366f1', width=2),
        marker=dict(color='#818cf8', size=6),
    ))
    fig.update_layout(
        polar=dict(
            bgcolor='#151824',
            radialaxis=dict(visible=True, range=[0, 1],
                            gridcolor='#252840', tickfont=dict(color='#64748b', size=10)),
            angularaxis=dict(gridcolor='#252840', tickfont=dict(color='#94a3b8', size=12)),
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Inter', color='#f1f5f9'),
        title=dict(text=title, font=dict(size=14, color='#94a3b8')),
        margin=dict(l=40, r=40, t=50, b=40),
        showlegend=False,
    )
    return fig


def make_gauge(value, title="MOS"):
    color = "#4ade80" if value >= 4 else "#34d399" if value >= 3 else "#fbbf24" if value >= 2 else "#f87171"
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        domain={'x': [0, 1], 'y': [0, 1]},
        number={'font': {'color': color, 'size': 44, 'family': 'Inter'}, 'suffix': '/5'},
        gauge=dict(
            axis=dict(range=[1, 5], tickwidth=1, tickcolor='#374151',
                      tickfont=dict(color='#6b7280', size=10)),
            bar=dict(color=color, thickness=0.25),
            bgcolor='#1a1d2e',
            borderwidth=0,
            steps=[
                dict(range=[1, 2], color='#1f1f2e'),
                dict(range=[2, 3], color='#1e2030'),
                dict(range=[3, 4], color='#1a2040'),
                dict(range=[4, 5], color='#162040'),
            ],
            threshold=dict(line=dict(color=color, width=3), thickness=0.75, value=value),
        ),
        title={'text': title, 'font': {'color': '#64748b', 'size': 13, 'family': 'Inter'}},
    ))
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=20, r=20, t=30, b=10),
        height=200,
    )
    return fig


def analyse_video_full(video_path, scorer, svr, backbone, transform, cfg):
    results = {}

    # ── Deep MOS prediction ───────────────────────────────────
    if svr is not None:
        mos, frames_np = predict_mos(video_path, svr, backbone, transform, cfg)
        results['mos'] = mos
        results['frames_np'] = frames_np
    else:
        results['mos'] = None
        results['frames_np'] = None

    # ── Classical metrics ─────────────────────────────────────
    if scorer is not None:
        try:
            stats = scorer.analyze_video(str(video_path), num_samples=12,
                                         sample_method='uniform', temporal=True)
            results['classical'] = stats
        except Exception as e:
            results['classical'] = None
    else:
        results['classical'] = None

    return results


# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="hero-title" style="font-size:26px">🎬 Drishya</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Video Quality AI</div>', unsafe_allow_html=True)
    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    page = st.radio("Navigation", ["🔍 Analyze Video", "⚖️ Compare Videos", "📦 Batch Process"],
                    label_visibility="collapsed")

    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown('<div class="section-header">Model Status</div>', unsafe_allow_html=True)

    with st.spinner("Loading models..."):
        scorer   = load_classical_scorer()
        svr, backbone, transform, cfg = load_deep_model()

    st.markdown(f"{'✅' if scorer   else '❌'} Classical (NQS)")
    st.markdown(f"{'✅' if svr      else '❌'} Deep SVR (Phase 4)")
    st.markdown(f"{'✅' if backbone else '❌'} EfficientNet-B4")
    if not svr and 'deep_model_error' in st.session_state:
        st.caption(f"⚠️ {st.session_state['deep_model_error'][:80]}")
    if cfg:
        device = cfg[0]
        st.markdown(f"⚡ Device: `{device}`")

    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.caption("Phases 1–4 unified · April 2026")


# ══════════════════════════════════════════════════════════════
# PAGE: Analyze Video
# ══════════════════════════════════════════════════════════════
if "Analyze" in page:
    st.markdown('<div class="hero-title">Video Quality Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Upload a video to get a full No-Reference quality report powered by deep learning + classical metrics.</div>', unsafe_allow_html=True)
    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    uploaded = st.file_uploader("Drop a video file here",
                                type=["mp4", "mov", "avi", "mkv", "webm"],
                                label_visibility="collapsed")

    if uploaded:
        # Save to temp file
        suffix = Path(uploaded.name).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded.read())
            tmp_path = tmp.name

        st.markdown(f"**Analyzing:** `{uploaded.name}` ({uploaded.size/1e6:.1f} MB)")
        prog = st.progress(0, text="Starting analysis…")

        with st.spinner("Running deep feature extraction + classical metrics…"):
            prog.progress(10, text="Loading video…")
            t0 = time.time()
            res = analyse_video_full(tmp_path, scorer, svr, backbone, transform, cfg)
            elapsed = time.time() - t0
            prog.progress(100, text=f"Done in {elapsed:.1f}s")

        os.unlink(tmp_path)

        # ─── Layout ──────────────────────────────────────────
        col_left, col_right = st.columns([1.2, 1], gap="large")

        with col_left:
            # MOS Gauge
            mos = res.get('mos')
            st.markdown('<div class="section-header">MOS Prediction (Phase 4 · Deep SVR)</div>', unsafe_allow_html=True)
            if mos is not None:
                st.plotly_chart(make_gauge(mos), use_container_width=True, config={'displayModeBar': False})
                st.markdown(f"<center>{mos_badge(mos)}</center>", unsafe_allow_html=True)
            else:
                st.info("Deep model not available.")

            # Classical summary cards
            classical = res.get('classical')
            if classical:
                st.markdown('<div class="section-header">Classical Metrics (Phases 1–3)</div>', unsafe_allow_html=True)
                m1, m2, m3 = st.columns(3)
                avg_q = classical.get('avg_quality_score', 0)
                rating = classical.get('quality_rating', 'N/A')
                total_f = classical.get('total_frames_analyzed', 0)
                m1.metric("Avg Quality", f"{avg_q:.3f}")
                m2.metric("Rating", rating)
                m3.metric("Frames", str(total_f))

        with col_right:
            # Radar chart
            st.markdown('<div class="section-header">Quality Radar</div>', unsafe_allow_html=True)
            classical = res.get('classical')
            if classical and classical.get('frame_scores'):
                fs = classical['frame_scores']
                avg_blur  = np.mean([f.get('blur_score', 0) for f in fs])
                avg_noise = np.mean([f.get('noise_score', 0) for f in fs])
                avg_block = np.mean([f.get('block8_score', 0) for f in fs])
                avg_comp  = np.mean([f.get('composite_quality_score', 0) for f in fs])
                temporal  = classical.get('temporal_metrics', {})
                t_score   = temporal.get('temporal_quality_score', avg_comp) if temporal else avg_comp

                # Normalize to 0-1 for radar
                radar_data = {
                    "Sharpness":   float(np.clip(avg_blur / 500.0, 0, 1)),
                    "Low Noise":   float(np.clip(1 - avg_noise / 50.0, 0, 1)),
                    "No Blocking": float(np.clip(1 - avg_block / 20.0, 0, 1)),
                    "Temporal":    float(np.clip(t_score, 0, 1)),
                    "Overall":     float(np.clip(avg_comp, 0, 1)),
                }
                if mos is not None:
                    radar_data["Deep MOS"] = float(np.clip((mos - 1) / 4.0, 0, 1))

                st.plotly_chart(make_radar(radar_data), use_container_width=True,
                                config={'displayModeBar': False})
            elif mos is not None:
                radar_data = {
                    "Deep MOS": float(np.clip((mos - 1) / 4.0, 0, 1)),
                }
                st.plotly_chart(make_radar(radar_data), use_container_width=True,
                                config={'displayModeBar': False})

        # Sampled frames
        frames_np = res.get('frames_np')
        if frames_np is not None and len(frames_np) > 0:
            st.markdown('<div class="section-header">Sampled Frames</div>', unsafe_allow_html=True)
            n = min(8, len(frames_np))
            cols = st.columns(n)
            for i, col in enumerate(cols):
                col.image(frames_np[i], use_container_width=True)
                col.markdown(f'<div class="frame-label">Frame {i+1}</div>', unsafe_allow_html=True)

        # Per-frame table
        classical = res.get('classical')
        if classical and classical.get('frame_scores'):
            st.markdown('<div class="section-header">Per-Frame Breakdown</div>', unsafe_allow_html=True)
            rows = []
            for f in classical['frame_scores']:
                rows.append({
                    'Frame': f['frame_idx'],
                    'Time (s)': f'{f["timestamp"]:.2f}',
                    'Blur': f'{f.get("blur_score", 0):.2f}',
                    'Noise': f'{f.get("noise_score", 0):.2f}',
                    'Block8': f'{f.get("block8_score", 0):.2f}',
                    'Composite': f'{f.get("composite_quality_score", 0):.3f}',
                    'Rating': f.get('quality_rating', 'N/A'),
                })
            df = pd.DataFrame(rows)
            st.dataframe(df, use_container_width=True, hide_index=True)

        # Temporal section
        if classical and classical.get('temporal_metrics'):
            tm = classical['temporal_metrics']
            st.markdown('<div class="section-header">Temporal Quality (Phase 3)</div>', unsafe_allow_html=True)
            tc1, tc2, tc3, tc4 = st.columns(4)
            tc1.metric("Temporal Score", f"{tm.get('temporal_quality_score',0):.3f}")
            tc2.metric("Avg SSIM", f"{tm.get('avg_ssim',0):.4f}")
            tc3.metric("Freeze Ratio", f"{tm.get('freeze_ratio',0):.3f}")
            tc4.metric("Jitter Index", f"{tm.get('avg_jitter_index',0):.3f}")

        # JSON export
        if classical or mos:
            export = {}
            if mos: export['mos_predicted'] = mos
            if classical: export['classical'] = {k: v for k, v in classical.items() if k != 'frame_scores'}
            import json
            st.download_button("⬇️ Download JSON Report",
                               data=json.dumps(export, indent=2),
                               file_name=f"{Path(uploaded.name).stem}_drishya_report.json",
                               mime="application/json")


# ══════════════════════════════════════════════════════════════
# PAGE: Compare Videos
# ══════════════════════════════════════════════════════════════
elif "Compare" in page:
    st.markdown('<div class="hero-title">Side-by-Side Comparison</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Upload two videos to compare their quality scores head-to-head.</div>', unsafe_allow_html=True)
    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    col_a, col_b = st.columns(2, gap="large")
    with col_a:
        st.markdown("**Video A**")
        up_a = st.file_uploader("Video A", type=["mp4","mov","avi","mkv","webm"], key="va", label_visibility="collapsed")
    with col_b:
        st.markdown("**Video B**")
        up_b = st.file_uploader("Video B", type=["mp4","mov","avi","mkv","webm"], key="vb", label_visibility="collapsed")

    if up_a and up_b:
        def save_tmp(up):
            suffix = Path(up.name).suffix
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
            tmp.write(up.read()); tmp.close()
            return tmp.name

        with st.spinner("Analyzing both videos…"):
            p_a, p_b = save_tmp(up_a), save_tmp(up_b)
            res_a = analyse_video_full(p_a, scorer, svr, backbone, transform, cfg)
            res_b = analyse_video_full(p_b, scorer, svr, backbone, transform, cfg)
            os.unlink(p_a); os.unlink(p_b)

        mos_a = res_a.get('mos'); mos_b = res_b.get('mos')
        cl_a  = res_a.get('classical') or {}; cl_b = res_b.get('classical') or {}
        avg_a = cl_a.get('avg_quality_score', 0); avg_b = cl_b.get('avg_quality_score', 0)

        # Winner banner
        if mos_a and mos_b:
            winner = "A" if mos_a >= mos_b else "B"
            st.success(f"🏆 Video **{winner}** is higher quality (MOS: A={mos_a:.2f} vs B={mos_b:.2f})")

        col_a, col_b = st.columns(2, gap="large")
        with col_a:
            st.markdown(f"**{up_a.name}**")
            if mos_a: st.plotly_chart(make_gauge(mos_a, "MOS — A"), use_container_width=True, config={'displayModeBar': False})
            st.metric("Classical Score", f"{avg_a:.3f}")
            st.metric("Rating", cl_a.get('quality_rating', 'N/A'))
            if res_a.get('frames_np') is not None:
                st.image(res_a['frames_np'][0], caption="First frame", use_container_width=True)
        with col_b:
            st.markdown(f"**{up_b.name}**")
            if mos_b: st.plotly_chart(make_gauge(mos_b, "MOS — B"), use_container_width=True, config={'displayModeBar': False})
            st.metric("Classical Score", f"{avg_b:.3f}")
            st.metric("Rating", cl_b.get('quality_rating', 'N/A'))
            if res_b.get('frames_np') is not None:
                st.image(res_b['frames_np'][0], caption="First frame", use_container_width=True)

        # Radar overlay
        def extract_radar(mos, cl):
            fs = cl.get('frame_scores', [])
            if not fs: return {"Deep MOS": ((mos or 1) - 1) / 4}
            avg_blur  = np.mean([f.get('blur_score',0) for f in fs])
            avg_noise = np.mean([f.get('noise_score',0) for f in fs])
            avg_block = np.mean([f.get('block8_score',0) for f in fs])
            avg_comp  = np.mean([f.get('composite_quality_score',0) for f in fs])
            d = {
                "Sharpness":   float(np.clip(avg_blur/500,0,1)),
                "Low Noise":   float(np.clip(1-avg_noise/50,0,1)),
                "No Blocking": float(np.clip(1-avg_block/20,0,1)),
                "Overall":     float(np.clip(avg_comp,0,1)),
            }
            if mos: d["Deep MOS"] = float(np.clip((mos-1)/4,0,1))
            return d

        ra = extract_radar(mos_a, cl_a)
        rb = extract_radar(mos_b, cl_b)
        cats = list(ra.keys()); vals_a = list(ra.values()); vals_b = list(rb.values())

        fig = go.Figure()
        for vals, name, color in [(vals_a+vals_a[:1], up_a.name, "#6366f1"),
                                   (vals_b+vals_b[:1], up_b.name, "#06b6d4")]:
            fig.add_trace(go.Scatterpolar(
                r=vals, theta=cats+cats[:1], name=name, fill='toself',
                fillcolor=color.replace(')', ',0.15)').replace('rgb','rgba') if 'rgb' in color else color+'26',
                line=dict(color=color, width=2), marker=dict(size=5),
            ))
        fig.update_layout(
            polar=dict(bgcolor='#151824',
                radialaxis=dict(range=[0,1], gridcolor='#252840', tickfont=dict(color='#64748b', size=10)),
                angularaxis=dict(gridcolor='#252840', tickfont=dict(color='#94a3b8', size=12))),
            paper_bgcolor='rgba(0,0,0,0)', legend=dict(font=dict(color='#94a3b8')),
            margin=dict(l=40,r=40,t=40,b=40), showlegend=True,
        )
        st.markdown('<div class="section-header">Quality Radar Overlay</div>', unsafe_allow_html=True)
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})


# ══════════════════════════════════════════════════════════════
# PAGE: Batch Process
# ══════════════════════════════════════════════════════════════
elif "Batch" in page:
    st.markdown('<div class="hero-title">Batch Processing</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Analyze an entire folder of videos and export results as a CSV spreadsheet.</div>', unsafe_allow_html=True)
    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    folder = st.text_input("📁 Path to video folder", placeholder="e.g. C:/Videos or /workspace/datasets/konvid_150k_b")
    max_vids = st.slider("Maximum videos to process", 1, 200, 20)

    if folder and st.button("▶ Start Batch Analysis"):
        folder_path = Path(folder)
        if not folder_path.exists():
            st.error(f"Folder not found: {folder}")
        else:
            exts = {'.mp4', '.mov', '.avi', '.mkv', '.webm'}
            videos = [f for f in sorted(folder_path.iterdir()) if f.suffix.lower() in exts][:max_vids]

            if not videos:
                st.warning("No video files found in that folder.")
            else:
                st.info(f"Found {len(videos)} videos. Processing…")
                prog = st.progress(0)
                status_box = st.empty()
                rows = []

                for i, vid in enumerate(videos):
                    status_box.markdown(f"Processing `{vid.name}` ({i+1}/{len(videos)})…")
                    try:
                        res = analyse_video_full(vid, scorer, svr, backbone, transform, cfg)
                        mos = res.get('mos')
                        cl  = res.get('classical') or {}
                        tm  = cl.get('temporal_metrics') or {}
                        rows.append({
                            'filename':       vid.name,
                            'mos_predicted':  mos,
                            'avg_quality':    cl.get('avg_quality_score'),
                            'quality_rating': cl.get('quality_rating'),
                            'avg_ssim':       tm.get('avg_ssim'),
                            'freeze_ratio':   tm.get('freeze_ratio'),
                            'jitter_index':   tm.get('avg_jitter_index'),
                        })
                    except Exception as e:
                        rows.append({'filename': vid.name, 'error': str(e)})
                    prog.progress((i + 1) / len(videos))

                status_box.empty()
                prog.progress(1.0, text="✅ Complete!")

                df = pd.DataFrame(rows)
                st.dataframe(df, use_container_width=True)

                csv = df.to_csv(index=False)
                st.download_button("⬇️ Download Results CSV", data=csv,
                                   file_name="drishya_batch_results.csv", mime="text/csv")

                if 'mos_predicted' in df.columns:
                    valid = df.dropna(subset=['mos_predicted'])
                    if len(valid):
                        fig = px.histogram(valid, x='mos_predicted', nbins=20,
                                           title='MOS Score Distribution',
                                           color_discrete_sequence=['#6366f1'])
                        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)',
                                          plot_bgcolor='#151824',
                                          font=dict(color='#94a3b8'))
                        st.plotly_chart(fig, use_container_width=True)

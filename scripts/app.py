"""
app.py  —  Streamlit Bird Species Classifier + Migration Predictor
Run:  streamlit run app.py
"""

import io
import json
import datetime

import streamlit as st
import torch
import torchaudio
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from models         import BirdClassifier
from migration_data import (
    get_phenology,
    seasonal_confidence_adjustment,
    migration_calendar_text,
    MONTH_NAMES,
    SPECIES_PHENOLOGY,
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Bird Species Classifier",
    page_icon="🐦",
    layout="wide",
)

# ── Constants ─────────────────────────────────────────────────────────────────
DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TARGET_SR      = 16000
CONF_THRESHOLD = 0.55
TOP_K_CHUNKS   = 5
MODEL_PATH     = "best_model.pth"
LABEL_PATH     = "label_to_index.json"


# ── Load resources (cached) ───────────────────────────────────────────────────
@st.cache_resource
def load_resources():
    with open(LABEL_PATH) as f:
        l2i = json.load(f)
    i2l = {v: k for k, v in l2i.items()}

    sample_size = 2048   # WAV2VEC2_LARGE → 2048-dim embedding
    model = BirdClassifier(input_dim=sample_size, num_classes=len(l2i)).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
    model.eval()

    bundle  = torchaudio.pipelines.WAV2VEC2_LARGE
    wav2vec = bundle.get_model().to(DEVICE)
    wav2vec.eval()

    return model, wav2vec, l2i, i2l


# ── Inference ─────────────────────────────────────────────────────────────────
def predict(audio_bytes: bytes, model, wav2vec, i2l):
    buf = io.BytesIO(audio_bytes)
    waveform, sr = torchaudio.load(buf)

    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != TARGET_SR:
        waveform = torchaudio.transforms.Resample(sr, TARGET_SR)(waveform)

    # Overlapping chunks
    chunk_size = TARGET_SR * 3
    stride     = TARGET_SR * 1
    chunks = []
    for i in range(0, waveform.shape[1] - chunk_size + 1, stride):
        chunk = waveform[:, i : i + chunk_size]
        rms = chunk.pow(2).mean().sqrt()
        if rms > 0.005:
            chunks.append(chunk)

    if not chunks:
        chunks = [waveform[:, :chunk_size]]

    embeddings = []
    with torch.no_grad():
        for chunk in chunks:
            chunk = chunk.to(DEVICE)
            features, _ = wav2vec.extract_features(chunk)
            x = features[-1]
            emb = torch.cat([x.mean(dim=1), x.max(dim=1).values], dim=1)
            emb = F.normalize(emb, dim=1)
            embeddings.append(emb)

    embeddings = torch.cat(embeddings, dim=0)
    # Aggregate: attention-weighted mean based on prediction entropy
    with torch.no_grad():
        logits_per_chunk = model(embeddings)
        probs_per_chunk  = F.softmax(logits_per_chunk, dim=1)
        # Low entropy = high confidence = upweight that chunk
        entropy    = -(probs_per_chunk * probs_per_chunk.log().clamp(-10)).sum(1)
        weights    = F.softmax(-entropy, dim=0).unsqueeze(1)
        final_emb  = (embeddings * weights).sum(0, keepdim=True)

    out   = model(final_emb)
    probs = F.softmax(out, dim=1).squeeze()
    return probs, waveform.squeeze().numpy()


# ── Waveform plot ─────────────────────────────────────────────────────────────
def waveform_figure(wave: np.ndarray) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 1.8))
    t = np.linspace(0, len(wave) / TARGET_SR, len(wave))
    ax.plot(t, wave, linewidth=0.3, color="#1f77b4", alpha=0.7)
    ax.set_xlabel("Time (s)", fontsize=8)
    ax.set_ylabel("Amplitude", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.set_xlim(0, t[-1])
    fig.tight_layout()
    return fig


# ── Migration calendar plot ───────────────────────────────────────────────────
def migration_calendar_figure(species_label: str, current_month: int) -> plt.Figure | None:
    pheno = get_phenology(species_label)
    if pheno is None:
        return None

    months   = list(range(1, 13))
    presence = [pheno["presence"].get(m, 0.0) for m in months]
    colors   = ["#1a9850" if p >= 0.7 else "#fee08b" if p >= 0.3 else "#d73027" if p > 0 else "#e0e0e0" for p in presence]
    colors[current_month - 1] = "#2166ac"   # highlight current month

    fig, ax = plt.subplots(figsize=(8, 2.2))
    bars = ax.bar(MONTH_NAMES[1:], presence, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Relative presence", fontsize=8)
    ax.set_title(f"{pheno['common_name']} — monthly occurrence", fontsize=9)
    ax.tick_params(labelsize=7)
    ax.axvline(x=current_month - 1, color="#2166ac", linewidth=1.5, linestyle="--", alpha=0.5)

    # Legend
    from matplotlib.patches import Patch
    legend = [
        Patch(color="#1a9850", label="Common (≥70%)"),
        Patch(color="#fee08b", label="Present (30-70%)"),
        Patch(color="#d73027", label="Rare (<30%)"),
        Patch(color="#2166ac", label="Current month"),
    ]
    ax.legend(handles=legend, fontsize=7, loc="upper right", framealpha=0.8)
    fig.tight_layout()
    return fig


# ── UI ────────────────────────────────────────────────────────────────────────
st.title("🐦 Bird Species Classifier + Migration Predictor")
st.caption("Upload a bird recording to identify the species and see its migration calendar.")

try:
    model, wav2vec, label_to_index, index_to_label = load_resources()
except FileNotFoundError as e:
    st.error(f"Model or label file not found: {e}\nPlease train the model first.")
    st.stop()

# Sidebar
with st.sidebar:
    st.header("Settings")
    conf_threshold = st.slider("Confidence threshold", 0.30, 0.90, CONF_THRESHOLD, 0.05)
    current_month  = st.selectbox(
        "Current month (for migration context)",
        options=list(range(1, 13)),
        index=datetime.date.today().month - 1,
        format_func=lambda m: MONTH_NAMES[m],
    )
    st.markdown("---")
    st.caption(f"Model: {len(label_to_index)} species  |  Device: {DEVICE}")

# Main
col1, col2 = st.columns([1, 1])

with col1:
    uploaded = st.file_uploader("Upload audio (WAV or MP3)", type=["wav", "mp3"])

if uploaded is not None:
    audio_bytes = uploaded.read()
    st.audio(audio_bytes)

    with st.spinner("Extracting features and classifying…"):
        probs, wave = predict(audio_bytes, model, wav2vec, index_to_label)

    # Waveform
    st.subheader("Audio waveform")
    st.pyplot(waveform_figure(wave))

    confidence, pred_idx = probs.max(0)
    pred_label = index_to_label[pred_idx.item()]

    # Seasonal adjustment
    adj_conf, season_note = seasonal_confidence_adjustment(
        pred_label, current_month, confidence.item()
    )

    # ── Prediction result ─────────────────────────────────────────────────────
    st.subheader("Prediction")
    if adj_conf < conf_threshold:
        st.warning(
            f"⚠️ Unknown / uncertain species\n\n"
            f"Best guess: **{pred_label}** ({confidence.item()*100:.1f}%)\n\n"
            f"Seasonal note: {season_note}"
        )
    else:
        st.success(f"**{pred_label}**")
        col_a, col_b = st.columns(2)
        col_a.metric("Raw confidence",      f"{confidence.item()*100:.1f}%")
        col_b.metric("Season-adjusted conf", f"{adj_conf*100:.1f}%")
        st.caption(f"📅 {season_note}")

    # ── Top-5 bar chart ───────────────────────────────────────────────────────
    st.subheader("Top-5 predictions")
    top5_vals, top5_idx = probs.topk(min(5, len(probs)))
    fig2, ax2 = plt.subplots(figsize=(7, 2.5))
    names  = [index_to_label[i.item()] for i in top5_idx]
    scores = [v.item() * 100 for v in top5_vals]
    colors2 = ["#1f77b4" if i == 0 else "#aec7e8" for i in range(len(names))]
    ax2.barh(names[::-1], scores[::-1], color=colors2[::-1], edgecolor="white")
    ax2.set_xlabel("Confidence (%)", fontsize=8)
    ax2.tick_params(labelsize=8)
    ax2.set_xlim(0, 100)
    fig2.tight_layout()
    st.pyplot(fig2)

    # ── Migration calendar ────────────────────────────────────────────────────
    st.subheader("Migration calendar")
    pheno = get_phenology(pred_label)
    if pheno:
        cal_fig = migration_calendar_figure(pred_label, current_month)
        if cal_fig:
            st.pyplot(cal_fig)

        with st.expander("Full migration details"):
            st.markdown(migration_calendar_text(pred_label))
    else:
        st.info(
            "No migration data found for this species in the database.\n\n"
            "Add it to `migration_data.py` → `SPECIES_PHENOLOGY`."
        )

    # ── Browse all species calendars ──────────────────────────────────────────
    st.markdown("---")
    with st.expander("Browse migration calendars for all species in database"):
        selected = st.selectbox(
            "Select species",
            options=sorted(SPECIES_PHENOLOGY.keys()),
            format_func=lambda k: SPECIES_PHENOLOGY[k]["common_name"],
        )
        fig_browse = migration_calendar_figure(selected, current_month)
        if fig_browse:
            st.pyplot(fig_browse)
        st.markdown(migration_calendar_text(selected))

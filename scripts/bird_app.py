"""
bird_app.py — Bird Species Classifier + Migration Predictor
Run: python -m streamlit run scripts/bird_app.py
"""

# ===== FINAL LIBROSA OVERRIDE =====
import sys
import types
import soundfile as sf
import resampy
import numpy as np

fake_librosa = types.ModuleType("librosa")

def load(path, sr=48000, mono=True, offset=0.0, duration=None, *args, **kwargs):
    data, orig_sr = sf.read(path, dtype="float32")

    if mono and data.ndim > 1:
        data = np.mean(data, axis=1)

    start = int(offset * orig_sr)

    if duration is not None:
        end = start + int(duration * orig_sr)
        data = data[start:end]
    else:
        data = data[start:]

    if orig_sr != sr:
        data = resampy.resample(data, orig_sr, sr)

    if np.max(np.abs(data)) > 0:
        data = data / np.max(np.abs(data))

    return data, sr

def get_duration(y=None, sr=48000):
    return len(y) / sr if y is not None else 0

fake_librosa.load = load
fake_librosa.get_duration = get_duration

fake_core = types.ModuleType("librosa.core")
fake_core.load = load
fake_librosa.core = fake_core

sys.modules["librosa"] = fake_librosa
sys.modules["librosa.core"] = fake_core
# ===== END FIX =====


# ===== IMPORTS =====
import os, tempfile, datetime
import torch
import torchaudio
import torchaudio.transforms as T

import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

st.set_page_config(page_title="Bird Classifier + Migration", page_icon="🐦", layout="wide")

TARGET_SR = 48000


# ===== MIGRATION DATA =====
MONTH_NAMES = ["","Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

def _resident():
    return {
        "presence": {m: 1.0 for m in range(1,13)},
        "migration_type": "resident",
        "notes": "Present year-round"
    }

SPECIES_PHENOLOGY = {

    "Andean Guan": _resident(),
    "White-bellied Nothura": _resident(),
    "Black Tinamou": _resident(),
    "Little Tinamou": _resident(),
    "Brushland Tinamou": _resident(),

    "White-throated Tinamou": {
        "presence": {1:0.6,2:0.7,3:0.8,4:1.0,5:1.0,6:1.0,7:0.9,8:0.8,9:0.7,10:0.6,11:0.6,12:0.6},
        "migration_type": "partial migrant",
        "notes": "Moves to lower elevations seasonally"
    },

    "Emu": {
        "presence": {1:0.7,2:0.6,3:0.5,4:0.8,5:1.0,6:1.0,7:1.0,8:0.9,9:0.8,10:0.7,11:0.6,12:0.7},
        "migration_type": "nomadic",
        "notes": "Movement depends on rainfall"
    },

    "Southern Cassowary": _resident(),
    "Great Spotted Kiwi": _resident(),

    "Spixs Guan": {
        "presence": {1:0.5,2:0.6,3:0.7,4:0.9,5:1.0,6:1.0,7:1.0,8:0.9,9:0.7,10:0.6,11:0.5,12:0.5},
        "migration_type": "local migrant",
        "notes": "Short distance forest movement"
    }
}


# ===== FIXED NAME MATCHING =====
def get_phenology(name):
    name = name.strip()

    # Handle names like "Andean Guan (Penelope montagnii)"
    if "(" in name:
        name = name.split("(")[0].strip()

    return SPECIES_PHENOLOGY.get(name, None)


def plot_migration(presence, current_month):
    values = [presence.get(m, 0) for m in range(1,13)]
    colors = ["#4caf50" if v > 0.5 else "#ccc" for v in values]
    colors[current_month-1] = "#2196f3"

    fig, ax = plt.subplots(figsize=(8,2))
    ax.bar(MONTH_NAMES[1:], values, color=colors)
    ax.set_ylim(0,1)
    ax.set_ylabel("Presence")
    ax.set_title("Migration Pattern")
    return fig


# ===== MODEL =====
@st.cache_resource
def load_analyzer():
    from birdnetlib.analyzer import Analyzer
    return Analyzer()

analyzer = load_analyzer()


# ===== AUDIO =====
def to_wav(audio_bytes, name):
    suffix = ".wav" if name.lower().endswith(".wav") else ".mp3"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
        f.write(audio_bytes)
        in_path = f.name

    waveform, sr = torchaudio.load(in_path)
    os.unlink(in_path)

    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    if sr != TARGET_SR:
        waveform = T.Resample(sr, TARGET_SR)(waveform)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
        out_path = f.name

    torchaudio.save(out_path, waveform, TARGET_SR)
    return out_path, waveform.squeeze(0).numpy(), TARGET_SR


# ===== UI =====
st.title("🐦 Bird Classifier + Migration Predictor")

current_month = st.selectbox(
    "Select current month",
    options=list(range(1,13)),
    index=datetime.date.today().month-1,
    format_func=lambda m: MONTH_NAMES[m]
)

uploaded = st.file_uploader("Upload bird audio", type=["wav","mp3"])

if uploaded:
    audio_bytes = uploaded.read()
    st.audio(audio_bytes)

    wav_path, wave_np, sr = to_wav(audio_bytes, uploaded.name)

    from birdnetlib import Recording
    recording = Recording(analyzer, wav_path)
    recording.analyze()
    detections = recording.detections

    os.unlink(wav_path)

    if not detections:
        st.warning("No birds detected.")
    else:
        detections = sorted(detections, key=lambda x: x["confidence"], reverse=True)

        st.subheader("Top Prediction")

        top = detections[0]
        name = top["common_name"].strip()   # ✅ FIX APPLIED
        conf = top["confidence"]

        st.success(f"{name} ({conf*100:.1f}%)")

        pheno = get_phenology(name)

        if pheno:
            presence = pheno["presence"].get(current_month, 0)
            adj_conf = min(conf * (0.7 + 0.3 * presence), 1.0)

            col1, col2 = st.columns(2)
            col1.metric("Model confidence", f"{conf*100:.1f}%")
            col2.metric("Season-adjusted", f"{adj_conf*100:.1f}%")

            st.info(f"Migration type: {pheno['migration_type']}")
            st.caption(pheno["notes"])

            st.pyplot(plot_migration(pheno["presence"], current_month))

        else:
            st.warning("No migration data available for this species.")

        st.subheader("All Predictions")
        for d in detections[:10]:
            st.write(f"{d['common_name']} - {d['confidence']*100:.1f}%")
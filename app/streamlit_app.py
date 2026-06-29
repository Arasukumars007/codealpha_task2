"""
streamlit_app.py
================
Interactive web demo for Emotion Recognition from Speech.

Features
--------
• Upload audio file (.wav / .mp3 / .ogg)
• Live microphone recording (via streamlit-audiorec)
• Waveform & Mel-Spectrogram visualisation
• Emotion prediction with animated confidence bars
• Model selector (CNN / LSTM / CNN-LSTM)
• Prediction history table

Run
---
  streamlit run app/streamlit_app.py
"""

import os
import sys
import io
import tempfile
import logging
from pathlib import Path

# Fix for Windows asyncio ProactorEventLoop AssertionError on shutdown
if sys.platform == 'win32':
    import asyncio
    try:
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    except Exception:
        pass

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import librosa
import librosa.display
import soundfile as sf

# ── Path setup ───────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data_loader import load_config
from src.feature_extractor import load_audio, extract_mel_spectrogram_only
from src.predict import predict_emotion, load_model, list_available_models

logging.basicConfig(level=logging.WARNING)

# ─────────────────────────────────────────────────────────────
# Page config — must be FIRST Streamlit call
# ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="SpeechSense — Emotion Recognition",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────────────────────

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');

  html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
  }

  /* Background */
  .stApp {
    background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    color: #e0e0e0;
  }

  /* Sidebar */
  [data-testid="stSidebar"] {
    background: rgba(255,255,255,0.05);
    backdrop-filter: blur(12px);
    border-right: 1px solid rgba(255,255,255,0.1);
  }

  /* Cards */
  .metric-card {
    background: rgba(255,255,255,0.08);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 16px;
    padding: 24px;
    text-align: center;
    backdrop-filter: blur(8px);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
  }
  .metric-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 12px 32px rgba(0,0,0,0.4);
  }
  .metric-card h1 {
    font-size: 3rem;
    margin: 0;
  }
  .metric-card h3 {
    margin: 4px 0 0 0;
    color: #a0a0c0;
    font-weight: 400;
    font-size: 0.95rem;
    letter-spacing: 0.05em;
    text-transform: uppercase;
  }

  /* Emotion result banner */
  .emotion-banner {
    background: linear-gradient(90deg, #6c63ff 0%, #48cae4 100%);
    border-radius: 16px;
    padding: 28px 36px;
    text-align: center;
    margin: 20px 0;
    box-shadow: 0 8px 32px rgba(108,99,255,0.4);
    animation: slideIn 0.5s ease;
  }
  .emotion-banner h2 {
    font-size: 2.4rem;
    font-weight: 800;
    margin: 0;
    text-shadow: 0 2px 8px rgba(0,0,0,0.3);
  }
  .emotion-banner p {
    font-size: 1.1rem;
    margin: 8px 0 0 0;
    opacity: 0.85;
  }

  /* Progress bar container */
  .prob-row {
    display: flex;
    align-items: center;
    gap: 12px;
    margin: 6px 0;
    font-size: 0.95rem;
  }
  .prob-label {
    width: 100px;
    text-align: right;
    text-transform: capitalize;
    font-weight: 500;
  }
  .prob-bar-bg {
    flex: 1;
    background: rgba(255,255,255,0.1);
    border-radius: 8px;
    height: 18px;
    overflow: hidden;
  }
  .prob-bar-fill {
    height: 100%;
    border-radius: 8px;
    transition: width 0.8s cubic-bezier(.4,0,.2,1);
  }
  .prob-pct {
    width: 52px;
    text-align: right;
    font-weight: 600;
  }

  /* Section headers */
  .section-header {
    font-size: 1.3rem;
    font-weight: 700;
    margin: 28px 0 12px 0;
    padding-left: 12px;
    border-left: 4px solid #6c63ff;
    color: #e8e8ff;
  }

  /* Keyframe animation */
  @keyframes slideIn {
    from { opacity: 0; transform: translateY(20px); }
    to   { opacity: 1; transform: translateY(0); }
  }

  /* Upload area */
  [data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.05);
    border: 2px dashed rgba(108,99,255,0.5);
    border-radius: 14px;
    padding: 16px;
  }

  /* Selectbox */
  .stSelectbox label { color: #c0c0e0 !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────

EMOTION_EMOJI = {
    "neutral":   "😐", "calm":      "😌", "happy":     "😄",
    "sad":       "😢", "angry":     "😠", "fearful":   "😨",
    "disgust":   "🤢", "surprised": "😲", "fear":      "😨",
    "boredom":   "😑", "joy":       "😄",
}

EMOTION_COLORS = {
    "neutral":   "#90cdf4", "calm":      "#68d391", "happy":     "#f6e05e",
    "sad":       "#76e4f7", "angry":     "#fc8181", "fearful":   "#b794f4",
    "disgust":   "#68d391", "surprised": "#fbd38d", "fear":      "#b794f4",
    "boredom":   "#a0aec0", "joy":       "#f6e05e",
}

# ─────────────────────────────────────────────────────────────
# Cached loaders
# ─────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading configuration …")
def get_config():
    config_path = ROOT / "config.yaml"
    return load_config(str(config_path))


@st.cache_resource(show_spinner="Loading model …")
def get_model(arch: str, config_key: str):
    cfg = get_config()
    return load_model(arch, cfg)


@st.cache_data(show_spinner=False)
def get_label_list(dataset_name: str, cfg: dict):
    if dataset_name == "ravdess":
        em = cfg["ravdess_emotions"]
        return [em[k] for k in sorted(em.keys())]
    elif dataset_name == "tess":
        em = cfg["tess_emotions"]
        return sorted(set(em.values()))
    elif dataset_name == "emodb":
        em = cfg["emodb_emotions"]
        return sorted(set(em.values()))
    else:
        return [
            "neutral","calm","happy","sad",
            "angry","fearful","disgust","surprised"
        ]


# ─────────────────────────────────────────────────────────────
# Visualisation helpers
# ─────────────────────────────────────────────────────────────

def plot_waveform(y: np.ndarray, sr: int) -> plt.Figure:
    """Return a matplotlib figure of the audio waveform."""
    fig, ax = plt.subplots(figsize=(10, 2.5))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#1a1a2e")

    t = np.linspace(0, len(y) / sr, len(y))
    ax.fill_between(t, y, alpha=0.6, color="#6c63ff")
    ax.plot(t, y, color="#9d8fff", linewidth=0.8, alpha=0.9)

    ax.set_xlim(0, len(y) / sr)
    ax.set_xlabel("Time (s)", color="#c0c0e0")
    ax.set_ylabel("Amplitude", color="#c0c0e0")
    ax.tick_params(colors="#c0c0e0")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333355")

    plt.tight_layout(pad=0.5)
    return fig


def plot_mel_spectrogram(y: np.ndarray, sr: int, hop_length: int = 512) -> plt.Figure:
    """Return a Mel-Spectrogram figure."""
    mel = librosa.feature.melspectrogram(y=y, sr=sr, hop_length=hop_length, n_mels=128)
    mel_db = librosa.power_to_db(mel, ref=np.max)

    fig, ax = plt.subplots(figsize=(10, 3.5))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#1a1a2e")

    img = librosa.display.specshow(
        mel_db, sr=sr, hop_length=hop_length,
        x_axis="time", y_axis="mel",
        cmap="magma", ax=ax,
    )
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    ax.set_title("Mel-Spectrogram", color="#c0c0e0")
    ax.tick_params(colors="#c0c0e0")
    ax.set_xlabel("Time (s)", color="#c0c0e0")
    ax.set_ylabel("Frequency (Hz)", color="#c0c0e0")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333355")

    plt.tight_layout(pad=0.5)
    return fig


def render_probability_bars(prob_dict: dict):
    """Render custom styled probability bars via HTML."""
    sorted_items = sorted(prob_dict.items(), key=lambda x: x[1], reverse=True)
    html_parts = ['<div style="margin-top:12px;">']

    for emotion, prob in sorted_items:
        pct     = prob * 100
        color   = EMOTION_COLORS.get(emotion, "#6c63ff")
        emoji   = EMOTION_EMOJI.get(emotion, "🎵")
        opacity = max(0.4, prob * 1.5)

        html_parts.append(f"""
        <div class="prob-row">
          <span class="prob-label">{emoji} {emotion}</span>
          <div class="prob-bar-bg">
            <div class="prob-bar-fill"
                 style="width:{pct:.1f}%;
                        background:linear-gradient(90deg, {color}cc, {color});
                        opacity:{opacity};">
            </div>
          </div>
          <span class="prob-pct">{pct:.1f}%</span>
        </div>
        """)

    html_parts.append("</div>")
    st.markdown("".join(html_parts), unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────

def render_sidebar(cfg: dict) -> tuple:
    """Render sidebar controls. Returns (arch, dataset, show_spec, show_wave)."""
    with st.sidebar:
        st.markdown("""
        <div style='text-align:center; padding:20px 0;'>
          <div style='font-size:3rem;'>🎙️</div>
          <h2 style='margin:8px 0 4px; color:#e8e8ff;'>SpeechSense</h2>
          <p style='color:#8080b0; font-size:0.85rem; margin:0;'>
            Emotion Recognition from Speech
          </p>
        </div>
        """, unsafe_allow_html=True)

        st.divider()

        st.markdown("### ⚙️ Model Settings")
        # Put trained models first so the default is always a working model
        available = list_available_models(cfg["training"]["checkpoint_dir"])
        all_archs  = ["cnn_lstm", "transformer", "cnn", "lstm"]
        ordered    = [a for a in all_archs if a in available] + \
                     [a for a in all_archs if a not in available]
        default_idx = ordered.index("lstm") if "lstm" in ordered else 0
        arch = st.selectbox(
            "Architecture",
            ordered,
            index=default_idx,
            help="Only models with ✅ in Model Status below can run inference.",
        )

        dataset = st.selectbox(
            "Dataset Label Set",
            ["ravdess", "tess", "emodb"],
            index=0,
            help="Choose which dataset's emotion labels to use.",
        )

        st.divider()
        st.markdown("### 🎨 Visualisations")
        show_wave = st.toggle("Show Waveform",      value=True)
        show_spec = st.toggle("Show Mel-Spectrogram", value=True)

        st.divider()
        st.markdown("### ℹ️ Model Status")
        for a in all_archs:
            status = "✅ Trained" if a in available else "❌ Not trained"
            st.markdown(f"**{a.upper()}** — {status}")

        st.divider()
        st.markdown(
            "<p style='color:#606080; font-size:0.75rem; text-align:center;'>"
            "Built with TensorFlow · librosa · Streamlit"
            "</p>",
            unsafe_allow_html=True,
        )

    return arch, dataset, show_spec, show_wave


# ─────────────────────────────────────────────────────────────
# Main App
# ─────────────────────────────────────────────────────────────

def main():
    cfg = get_config()

    # Sidebar
    arch, dataset, show_spec, show_wave = render_sidebar(cfg)

    # ── Header ───────────────────────────────────────────────
    st.markdown("""
    <div style='text-align:center; padding:32px 0 16px;'>
      <h1 style='font-size:2.8rem; font-weight:800; margin:0;
                 background:linear-gradient(90deg,#6c63ff,#48cae4);
                 -webkit-background-clip:text; -webkit-text-fill-color:transparent;'>
        🎙️ SpeechSense
      </h1>
      <p style='color:#8080b0; font-size:1.1rem; margin:8px 0 0;'>
        Detect human emotions from speech using deep learning
      </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Model availability warning ────────────────────────────
    available_models = list_available_models(cfg["training"]["checkpoint_dir"])
    if arch not in available_models:
        st.warning(
            f"⚠️ **{arch.upper()} model is not trained yet.**  "
            f"Please select a trained model from the sidebar.  \n"
            f"✅ **Currently trained:** {', '.join(available_models) if available_models else 'None — run: python run.py train'}",
            icon="⚠️"
        )

    # ── Stats row ────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)
    stats = [
        ("8",     "Emotions"),
        ("40",    "MFCCs"),
        ("3",     "Models"),
        ("CNN+LSTM", "Architecture"),
    ]
    for col, (val, label) in zip([col1, col2, col3, col4], stats):
        with col:
            st.markdown(f"""
            <div class="metric-card">
              <h1>{val}</h1>
              <h3>{label}</h3>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Input tabs ───────────────────────────────────────────
    tab_file, tab_mic, tab_history = st.tabs([
        "📁 Upload Audio", "🎤 Record Microphone", "📊 History"
    ])

    # ── Initialise session state ──────────────────────────────
    if "history" not in st.session_state:
        st.session_state.history = []

    # ═══════════════════════════════════════════════════════════
    # TAB 1: File Upload
    # ═══════════════════════════════════════════════════════════
    with tab_file:
        st.markdown('<div class="section-header">Upload an audio file</div>',
                    unsafe_allow_html=True)

        uploaded = st.file_uploader(
            "Drag and drop or browse",
            type=["wav", "mp3", "ogg", "flac", "m4a"],
            help="Supports WAV, MP3, OGG, FLAC, M4A",
        )

        if uploaded:
            # Save to temp file
            suffix = Path(uploaded.name).suffix
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                tmp.write(uploaded.getbuffer())
                tmp_path = tmp.name

            # Audio player
            st.markdown('<div class="section-header">▶ Playback</div>',
                        unsafe_allow_html=True)
            st.audio(uploaded, format=f"audio/{suffix.lstrip('.')}")

            # Load waveform
            y_wave = load_audio(
                tmp_path,
                sr=cfg["audio"]["sample_rate"],
                duration=cfg["audio"]["duration"],
            )
            sr = cfg["audio"]["sample_rate"]

            # Visualisations
            if show_wave:
                st.markdown('<div class="section-header">〰 Waveform</div>',
                            unsafe_allow_html=True)
                fig_wave = plot_waveform(y_wave, sr)
                st.pyplot(fig_wave, use_container_width=True)
                plt.close(fig_wave)

            if show_spec:
                st.markdown('<div class="section-header">🌈 Mel-Spectrogram</div>',
                            unsafe_allow_html=True)
                fig_mel = plot_mel_spectrogram(y_wave, sr)
                st.pyplot(fig_mel, use_container_width=True)
                plt.close(fig_mel)

            # Prediction
            st.markdown('<div class="section-header">🧠 Prediction</div>',
                        unsafe_allow_html=True)

            available = list_available_models(cfg["training"]["checkpoint_dir"])
            if arch not in available:
                st.warning(
                    f"⚠️ No trained **{arch.upper()}** model found. "
                    f"Train first: `python src/train.py --model {arch}`"
                )
            else:
                with st.spinner("Analysing emotion …"):
                    label_list = get_label_list(dataset, cfg)
                    model      = get_model(arch, arch)
                    result     = predict_emotion(
                        tmp_path, cfg,
                        arch=arch, model=model, label_list=label_list,
                    )

                emotion = result["predicted_emotion"]
                conf    = result["confidence"]
                emoji   = EMOTION_EMOJI.get(emotion, "🎵")

                # Result banner
                st.markdown(f"""
                <div class="emotion-banner">
                  <h2>{emoji} {emotion.upper()}</h2>
                  <p>Confidence: {conf*100:.1f}% &nbsp;|&nbsp;
                     Model: {arch.upper()} &nbsp;|&nbsp;
                     File: {uploaded.name}</p>
                </div>
                """, unsafe_allow_html=True)

                # Probability bars
                st.markdown("**Emotion probabilities:**")
                render_probability_bars(result["probabilities"])

                # Add to history
                st.session_state.history.append({
                    "Source":    uploaded.name,
                    "Model":     arch.upper(),
                    "Emotion":   f"{emoji} {emotion}",
                    "Confidence": f"{conf*100:.1f}%",
                })

            # Cleanup
            try:
                os.unlink(tmp_path)
            except Exception:
                pass

    # ═══════════════════════════════════════════════════════════
    # TAB 2: Microphone
    # ═══════════════════════════════════════════════════════════
    with tab_mic:
        st.markdown('<div class="section-header">🎤 Record from Microphone</div>',
                    unsafe_allow_html=True)

        st.info(
            "**Instructions:** Click **Start Recording**, speak clearly for "
            "a few seconds expressing an emotion, then click **Stop**."
        )

        # Try to use streamlit-audiorec
        try:
            from streamlit_audiorec import st_audiorec
            wav_audio_data = st_audiorec()

            if wav_audio_data is not None:
                st.audio(wav_audio_data, format="audio/wav")

                # Load waveform from bytes
                audio_bytes = io.BytesIO(wav_audio_data)
                y_wave, sr_loaded = sf.read(audio_bytes)
                if y_wave.ndim > 1:
                    y_wave = y_wave.mean(axis=1)
                y_wave = y_wave.astype(np.float32)

                # Resample if needed
                sr = cfg["audio"]["sample_rate"]
                if sr_loaded != sr:
                    import librosa
                    y_wave = librosa.resample(y_wave, orig_sr=sr_loaded, target_sr=sr)

                if show_wave:
                    st.markdown('<div class="section-header">〰 Waveform</div>',
                                unsafe_allow_html=True)
                    fig = plot_waveform(y_wave, sr)
                    st.pyplot(fig, use_container_width=True)
                    plt.close(fig)

                if show_spec:
                    st.markdown('<div class="section-header">🌈 Mel-Spectrogram</div>',
                                unsafe_allow_html=True)
                    fig = plot_mel_spectrogram(y_wave, sr)
                    st.pyplot(fig, use_container_width=True)
                    plt.close(fig)

                available = list_available_models(cfg["training"]["checkpoint_dir"])
                if arch not in available:
                    st.warning(f"⚠️ No trained {arch.upper()} model found.")
                else:
                    with st.spinner("Analysing …"):
                        label_list = get_label_list(dataset, cfg)
                        model      = get_model(arch, arch)
                        result     = predict_emotion(
                            y_wave, cfg,
                            arch=arch, model=model, label_list=label_list,
                        )

                    emotion = result["predicted_emotion"]
                    emoji   = EMOTION_EMOJI.get(emotion, "🎵")
                    conf    = result["confidence"]

                    st.markdown(f"""
                    <div class="emotion-banner">
                      <h2>{emoji} {emotion.upper()}</h2>
                      <p>Confidence: {conf*100:.1f}% &nbsp;|&nbsp; Model: {arch.upper()}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    render_probability_bars(result["probabilities"])

                    st.session_state.history.append({
                        "Source":     "🎤 Microphone",
                        "Model":      arch.upper(),
                        "Emotion":    f"{emoji} {emotion}",
                        "Confidence": f"{conf*100:.1f}%",
                    })

        except ImportError:
            st.warning(
                "**streamlit-audiorec** is not installed.\n\n"
                "Install it with:\n```\npip install streamlit-audiorec\n```\n\n"
                "Then restart the Streamlit app."
            )
            st.markdown(
                "Alternatively, use **Upload Audio** tab to upload a pre-recorded file."
            )

    # ═══════════════════════════════════════════════════════════
    # TAB 3: History
    # ═══════════════════════════════════════════════════════════
    with tab_history:
        st.markdown('<div class="section-header">📊 Prediction History</div>',
                    unsafe_allow_html=True)

        if st.session_state.history:
            import pandas as pd
            df_hist = pd.DataFrame(st.session_state.history[::-1])  # newest first
            st.dataframe(
                df_hist,
                use_container_width=True,
                hide_index=True,
            )
            if st.button("🗑️ Clear History"):
                st.session_state.history = []
                st.rerun()
        else:
            st.markdown(
                "<p style='color:#6060a0; text-align:center; margin-top:48px;'>"
                "No predictions yet. Upload a file or record from microphone."
                "</p>",
                unsafe_allow_html=True,
            )

    # ── Footer ───────────────────────────────────────────────
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("""
    <div style='text-align:center; color:#404068; font-size:0.8rem; padding:20px;
                border-top:1px solid rgba(255,255,255,0.06);'>
      SpeechSense &nbsp;|&nbsp; Emotion Recognition from Speech &nbsp;|&nbsp;
      TensorFlow · librosa · Streamlit
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()

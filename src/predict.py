"""
predict.py
==========
Inference module for emotion recognition — file or microphone input.

Usage
-----
  # Predict from a WAV file
  python src/predict.py --file path/to/audio.wav

  # Predict from a WAV file using a specific model
  python src/predict.py --file path/to/audio.wav --model cnn

  # Record from microphone (requires PyAudio)
  python src/predict.py --mic --duration 4

  # List available models
  python src/predict.py --list-models
"""

import os
import sys

# Configure UTF-8 streams on Windows to prevent Unicode encoding crashes on older terminals
if sys.platform == 'win32':
    if hasattr(sys.stdout, 'reconfigure'):
        try:
            sys.stdout.reconfigure(encoding='utf-8')
        except Exception:
            pass
    if hasattr(sys.stderr, 'reconfigure'):
        try:
            sys.stderr.reconfigure(encoding='utf-8')
        except Exception:
            pass

import argparse
import logging
import tempfile
import numpy as np
import yaml
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_loader import load_config
from src.feature_extractor import load_audio, extract_features, extract_mel_spectrogram_only
from src.train import prepare_inputs
# Import transformer_model to register the LabelSmoothedCCE loss class
# in Keras's serialization registry (needed for load_model() to work).
from src.models import transformer_model as _tm  # noqa: F401

import tensorflow as tf

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("predict")


# ─────────────────────────────────────────────────────────────
# Model discovery
# ─────────────────────────────────────────────────────────────

def list_available_models(checkpoint_dir: str = "models") -> dict:
    """Scan checkpoint directory and return {arch: path} for available models."""
    available = {}
    if not os.path.exists(checkpoint_dir):
        return available

    for arch in ["cnn", "lstm", "cnn_lstm", "transformer"]:
        for suffix in ["_best.keras", "_final.keras"]:
            path = os.path.join(checkpoint_dir, f"{arch}{suffix}")
            if os.path.exists(path):
                available[arch] = path
                break  # prefer _best over _final

    return available


def load_model(arch: str, config: dict) -> tf.keras.Model:
    """Load the trained Keras model for the given architecture."""
    models_dir = config["training"]["checkpoint_dir"]
    available  = list_available_models(models_dir)

    if arch not in available:
        raise FileNotFoundError(
            f"No trained model found for architecture '{arch}'.\n"
            f"Available: {list(available.keys()) or 'none'}\n"
            f"Train first: python src/train.py --model {arch}"
        )

    path = available[arch]
    logger.info(f"Loading model: {path}")
    return tf.keras.models.load_model(path)


# ─────────────────────────────────────────────────────────────
# Feature extraction for inference
# ─────────────────────────────────────────────────────────────

def extract_inference_features(
    y_wave: np.ndarray,
    arch: str,
    config: dict,
) -> np.ndarray:
    """
    Extract features from a waveform and prepare for a specific architecture.

    Returns
    -------
    np.ndarray with a batch dimension: (1, *input_shape)
    """
    audio_cfg = config["audio"]
    feat_cfg  = config["features"]

    # Full feature map
    feat = extract_features(
        y_wave,
        sr=audio_cfg["sample_rate"],
        n_mfcc=feat_cfg["n_mfcc"],
        n_mels=feat_cfg["n_mels"],
        n_chroma=feat_cfg["n_chroma"],
        n_fft=audio_cfg["n_fft"],
        hop_length=audio_cfg["hop_length"],
        use_delta=feat_cfg["use_delta"],
        use_chroma=feat_cfg["use_chroma"],
        use_mel=feat_cfg["use_mel"],
        use_zcr=feat_cfg["use_zcr"],
        use_rmse=feat_cfg["use_rmse"],
        time_frames=feat_cfg["time_frames"],
    )  # shape: (n_features, time_frames)

    # Expand to batch of 1 → (1, n_features, time_frames)
    X = feat[np.newaxis, ...]

    # Reshape for architecture
    X_prep = prepare_inputs(X, arch, config)
    return X_prep


# ─────────────────────────────────────────────────────────────
# Microphone recording
# ─────────────────────────────────────────────────────────────

def record_from_microphone(
    duration: float = 4.0,
    sr: int = 22050,
) -> np.ndarray:
    """
    Record audio from the default microphone.

    Requires PyAudio to be installed.
    Returns a float32 mono waveform.
    """
    try:
        import pyaudio
        import wave
        import struct
    except ImportError:
        raise ImportError(
            "PyAudio is required for microphone recording.\n"
            "Install with: pip install pyaudio"
        )

    CHUNK       = 1024
    FORMAT      = pyaudio.paInt16
    CHANNELS    = 1
    RATE        = sr
    RECORD_SECS = duration

    pa = pyaudio.PyAudio()
    stream = pa.open(
        format=FORMAT, channels=CHANNELS,
        rate=RATE, input=True,
        frames_per_buffer=CHUNK
    )

    logger.info(f"🎤 Recording for {RECORD_SECS:.1f} seconds …")
    frames = []
    n_chunks = int(RATE / CHUNK * RECORD_SECS)
    for _ in range(n_chunks):
        data = stream.read(CHUNK)
        frames.append(data)

    logger.info("✅ Recording complete.")
    stream.stop_stream()
    stream.close()
    pa.terminate()

    # Convert int16 bytes → float32 numpy array
    raw = b"".join(frames)
    samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    return samples


# ─────────────────────────────────────────────────────────────
# Core prediction function
# ─────────────────────────────────────────────────────────────

def predict_emotion(
    audio_input,            # str (file path) or np.ndarray (waveform)
    config: dict,
    arch: str = None,
    model=None,
    label_list: list = None,
    top_k: int = 3,
) -> dict:
    """
    Predict emotion from a file path or waveform array.

    Parameters
    ----------
    audio_input : str | np.ndarray
        Path to audio file OR pre-loaded waveform array.
    config      : loaded config dict
    arch        : model architecture; defaults to config["model"]["architecture"]
    model       : pre-loaded Keras model (skips disk load if provided)
    label_list  : list of emotion names in label-index order
    top_k       : number of top predictions to return

    Returns
    -------
    dict with keys:
        predicted_emotion  : str  — top prediction
        confidence         : float (0-1)
        top_k              : list of (emotion, confidence) tuples
        probabilities      : dict {emotion: prob}
    """
    if arch is None:
        arch = config["model"]["architecture"]

    # ── Load model ───────────────────────────────────────────
    if model is None:
        model = load_model(arch, config)

    # ── Build label list ─────────────────────────────────────
    if label_list is None:
        # Infer from RAVDESS config as default
        emotion_map = config.get("ravdess_emotions", {})
        label_list  = [emotion_map[k] for k in sorted(emotion_map.keys())]

    # ── Load audio ───────────────────────────────────────────
    audio_cfg = config["audio"]
    if isinstance(audio_input, str):
        y_wave = load_audio(
            audio_input,
            sr=audio_cfg["sample_rate"],
            duration=audio_cfg["duration"],
        )
    else:
        y_wave = audio_input
        # Ensure correct length
        target_len = int(audio_cfg["sample_rate"] * audio_cfg["duration"])
        if len(y_wave) > target_len:
            y_wave = y_wave[:target_len]
        elif len(y_wave) < target_len:
            y_wave = np.pad(y_wave, (0, target_len - len(y_wave)))

    # ── Feature extraction ───────────────────────────────────
    X_prep = extract_inference_features(y_wave, arch, config)

    # ── Inference ────────────────────────────────────────────
    probs = model.predict(X_prep, verbose=0)[0]  # shape: (num_classes,)

    # Build results
    prob_dict     = {label_list[i]: float(probs[i]) for i in range(len(label_list))}
    top_indices   = np.argsort(probs)[::-1][:top_k]
    top_preds     = [(label_list[i], float(probs[i])) for i in top_indices]
    best_idx      = top_indices[0]

    return {
        "predicted_emotion": label_list[best_idx],
        "confidence":        float(probs[best_idx]),
        "top_k":             top_preds,
        "probabilities":     prob_dict,
    }


# ─────────────────────────────────────────────────────────────
# Pretty-print result
# ─────────────────────────────────────────────────────────────

EMOTION_EMOJI = {
    "neutral":   "😐",
    "calm":      "😌",
    "happy":     "😄",
    "sad":       "😢",
    "angry":     "😠",
    "fearful":   "😨",
    "disgust":   "🤢",
    "surprised": "😲",
    "fear":      "😨",
    "boredom":   "😑",
    "joy":       "😄",
}

def print_result(result: dict):
    """Pretty-print prediction result to console."""
    emotion = result["predicted_emotion"]
    emoji   = EMOTION_EMOJI.get(emotion, "🎵")
    conf    = result["confidence"] * 100

    print("\n" + "="*50)
    print(f"  {emoji}  Predicted Emotion : {emotion.upper()}")
    print(f"  Confidence        : {conf:.1f}%")
    print("-"*50)
    print("  Top predictions:")
    for rank, (emo, prob) in enumerate(result["top_k"], 1):
        bar = "█" * int(prob * 20)
        print(f"  {rank}. {emo:<12} {prob*100:5.1f}%  {bar}")
    print("="*50 + "\n")


# ─────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Emotion Recognition Inference")
    parser.add_argument("--config",       default="config.yaml")
    parser.add_argument("--model",        default=None,  choices=["cnn", "lstm", "cnn_lstm", "transformer"])
    parser.add_argument("--file",         default=None,  help="Path to audio file (.wav/.mp3)")
    parser.add_argument("--mic",          action="store_true", help="Record from microphone")
    parser.add_argument("--duration",     default=4.0, type=float, help="Mic recording duration (s)")
    parser.add_argument("--list-models",  action="store_true", help="List available trained models")
    args = parser.parse_args()

    cfg  = load_config(args.config)
    arch = args.model or cfg["model"]["architecture"]

    if args.list_models:
        available = list_available_models(cfg["training"]["checkpoint_dir"])
        if available:
            print("Available trained models:")
            for a, p in available.items():
                print(f"  {a:12} → {p}")
        else:
            print("No trained models found. Run: python src/train.py")
        sys.exit(0)

    # ── Build label list ─────────────────────────────────────
    dataset_name = cfg["dataset"]["name"]
    if dataset_name == "ravdess":
        em = cfg["ravdess_emotions"]
        label_list = [em[k] for k in sorted(em.keys())]
    elif dataset_name == "tess":
        em = cfg["tess_emotions"]
        label_list = sorted(set(em.values()))
    elif dataset_name == "emodb":
        em = cfg["emodb_emotions"]
        label_list = sorted(set(em.values()))
    else:
        label_list = None   # will be inferred from dataset

    # ── Get audio input ──────────────────────────────────────
    if args.file:
        if not os.path.exists(args.file):
            print(f"File not found: {args.file}")
            sys.exit(1)
        audio_input = args.file

    elif args.mic:
        sr     = cfg["audio"]["sample_rate"]
        y_wave = record_from_microphone(duration=args.duration, sr=sr)
        audio_input = y_wave

    else:
        parser.print_help()
        sys.exit(0)

    # ── Predict ──────────────────────────────────────────────
    result = predict_emotion(audio_input, cfg, arch=arch, label_list=label_list)
    print_result(result)

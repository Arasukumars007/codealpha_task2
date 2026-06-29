"""
feature_extractor.py
====================
Extracts acoustic features from audio files for emotion recognition.

Features extracted
------------------
- MFCCs (Mel-Frequency Cepstral Coefficients) — 40 coefficients
- Delta MFCCs  — temporal derivatives (1st order)
- Delta-Delta MFCCs — temporal derivatives (2nd order)
- Chroma STFT  — 12 pitch-class energy features
- Mel-Spectrogram — 128-band spectral energy map
- Zero-Crossing Rate (ZCR)
- Root Mean Square Energy (RMSE)

Two output modes
----------------
1. Flattened 1-D vector  → for classical ML baselines
2. 2-D spectrogram array → for CNN / CNN-LSTM models (shape: [n_mels, T])

All features are normalized (z-score) per sample.
"""

import os
import logging
import pickle
from pathlib import Path
from typing import Tuple, List, Optional

import numpy as np
import librosa
import yaml
from tqdm import tqdm

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# Audio loading & augmentation
# ─────────────────────────────────────────────────────────────

def load_audio(
    filepath: str,
    sr: int = 22050,
    duration: float = 3.0,
) -> np.ndarray:
    """
    Load a WAV/MP3 file, fix duration by padding or truncating.

    Returns a float32 mono waveform of length sr*duration.
    """
    target_len = int(sr * duration)
    try:
        y, _ = librosa.load(filepath, sr=sr, mono=True, duration=duration + 0.1)
    except Exception as e:
        logger.error(f"Failed to load {filepath}: {e}")
        return np.zeros(target_len, dtype=np.float32)

    # Truncate if longer than target
    if len(y) > target_len:
        y = y[:target_len]
    # Zero-pad if shorter
    elif len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)), mode="constant")

    return y.astype(np.float32)


def augment_audio(
    y: np.ndarray,
    sr: int,
    noise_factor: float = 0.005,
    pitch_steps: Optional[int] = None,
    stretch_rate: Optional[float] = None,
) -> np.ndarray:
    """
    Apply one or more augmentation strategies to a waveform.

    Parameters
    ----------
    y            : Input waveform
    sr           : Sample rate
    noise_factor : Gaussian noise amplitude (0 = disabled)
    pitch_steps  : Number of semitones to shift (None = disabled)
    stretch_rate : Time-stretch ratio (None = disabled, 1.0 = no change)

    Returns
    -------
    Augmented waveform (same length as input).
    """
    target_len = len(y)

    if noise_factor > 0:
        noise = np.random.randn(len(y)).astype(np.float32) * noise_factor
        y = y + noise

    if stretch_rate is not None and stretch_rate != 1.0:
        y = librosa.effects.time_stretch(y, rate=stretch_rate)
        # Re-fix length after stretching
        if len(y) > target_len:
            y = y[:target_len]
        elif len(y) < target_len:
            y = np.pad(y, (0, target_len - len(y)), mode="constant")

    if pitch_steps is not None and pitch_steps != 0:
        y = librosa.effects.pitch_shift(y, sr=sr, n_steps=pitch_steps)

    return y.astype(np.float32)


# ─────────────────────────────────────────────────────────────
# Feature extraction
# ─────────────────────────────────────────────────────────────

def extract_features(
    y: np.ndarray,
    sr: int = 22050,
    n_mfcc: int = 40,
    n_mels: int = 128,
    n_chroma: int = 12,
    n_fft: int = 2048,
    hop_length: int = 512,
    use_delta: bool = True,
    use_chroma: bool = True,
    use_mel: bool = True,
    use_zcr: bool = True,
    use_rmse: bool = True,
    time_frames: int = 130,
) -> np.ndarray:
    """
    Extract a 2-D feature map from a waveform.

    Returns
    -------
    np.ndarray of shape (n_features, time_frames) where n_features depends
    on which features are enabled:
      MFCCs       : n_mfcc  (default 40)
      Delta       : n_mfcc  (if use_delta=True)
      Delta-Delta : n_mfcc  (if use_delta=True)
      Chroma      : n_chroma (if use_chroma=True, default 12)
      Mel-Spec    : n_mels  (if use_mel=True, default 128)
      ZCR         : 1       (if use_zcr=True)
      RMSE        : 1       (if use_rmse=True)
    """
    feature_list = []

    # ── MFCCs ──────────────────────────────────────────────
    mfcc = librosa.feature.mfcc(
        y=y, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length
    )
    feature_list.append(mfcc)

    if use_delta:
        delta1 = librosa.feature.delta(mfcc, order=1)
        delta2 = librosa.feature.delta(mfcc, order=2)
        feature_list.append(delta1)
        feature_list.append(delta2)

    # ── Chroma STFT ────────────────────────────────────────
    if use_chroma:
        chroma = librosa.feature.chroma_stft(
            y=y, sr=sr, n_chroma=n_chroma, n_fft=n_fft, hop_length=hop_length
        )
        feature_list.append(chroma)

    # ── Mel-Spectrogram (log scale) ─────────────────────────
    if use_mel:
        mel = librosa.feature.melspectrogram(
            y=y, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length
        )
        mel_db = librosa.power_to_db(mel, ref=np.max)
        feature_list.append(mel_db)

    # ── Zero-Crossing Rate ──────────────────────────────────
    if use_zcr:
        zcr = librosa.feature.zero_crossing_rate(y, hop_length=hop_length)
        feature_list.append(zcr)

    # ── Root Mean Square Energy ─────────────────────────────
    if use_rmse:
        rmse = librosa.feature.rms(y=y, hop_length=hop_length)
        feature_list.append(rmse)

    # ── Stack and fix time axis ─────────────────────────────
    # Each feature: shape (n_feat_i, T_i) — pad/truncate T to time_frames
    fixed = []
    for feat in feature_list:
        T = feat.shape[1]
        if T > time_frames:
            feat = feat[:, :time_frames]
        elif T < time_frames:
            pad_width = time_frames - T
            feat = np.pad(feat, ((0, 0), (0, pad_width)), mode="constant")
        fixed.append(feat)

    combined = np.vstack(fixed)  # shape: (total_features, time_frames)

    # ── Z-score normalization (per feature row) ─────────────
    mean = combined.mean(axis=1, keepdims=True)
    std  = combined.std(axis=1, keepdims=True) + 1e-8
    combined = (combined - mean) / std

    return combined.astype(np.float32)


def extract_mel_spectrogram_only(
    y: np.ndarray,
    sr: int = 22050,
    n_mels: int = 128,
    n_fft: int = 2048,
    hop_length: int = 512,
    time_frames: int = 130,
) -> np.ndarray:
    """
    Extract just the log Mel-Spectrogram (for pure CNN input).

    Returns
    -------
    np.ndarray of shape (n_mels, time_frames).
    """
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)

    T = mel_db.shape[1]
    if T > time_frames:
        mel_db = mel_db[:, :time_frames]
    elif T < time_frames:
        mel_db = np.pad(mel_db, ((0, 0), (0, time_frames - T)), mode="constant")

    # Normalize to [-1, 1]
    mel_db = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-8) * 2 - 1
    return mel_db.astype(np.float32)


# ─────────────────────────────────────────────────────────────
# Batch processing
# ─────────────────────────────────────────────────────────────

def build_feature_dataset(
    df,
    config: dict,
    augment: bool = False,
    cache_path: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build (X, y) arrays from a DataFrame of audio files.

    Parameters
    ----------
    df         : DataFrame with [filepath, label] columns
    config     : Loaded config dict
    augment    : Whether to apply data augmentation
    cache_path : If provided, save/load cached .npz file

    Returns
    -------
    X : np.ndarray of shape (N, n_features, time_frames)
    y : np.ndarray of shape (N,) — integer labels
    """
    # ── Load from cache if available ────────────────────────
    if cache_path and os.path.exists(cache_path):
        logger.info(f"Loading features from cache: {cache_path}")
        data = np.load(cache_path)
        return data["X"], data["y"]

    audio_cfg = config["audio"]
    feat_cfg  = config["features"]
    aug_cfg   = config["augmentation"]

    sr          = audio_cfg["sample_rate"]
    duration    = audio_cfg["duration"]
    hop_length  = audio_cfg["hop_length"]
    n_fft       = audio_cfg["n_fft"]
    time_frames = feat_cfg["time_frames"]

    num_samples = len(df)
    if augment and aug_cfg["enabled"]:
        num_samples *= 3

    X = None
    y = np.empty((num_samples,), dtype=np.int32)
    idx = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Extracting features"):
        y_wave = load_audio(row["filepath"], sr=sr, duration=duration)

        # Original sample
        feat = extract_features(
            y_wave, sr=sr,
            n_mfcc=feat_cfg["n_mfcc"],
            n_mels=feat_cfg["n_mels"],
            n_chroma=feat_cfg["n_chroma"],
            n_fft=n_fft,
            hop_length=hop_length,
            use_delta=feat_cfg["use_delta"],
            use_chroma=feat_cfg["use_chroma"],
            use_mel=feat_cfg["use_mel"],
            use_zcr=feat_cfg["use_zcr"],
            use_rmse=feat_cfg["use_rmse"],
            time_frames=time_frames,
        )
        if X is None:
            X = np.empty((num_samples, feat.shape[0], feat.shape[1]), dtype=np.float32)

        X[idx] = feat
        y[idx] = row["label"]
        idx += 1

        # Augmented samples
        if augment and aug_cfg["enabled"]:
            # Gaussian noise
            y_noisy = augment_audio(y_wave.copy(), sr, noise_factor=aug_cfg["noise_factor"])
            feat_noisy = extract_features(y_noisy, sr=sr,
                    n_mfcc=feat_cfg["n_mfcc"], n_mels=feat_cfg["n_mels"],
                    n_chroma=feat_cfg["n_chroma"], n_fft=n_fft,
                    hop_length=hop_length, use_delta=feat_cfg["use_delta"],
                    use_chroma=feat_cfg["use_chroma"], use_mel=feat_cfg["use_mel"],
                    use_zcr=feat_cfg["use_zcr"], use_rmse=feat_cfg["use_rmse"],
                    time_frames=time_frames)
            X[idx] = feat_noisy
            y[idx] = row["label"]
            idx += 1

            # Pitch shift (random ±2 semitones)
            pitch = np.random.choice(aug_cfg["pitch_shift_steps"])
            y_pitched = augment_audio(y_wave.copy(), sr, pitch_steps=int(pitch))
            feat_pitched = extract_features(y_pitched, sr=sr,
                    n_mfcc=feat_cfg["n_mfcc"], n_mels=feat_cfg["n_mels"],
                    n_chroma=feat_cfg["n_chroma"], n_fft=n_fft,
                    hop_length=hop_length, use_delta=feat_cfg["use_delta"],
                    use_chroma=feat_cfg["use_chroma"], use_mel=feat_cfg["use_mel"],
                    use_zcr=feat_cfg["use_zcr"], use_rmse=feat_cfg["use_rmse"],
                    time_frames=time_frames)
            X[idx] = feat_pitched
            y[idx] = row["label"]
            idx += 1

    logger.info(f"Feature array shape: {X.shape}")

    # ── Cache results ────────────────────────────────────────
    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        np.savez_compressed(cache_path, X=X, y=y)
        logger.info(f"Features cached to: {cache_path}")

    return X, y


if __name__ == "__main__":
    import librosa.display
    import matplotlib.pyplot as plt

    logging.basicConfig(level=logging.INFO)
    # Quick smoke test on a single demo file
    demo_path = librosa.ex("trumpet")
    y_wave = load_audio(demo_path, sr=22050, duration=3.0)
    feat   = extract_features(y_wave, sr=22050)
    print(f"Feature shape: {feat.shape}")

    mel = extract_mel_spectrogram_only(y_wave)
    print(f"Mel-spec shape: {mel.shape}")

    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mel, sr=22050, hop_length=512, x_axis="time", y_axis="mel")
    plt.colorbar(format="%+2.0f dB")
    plt.title("Mel-Spectrogram")
    plt.tight_layout()
    plt.savefig("demo_mel.png")
    print("Saved demo_mel.png")

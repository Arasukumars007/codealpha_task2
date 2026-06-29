"""
data_loader.py
==============
Handles dataset discovery, filename parsing, and label extraction for
RAVDESS, TESS, and EMO-DB datasets. Returns a unified pandas DataFrame
with columns: [filepath, emotion, dataset].

Supported datasets
------------------
RAVDESS : Audio-Speech files follow the naming convention
          03-01-{emotion:02d}-{intensity}-{statement}-{repetition}-{actor}.wav
          where emotion codes 01-08 map to neutral..surprised.

TESS    : Files are named as <word>_<emotion>.wav inside per-actor folders.
          e.g.  YAF_angry/YAF_back_angry.wav

EMO-DB  : Files follow the Berlin EmoDB convention:
          {speaker}{code}{text}{version}.wav  (character at index 5 = emotion)
          e.g.  03a01Wa.wav  -> emotion = 'W' = angry
"""

import os
import re
import logging
import yaml
import pandas as pd
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# Config helpers
# ─────────────────────────────────────────────────────────────

def load_config(config_path: str = "config.yaml") -> dict:
    """Load YAML configuration file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# ─────────────────────────────────────────────────────────────
# RAVDESS
# ─────────────────────────────────────────────────────────────

def load_ravdess(data_dir: str, config: dict) -> pd.DataFrame:
    """
    Parse RAVDESS audio-speech files.

    Expected folder structure:
        data_dir/
            Actor_01/
                03-01-01-01-01-01-01.wav
                ...
            Actor_02/
                ...

    Parameters
    ----------
    data_dir : str
        Path to the RAVDESS root (containing Actor_XX sub-folders).
    config : dict
        Loaded config.yaml contents.

    Returns
    -------
    pd.DataFrame with columns [filepath, emotion, dataset].
    """
    emotion_map = {int(k): v for k, v in config["ravdess_emotions"].items()}
    records = []

    data_path = Path(data_dir)
    wav_files = list(data_path.rglob("*.wav"))

    if not wav_files:
        logger.warning(f"No .wav files found in RAVDESS dir: {data_dir}")
        return pd.DataFrame(columns=["filepath", "emotion", "dataset"])

    for wav_path in wav_files:
        parts = wav_path.stem.split("-")
        if len(parts) != 7:
            logger.debug(f"Skipping non-standard file: {wav_path.name}")
            continue

        modality = int(parts[0])
        if modality != 3:          # 03 = audio-speech only
            continue

        emotion_code = int(parts[2])
        emotion = emotion_map.get(emotion_code)
        if emotion is None:
            logger.warning(f"Unknown emotion code {emotion_code} in {wav_path.name}")
            continue

        records.append({
            "filepath": str(wav_path),
            "emotion": emotion,
            "dataset": "ravdess",
        })

    df = pd.DataFrame(records)
    logger.info(f"RAVDESS: loaded {len(df)} files | emotions: {sorted(df['emotion'].unique())}")
    return df


# ─────────────────────────────────────────────────────────────
# TESS
# ─────────────────────────────────────────────────────────────

def load_tess(data_dir: str, config: dict) -> pd.DataFrame:
    """
    Parse TESS dataset.

    Expected folder structure:
        data_dir/
            YAF_angry/
                YAF_back_angry.wav ...
            OAF_happy/
                OAF_back_happy.wav ...
            ...

    Parameters
    ----------
    data_dir : str
        Path to the TESS root directory.
    config : dict
        Loaded config.yaml contents.

    Returns
    -------
    pd.DataFrame with columns [filepath, emotion, dataset].
    """
    emotion_map = {k.lower(): v for k, v in config["tess_emotions"].items()}
    records = []

    data_path = Path(data_dir)
    for wav_path in data_path.rglob("*.wav"):
        # Emotion is the last underscore-separated token in the filename
        stem_parts = wav_path.stem.lower().split("_")
        raw_emotion = stem_parts[-1]
        emotion = emotion_map.get(raw_emotion)
        if emotion is None:
            logger.debug(f"TESS: unknown emotion '{raw_emotion}' in {wav_path.name}")
            continue

        records.append({
            "filepath": str(wav_path),
            "emotion": emotion,
            "dataset": "tess",
        })

    df = pd.DataFrame(records)
    logger.info(f"TESS: loaded {len(df)} files | emotions: {sorted(df['emotion'].unique())}")
    return df


# ─────────────────────────────────────────────────────────────
# EMO-DB
# ─────────────────────────────────────────────────────────────

def load_emodb(data_dir: str, config: dict) -> pd.DataFrame:
    """
    Parse Berlin EMO-DB dataset.

    File naming convention (wav folder):
        {speaker:2d}{text:3s}{emotion:1s}{version:1s}.wav
        Character index 5 (0-based) is the emotion code.
        e.g.  03a01Wa.wav  → W = Wut (anger)

    Parameters
    ----------
    data_dir : str
        Path to the EMO-DB root (or 'wav' sub-folder directly).
    config : dict
        Loaded config.yaml contents.

    Returns
    -------
    pd.DataFrame with columns [filepath, emotion, dataset].
    """
    emotion_map = config["emodb_emotions"]
    records = []

    data_path = Path(data_dir)
    # Support both root and wav/ sub-folder
    search_paths = [data_path, data_path / "wav"]

    wav_files = []
    for sp in search_paths:
        if sp.exists():
            wav_files.extend(list(sp.glob("*.wav")))

    if not wav_files:
        logger.warning(f"No .wav files found in EMO-DB dir: {data_dir}")
        return pd.DataFrame(columns=["filepath", "emotion", "dataset"])

    for wav_path in wav_files:
        stem = wav_path.stem
        if len(stem) < 6:
            continue
        emotion_code = stem[5]  # 6th character (0-indexed = 5)
        emotion = emotion_map.get(emotion_code)
        if emotion is None:
            logger.debug(f"EMO-DB: unknown code '{emotion_code}' in {wav_path.name}")
            continue

        records.append({
            "filepath": str(wav_path),
            "emotion": emotion,
            "dataset": "emodb",
        })

    df = pd.DataFrame(records)
    logger.info(f"EMO-DB: loaded {len(df)} files | emotions: {sorted(df['emotion'].unique())}")
    return df


# ─────────────────────────────────────────────────────────────
# Unified loader
# ─────────────────────────────────────────────────────────────

def load_dataset(config: dict) -> pd.DataFrame:
    """
    Load one or more datasets as configured in config.yaml.

    config["dataset"]["name"] can be:
        "ravdess" | "tess" | "emodb" | "combined"

    For "combined", all datasets whose sub-folders exist under raw_dir
    will be merged.

    Returns
    -------
    pd.DataFrame with columns [filepath, emotion, dataset].
    """
    name = config["dataset"]["name"].lower()
    raw_dir = config["dataset"]["raw_dir"]

    loaders = {
        "ravdess": (load_ravdess, os.path.join(raw_dir, "ravdess")),
        "tess":    (load_tess,    os.path.join(raw_dir, "tess")),
        "emodb":   (load_emodb,   os.path.join(raw_dir, "emodb")),
    }

    if name == "combined":
        frames = []
        for key, (fn, path) in loaders.items():
            if os.path.exists(path):
                frames.append(fn(path, config))
            else:
                logger.info(f"Skipping {key}: folder not found at {path}")
        if not frames:
            raise FileNotFoundError(
                f"No dataset folders found under {raw_dir}. "
                "Please download and place datasets in data/raw/<dataset_name>/"
            )
        df = pd.concat(frames, ignore_index=True)
    elif name in loaders:
        fn, path = loaders[name]
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Dataset folder not found: {path}\n"
                f"Please download {name.upper()} and place it at: {path}"
            )
        df = fn(path, config)
    else:
        raise ValueError(f"Unknown dataset name '{name}'. Choose from: ravdess, tess, emodb, combined")

    # Sanity check
    if df.empty:
        raise ValueError("Dataset loaded but is empty. Check your data directory structure.")

    # Encode labels
    label_list = sorted(df["emotion"].unique())
    df["label"] = df["emotion"].map({e: i for i, e in enumerate(label_list)})
    df["label_name"] = df["emotion"]

    logger.info(
        f"\nDataset summary:\n"
        f"  Total files : {len(df)}\n"
        f"  Emotions    : {label_list}\n"
        f"  Distribution:\n{df['emotion'].value_counts().to_string()}\n"
    )
    return df, label_list


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    cfg = load_config()
    df, labels = load_dataset(cfg)
    print(df.head())
    print("Labels:", labels)

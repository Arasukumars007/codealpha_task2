"""
download_data.py
================
Helper script to guide dataset download and verify folder structure.

RAVDESS, TESS and EMO-DB cannot be downloaded via a simple script
(they require form acceptance or Kaggle credentials). This script
provides direct links and verifies your local structure once placed.

Usage
-----
  # Check if datasets are correctly placed
  python scripts/download_data.py --check

  # Download RAVDESS audio-only (requires Kaggle CLI)
  python scripts/download_data.py --kaggle ravdess

  # Show instructions for manual download
  python scripts/download_data.py --info ravdess
"""

import os
import sys
import argparse
import zipfile
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

ROOT     = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "raw"

# ─────────────────────────────────────────────────────────────
# Dataset metadata
# ─────────────────────────────────────────────────────────────

DATASETS = {
    "ravdess": {
        "name":    "RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)",
        "folder":  DATA_DIR / "ravdess",
        "url":     "https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio",
        "kaggle":  "uwrfkaggler/ravdess-emotional-speech-audio",
        "size":    "~1 GB (audio-speech only)",
        "emotions": ["neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised"],
        "actors":  24,
        "expected_structure": """
  data/raw/ravdess/
    Actor_01/
      03-01-01-01-01-01-01.wav
      03-01-02-01-01-01-01.wav
      ...
    Actor_02/
      ...
    ...
    Actor_24/
""",
        "manual_steps": """
  1. Visit: https://zenodo.org/record/1188976
     OR:    https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio
  2. Download: 'Audio_Speech_Actors_01-24.zip' (audio-speech only, ~1 GB)
  3. Extract to: data/raw/ravdess/
  4. You should see folders: Actor_01/ Actor_02/ ... Actor_24/
""",
    },

    "tess": {
        "name":    "TESS (Toronto Emotional Speech Set)",
        "folder":  DATA_DIR / "tess",
        "url":     "https://www.kaggle.com/datasets/ejlok1/toronto-emotional-speech-set-tess",
        "kaggle":  "ejlok1/toronto-emotional-speech-set-tess",
        "size":    "~4 GB",
        "emotions": ["neutral", "happy", "sad", "angry", "fearful", "disgust", "surprised"],
        "actors":  2,
        "expected_structure": """
  data/raw/tess/
    TESS Toronto emotional speech set data/
      OAF_angry/
        OAF_back_angry.wav ...
      OAF_disgust/
        ...
      YAF_happy/
        ...
""",
        "manual_steps": """
  1. Visit: https://www.kaggle.com/datasets/ejlok1/toronto-emotional-speech-set-tess
  2. Download the dataset zip
  3. Extract to: data/raw/tess/
  4. You should see folders like: OAF_angry/, YAF_happy/, etc.
  5. Our loader will recursively find all .wav files.
""",
    },

    "emodb": {
        "name":    "EMO-DB (Berlin Database of Emotional Speech)",
        "folder":  DATA_DIR / "emodb",
        "url":     "http://emodb.bilderbar.info/download/",
        "kaggle":  None,
        "size":    "~500 MB",
        "emotions": ["neutral", "angry", "fearful", "happy", "sad", "disgust", "boredom"],
        "actors":  10,
        "expected_structure": """
  data/raw/emodb/
    wav/
      03a01Fa.wav
      03a01Na.wav
      ...
""",
        "manual_steps": """
  1. Visit: http://emodb.bilderbar.info/download/
  2. Download: 'emodb_full.zip' (or download/download0.zip for audio only)
  3. Extract to: data/raw/emodb/
  4. You should see a 'wav/' subfolder containing .wav files.
""",
    },
}

# ─────────────────────────────────────────────────────────────
# Structure verification
# ─────────────────────────────────────────────────────────────

def check_dataset(key: str, meta: dict) -> tuple:
    """Check if a dataset folder exists and count audio files."""
    folder = meta["folder"]
    if not folder.exists():
        return False, 0

    wav_files = list(folder.rglob("*.wav"))
    return True, len(wav_files)


def check_all():
    """Print a status table for all datasets."""
    print("\n" + "="*60)
    print("  DATASET STATUS CHECK")
    print("="*60)

    all_ok = True
    for key, meta in DATASETS.items():
        exists, n_files = check_dataset(key, meta)
        status = "[OK] FOUND" if exists and n_files > 0 else "[!!] MISSING"
        if not exists or n_files == 0:
            all_ok = False
        print(f"\n  {key.upper():<12} {status}")
        print(f"  Folder    : {meta['folder']}")
        if exists:
            print(f"  WAV files : {n_files}")
        else:
            print(f"  -> Run: python scripts/download_data.py --info {key}")

    print("\n" + "="*60)
    if all_ok:
        print("  [OK] All datasets are ready!")
    else:
        print("  [!!] Some datasets are missing. See instructions above.")
    print("="*60 + "\n")


# ─────────────────────────────────────────────────────────────
# Manual instructions
# ─────────────────────────────────────────────────────────────

def show_info(key: str):
    """Print detailed download instructions for a dataset."""
    meta = DATASETS[key]
    print(f"\n{'='*60}")
    print(f"  {meta['name']}")
    print(f"{'='*60}")
    print(f"  Size     : {meta['size']}")
    print(f"  Emotions : {', '.join(meta['emotions'])}")
    print(f"  Actors   : {meta['actors']}")
    print(f"\n  Download URL:\n  {meta['url']}")
    print(f"\n  Manual Steps:{meta['manual_steps']}")
    print(f"  Expected Structure:{meta['expected_structure']}")
    print(f"{'='*60}\n")


# ─────────────────────────────────────────────────────────────
# Kaggle download (requires kaggle CLI)
# ─────────────────────────────────────────────────────────────

def download_kaggle(key: str):
    """Attempt to download dataset using the Kaggle CLI."""
    meta = DATASETS[key]
    if meta["kaggle"] is None:
        print(f"  No Kaggle source for {key}. See: {meta['url']}")
        return

    target = meta["folder"]
    target.mkdir(parents=True, exist_ok=True)

    print(f"\n  Downloading {key.upper()} via Kaggle CLI …")
    print(f"  Target: {target}")
    print("  (Requires: pip install kaggle  +  ~/.kaggle/kaggle.json API key)\n")

    cmd = f'kaggle datasets download -d {meta["kaggle"]} -p "{target}" --unzip'
    print(f"  Running: {cmd}\n")
    ret = os.system(cmd)

    if ret == 0:
        _, n_files = check_dataset(key, meta)
        print(f"\n  [OK] Download complete! Found {n_files} WAV files.")
    else:
        print(f"\n  [!!] Download failed (exit code {ret}).")
        print("     Make sure kaggle CLI is installed and configured.")
        show_info(key)


# ─────────────────────────────────────────────────────────────
# Create folder skeleton
# ─────────────────────────────────────────────────────────────

def create_folder_structure():
    """Create the expected data directory tree."""
    dirs = [
        DATA_DIR / "ravdess",
        DATA_DIR / "tess",
        DATA_DIR / "emodb",
        ROOT / "data" / "processed",
        ROOT / "models",
        ROOT / "results" / "plots",
        ROOT / "results" / "reports",
        ROOT / "results" / "tensorboard",
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)

    print("  [OK] Folder structure created:")
    for d in dirs:
        print(f"     {d}")


# ─────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Dataset helper for Emotion Recognition from Speech"
    )
    parser.add_argument("--check",  action="store_true",
                        help="Check if datasets are correctly placed")
    parser.add_argument("--info",   choices=["ravdess", "tess", "emodb"],
                        help="Show download instructions for a dataset")
    parser.add_argument("--kaggle", choices=["ravdess", "tess"],
                        help="Download via Kaggle CLI")
    parser.add_argument("--setup",  action="store_true",
                        help="Create directory structure")
    args = parser.parse_args()

    if args.setup:
        create_folder_structure()

    if args.check:
        check_all()
    elif args.info:
        show_info(args.info)
    elif args.kaggle:
        download_kaggle(args.kaggle)
    else:
        # Default: show status + instructions
        check_all()
        print("Usage:")
        print("  python scripts/download_data.py --check")
        print("  python scripts/download_data.py --info ravdess")
        print("  python scripts/download_data.py --kaggle ravdess")
        print("  python scripts/download_data.py --setup")

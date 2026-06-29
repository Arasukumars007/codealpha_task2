"""
check_models.py
===============
Full project health-check:
  • Lists all available trained models
  • Loads each model and prints its summary
  • Runs a dummy prediction on each (using synthetic 3-second sine audio)
  • Prints a pass/fail report
"""

import os
import sys
import numpy as np

# ── suppress TF noise ──────────────────────────────────────────
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from pathlib import Path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import tensorflow as tf
tf.get_logger().setLevel("ERROR")

from src.data_loader import load_config
from src.predict import list_available_models, predict_emotion

# ── colours ───────────────────────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

def banner(text, color=CYAN):
    line = "=" * 60
    print(f"\n{color}{BOLD}{line}")
    print(f"  {text}")
    print(f"{line}{RESET}")

def ok(msg):  print(f"  {GREEN}✅ {msg}{RESET}")
def fail(msg):print(f"  {RED}❌ {msg}{RESET}")
def info(msg):print(f"  {YELLOW}ℹ  {msg}{RESET}")


def make_synthetic_audio(sr=22050, duration=3.0):
    """Return a 3-second 440 Hz sine wave at 22050 Hz sample rate."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    y = 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
    return y, sr


def run_checks():
    banner("🎙️  EMOTION RECOGNITION — PROJECT HEALTH CHECK")

    # ── 1. Load config ────────────────────────────────────────
    cfg_path = ROOT / "config.yaml"
    try:
        cfg = load_config(str(cfg_path))
        ok(f"config.yaml loaded  →  dataset={cfg['dataset']['name']}, "
           f"sr={cfg['audio']['sample_rate']}, "
           f"features={cfg['features']['n_mfcc']} MFCCs")
    except Exception as e:
        fail(f"config.yaml failed: {e}")
        sys.exit(1)

    # ── 2. Discover models ───────────────────────────────────
    banner("📂  MODEL DISCOVERY")
    models_dir = cfg["training"]["checkpoint_dir"]
    available  = list_available_models(models_dir)

    ALL_ARCHS = ["cnn", "lstm", "cnn_lstm", "transformer"]
    for arch in ALL_ARCHS:
        if arch in available:
            ok(f"{arch.upper():<12} → {available[arch]}")
        else:
            info(f"{arch.upper():<12} → NOT TRAINED (run: python run.py train --model {arch})")

    if not available:
        fail("No trained models found at all!")
        sys.exit(1)

    # ── 3. Build label list ──────────────────────────────────
    em = cfg["ravdess_emotions"]
    label_list = [em[k] for k in sorted(em.keys())]
    info(f"Labels ({len(label_list)}): {label_list}")

    # ── 4. Per-model: load + summary + predict ───────────────
    results = {}
    y_wave, sr = make_synthetic_audio(
        sr=cfg["audio"]["sample_rate"],
        duration=cfg["audio"]["duration"],
    )

    for arch, model_path in available.items():
        banner(f"🧠  TESTING  {arch.upper()}", color=YELLOW)

        # — Load —
        try:
            model = tf.keras.models.load_model(model_path)
            ok(f"Model loaded from {model_path}")
        except Exception as e:
            fail(f"Load FAILED: {e}")
            results[arch] = "LOAD FAILED"
            continue

        # — Summary —
        print()
        try:
            model.summary(line_length=72, print_fn=lambda s: print(f"    {s}"))
        except Exception:
            pass
        print()

        # — Input / output shapes —
        try:
            in_shape  = model.input_shape
            out_shape = model.output_shape
            ok(f"Input shape : {in_shape}")
            ok(f"Output shape: {out_shape}  ({out_shape[-1]} classes)")
        except Exception as e:
            info(f"Shape info unavailable: {e}")

        # — Predict on synthetic audio —
        try:
            result = predict_emotion(
                y_wave, cfg,
                arch=arch,
                model=model,
                label_list=label_list,
            )
            emotion = result["predicted_emotion"]
            conf    = result["confidence"] * 100
            top3    = result["top_k"]

            ok(f"Prediction  : {emotion.upper()}  (confidence {conf:.1f}%)")
            print(f"\n    {CYAN}Top-3 predictions:{RESET}")
            for rank, (emo, prob) in enumerate(top3, 1):
                bar = "█" * int(prob * 30)
                print(f"      {rank}. {emo:<12} {prob*100:5.1f}%  {bar}")
            print()

            results[arch] = "PASS"

        except Exception as e:
            fail(f"Prediction FAILED: {e}")
            results[arch] = f"PREDICT FAILED: {e}"

    # ── 5. Summary report ────────────────────────────────────
    banner("📊  FINAL REPORT")
    total = len(results)
    passed = sum(1 for v in results.values() if v == "PASS")

    for arch in ALL_ARCHS:
        if arch in results:
            status = results[arch]
            if status == "PASS":
                ok(f"{arch.upper():<12} — PASS")
            else:
                fail(f"{arch.upper():<12} — {status}")
        else:
            info(f"{arch.upper():<12} — NOT TRAINED (skipped)")

    print()
    score = int(passed / total * 100) if total > 0 else 0
    if passed == total:
        print(f"  {GREEN}{BOLD}🎉 ALL {total} TRAINED MODEL(S) PASSED!  Score: {score}% ✅{RESET}")
    else:
        print(f"  {YELLOW}{BOLD}⚠️  {passed}/{total} models passed.  Score: {score}%{RESET}")
    print()


if __name__ == "__main__":
    run_checks()

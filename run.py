"""
run.py
======
Top-level convenience launcher for the Emotion Recognition pipeline.

Usage
-----
  # Train a model (default: cnn_lstm from config.yaml)
  python run.py train

  # Train a specific model
  python run.py train --model transformer
  python run.py train --model cnn_lstm --epochs 50

  # Evaluate a trained model
  python run.py evaluate --model transformer
  python run.py evaluate --model cnn_lstm

  # Predict emotion from a file
  python run.py predict --file path/to/audio.wav --model transformer

  # Launch the Streamlit web app
  python run.py app

  # Verify the transformer model builds correctly
  python run.py test-model

  # Show model summaries for all architectures
  python run.py summarize
"""

import sys
import os
import subprocess
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def run_train(extra_args):
    # Always use cache if the processed features file exists
    cache_path = ROOT / "data" / "processed" / "ravdess_aug_features.npz"
    if cache_path.exists() and "--cache" not in extra_args:
        extra_args = ["--cache"] + extra_args
    cmd = [sys.executable, str(ROOT / "src" / "train.py")] + extra_args
    print(f"\n>>> {' '.join(cmd)}\n")
    subprocess.run(cmd, cwd=str(ROOT), check=True)


def run_evaluate(extra_args):
    cmd = [sys.executable, str(ROOT / "src" / "evaluate.py")] + extra_args
    print(f"\n>>> {' '.join(cmd)}\n")
    subprocess.run(cmd, cwd=str(ROOT), check=True)


def run_predict(extra_args):
    cmd = [sys.executable, str(ROOT / "src" / "predict.py")] + extra_args
    print(f"\n>>> {' '.join(cmd)}\n")
    subprocess.run(cmd, cwd=str(ROOT), check=True)


def run_app():
    # Use python -m streamlit for reliability on Windows
    cmd = [sys.executable, "-m", "streamlit", "run",
           str(ROOT / "app" / "streamlit_app.py"),
           "--server.headless", "false"]
    print(f"\n>>> {' '.join(cmd)}\n")
    try:
        subprocess.run(cmd, cwd=str(ROOT), check=True)
    except KeyboardInterrupt:
        print("\n[App stopped by user]")


def test_model():
    """Run a quick forward-pass smoke test on all four architectures."""
    import tensorflow as tf
    from src.models.cnn_model import build_cnn_model
    from src.models.lstm_model import build_lstm_model
    from src.models.cnn_lstm_model import build_cnn_lstm_model
    from src.models.transformer_model import build_transformer_model

    archs = {
        "CNN":          (build_cnn_model,         (128, 130, 1)),
        "LSTM":         (build_lstm_model,         (130, 40)),
        "CNN-LSTM":     (build_cnn_lstm_model,     (262, 130)),
        "Transformer":  (build_transformer_model,  (262, 130)),
    }

    num_classes = 8
    print("\n" + "="*55)
    print("  Smoke-testing all model architectures ...")
    print("="*55)

    for name, (builder, shape) in archs.items():
        model = builder(input_shape=shape, num_classes=num_classes)
        dummy = tf.random.normal((2, *shape))
        out   = model(dummy, training=False)
        params = model.count_params()
        print(f"  [OK]  {name:<14} | input {str(shape):<16} | "
              f"output {out.shape} | params {params:,}")

    print("="*55)
    print("  All architectures OK!\n")


def summarize():
    """Print summary for all four model architectures."""
    from src.models.cnn_model import build_cnn_model
    from src.models.lstm_model import build_lstm_model
    from src.models.cnn_lstm_model import build_cnn_lstm_model
    from src.models.transformer_model import build_transformer_model

    num_classes = 8
    builders = [
        ("CNN",         build_cnn_model,         (128, 130, 1)),
        ("LSTM",        build_lstm_model,         (130, 40)),
        ("CNN-LSTM",    build_cnn_lstm_model,     (262, 130)),
        ("Transformer", build_transformer_model,  (262, 130)),
    ]
    for name, builder, shape in builders:
        print(f"\n{'='*60}")
        print(f"  {name} Model  —  input_shape={shape}")
        print(f"{'='*60}")
        model = builder(input_shape=shape, num_classes=num_classes)
        model.summary()


# ─────────────────────────────────────────────────────────────
# CLI dispatch
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Emotion Recognition — pipeline launcher",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "command",
        choices=["train", "evaluate", "predict", "app", "test-model", "summarize"],
        help=(
            "train      — Train a model\n"
            "evaluate   — Evaluate a trained model on the test set\n"
            "predict    — Run inference on a WAV file\n"
            "app        — Launch the Streamlit web demo\n"
            "test-model — Smoke-test all four model architectures\n"
            "summarize  — Print Keras model.summary() for all architectures"
        ),
    )

    # Capture the sub-command then pass remaining args downstream
    args, extra = parser.parse_known_args()

    if args.command == "train":
        run_train(extra)
    elif args.command == "evaluate":
        run_evaluate(extra)
    elif args.command == "predict":
        run_predict(extra)
    elif args.command == "app":
        run_app()
    elif args.command == "test-model":
        test_model()
    elif args.command == "summarize":
        summarize()


if __name__ == "__main__":
    main()

"""
train.py
========
Full training pipeline for emotion recognition models.

Usage
-----
  python src/train.py                          # uses config.yaml defaults
  python src/train.py --model cnn_lstm         # override model architecture
  python src/train.py --model cnn --epochs 30  # override multiple settings
  python src/train.py --dataset tess           # use TESS dataset

Outputs
-------
  models/<arch>_best.keras        — best checkpoint (val accuracy)
  models/<arch>_final.keras       — final weights after training
  results/plots/training_<arch>.png — loss & accuracy curves
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
import yaml
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow.keras import callbacks as keras_callbacks

# Limit TensorFlow CPU memory to prevent Windows access-violation crash.
# oneDNN is disabled because it uses excessive memory buffers on CPU-only runs.
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

try:
    cpus = tf.config.list_physical_devices("CPU")
    if cpus:
        tf.config.set_logical_device_configuration(
            cpus[0],
            [tf.config.LogicalDeviceConfiguration(memory_limit=4096)]  # 4 GB cap
        )
except Exception:
    pass  # May fail on some TF versions — non-fatal

# Disable tf.summary writes if tensorboard package is absent.
# TensorFlow calls tf.summary.scalar internally during model.fit() and crashes
# with TBNotInstalledError when the tensorboard package is missing.
try:
    import tensorboard  # noqa: F401
except ImportError:
    tf.summary.record_if(False)


# ── Add project root to path ─────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_loader import load_config, load_dataset
from src.feature_extractor import build_feature_dataset
from src.models.cnn_model import build_cnn_model
from src.models.lstm_model import build_lstm_model
from src.models.cnn_lstm_model import build_cnn_lstm_model
from src.models.transformer_model import build_transformer_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train")


# ─────────────────────────────────────────────────────────────
# Argument parsing
# ─────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Train Emotion Recognition Model")
    parser.add_argument("--config",   default="config.yaml", help="Path to config YAML")
    parser.add_argument("--model",    default=None, choices=["cnn", "lstm", "cnn_lstm", "transformer"])
    parser.add_argument("--dataset",  default=None, choices=["ravdess", "tess", "emodb", "combined"])
    parser.add_argument("--epochs",   default=None, type=int)
    parser.add_argument("--batch",    default=None, type=int)
    parser.add_argument("--lr",       default=None, type=float, help="Learning rate")
    parser.add_argument("--no-augment", action="store_true", help="Disable augmentation")
    parser.add_argument("--cache",    action="store_true", help="Cache extracted features")
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────
# Model factory
# ─────────────────────────────────────────────────────────────

def get_model(arch: str, input_shape: tuple, num_classes: int, config: dict):
    """Instantiate the requested model architecture."""
    model_cfg = config["model"]
    dropout   = model_cfg["dropout_rate"]

    if arch == "cnn":
        # CNN expects (n_mels, time_frames, 1)
        n_mels = config["features"]["n_mels"]
        T      = config["features"]["time_frames"]
        cnn_shape = (n_mels, T, 1)
        return build_cnn_model(
            input_shape=cnn_shape,
            num_classes=num_classes,
            dropout_rate=dropout,
        )

    elif arch == "lstm":
        # LSTM expects (time_frames, n_mfcc)
        T      = config["features"]["time_frames"]
        n_mfcc = config["features"]["n_mfcc"]
        lstm_shape = (T, n_mfcc)
        return build_lstm_model(
            input_shape=lstm_shape,
            num_classes=num_classes,
            lstm_units=model_cfg.get("lstm_units", [256, 128]),
            dropout_rate=dropout,
        )

    elif arch == "cnn_lstm":
        # CNN-LSTM expects (n_features, time_frames)
        return build_cnn_lstm_model(
            input_shape=input_shape,
            num_classes=num_classes,
            cnn_filters=model_cfg.get("cnn_filters", [64, 128]),
            lstm_units=model_cfg.get("lstm_units", [128, 64]),
            dropout_rate=dropout,
        )

    elif arch == "transformer":
        # Transformer expects (n_features, time_frames) — same as cnn_lstm
        tf_cfg = config.get("transformer", {})
        return build_transformer_model(
            input_shape=input_shape,
            num_classes=num_classes,
            d_model=tf_cfg.get("d_model", 128),
            num_heads=tf_cfg.get("num_heads", 4),
            num_layers=tf_cfg.get("num_layers", 4),
            d_ff=tf_cfg.get("d_ff", 256),
            dropout_rate=tf_cfg.get("dropout_rate", 0.3),
        )
    else:
        raise ValueError(f"Unknown architecture: {arch}")


# ─────────────────────────────────────────────────────────────
# Input preprocessing per architecture
# ─────────────────────────────────────────────────────────────

def prepare_inputs(X: np.ndarray, arch: str, config: dict):
    """
    Reshape/slice X to match expected model input.

    X shape from build_feature_dataset: (N, n_features, time_frames)

    - cnn      : slice Mel rows → (N, n_mels, T, 1)
    - lstm     : slice MFCC rows → (N, T, n_mfcc)
    - cnn_lstm : use full X → (N, n_features, T)
    """
    n_mfcc = config["features"]["n_mfcc"]
    n_mels = config["features"]["n_mels"]
    use_delta = config["features"]["use_delta"]

    if arch == "cnn":
        # Mel-Spectrogram starts after MFCC rows
        mfcc_rows = n_mfcc * 3 if use_delta else n_mfcc   # mfcc + delta + delta2
        chroma_rows = 12 if config["features"]["use_chroma"] else 0
        mel_start = mfcc_rows + chroma_rows
        mel_end   = mel_start + n_mels
        X_mel = X[:, mel_start:mel_end, :]         # (N, n_mels, T)
        return X_mel[:, :, :, np.newaxis]          # (N, n_mels, T, 1)

    elif arch == "lstm":
        # Use only the first n_mfcc rows (base MFCCs, no delta)
        X_mfcc = X[:, :n_mfcc, :]                 # (N, n_mfcc, T)
        return np.transpose(X_mfcc, (0, 2, 1))    # (N, T, n_mfcc)

    else:  # cnn_lstm or transformer
        return X                                   # (N, n_features, T)


# ─────────────────────────────────────────────────────────────
# Callbacks
# ─────────────────────────────────────────────────────────────

def build_callbacks(config: dict, arch: str) -> list:
    """Build Keras training callbacks."""
    train_cfg = config["training"]
    ckpt_dir  = train_cfg["checkpoint_dir"]
    os.makedirs(ckpt_dir, exist_ok=True)

    best_path  = os.path.join(ckpt_dir, f"{arch}_best.keras")
    final_path = os.path.join(ckpt_dir, f"{arch}_final.keras")

    cb_list = [
        # Save best model by val_accuracy
        keras_callbacks.ModelCheckpoint(
            filepath=best_path,
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
            verbose=1,
        ),
        # Reduce LR on plateau
        keras_callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=train_cfg["lr_factor"],
            patience=train_cfg["lr_patience"],
            min_lr=1e-6,
            verbose=1,
        ),
        # Early stopping
        keras_callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=train_cfg["early_stop_patience"],
            restore_best_weights=True,
            verbose=1,
        ),
        # TensorBoard logging
        keras_callbacks.TensorBoard(
            log_dir=os.path.join("results", "tensorboard", arch),
            histogram_freq=1,
        ),
    ]

    # TensorBoard logging — optional (only added if tensorboard is installed)
    try:
        import tensorboard  # noqa: F401
        cb_list.append(
            keras_callbacks.TensorBoard(
                log_dir=os.path.join("results", "tensorboard", arch),
                histogram_freq=1,
            )
        )
    except ImportError:
        logger.info("tensorboard not installed — skipping TensorBoard callback")

    return cb_list, best_path


# ─────────────────────────────────────────────────────────────
# Training curves plot
# ─────────────────────────────────────────────────────────────

def plot_training_history(history, arch: str, output_dir: str):
    """Save training & validation loss/accuracy curves."""
    os.makedirs(output_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"{arch.upper()} — Training History", fontsize=14, fontweight="bold")

    # Loss
    axes[0].plot(history.history["loss"],     label="Train Loss",  linewidth=2)
    axes[0].plot(history.history["val_loss"], label="Val Loss",    linewidth=2, linestyle="--")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Accuracy
    axes[1].plot(history.history["accuracy"],     label="Train Acc", linewidth=2)
    axes[1].plot(history.history["val_accuracy"], label="Val Acc",   linewidth=2, linestyle="--")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(output_dir, f"training_{arch}.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Training curves saved → {out_path}")


# ─────────────────────────────────────────────────────────────
# Main training function
# ─────────────────────────────────────────────────────────────

def train(config: dict, arch: str, use_augment: bool = True, use_cache: bool = False):
    """
    Full training pipeline.

    Returns
    -------
    model     : trained Keras model
    history   : Keras History object
    X_test    : test feature array
    y_test    : test labels
    label_list: list of emotion names
    """
    logger.info(f"=== Training {arch.upper()} model ===")
    logger.info(f"Dataset: {config['dataset']['name']}")

    # ── 1. Load dataset ──────────────────────────────────────
    df, label_list = load_dataset(config)
    num_classes = len(label_list)
    logger.info(f"Classes ({num_classes}): {label_list}")

    # ── 2. Feature extraction ────────────────────────────────
    cache_path = None
    if use_cache:
        aug_suffix = "_aug" if use_augment else ""
        cache_path = os.path.join(
            config["dataset"]["processed_dir"],
            f"{config['dataset']['name']}{aug_suffix}_features.npz"
        )

    X, y = build_feature_dataset(
        df, config,
        augment=use_augment,
        cache_path=cache_path,
    )
    logger.info(f"Features shape: {X.shape}, Labels shape: {y.shape}")

    # ── 3. Prepare architecture-specific inputs ──────────────
    X_prep = prepare_inputs(X, arch, config)
    logger.info(f"Model input shape: {X_prep.shape[1:]}")

    # ── 4. Train / Val / Test split ──────────────────────────
    cfg = config["dataset"]
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X_prep, y,
        test_size=cfg["test_size"],
        random_state=cfg["random_state"],
        stratify=y,
    )
    val_ratio_adjusted = cfg["val_size"] / (1 - cfg["test_size"])
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val,
        test_size=val_ratio_adjusted,
        random_state=cfg["random_state"],
        stratify=y_train_val,
    )
    logger.info(f"Split → Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

    # ── 5. Compute class weights ─────────────────────────────
    class_weights_array = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(num_classes),
        y=y_train,
    )
    class_weight_dict = {int(i): float(w) for i, w in enumerate(class_weights_array)}
    logger.info(f"Class weights: {class_weight_dict}")

    # ── 6. Build model ───────────────────────────────────────
    input_shape = X_prep.shape[1:]
    model = get_model(arch, input_shape, num_classes, config)
    model.summary(print_fn=logger.info)

    # ── 7. Update LR if overridden ───────────────────────────
    lr = config["training"]["learning_rate"]
    try:
        if hasattr(model.optimizer.learning_rate, "assign"):
            model.optimizer.learning_rate.assign(lr)
        else:
            model.optimizer.learning_rate = lr
    except Exception as e:
        logger.warning(f"Could not assign learning rate via standard Keras 3 variable update: {e}. Trying backend fallback...")
        try:
            tf.keras.backend.set_value(model.optimizer.learning_rate, lr)
        except Exception as err:
            logger.error(f"Failed to set learning rate: {err}")
    # ── 8. Training ──────────────────────────────────────────
    cb_list, best_path = build_callbacks(config, arch)
    train_cfg = config["training"]

    # Bulletproof guard: silence TensorBoard-missing errors inside TF's training loop.
    # TF calls tf.summary.scalar internally even when no TensorBoard callback is used.
    try:
        import tensorboard  # noqa: F401
    except ImportError:
        # Monkey-patch summary ops to no-ops so model.fit() never hits TBNotInstalledError
        _noop = lambda *a, **kw: None  # noqa: E731
        tf.summary.scalar    = _noop
        tf.summary.histogram = _noop
        tf.summary.image     = _noop
        tf.summary.audio     = _noop
        tf.summary.text      = _noop
        tf.summary.record_if(False)

    history = model.fit(
        X_train, y_train,
        epochs=train_cfg["epochs"],
        batch_size=train_cfg["batch_size"],
        validation_data=(X_val, y_val),
        class_weight=class_weight_dict,
        callbacks=cb_list,
        verbose=1,
    )

    # ── 9. Save final model ──────────────────────────────────
    final_path = os.path.join(train_cfg["checkpoint_dir"], f"{arch}_final.keras")
    model.save(final_path)
    logger.info(f"Final model saved → {final_path}")
    logger.info(f"Best model saved  → {best_path}")

    # ── 10. Plot training curves ─────────────────────────────
    plot_training_history(
        history, arch,
        output_dir=config["evaluation"]["plots_dir"],
    )

    return model, history, X_test, y_test, label_list


# ─────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = parse_args()
    cfg  = load_config(args.config)

    # Apply CLI overrides
    if args.model:
        cfg["model"]["architecture"] = args.model
    if args.dataset:
        cfg["dataset"]["name"] = args.dataset
    if args.epochs:
        cfg["training"]["epochs"] = args.epochs
    if args.batch:
        cfg["training"]["batch_size"] = args.batch
    if args.lr:
        cfg["training"]["learning_rate"] = args.lr
    if args.no_augment:
        cfg["augmentation"]["enabled"] = False

    arch = cfg["model"]["architecture"]

    # GPU memory growth (prevents OOM on some GPUs)
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logger.info(f"GPUs available: {[g.name for g in gpus]}")
    else:
        logger.info("No GPU found — running on CPU")

    model, history, X_test, y_test, label_list = train(
        cfg, arch,
        use_augment=not args.no_augment,
        use_cache=args.cache,
    )

    # Quick test-set evaluation
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    logger.info(f"\n{'='*50}")
    logger.info(f"Test Loss    : {loss:.4f}")
    logger.info(f"Test Accuracy: {acc*100:.2f}%")
    logger.info(f"{'='*50}")
    logger.info("\nRun evaluate.py for full metrics and confusion matrix.")

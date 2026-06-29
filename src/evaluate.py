"""
evaluate.py
===========
Full evaluation suite for trained emotion recognition models.

Generates
---------
  results/plots/confusion_matrix_<arch>.png  — heatmap
  results/plots/roc_curves_<arch>.png        — per-class ROC-AUC
  results/reports/classification_<arch>.csv  — precision/recall/F1
  results/reports/summary_<arch>.txt         — human-readable summary

Usage
-----
  python src/evaluate.py --model cnn_lstm
  python src/evaluate.py --model cnn --checkpoint models/cnn_best.keras
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
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
from pathlib import Path
from itertools import cycle
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
)
from sklearn.preprocessing import label_binarize
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_loader import load_config, load_dataset
from src.feature_extractor import build_feature_dataset
from src.train import prepare_inputs
# Import transformer_model to register the LabelSmoothedCCE loss class
# in Keras's serialization registry (needed for load_model() to work).
from src.models import transformer_model as _tm  # noqa: F401

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("evaluate")

# ── Seaborn style ────────────────────────────────────────────
sns.set_theme(style="whitegrid", font_scale=1.1)
PALETTE = sns.color_palette("husl", 10)


# ─────────────────────────────────────────────────────────────
# Confusion Matrix
# ─────────────────────────────────────────────────────────────

def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_list: list,
    arch: str,
    output_dir: str,
):
    """Save a normalised confusion matrix heatmap."""
    os.makedirs(output_dir, exist_ok=True)

    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle(
        f"{arch.upper()} — Confusion Matrix",
        fontsize=16, fontweight="bold", y=1.01
    )

    for ax, data, title, fmt in zip(
        axes,
        [cm, cm_norm],
        ["Counts", "Normalised (%)"],
        ["d", ".1%"],
    ):
        sns.heatmap(
            data if fmt == "d" else cm_norm * 100,
            annot=True, fmt=fmt if fmt == "d" else ".1f",
            cmap="Blues",
            xticklabels=[l.capitalize() for l in label_list],
            yticklabels=[l.capitalize() for l in label_list],
            linewidths=0.5,
            ax=ax,
        )
        ax.set_title(title, fontsize=13)
        ax.set_xlabel("Predicted", fontsize=11)
        ax.set_ylabel("Actual", fontsize=11)
        ax.tick_params(axis="x", rotation=35)
        ax.tick_params(axis="y", rotation=0)

    plt.tight_layout()
    out_path = os.path.join(output_dir, f"confusion_matrix_{arch}.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Confusion matrix saved → {out_path}")


# ─────────────────────────────────────────────────────────────
# ROC Curves
# ─────────────────────────────────────────────────────────────

def plot_roc_curves(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    label_list: list,
    arch: str,
    output_dir: str,
):
    """Save per-class ROC curves with macro-average AUC."""
    os.makedirs(output_dir, exist_ok=True)
    n_classes = len(label_list)

    # Binarise labels for OvR ROC
    y_bin = label_binarize(y_true, classes=np.arange(n_classes))

    fpr, tpr, roc_auc = {}, {}, {}
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], y_prob[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Macro average
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= n_classes
    macro_auc = auc(all_fpr, mean_tpr)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = cycle(sns.color_palette("husl", n_classes))

    for i, (color, label) in enumerate(zip(colors, label_list)):
        ax.plot(
            fpr[i], tpr[i],
            color=color, lw=2,
            label=f"{label.capitalize()} (AUC = {roc_auc[i]:.3f})",
        )

    ax.plot(
        all_fpr, mean_tpr,
        color="black", lw=3, linestyle="--",
        label=f"Macro-average (AUC = {macro_auc:.3f})",
    )
    ax.plot([0, 1], [0, 1], "k:", lw=1, alpha=0.5)

    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title(f"{arch.upper()} — ROC Curves (One-vs-Rest)", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(output_dir, f"roc_curves_{arch}.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"ROC curves saved → {out_path}")
    return macro_auc


# ─────────────────────────────────────────────────────────────
# Per-class bar chart
# ─────────────────────────────────────────────────────────────

def plot_per_class_metrics(report_df: pd.DataFrame, arch: str, output_dir: str):
    """Bar chart of precision, recall, F1 per emotion class."""
    os.makedirs(output_dir, exist_ok=True)

    # Drop aggregate rows
    class_df = report_df[~report_df.index.isin(["accuracy", "macro avg", "weighted avg"])]
    x = np.arange(len(class_df))
    w = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - w, class_df["precision"], w, label="Precision", color=PALETTE[0])
    ax.bar(x,     class_df["recall"],    w, label="Recall",    color=PALETTE[2])
    ax.bar(x + w, class_df["f1-score"],  w, label="F1-Score",  color=PALETTE[5])

    ax.set_xticks(x)
    ax.set_xticklabels([l.capitalize() for l in class_df.index], rotation=30)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title(f"{arch.upper()} — Per-Class Metrics", fontsize=14, fontweight="bold")
    ax.legend()
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(output_dir, f"per_class_metrics_{arch}.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Per-class metrics chart saved → {out_path}")


# ─────────────────────────────────────────────────────────────
# Main evaluation function
# ─────────────────────────────────────────────────────────────

def evaluate(config: dict, arch: str, checkpoint_path: str = None):
    """
    Load a trained model, run it on the test set, and generate all reports.

    Parameters
    ----------
    config          : loaded config dict
    arch            : model architecture name ("cnn" | "lstm" | "cnn_lstm")
    checkpoint_path : path to .keras file; defaults to models/<arch>_best.keras
    """
    # ── Resolve checkpoint ───────────────────────────────────
    if checkpoint_path is None:
        checkpoint_path = os.path.join(
            config["training"]["checkpoint_dir"],
            f"{arch}_best.keras",
        )
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}\n"
            f"Train the model first: python src/train.py --model {arch}"
        )
    logger.info(f"Loading model from: {checkpoint_path}")
    model = tf.keras.models.load_model(checkpoint_path)

    # ── Load data ────────────────────────────────────────────
    logger.info("Loading dataset & extracting features …")
    df, label_list = load_dataset(config)
    X, y = build_feature_dataset(df, config, augment=False)
    X_prep = prepare_inputs(X, arch, config)

    # Re-create test split (same seed → same split as training)
    from sklearn.model_selection import train_test_split
    cfg = config["dataset"]
    _, X_test, _, y_test = train_test_split(
        X_prep, y,
        test_size=cfg["test_size"],
        random_state=cfg["random_state"],
        stratify=y,
    )
    logger.info(f"Test set size: {len(X_test)}")

    # ── Predictions ──────────────────────────────────────────
    y_prob = model.predict(X_test, batch_size=32, verbose=0)
    y_pred = np.argmax(y_prob, axis=1)

    # ── Metrics ──────────────────────────────────────────────
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    logger.info(f"Test Loss    : {test_loss:.4f}")
    logger.info(f"Test Accuracy: {test_acc*100:.2f}%")

    report_str = classification_report(
        y_test, y_pred,
        target_names=[l.capitalize() for l in label_list],
        digits=4,
    )
    logger.info(f"\nClassification Report:\n{report_str}")

    # DataFrame version
    report_dict = classification_report(
        y_test, y_pred,
        target_names=label_list,
        digits=4,
        output_dict=True,
    )
    report_df = pd.DataFrame(report_dict).T

    # ── Save reports ─────────────────────────────────────────
    reports_dir = config["evaluation"]["reports_dir"]
    plots_dir   = config["evaluation"]["plots_dir"]
    os.makedirs(reports_dir, exist_ok=True)
    os.makedirs(plots_dir,   exist_ok=True)

    csv_path = os.path.join(reports_dir, f"classification_{arch}.csv")
    report_df.to_csv(csv_path)
    logger.info(f"Classification report saved → {csv_path}")

    txt_path = os.path.join(reports_dir, f"summary_{arch}.txt")
    with open(txt_path, "w") as f:
        f.write(f"Model        : {arch.upper()}\n")
        f.write(f"Dataset      : {config['dataset']['name']}\n")
        f.write(f"Test Accuracy: {test_acc*100:.2f}%\n")
        f.write(f"Test Loss    : {test_loss:.4f}\n\n")
        f.write(report_str)
    logger.info(f"Summary saved → {txt_path}")

    # ── Plots ─────────────────────────────────────────────────
    plot_confusion_matrix(y_test, y_pred, label_list, arch, plots_dir)
    macro_auc = plot_roc_curves(y_test, y_prob, label_list, arch, plots_dir)
    plot_per_class_metrics(report_df, arch, plots_dir)

    logger.info(f"\nMacro-average AUC: {macro_auc:.4f}")
    logger.info("Evaluation complete.")

    return {
        "accuracy": test_acc,
        "loss": test_loss,
        "macro_auc": macro_auc,
        "report_df": report_df,
    }


# ─────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Emotion Recognition Model")
    parser.add_argument("--config",     default="config.yaml")
    parser.add_argument("--model",      default=None, choices=["cnn", "lstm", "cnn_lstm", "transformer"])
    parser.add_argument("--checkpoint", default=None, help="Path to .keras checkpoint")
    parser.add_argument("--dataset",    default=None)
    args = parser.parse_args()

    cfg  = load_config(args.config)
    arch = args.model or cfg["model"]["architecture"]

    if args.dataset:
        cfg["dataset"]["name"] = args.dataset

    results = evaluate(cfg, arch, checkpoint_path=args.checkpoint)
    print(f"\n✅ Accuracy : {results['accuracy']*100:.2f}%")
    print(f"✅ AUC      : {results['macro_auc']:.4f}")

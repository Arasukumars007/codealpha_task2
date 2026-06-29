"""
tests/test_models.py
====================
Unit tests for all model architectures.
"""

import sys
import numpy as np
import pytest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf

from src.models.cnn_model import build_cnn_model
from src.models.lstm_model import build_lstm_model
from src.models.cnn_lstm_model import build_cnn_lstm_model
from src.models.transformer_model import build_transformer_model

NUM_CLASSES = 8
BATCH_SIZE  = 4


# ─────────────────────────────────────────────────────────────
# CNN model
# ─────────────────────────────────────────────────────────────

class TestCNNModel:
    def test_builds_without_error(self):
        model = build_cnn_model(input_shape=(128, 130, 1), num_classes=NUM_CLASSES)
        assert model is not None

    def test_output_shape(self):
        model = build_cnn_model(input_shape=(128, 130, 1), num_classes=NUM_CLASSES)
        x = tf.random.normal((BATCH_SIZE, 128, 130, 1))
        out = model(x, training=False)
        assert out.shape == (BATCH_SIZE, NUM_CLASSES)

    def test_output_sums_to_one(self):
        """Softmax output should sum to 1."""
        model = build_cnn_model(input_shape=(128, 130, 1), num_classes=NUM_CLASSES)
        x = tf.random.normal((BATCH_SIZE, 128, 130, 1))
        out = model(x, training=False).numpy()
        np.testing.assert_allclose(out.sum(axis=1), 1.0, atol=1e-5)

    def test_has_optimizer(self):
        model = build_cnn_model(input_shape=(128, 130, 1), num_classes=NUM_CLASSES)
        assert model.optimizer is not None


# ─────────────────────────────────────────────────────────────
# LSTM model
# ─────────────────────────────────────────────────────────────

class TestLSTMModel:
    def test_builds_without_error(self):
        model = build_lstm_model(input_shape=(130, 40), num_classes=NUM_CLASSES)
        assert model is not None

    def test_output_shape(self):
        model = build_lstm_model(input_shape=(130, 40), num_classes=NUM_CLASSES)
        x = tf.random.normal((BATCH_SIZE, 130, 40))
        out = model(x, training=False)
        assert out.shape == (BATCH_SIZE, NUM_CLASSES)

    def test_output_sums_to_one(self):
        model = build_lstm_model(input_shape=(130, 40), num_classes=NUM_CLASSES)
        x = tf.random.normal((BATCH_SIZE, 130, 40))
        out = model(x, training=False).numpy()
        np.testing.assert_allclose(out.sum(axis=1), 1.0, atol=1e-5)

    def test_custom_lstm_units(self):
        model = build_lstm_model(
            input_shape=(130, 40), num_classes=NUM_CLASSES,
            lstm_units=[128, 64]
        )
        x = tf.random.normal((BATCH_SIZE, 130, 40))
        out = model(x, training=False)
        assert out.shape == (BATCH_SIZE, NUM_CLASSES)


# ─────────────────────────────────────────────────────────────
# CNN-LSTM model
# ─────────────────────────────────────────────────────────────

class TestCNNLSTMModel:
    def test_builds_without_error(self):
        model = build_cnn_lstm_model(input_shape=(262, 130), num_classes=NUM_CLASSES)
        assert model is not None

    def test_output_shape(self):
        model = build_cnn_lstm_model(input_shape=(262, 130), num_classes=NUM_CLASSES)
        x = tf.random.normal((BATCH_SIZE, 262, 130))
        out = model(x, training=False)
        assert out.shape == (BATCH_SIZE, NUM_CLASSES)

    def test_output_sums_to_one(self):
        model = build_cnn_lstm_model(input_shape=(262, 130), num_classes=NUM_CLASSES)
        x = tf.random.normal((BATCH_SIZE, 262, 130))
        out = model(x, training=False).numpy()
        np.testing.assert_allclose(out.sum(axis=1), 1.0, atol=1e-5)

    def test_custom_filters_and_units(self):
        model = build_cnn_lstm_model(
            input_shape=(262, 130), num_classes=NUM_CLASSES,
            cnn_filters=[32, 64], lstm_units=[128, 64]
        )
        x = tf.random.normal((BATCH_SIZE, 262, 130))
        out = model(x, training=False)
        assert out.shape == (BATCH_SIZE, NUM_CLASSES)


# ─────────────────────────────────────────────────────────────
# Transformer model
# ─────────────────────────────────────────────────────────────

class TestTransformerModel:
    def test_builds_without_error(self):
        model = build_transformer_model(input_shape=(262, 130), num_classes=NUM_CLASSES)
        assert model is not None

    def test_output_shape(self):
        model = build_transformer_model(input_shape=(262, 130), num_classes=NUM_CLASSES)
        x = tf.random.normal((BATCH_SIZE, 262, 130))
        out = model(x, training=False)
        assert out.shape == (BATCH_SIZE, NUM_CLASSES)

    def test_output_sums_to_one(self):
        model = build_transformer_model(input_shape=(262, 130), num_classes=NUM_CLASSES)
        x = tf.random.normal((BATCH_SIZE, 262, 130))
        out = model(x, training=False).numpy()
        np.testing.assert_allclose(out.sum(axis=1), 1.0, atol=1e-5)

    def test_custom_hyperparams(self):
        model = build_transformer_model(
            input_shape=(262, 130), num_classes=NUM_CLASSES,
            d_model=64, num_heads=4, num_layers=2, d_ff=128,
        )
        x = tf.random.normal((BATCH_SIZE, 262, 130))
        out = model(x, training=False)
        assert out.shape == (BATCH_SIZE, NUM_CLASSES)

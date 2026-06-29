"""
transformer_model.py
====================
Transformer-based model for emotion recognition from speech.

Input : 2-D feature map (n_features, time_frames) — same as CNN-LSTM
Output: Softmax probabilities over emotion classes.

Architecture
------------
  Linear projection: (n_features, time_frames) → (time_frames, d_model)
  Positional Encoding (sinusoidal)
  N × Transformer Encoder Block:
      Multi-Head Self-Attention (h heads)
      Add & Layer Norm
      Feed-Forward (d_ff → d_model) with GELU
      Add & Layer Norm
      Dropout
  Global Average + Max Pooling (concatenated)
  Dense(128) → GELU → Dropout
  Dense(64)  → GELU → Dropout
  Dense(num_classes) → Softmax

Why Transformers work for SER
------------------------------
1. Self-attention captures long-range temporal dependencies without
   the sequential bottleneck of LSTMs.
2. Multi-head attention learns multiple types of prosodic relationships
   simultaneously (pitch contours, energy bursts, rhythm).
3. Mean+Max pooling gives richer temporal aggregation than a single
   global pooling operation.
"""

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
import numpy as np


# ─────────────────────────────────────────────────────────────
# Serializable Label-Smoothed Loss (module-level for save/load)
# ─────────────────────────────────────────────────────────────

@tf.keras.utils.register_keras_serializable(package="emotion_recognition")
class LabelSmoothedCCE(tf.keras.losses.Loss):
    """Sparse cross-entropy with label smoothing.

    Registered at module level so models compiled with this loss can
    be saved and reloaded without specifying custom_objects.
    """

    def __init__(self, smoothing: float = 0.05, **kwargs):
        super().__init__(name="label_smoothed_cce", **kwargs)
        self.smoothing = smoothing

    def call(self, y_true, y_pred):
        n_cls = tf.shape(y_pred)[-1]
        y_oh  = tf.one_hot(tf.cast(y_true, tf.int32), n_cls)
        eps   = self.smoothing
        y_sm  = y_oh * (1.0 - eps) + eps / tf.cast(n_cls, tf.float32)
        return -tf.reduce_mean(
            tf.reduce_sum(y_sm * tf.math.log(y_pred + 1e-7), axis=-1)
        )

    def get_config(self):
        cfg = super().get_config()
        cfg["smoothing"] = self.smoothing
        return cfg


# ─────────────────────────────────────────────────────────────
# Positional Encoding
# ─────────────────────────────────────────────────────────────

class PositionalEncoding(layers.Layer):
    """
    Sinusoidal positional encoding as in 'Attention Is All You Need'.

    Adds position-dependent signal to each time frame so the model
    can distinguish temporal order without recurrence.
    """

    def __init__(self, max_len: int = 512, **kwargs):
        super().__init__(**kwargs)
        self.max_len = max_len

    def build(self, input_shape):
        d_model = input_shape[-1]
        # Build (max_len, d_model) encoding matrix
        positions = np.arange(self.max_len)[:, np.newaxis]      # (T, 1)
        dims      = np.arange(d_model)[np.newaxis, :]            # (1, d)
        angles    = positions / np.power(10000, (2 * (dims // 2)) / d_model)
        angles[:, 0::2] = np.sin(angles[:, 0::2])
        angles[:, 1::2] = np.cos(angles[:, 1::2])

        # Store as a non-trainable weight for proper serialization
        pe_init = angles[np.newaxis, :, :].astype(np.float32)  # (1, T, d)
        self.pe = self.add_weight(
            name="positional_encoding",
            shape=pe_init.shape,
            initializer=tf.keras.initializers.Constant(pe_init),
            trainable=False,
        )
        super().build(input_shape)

    def call(self, x):
        # x: (batch, T, d_model)
        seq_len = tf.shape(x)[1]
        return x + self.pe[:, :seq_len, :]

    def get_config(self):
        cfg = super().get_config()
        cfg["max_len"] = self.max_len
        return cfg


# ─────────────────────────────────────────────────────────────
# Transformer Encoder Block
# ─────────────────────────────────────────────────────────────

class TransformerEncoderBlock(layers.Layer):
    """
    Single Transformer encoder block:
      Multi-Head Attention → Add & Norm → FFN → Add & Norm → Dropout
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout_rate: float = 0.1,
        l2_reg: float = 1e-4,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.d_model      = d_model
        self.num_heads    = num_heads
        self.d_ff         = d_ff
        self.dropout_rate = dropout_rate
        self.l2_reg       = l2_reg

        self.attn      = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=d_model // num_heads,
            dropout=dropout_rate,
        )
        self.ffn1      = layers.Dense(
            d_ff, activation="gelu",
            kernel_regularizer=regularizers.l2(l2_reg),
        )
        self.ffn2      = layers.Dense(
            d_model,
            kernel_regularizer=regularizers.l2(l2_reg),
        )
        self.norm1     = layers.LayerNormalization(epsilon=1e-6)
        self.norm2     = layers.LayerNormalization(epsilon=1e-6)
        self.drop1     = layers.Dropout(dropout_rate)
        self.drop2     = layers.Dropout(dropout_rate)

    def call(self, x, training=False):
        # Multi-Head Self-Attention
        attn_out = self.attn(x, x, training=training)
        attn_out = self.drop1(attn_out, training=training)
        x        = self.norm1(x + attn_out)

        # Position-wise Feed-Forward
        ffn_out  = self.ffn2(self.ffn1(x))
        ffn_out  = self.drop2(ffn_out, training=training)
        x        = self.norm2(x + ffn_out)
        return x

    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            "d_model":      self.d_model,
            "num_heads":    self.num_heads,
            "d_ff":         self.d_ff,
            "dropout_rate": self.dropout_rate,
            "l2_reg":       self.l2_reg,
        })
        return cfg


# ─────────────────────────────────────────────────────────────
# Transformer Emotion Recognizer
# ─────────────────────────────────────────────────────────────

def build_transformer_model(
    input_shape: tuple,
    num_classes: int,
    d_model: int = 128,
    num_heads: int = 4,
    num_layers: int = 4,
    d_ff: int = 256,
    dropout_rate: float = 0.3,
    l2_reg: float = 1e-4,
) -> tf.keras.Model:
    """
    Build the Transformer-based emotion recognition model.

    Parameters
    ----------
    input_shape  : (n_features, time_frames) — e.g. (222, 130)
    num_classes  : number of emotion categories
    d_model      : Transformer embedding dimension (must be divisible by num_heads)
    num_heads    : number of self-attention heads
    num_layers   : number of stacked Transformer encoder blocks
    d_ff         : inner dimension of the position-wise feed-forward network
    dropout_rate : dropout probability
    l2_reg       : L2 regularization coefficient

    Returns
    -------
    Compiled Keras model.
    """
    n_features, time_frames = input_shape

    # ── Input ────────────────────────────────────────────────
    inputs = layers.Input(shape=input_shape, name="feature_map_input")

    # Transpose: (batch, n_features, time_frames) → (batch, time_frames, n_features)
    x = layers.Permute((2, 1), name="transpose")(inputs)

    # ── Linear projection → d_model ──────────────────────────
    # Maps each time frame's n_features-dim vector to d_model dimensions.
    x = layers.Dense(
        d_model,
        kernel_regularizer=regularizers.l2(l2_reg),
        name="input_projection",
    )(x)
    x = layers.LayerNormalization(epsilon=1e-6, name="input_norm")(x)
    x = layers.Dropout(dropout_rate * 0.5, name="input_drop")(x)

    # ── Positional Encoding ──────────────────────────────────
    x = PositionalEncoding(max_len=time_frames + 16, name="pos_enc")(x)

    # ── Transformer Encoder Blocks ───────────────────────────
    for i in range(num_layers):
        x = TransformerEncoderBlock(
            d_model=d_model,
            num_heads=num_heads,
            d_ff=d_ff,
            dropout_rate=dropout_rate,
            l2_reg=l2_reg,
            name=f"transformer_block_{i+1}",
        )(x)

    # ── Temporal Aggregation ─────────────────────────────────
    # Mean + max pooling concatenated for richer pooling
    x_mean = layers.GlobalAveragePooling1D(name="gap")(x)
    x_max  = layers.GlobalMaxPooling1D(name="gmp")(x)
    x = layers.Concatenate(name="pool_concat")([x_mean, x_max])

    # ── Classification Head ──────────────────────────────────
    x = layers.Dense(
        128, activation="gelu",
        kernel_regularizer=regularizers.l2(l2_reg),
        name="fc1",
    )(x)
    x = layers.Dropout(dropout_rate, name="drop_fc1")(x)

    x = layers.Dense(
        64, activation="gelu",
        kernel_regularizer=regularizers.l2(l2_reg),
        name="fc2",
    )(x)
    x = layers.Dropout(dropout_rate * 0.5, name="drop_fc2")(x)

    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = models.Model(inputs, outputs, name="Transformer_EmotionRecognizer")

    # ── Optimiser: AdamW if available, else Adam ─────────────
    try:
        optimizer = tf.keras.optimizers.AdamW(
            learning_rate=5e-4,
            weight_decay=1e-4,
        )
    except (AttributeError, TypeError):
        optimizer = tf.keras.optimizers.Adam(learning_rate=5e-4)

    # ── Loss: label-smoothed cross-entropy ───────────────────
    # Use the module-level LabelSmoothedCCE class (registered with Keras)
    # so models are serializable across save/load cycles.
    model.compile(
        optimizer=optimizer,
        loss=LabelSmoothedCCE(smoothing=0.05),
        metrics=["accuracy"],
    )
    return model


# ─────────────────────────────────────────────────────────────
# Self-test
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Building Transformer model …")
    model = build_transformer_model(
        input_shape=(262, 130),
        num_classes=8,
        d_model=128,
        num_heads=4,
        num_layers=4,
        d_ff=256,
        dropout_rate=0.3,
    )
    model.summary()

    # Quick forward pass smoke-test
    dummy = tf.random.normal((4, 262, 130))
    out   = model(dummy, training=False)
    print(f"\nForward pass OK — output shape: {out.shape}")
    print(f"Total parameters: {model.count_params():,}")

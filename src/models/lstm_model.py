"""
lstm_model.py
=============
Bidirectional LSTM for emotion recognition from sequences of MFCC frames.

Input : MFCC sequence (time_frames, n_mfcc)  — e.g. (130, 40)
Output: Softmax probabilities over emotion classes

Architecture
------------
  Bidirectional LSTM(256) → Dropout
  Bidirectional LSTM(128) → Dropout
  Dense(128) → ReLU → Dropout
  Dense(num_classes) → Softmax

Why Bidirectional?
  Speech emotion cues appear at multiple temporal positions. BiLSTM
  reads the sequence both forward and backward, capturing past and
  future context simultaneously.
"""

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers


def build_lstm_model(
    input_shape: tuple,
    num_classes: int,
    lstm_units: list = None,
    dropout_rate: float = 0.4,
    l2_reg: float = 1e-4,
) -> tf.keras.Model:
    """
    Build a Bidirectional LSTM model for emotion classification.

    Parameters
    ----------
    input_shape : (time_frames, n_features) — shape of one sequence sample
    num_classes : number of emotion categories
    lstm_units  : list of unit counts per LSTM layer (default [256, 128])
    dropout_rate: dropout probability
    l2_reg      : L2 regularization coefficient

    Returns
    -------
    Compiled Keras model.
    """
    if lstm_units is None:
        lstm_units = [256, 128]

    inputs = layers.Input(shape=input_shape, name="mfcc_sequence_input")
    x = inputs

    # ── LSTM layers ──────────────────────────────────────────
    for i, units in enumerate(lstm_units):
        return_sequences = (i < len(lstm_units) - 1)  # True for all but last
        x = layers.Bidirectional(
            layers.LSTM(
                units,
                return_sequences=return_sequences,
                dropout=dropout_rate * 0.5,
                recurrent_dropout=0.0,         # keep GPU-compatible
                kernel_regularizer=regularizers.l2(l2_reg),
            ),
            name=f"bilstm_{i+1}"
        )(x)
        x = layers.Dropout(dropout_rate, name=f"drop_lstm_{i+1}")(x)

    # ── Classification head ──────────────────────────────────
    x = layers.Dense(
        128, activation="relu",
        kernel_regularizer=regularizers.l2(l2_reg),
        name="fc1"
    )(x)
    x = layers.Dropout(dropout_rate, name="drop_fc")(x)

    x = layers.Dense(
        64, activation="relu",
        kernel_regularizer=regularizers.l2(l2_reg),
        name="fc2"
    )(x)
    x = layers.Dropout(dropout_rate * 0.5, name="drop_fc2")(x)

    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = models.Model(inputs, outputs, name="BiLSTM_EmotionRecognizer")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


if __name__ == "__main__":
    # input: 130 time frames, 40 MFCC features
    model = build_lstm_model(input_shape=(130, 40), num_classes=8)
    model.summary()

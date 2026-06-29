"""
cnn_lstm_model.py
=================
Hybrid CNN-LSTM model — the PRIMARY architecture for emotion recognition.

Input : 2-D feature map (n_features, time_frames) reshaped to
        (time_frames, n_features, 1) for CNN processing.
Output: Softmax probabilities over emotion classes.

Architecture
------------
  Reshape input → (time_frames, n_features, 1)
  TimeDistributed(Conv1D(64)) → BN → ReLU → Dropout
  TimeDistributed(Conv1D(128))→ BN → ReLU → Dropout
  TimeDistributed(GlobalMaxPool1D)
  Bidirectional LSTM(128) → Dropout
  Bidirectional LSTM(64)  → Dropout
  Dense(128) → ReLU → Dropout
  Dense(num_classes) → Softmax

Why this works
--------------
1. CNN layers extract local frequency-domain patterns at each time frame.
2. LSTM layers model how those patterns evolve over time.
3. Together: best spectral + temporal modeling.
"""

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers


def build_cnn_lstm_model(
    input_shape: tuple,
    num_classes: int,
    cnn_filters: list = None,
    lstm_units: list = None,
    dropout_rate: float = 0.4,
    l2_reg: float = 1e-4,
) -> tf.keras.Model:
    """
    Build the CNN-LSTM hybrid model.

    Parameters
    ----------
    input_shape  : (n_features, time_frames) — e.g. (222, 130) for all features
    num_classes  : number of emotion categories
    cnn_filters  : list of filter counts for each TimeDistributed Conv1D block
    lstm_units   : list of unit counts per BiLSTM layer
    dropout_rate : dropout probability
    l2_reg       : L2 regularization coefficient

    Returns
    -------
    Compiled Keras model.
    """
    if cnn_filters is None:
        cnn_filters = [64, 128]
    if lstm_units is None:
        lstm_units = [128, 64]

    n_features, time_frames = input_shape

    # ── Input: (batch, n_features, time_frames) ─────────────
    inputs = layers.Input(shape=input_shape, name="feature_map_input")

    # Transpose to (batch, time_frames, n_features) for TimeDistributed
    x = layers.Permute((2, 1), name="transpose")(inputs)

    # Add channel dim: (batch, time_frames, n_features, 1)
    x = layers.Reshape((time_frames, n_features, 1), name="reshape")(x)

    # ── TimeDistributed CNN blocks ───────────────────────────
    # Each time frame is processed independently through the same CNN.
    # Conv1D along the frequency axis at each time step.
    for i, filters in enumerate(cnn_filters):
        x = layers.TimeDistributed(
            layers.Conv1D(
                filters, kernel_size=3, padding="same",
                kernel_regularizer=regularizers.l2(l2_reg),
            ),
            name=f"td_conv_{i+1}"
        )(x)
        x = layers.TimeDistributed(
            layers.BatchNormalization(),
            name=f"td_bn_{i+1}"
        )(x)
        x = layers.TimeDistributed(
            layers.Activation("relu"),
            name=f"td_relu_{i+1}"
        )(x)
        x = layers.TimeDistributed(
            layers.MaxPooling1D(pool_size=2),
            name=f"td_pool_{i+1}"
        )(x)
        x = layers.TimeDistributed(
            layers.Dropout(dropout_rate * 0.5),
            name=f"td_drop_{i+1}"
        )(x)

    # ── Reduce frequency axis: (batch, time_frames, filters) ─
    x = layers.TimeDistributed(
        layers.GlobalMaxPooling1D(),
        name="td_pool"
    )(x)

    # ── Bidirectional LSTM layers ────────────────────────────
    for i, units in enumerate(lstm_units):
        return_sequences = (i < len(lstm_units) - 1)
        x = layers.Bidirectional(
            layers.LSTM(
                units,
                return_sequences=return_sequences,
                dropout=dropout_rate * 0.5,
                recurrent_dropout=0.0,
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
    x = layers.Dropout(dropout_rate, name="drop_fc1")(x)

    x = layers.Dense(
        64, activation="relu",
        kernel_regularizer=regularizers.l2(l2_reg),
        name="fc2"
    )(x)
    x = layers.Dropout(dropout_rate * 0.5, name="drop_fc2")(x)

    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = models.Model(inputs, outputs, name="CNN_LSTM_EmotionRecognizer")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_cnn_lstm_mel_model(
    mel_shape: tuple,
    num_classes: int,
    dropout_rate: float = 0.4,
    l2_reg: float = 1e-4,
) -> tf.keras.Model:
    """
    Alternative CNN-LSTM model that takes raw Mel-Spectrogram input.
    Uses 2D CNN blocks followed by reshape + LSTM.

    Parameters
    ----------
    mel_shape   : (n_mels, time_frames, 1) e.g. (128, 130, 1)
    num_classes : number of emotion categories

    Returns
    -------
    Compiled Keras model.
    """
    inputs = layers.Input(shape=mel_shape, name="mel_input")
    x = inputs

    # ── 2D CNN blocks ────────────────────────────────────────
    for filters, pool_size in [(32, (2, 2)), (64, (2, 2)), (128, (2, 1))]:
        x = layers.Conv2D(
            filters, (3, 3), padding="same",
            kernel_regularizer=regularizers.l2(l2_reg)
        )(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.MaxPooling2D(pool_size)(x)
        x = layers.Dropout(dropout_rate * 0.5)(x)

    # ── Reshape for LSTM: merge freq+channel axes ────────────
    # x shape: (batch, reduced_mels, time_frames', filters)
    shape = x.shape
    # Merge last two dims: (batch, reduced_mels, time_frames', filters)
    # → permute to (batch, time_frames', reduced_mels * filters)
    x = layers.Permute((2, 1, 3))(x)  # (batch, T', F', C)
    x = layers.Reshape(
        (x.shape[1], x.shape[2] * x.shape[3]),
        name="reshape_for_lstm"
    )(x)

    # ── LSTM layers ──────────────────────────────────────────
    x = layers.Bidirectional(
        layers.LSTM(128, return_sequences=True, dropout=0.2)
    )(x)
    x = layers.Dropout(dropout_rate)(x)

    x = layers.Bidirectional(
        layers.LSTM(64, return_sequences=False, dropout=0.2)
    )(x)
    x = layers.Dropout(dropout_rate)(x)

    # ── Head ─────────────────────────────────────────────────
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = models.Model(inputs, outputs, name="CNN2D_LSTM_EmotionRecognizer")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


if __name__ == "__main__":
    # Primary model: all features stacked
    model = build_cnn_lstm_model(input_shape=(262, 130), num_classes=8)
    model.summary()

    print("\n--- Mel-only variant ---")
    model2 = build_cnn_lstm_mel_model(mel_shape=(128, 130, 1), num_classes=8)
    model2.summary()

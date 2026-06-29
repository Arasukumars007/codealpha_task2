"""
cnn_model.py
============
2-D Convolutional Neural Network for emotion recognition from speech.

Input : Mel-Spectrogram (n_mels, time_frames, 1)
Output: Softmax probabilities over emotion classes

Architecture
------------
  Conv2D(32) → BN → ReLU → MaxPool → Dropout
  Conv2D(64) → BN → ReLU → MaxPool → Dropout
  Conv2D(128)→ BN → ReLU → MaxPool → Dropout
  GlobalAveragePooling2D
  Dense(256) → ReLU → Dropout
  Dense(num_classes) → Softmax
"""

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers


def build_cnn_model(
    input_shape: tuple,
    num_classes: int,
    dropout_rate: float = 0.4,
    l2_reg: float = 1e-4,
) -> tf.keras.Model:
    """
    Build a 2-D CNN model for spectrogram-based emotion classification.

    Parameters
    ----------
    input_shape : (n_mels, time_frames, 1) — shape of one input sample
    num_classes : number of emotion categories
    dropout_rate: dropout probability after each block
    l2_reg      : L2 regularization coefficient for Conv and Dense layers

    Returns
    -------
    Compiled Keras model.
    """
    inputs = layers.Input(shape=input_shape, name="mel_spectrogram_input")
    x = inputs

    # ── Block 1 ─────────────────────────────────────────────
    x = layers.Conv2D(
        32, (3, 3), padding="same",
        kernel_regularizer=regularizers.l2(l2_reg),
        name="conv1"
    )(x)
    x = layers.BatchNormalization(name="bn1")(x)
    x = layers.Activation("relu", name="relu1")(x)
    x = layers.MaxPooling2D((2, 2), name="pool1")(x)
    x = layers.Dropout(dropout_rate, name="drop1")(x)

    # ── Block 2 ─────────────────────────────────────────────
    x = layers.Conv2D(
        64, (3, 3), padding="same",
        kernel_regularizer=regularizers.l2(l2_reg),
        name="conv2"
    )(x)
    x = layers.BatchNormalization(name="bn2")(x)
    x = layers.Activation("relu", name="relu2")(x)
    x = layers.MaxPooling2D((2, 2), name="pool2")(x)
    x = layers.Dropout(dropout_rate, name="drop2")(x)

    # ── Block 3 ─────────────────────────────────────────────
    x = layers.Conv2D(
        128, (3, 3), padding="same",
        kernel_regularizer=regularizers.l2(l2_reg),
        name="conv3"
    )(x)
    x = layers.BatchNormalization(name="bn3")(x)
    x = layers.Activation("relu", name="relu3")(x)
    x = layers.MaxPooling2D((2, 2), name="pool3")(x)
    x = layers.Dropout(dropout_rate, name="drop3")(x)

    # ── Block 4 (extra depth) ────────────────────────────────
    x = layers.Conv2D(
        256, (3, 3), padding="same",
        kernel_regularizer=regularizers.l2(l2_reg),
        name="conv4"
    )(x)
    x = layers.BatchNormalization(name="bn4")(x)
    x = layers.Activation("relu", name="relu4")(x)
    x = layers.Dropout(dropout_rate, name="drop4")(x)

    # ── Pooling + Classification head ───────────────────────
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dense(
        256, activation="relu",
        kernel_regularizer=regularizers.l2(l2_reg),
        name="fc1"
    )(x)
    x = layers.Dropout(dropout_rate, name="drop_fc")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = models.Model(inputs, outputs, name="CNN_EmotionRecognizer")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


if __name__ == "__main__":
    model = build_cnn_model(input_shape=(128, 130, 1), num_classes=8)
    model.summary()

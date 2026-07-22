"""
Model Architecture — ECGNet-SE-BiGRU
======================================
1-D CNN feature extractor + Squeeze-and-Excitation attention +
Bidirectional GRU temporal encoder + fully connected classification head.
"""

import tensorflow as tf
from keras import layers, models


def squeeze_excitation_block(
    input_tensor: tf.Tensor,
    reduction_ratio: int = 8,
) -> tf.Tensor:
    """
    1-D Squeeze-and-Excitation attention block.
    Input shape : (Batch, Length, Channels)
    Output shape: (Batch, Length, Channels)
    """
    num_channels = input_tensor.shape[-1]

    # Squeeze: global context descriptor
    context_vector = layers.GlobalAveragePooling1D()(input_tensor)

    # Excitation: two-layer bottleneck MLP
    excitation = layers.Dense(
        num_channels // reduction_ratio,
        activation="relu",
        kernel_initializer="he_normal",
    )(context_vector)
    excitation = layers.Dense(num_channels, activation="sigmoid")(excitation)
    excitation = layers.Reshape((1, num_channels))(excitation)

    # Feature recalibration
    recalibrated = layers.Multiply()([input_tensor, excitation])
    return recalibrated


def build_ecg_classifier(
    input_shape: tuple = (2049, 1),
    num_classes: int = 3,
) -> tf.keras.Model:
    """
    1-D CNN with SE-Attention and Bidirectional GRU head.

    Stage 1 : Conv1D(32,  k=7) → BN → ReLU → MaxPool(2) → Dropout(0.2)
    Stage 2 : Conv1D(64,  k=5) → BN → ReLU → MaxPool(2) → Dropout(0.2)
    Stage 3 : Conv1D(128, k=3) → BN → ReLU → MaxPool(2) → Dropout(0.2)
    Stage 4 : Conv1D(256, k=3) → BN → ReLU → SE-Block
    Head    : BiGRU(64) → Dense(128) → Dropout(0.5) → Softmax(num_classes)
    """
    inputs = layers.Input(shape=input_shape)

    # Stage 1
    x = layers.Conv1D(32, kernel_size=7, padding="same", kernel_initializer="he_normal")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Dropout(0.2)(x)

    # Stage 2
    x = layers.Conv1D(64, kernel_size=5, padding="same", kernel_initializer="he_normal")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Dropout(0.2)(x)

    # Stage 3
    x = layers.Conv1D(128, kernel_size=3, padding="same", kernel_initializer="he_normal")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Dropout(0.2)(x)

    # Stage 4
    x = layers.Conv1D(256, kernel_size=3, padding="same", kernel_initializer="he_normal")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    # Attention module
    x = squeeze_excitation_block(x, reduction_ratio=8)

    # Sequence encoder
    x = layers.Bidirectional(layers.GRU(64, return_sequences=False))(x)

    # Classification head
    x = layers.Dense(128, activation="relu", kernel_initializer="he_normal")(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    return models.Model(inputs=inputs, outputs=outputs, name="ECGNet_SE_BiGRU")

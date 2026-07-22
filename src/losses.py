"""
Custom Loss Function — Focal Loss with Label Smoothing
=========================================================
Used to address class imbalance and emphasize learning on
difficult / underrepresented samples during training.
"""

import tensorflow as tf


def focal_loss(gamma: float = 3.0, alpha: float = 1.0, label_smoothing: float = 0.05):
    """
    Multiclass Focal Loss with label smoothing.

    Parameters
    ----------
    gamma          : focusing exponent (higher → harder examples weighted more)
    alpha          : scaling factor
    label_smoothing: soft label coefficient
    """
    def focal_loss_fixed(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        y_true = tf.cast(y_true, tf.float32)
        num_classes = tf.cast(tf.shape(y_true)[-1], tf.float32)
        y_true = y_true * (1.0 - label_smoothing) + (label_smoothing / num_classes)
        y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1.0 - tf.keras.backend.epsilon())
        cross_entropy = -y_true * tf.math.log(y_pred)
        focal_weight = alpha * tf.math.pow(1.0 - y_pred, gamma)
        return tf.reduce_sum(focal_weight * cross_entropy, axis=-1)

    return focal_loss_fixed

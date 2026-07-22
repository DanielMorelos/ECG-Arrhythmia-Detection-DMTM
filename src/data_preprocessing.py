"""
Data Loading and Preprocessing
===============================
Handles HDF5 loading of ECG segments/labels/patient groups and
per-sample Z-score normalization, exactly as used in the original
training pipeline.
"""

import numpy as np
import h5py
import tensorflow as tf


def load_train_data(train_data_path: str):
    """
    Load training data from an HDF5 file.

    Expects the file to contain the datasets: "data", "labels", "patient".

    Parameters
    ----------
    train_data_path : str
        Path to the training HDF5 file.

    Returns
    -------
    X_train : np.ndarray, shape (N, 2049)
    y_train : np.ndarray, shape (N,)
    patient_groups : np.ndarray, shape (N,)
    """
    with h5py.File(train_data_path, "r") as h5_file:
        X_train = np.array(h5_file["data"], dtype=np.float32)
        y_train = np.squeeze(np.array(h5_file["labels"], dtype=np.int32))
        patient_groups = np.squeeze(np.array(h5_file["patient"], dtype=np.int32))
    return X_train, y_train, patient_groups


def load_test_data(test_data_path: str):
    """
    Load unlabeled test data from an HDF5 file.

    Expects the file to contain the dataset: "data".

    Parameters
    ----------
    test_data_path : str
        Path to the test HDF5 file.

    Returns
    -------
    X_test_raw : np.ndarray, shape (N, 2049)
    """
    with h5py.File(test_data_path, "r") as h5_file:
        X_test_raw = np.array(h5_file["data"], dtype=np.float32)
    return X_test_raw


def z_score_normalize(X: np.ndarray) -> np.ndarray:
    """Per-sample Z-score normalisation along the time axis."""
    mean = X.mean(axis=1, keepdims=True)
    std = X.std(axis=1, keepdims=True) + 1e-8
    return (X - mean) / std


def prepare_datasets(X_train_raw: np.ndarray, y_train: np.ndarray, X_test_raw: np.ndarray, num_classes: int):
    """
    Apply Z-score normalization, add the channel dimension, and one-hot
    encode the training labels — exactly as done in the original pipeline.

    Parameters
    ----------
    X_train_raw : np.ndarray, shape (N, 2049)
    y_train : np.ndarray, shape (N,)
    X_test_raw : np.ndarray, shape (M, 2049)
    num_classes : int

    Returns
    -------
    X_train : np.ndarray, shape (N, 2049)          normalized, 2D (kept for oversampling step)
    X_train_3d : np.ndarray, shape (N, 2049, 1)     normalized, with channel dim
    X_test_3d : np.ndarray, shape (M, 2049, 1)      normalized, with channel dim
    y_train_onehot : np.ndarray, shape (N, num_classes)
    """
    X_train = z_score_normalize(X_train_raw)
    X_test_raw = z_score_normalize(X_test_raw)

    # Add channel dimension → (N, 2049, 1)
    X_train_3d = np.expand_dims(X_train, axis=-1) if X_train.ndim == 2 else X_train
    X_test_3d = np.expand_dims(X_test_raw, axis=-1) if X_test_raw.ndim == 2 else X_test_raw

    # One-hot labels
    y_train_onehot = tf.keras.utils.to_categorical(y_train, num_classes=num_classes)

    return X_train, X_train_3d, X_test_3d, y_train_onehot

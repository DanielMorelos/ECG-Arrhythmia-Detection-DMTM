"""
Custom Keras Callbacks
========================
- TrainingStateCheckpoint : persists weights + optimizer state per epoch (resume support)
- F1MacroCallback         : computes val_f1_macro at the end of every epoch
- HistoryPersistenceCallback : appends epoch metrics to a JSON file (resume-safe history)
"""

import os
import json
import pickle

import numpy as np
import tensorflow as tf
from sklearn.metrics import f1_score


class TrainingStateCheckpoint(tf.keras.callbacks.Callback):
    """
    Persists model weights and optimiser state at each epoch end
    to allow seamless training resumption after interruptions.
    """

    def __init__(self, checkpoint_dir: str, fold_index: int):
        super().__init__()
        self.checkpoint_dir = checkpoint_dir
        self.fold_index = fold_index
        self.state_path = os.path.join(checkpoint_dir, f"training_state_fold_{fold_index}.pkl")

    def on_epoch_end(self, epoch: int, logs: dict = None):
        weights_path = os.path.join(
            self.checkpoint_dir, f"resume_weights_fold_{self.fold_index}.weights.h5"
        )
        self.model.save_weights(weights_path)

        state = {
            "epoch": epoch + 1,
            "fold": self.fold_index,
            "logs": logs,
            "lr": float(self.model.optimizer.learning_rate),
        }
        with open(self.state_path, "wb") as fp:
            pickle.dump(state, fp)


class F1MacroCallback(tf.keras.callbacks.Callback):
    """
    Computes and logs val_f1_macro at the end of each epoch.
    Injecting it into `logs` makes it available to all other callbacks
    (e.g. ModelCheckpoint, ReduceLROnPlateau) and to history.history.

    Parameters
    ----------
    val_data : tuple (X_val, y_val_onehot)
    """

    def __init__(self, val_data: tuple):
        super().__init__()
        self.X_val = val_data[0]
        self.y_val = val_data[1]  # one-hot

    def on_epoch_end(self, epoch: int, logs: dict = None):
        val_probs = self.model.predict(self.X_val, verbose=0)
        val_preds = np.argmax(val_probs, axis=-1)
        val_true = np.argmax(self.y_val, axis=-1)
        score = f1_score(val_true, val_preds, average="macro", zero_division=0)
        logs["val_f1_macro"] = score


class HistoryPersistenceCallback(tf.keras.callbacks.Callback):
    """
    Appends epoch metrics to a JSON file after every epoch so that
    training history survives Colab disconnections and multi-account runs.

    The JSON stores lists for every key in `logs` (loss, accuracy,
    val_loss, val_accuracy, val_f1_macro, …).  On resumption the new
    epochs are appended rather than overwriting previous data.

    Parameters
    ----------
    history_path : full path to the target .json file
    """

    def __init__(self, history_path: str):
        super().__init__()
        self.history_path = history_path

    def on_epoch_end(self, epoch: int, logs: dict = None):
        logs = logs or {}

        # Load existing data if the file already exists (resumed run)
        if os.path.exists(self.history_path):
            with open(self.history_path, "r") as fp:
                stored = json.load(fp)
        else:
            stored = {}

        # Append current epoch values
        for key, value in logs.items():
            stored.setdefault(key, []).append(float(value))

        with open(self.history_path, "w") as fp:
            json.dump(stored, fp, indent=2)

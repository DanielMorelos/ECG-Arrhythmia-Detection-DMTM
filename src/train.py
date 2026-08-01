"""
Training Script — 5-Fold Stratified Cross-Validation
=======================================================
Trains ECGNet-SE-BiGRU with Focal Loss across 5 stratified folds,
using RandomOverSampler on each training fold to address class imbalance.
Model checkpoints and training histories are persisted per fold to allow
seamless resumption after interruptions.

Usage
-----
python src/train.py \
    --train-data path/to/training_set.h5 \
    --test-data path/to/test_set.h5 \
    --checkpoint-dir path/to/checkpoints/
"""

import os
import argparse

import numpy as np
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import RandomOverSampler

from data_preprocessing import load_train_data, load_test_data, prepare_datasets
from model import build_ecg_classifier
from losses import focal_loss
from callbacks import TrainingStateCheckpoint, F1MacroCallback, HistoryPersistenceCallback


def parse_args():
    parser = argparse.ArgumentParser(description="Train ECGNet-SE-BiGRU with 5-fold stratified CV.")
    parser.add_argument("--train-data", type=str, required=True, help="Path to training HDF5 file.")
    parser.add_argument("--test-data", type=str, required=True, help="Path to test HDF5 file (unlabeled).")
    parser.add_argument("--checkpoint-dir", type=str, required=True, help="Directory to store checkpoints/history.")
    parser.add_argument("--input-length", type=int, default=2049, help="ECG segment length (time steps).")
    parser.add_argument("--num-classes", type=int, default=3, help="Number of target classes.")
    parser.add_argument("--n-folds", type=int, default=5, help="Number of stratified CV folds.")
    parser.add_argument("--batch-size", type=int, default=32, help="Training batch size.")
    parser.add_argument("--learning-rate", type=float, default=0.001, help="Initial learning rate (AdamW).")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay for AdamW.")
    parser.add_argument("--max-epochs", type=int, default=40, help="Maximum training epochs per fold.")
    parser.add_argument("--lr-patience", type=int, default=3, help="Patience for ReduceLROnPlateau.")
    parser.add_argument("--early-stop-patience", type=int, default=50, help="Patience for EarlyStopping.")
    parser.add_argument("--random-state", type=int, default=42, help="Random state for StratifiedKFold and oversampling.")
    return parser.parse_args()


def main():
    args = parse_args()

    INPUT_SHAPE = (args.input_length, 1)
    NUM_CLASSES = args.num_classes
    N_FOLDS = args.n_folds
    BATCH_SIZE = args.batch_size
    LEARNING_RATE = args.learning_rate
    MAX_EPOCHS = args.max_epochs
    LR_PATIENCE = args.lr_patience
    EARLY_STOP_PATIENCE = args.early_stop_patience

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # ── Data loading ──────────────────────────────────────────────────────
    print("Loading data from HDF5 files …")
    X_train_raw, y_train, patient_groups = load_train_data(args.train_data)
    X_test_raw = load_test_data(args.test_data)
    print(f"Dataset loaded — train: {X_train_raw.shape[0]} samples | test: {X_test_raw.shape[0]} samples")

    # ── Preprocessing ─────────────────────────────────────────────────────
    X_train, X_train_3d, X_test_3d, y_train_onehot = prepare_datasets(
        X_train_raw, y_train, X_test_raw, NUM_CLASSES
    )

    # ── Cross-validation ──────────────────────────────────────────────────
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=args.random_state)
    oof_probabilities = np.zeros((X_train.shape[0], NUM_CLASSES))

    print(f"\nStarting {N_FOLDS}-Fold cross-validation with Focal Loss …\n")

    for fold_idx, (train_indices, val_indices) in enumerate(skf.split(X_train, y_train)):
        print(f"\n{'─'*60}")
        print(f"  FOLD {fold_idx + 1} / {N_FOLDS}")
        print(f"{'─'*60}")

        X_fold_train, y_fold_train = X_train_3d[train_indices], y_train_onehot[train_indices]
        X_fold_val, y_fold_val = X_train_3d[val_indices], y_train_onehot[val_indices]

        # ── Build and compile model ───────────────────────────────────────
        model = build_ecg_classifier(input_shape=INPUT_SHAPE, num_classes=NUM_CLASSES)
        optimizer = tf.keras.optimizers.AdamW(learning_rate=LEARNING_RATE, weight_decay=args.weight_decay)
        model.compile(
            optimizer=optimizer,
            loss=focal_loss(),
            metrics=["accuracy"],
        )

        # ── Resume from checkpoint if available ───────────────────────────
        state_path = os.path.join(args.checkpoint_dir, f"training_state_fold_{fold_idx + 1}.pkl")
        weights_path = os.path.join(args.checkpoint_dir, f"resume_weights_fold_{fold_idx + 1}.weights.h5")
        start_epoch = 0

        if os.path.exists(state_path) and os.path.exists(weights_path):
            import pickle
            with open(state_path, "rb") as fp:
                saved_state = pickle.load(fp)
            start_epoch = saved_state["epoch"]
            model.load_weights(weights_path)
            model.optimizer.learning_rate.assign(saved_state["lr"])
            print(f"  Resuming from epoch {start_epoch}")
        else:
            print("  Starting from scratch")

        # ── Callbacks ──────────────────────────────────────────────────────
        best_model_path = os.path.join(args.checkpoint_dir, f"best_model_fold_{fold_idx + 1}.keras")
        history_path = os.path.join(args.checkpoint_dir, f"history_fold_{fold_idx + 1}.json")

        callbacks = [
            F1MacroCallback(val_data=(X_fold_val, y_fold_val)),  # must be first -> injects val_f1_macro into logs
            tf.keras.callbacks.ModelCheckpoint(
                filepath=best_model_path,
                monitor="val_loss",
                save_best_only=True,
                verbose=1,
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=LR_PATIENCE,
                verbose=1,
                min_lr=1e-6,
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=EARLY_STOP_PATIENCE,
                verbose=1,
                restore_best_weights=True,
            ),
            TrainingStateCheckpoint(checkpoint_dir=args.checkpoint_dir, fold_index=fold_idx + 1),
            HistoryPersistenceCallback(history_path=history_path),  # must be last -> all metrics already in logs
        ]

        # ── Class-balance oversampling ─────────────────────────────────────
        oversampler = RandomOverSampler(random_state=args.random_state)
        X_flat = X_fold_train.reshape(X_fold_train.shape[0], -1)
        X_resampled, y_resampled_int = oversampler.fit_resample(
            X_flat, np.argmax(y_fold_train, axis=-1)
        )
        X_resampled = X_resampled.reshape(-1, INPUT_SHAPE[0], 1)
        y_resampled = tf.keras.utils.to_categorical(y_resampled_int, num_classes=NUM_CLASSES)

        # ── Fit (on the oversampled training fold; validation stays untouched) ─
        model.fit(
            X_resampled, y_resampled,
            validation_data=(X_fold_val, y_fold_val),
            epochs=MAX_EPOCHS,
            initial_epoch=start_epoch,
            batch_size=BATCH_SIZE,
            callbacks=callbacks,
            verbose=1,
        )

        # ── OOF predictions ────────────────────────────────────────────────
        best_model = tf.keras.models.load_model(
            best_model_path,
            custom_objects={"focal_loss_fixed": focal_loss()},
        )
        oof_probabilities[val_indices] = best_model.predict(X_fold_val, batch_size=BATCH_SIZE)

    print("\nCross-validation training complete.")
    print(f"Checkpoints and histories saved to: {args.checkpoint_dir}")


if __name__ == "__main__":
    main()
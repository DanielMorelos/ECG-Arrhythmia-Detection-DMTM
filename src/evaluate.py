"""
Evaluation Script
===================
Reloads the best per-fold checkpoints from cross-validation, evaluates
OOF performance, fine-tunes the last 3 layers of each fold model,
re-evaluates the fine-tuned ensemble, and generates training-curve and
performance-dashboard figures.

Persists the OOF probability matrix, OOF true labels, and per-fold test
set probabilities to disk (as .npy files) so that threshold_optimization.py
can load them without needing to re-run inference.

Usage
-----
python src/evaluate.py \
    --train-data path/to/training_set.h5 \
    --test-data path/to/test_set.h5 \
    --checkpoint-dir path/to/checkpoints/
"""

import os
import json
import argparse

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

from data_preprocessing import load_train_data, load_test_data, prepare_datasets
from losses import focal_loss
from callbacks import F1MacroCallback, HistoryPersistenceCallback

CLASS_NAMES = ["Normal", "Arrhythmia", "Noise"]


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate, fine-tune, and ensemble ECGNet-SE-BiGRU cross-validation models.")
    parser.add_argument("--train-data", type=str, required=True, help="Path to training HDF5 file.")
    parser.add_argument("--test-data", type=str, required=True, help="Path to test HDF5 file (unlabeled).")
    parser.add_argument("--checkpoint-dir", type=str, required=True, help="Directory containing fold checkpoints.")
    parser.add_argument("--input-length", type=int, default=2049, help="ECG segment length (time steps).")
    parser.add_argument("--num-classes", type=int, default=3, help="Number of target classes.")
    parser.add_argument("--n-folds", type=int, default=5, help="Number of stratified CV folds.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for prediction.")
    parser.add_argument("--random-state", type=int, default=42, help="Random state for StratifiedKFold (must match training).")
    parser.add_argument("--finetune-epochs", type=int, default=15, help="Max epochs for fine-tuning stage.")
    parser.add_argument("--finetune-lr", type=float, default=1e-5, help="Learning rate for fine-tuning stage.")
    parser.add_argument("--finetune-patience", type=int, default=5, help="EarlyStopping patience for fine-tuning.")
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING CURVES — LOSS, ACCURACY AND F1 PER FOLD
# ─────────────────────────────────────────────────────────────────────────────
def plot_training_curves(checkpoint_dir: str, n_folds: int, suffix: str = "", title_prefix: str = "Training") -> None:
    """
    Loads the persisted JSON history for every fold and renders a
    3-row x N_FOLDS grid showing Loss, Accuracy and F1 Macro curves.
    """
    fig, axes = plt.subplots(3, n_folds, figsize=(5 * n_folds, 12))
    fig.suptitle(f"{title_prefix} Curves per Fold — Loss · Accuracy · F1 Macro", fontsize=14, fontweight="bold")

    colors = {"train": "#2c3e50", "val": "#e74c3c"}

    for fold_idx in range(n_folds):
        history_path = os.path.join(checkpoint_dir, f"history_fold_{fold_idx + 1}{suffix}.json")

        if not os.path.exists(history_path):
            for row in range(3):
                axes[row, fold_idx].set_visible(False)
            continue

        with open(history_path, "r") as fp:
            hist = json.load(fp)

        epochs = range(1, len(hist.get("loss", [])) + 1)

        ax = axes[0, fold_idx]
        ax.plot(epochs, hist.get("loss", []), color=colors["train"], label="Train")
        ax.plot(epochs, hist.get("val_loss", []), color=colors["val"], label="Val", linestyle="--")
        ax.set_title(f"Fold {fold_idx + 1} — Loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

        ax = axes[1, fold_idx]
        ax.plot(epochs, hist.get("accuracy", []), color=colors["train"], label="Train")
        ax.plot(epochs, hist.get("val_accuracy", []), color=colors["val"], label="Val", linestyle="--")
        ax.set_title(f"Fold {fold_idx + 1} — Accuracy")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy")
        ax.set_ylim([0, 1.0])
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

        ax = axes[2, fold_idx]
        val_f1 = hist.get("val_f1_macro", [])
        if val_f1:
            ax.plot(epochs, val_f1, color=colors["val"], label="Val F1 Macro", linestyle="--")
            best_epoch = int(np.argmax(val_f1)) + 1
            ax.axvline(best_epoch, color="#27ae60", linestyle=":", alpha=0.7, label=f"Best epoch ({best_epoch})")
            ax.axhline(max(val_f1), color="#8e44ad", linestyle=":", alpha=0.5, label=f"Peak ({max(val_f1):.4f})")
        ax.set_title(f"Fold {fold_idx + 1} — Val F1 Macro")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("F1 Macro")
        ax.set_ylim([0, 1.0])
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    filename = "training_curves.png" if not suffix else "finetuning_curves.png"
    save_path = os.path.join(checkpoint_dir, filename)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Training curves saved -> {save_path}")


def main():
    args = parse_args()

    INPUT_SHAPE = (args.input_length, 1)
    NUM_CLASSES = args.num_classes
    N_FOLDS = args.n_folds
    BATCH_SIZE = args.batch_size

    # ── Data loading ──────────────────────────────────────────────────────
    X_train_raw, y_train, patient_groups = load_train_data(args.train_data)
    X_test_raw = load_test_data(args.test_data)
    X_train, X_train_3d, X_test_3d, y_train_onehot = prepare_datasets(
        X_train_raw, y_train, X_test_raw, NUM_CLASSES
    )

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=args.random_state)

    # ── Training curves for the initial training stage ────────────────────
    plot_training_curves(args.checkpoint_dir, N_FOLDS, suffix="", title_prefix="Training")

    # ─────────────────────────────────────────────────────────────────────
    # EVALUATION — RELOAD BEST CHECKPOINTS
    # ─────────────────────────────────────────────────────────────────────
    print("\n\nReloading best checkpoints for evaluation …")

    for fold_idx, (train_indices, val_indices) in enumerate(skf.split(X_train, y_train)):
        checkpoint_path = os.path.join(args.checkpoint_dir, f"best_model_fold_{fold_idx + 1}.keras")

        if not os.path.exists(checkpoint_path):
            print(f"  Checkpoint for fold {fold_idx + 1} not found — skipping.")
            continue

        eval_model = tf.keras.models.load_model(
            checkpoint_path,
            custom_objects={"focal_loss_fixed": focal_loss(gamma=2.0, alpha=1.0, label_smoothing=0.05)},
        )

        X_fold_val = X_train_3d[val_indices]
        y_fold_val_true = y_train[val_indices]

        val_probs = eval_model.predict(X_fold_val, batch_size=BATCH_SIZE, verbose=0)
        val_preds = np.argmax(val_probs, axis=-1)
        macro_f1 = f1_score(y_fold_val_true, val_preds, average="macro")

        print(f"\n{'='*50}")
        print(f"  Fold {fold_idx + 1} — Validation Results")
        print(f"{'='*50}")
        print(f"  F1 Macro : {macro_f1:.4f}")
        print(classification_report(y_fold_val_true, val_preds, target_names=CLASS_NAMES))

    # ─────────────────────────────────────────────────────────────────────
    # FINE-TUNING — LAST-LAYER ADAPTATION
    # ─────────────────────────────────────────────────────────────────────
    print("\n\nFine-tuning best models (last 3 layers unfrozen) …")

    for fold_idx, (train_indices, val_indices) in enumerate(skf.split(X_train, y_train)):
        base_ckpt = os.path.join(args.checkpoint_dir, f"best_model_fold_{fold_idx + 1}.keras")
        finetuned_ckpt = os.path.join(args.checkpoint_dir, f"best_model_fold_{fold_idx + 1}_finetuned.keras")
        ft_history_path = os.path.join(args.checkpoint_dir, f"history_fold_{fold_idx + 1}_finetuned.json")

        if not os.path.exists(base_ckpt):
            print(f"  Base checkpoint for fold {fold_idx + 1} not found — skipping.")
            continue

        pretrained_model = tf.keras.models.load_model(
            base_ckpt,
            custom_objects={"focal_loss_fixed": focal_loss(gamma=3.0, alpha=1.0, label_smoothing=0.05)},
        )

        # Freeze all layers except the last three
        for layer in pretrained_model.layers[:-3]:
            layer.trainable = False

        pretrained_model.compile(
            optimizer=tf.keras.optimizers.AdamW(learning_rate=args.finetune_lr, weight_decay=0.01),
            loss=focal_loss(gamma=3.0, alpha=1.0, label_smoothing=0.05),
            metrics=["accuracy"],
        )

        X_fold_train, y_fold_train = X_train_3d[train_indices], y_train_onehot[train_indices]
        X_fold_val, y_fold_val = X_train_3d[val_indices], y_train_onehot[val_indices]

        pretrained_model.fit(
            X_fold_train, y_fold_train,
            validation_data=(X_fold_val, y_fold_val),
            epochs=args.finetune_epochs,
            batch_size=BATCH_SIZE,
            callbacks=[
                F1MacroCallback(val_data=(X_fold_val, y_fold_val)),
                tf.keras.callbacks.ModelCheckpoint(finetuned_ckpt, monitor="val_loss", save_best_only=True, verbose=1),
                tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=args.finetune_patience, restore_best_weights=True),
                HistoryPersistenceCallback(history_path=ft_history_path),
            ],
            verbose=1,
        )
        print(f"  Fold {fold_idx + 1} fine-tuned model saved: {finetuned_ckpt}")

    # Fine-tuning curves
    plot_training_curves(args.checkpoint_dir, N_FOLDS, suffix="_finetuned", title_prefix="Fine-Tuning")

    # ─────────────────────────────────────────────────────────────────────
    # ENSEMBLE EVALUATION — OOF METRICS WITH FINE-TUNED MODELS
    # ─────────────────────────────────────────────────────────────────────
    oof_prob_matrix = np.zeros((X_train.shape[0], NUM_CLASSES), dtype=np.float32)
    oof_true_labels = np.zeros(X_train.shape[0], dtype=np.int32)
    ensemble_test_probs = []

    fold_f1_scores = []
    fold_class_f1s = []
    fold_conf_matrices = []

    for fold_idx, (train_indices, val_indices) in enumerate(skf.split(X_train, y_train)):
        finetuned_ckpt = os.path.join(args.checkpoint_dir, f"best_model_fold_{fold_idx + 1}_finetuned.keras")
        base_ckpt = os.path.join(args.checkpoint_dir, f"best_model_fold_{fold_idx + 1}.keras")
        checkpoint_to_load = finetuned_ckpt if os.path.exists(finetuned_ckpt) else base_ckpt

        if not os.path.exists(checkpoint_to_load):
            print(f"  No checkpoint found for fold {fold_idx + 1} — skipping.")
            continue

        model = tf.keras.models.load_model(
            checkpoint_to_load,
            custom_objects={"focal_loss_fixed": focal_loss(gamma=3.0, alpha=1.0, label_smoothing=0.05)},
        )

        X_fold_val = X_train_3d[val_indices]
        y_fold_val_true = y_train[val_indices]

        fold_val_probs = model.predict(X_fold_val, batch_size=BATCH_SIZE, verbose=0)
        fold_val_preds = np.argmax(fold_val_probs, axis=-1)

        oof_prob_matrix[val_indices] = fold_val_probs
        oof_true_labels[val_indices] = y_fold_val_true

        macro_f1 = f1_score(y_fold_val_true, fold_val_preds, average="macro")
        class_f1s = f1_score(y_fold_val_true, fold_val_preds, average=None)
        conf_matrix = confusion_matrix(y_fold_val_true, fold_val_preds)

        fold_f1_scores.append(macro_f1)
        fold_class_f1s.append(class_f1s)
        fold_conf_matrices.append(conf_matrix)

        print(f"\n  Fold {fold_idx + 1} — F1 Macro: {macro_f1:.4f}")
        print(classification_report(y_fold_val_true, fold_val_preds, target_names=CLASS_NAMES))

        ensemble_test_probs.append(model.predict(X_test_3d, batch_size=BATCH_SIZE, verbose=0))

    oof_predictions = np.argmax(oof_prob_matrix, axis=-1)
    global_macro_f1 = f1_score(oof_true_labels, oof_predictions, average="macro")
    global_conf_mat = confusion_matrix(oof_true_labels, oof_predictions)

    print(f"\n{'='*60}")
    print("GLOBAL OOF METRICS")
    print(f"{'='*60}")
    print(f"  F1 Macro (global): {global_macro_f1:.4f}")
    print(classification_report(oof_true_labels, oof_predictions, target_names=CLASS_NAMES))

    # ─────────────────────────────────────────────────────────────────────
    # PERSIST ARTIFACTS FOR THRESHOLD OPTIMIZATION
    # ─────────────────────────────────────────────────────────────────────
    np.save(os.path.join(args.checkpoint_dir, "oof_prob_matrix.npy"), oof_prob_matrix)
    np.save(os.path.join(args.checkpoint_dir, "oof_true_labels.npy"), oof_true_labels)
    np.save(os.path.join(args.checkpoint_dir, "ensemble_test_probs.npy"), np.array(ensemble_test_probs))
    print(f"\nOOF probability matrix, true labels, and ensemble test probabilities saved to: {args.checkpoint_dir}")

    # ─────────────────────────────────────────────────────────────────────
    # PERFORMANCE DASHBOARD
    # ─────────────────────────────────────────────────────────────────────
    n_evaluated_folds = len(fold_f1_scores)
    fold_labels = [f"Fold {i + 1}" for i in range(n_evaluated_folds)]
    fold_class_f1_arr = np.array(fold_class_f1s)

    fig, axes = plt.subplots(3, 3, figsize=(18, 14))
    fig.suptitle("ECG Arrhythmia Classifier — Cross-Validation Performance Dashboard", fontsize=14, fontweight="bold")

    ax = axes[0, 0]
    bars = ax.bar(fold_labels, fold_f1_scores, color="#2c3e50", alpha=0.7)
    ax.axhline(np.mean(fold_f1_scores), color="#e74c3c", linestyle="--", label=f"Mean ({np.mean(fold_f1_scores):.4f})")
    ax.axhline(global_macro_f1, color="#27ae60", linestyle=":", label=f"Global OOF ({global_macro_f1:.4f})")
    for bar, val in zip(bars, fold_f1_scores):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005, f"{val:.3f}", ha="center", fontsize=9)
    ax.set_title("F1 Macro per Fold")
    ax.set_ylim([0.7, 1.0])
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[0, 1]
    x_pos = np.arange(n_evaluated_folds)
    bar_width = 0.25
    ax.bar(x_pos - bar_width, fold_class_f1_arr[:, 0], bar_width, label="Normal", color="#2ecc71", alpha=0.8)
    ax.bar(x_pos, fold_class_f1_arr[:, 1], bar_width, label="Arrhythmia", color="#e67e22", alpha=0.8)
    ax.bar(x_pos + bar_width, fold_class_f1_arr[:, 2], bar_width, label="Noise", color="#3498db", alpha=0.8)
    ax.set_title("Per-Class F1 Score per Fold")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(fold_labels)
    ax.set_ylim([0.7, 1.0])
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[0, 2]
    ConfusionMatrixDisplay(global_conf_mat, display_labels=CLASS_NAMES).plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title("Global OOF Confusion Matrix")

    for i in range(len(fold_conf_matrices)):
        row = 1 + (i // 3)
        col = i % 3
        ax = axes[row, col]
        cm_normalized = fold_conf_matrices[i].astype(float) / fold_conf_matrices[i].sum(axis=1, keepdims=True)
        ConfusionMatrixDisplay(cm_normalized, display_labels=CLASS_NAMES).plot(ax=ax, colorbar=False, cmap="Blues", values_format=".2f")
        ax.set_title(f"Fold {i + 1} — Normalised")

    axes[2, 2].axis("off")

    plt.tight_layout()
    dashboard_path = os.path.join(args.checkpoint_dir, "performance_dashboard.png")
    plt.savefig(dashboard_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"\nDashboard saved: {dashboard_path}")


if __name__ == "__main__":
    main()

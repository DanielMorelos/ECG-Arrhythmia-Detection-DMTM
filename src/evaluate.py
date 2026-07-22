"""
Evaluation, Fine-Tuning, and Ensemble Diagnostics
====================================================
Reloads the per-fold checkpoints produced by train.py, reports held-out
metrics, fine-tunes the classification head of each fold, and rebuilds
the out-of-fold (OOF) ensemble used downstream by threshold_optimization.py.

Pipeline
--------
1. Reload best_model_fold_{i}.keras and report per-fold validation metrics.
2. Fine-tune the last N layers of each fold's best checkpoint.
3. Re-evaluate the ensemble (fine-tuned checkpoint when available, base
   checkpoint otherwise) to build:
       - oof_prob_matrix.npy      (N_train, num_classes)
       - oof_true_labels.npy      (N_train,)
       - ensemble_test_probs.npy  (n_folds, N_test, num_classes)
   These three files are the required inputs of threshold_optimization.py.
4. Render training-curve and performance-dashboard figures to
   checkpoint_dir for visual inspection.

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
from sklearn.metrics import (
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import StratifiedKFold

from data_preprocessing import load_train_data, load_test_data, prepare_datasets
from losses import focal_loss
from callbacks import F1MacroCallback, HistoryPersistenceCallback

CLASS_NAMES = ["Normal", "Arrhythmia", "Noise"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Reload fold checkpoints, fine-tune, and rebuild the OOF ensemble."
    )
    parser.add_argument("--train-data", type=str, required=True, help="Path to training HDF5 file.")
    parser.add_argument("--test-data", type=str, required=True, help="Path to test HDF5 file (unlabeled).")
    parser.add_argument("--checkpoint-dir", type=str, required=True, help="Directory containing fold checkpoints from train.py.")
    parser.add_argument("--input-length", type=int, default=2049, help="ECG segment length (time steps).")
    parser.add_argument("--num-classes", type=int, default=3, help="Number of target classes.")
    parser.add_argument("--n-folds", type=int, default=5, help="Number of stratified CV folds (must match train.py).")
    parser.add_argument("--batch-size", type=int, default=32, help="Inference / fine-tuning batch size.")
    parser.add_argument("--random-state", type=int, default=42, help="Random state for StratifiedKFold (must match train.py).")
    parser.add_argument("--finetune-epochs", type=int, default=15, help="Maximum fine-tuning epochs per fold.")
    parser.add_argument("--finetune-lr", type=float, default=1e-5, help="Learning rate for fine-tuning.")
    parser.add_argument("--finetune-patience", type=int, default=5, help="EarlyStopping patience during fine-tuning.")
    parser.add_argument("--unfrozen-layers", type=int, default=3, help="Number of trailing layers left trainable during fine-tuning.")
    parser.add_argument("--skip-finetune", action="store_true", help="Skip fine-tuning and evaluate base checkpoints only.")
    return parser.parse_args()


def reload_and_evaluate_folds(checkpoint_dir, skf, X_train_3d, X_test_3d, y_train, gamma, batch_size):
    """
    Reloads each fold's best base checkpoint and reports held-out metrics.

    Returns
    -------
    fold_macro_f1_scores : list[float]
    fold_per_class_f1    : list[np.ndarray]
    """
    fold_macro_f1_scores = []
    fold_per_class_f1 = []

    for fold_idx, (_, val_indices) in enumerate(skf.split(X_train_3d, y_train)):
        checkpoint_path = os.path.join(checkpoint_dir, f"best_model_fold_{fold_idx + 1}.keras")
        if not os.path.exists(checkpoint_path):
            print(f"  Checkpoint for fold {fold_idx + 1} not found — skipping.")
            continue

        eval_model = tf.keras.models.load_model(
            checkpoint_path,
            custom_objects={"focal_loss_fixed": focal_loss(gamma=gamma)},
        )

        X_fold_val = X_train_3d[val_indices]
        y_fold_val_true = y_train[val_indices]

        val_probs = eval_model.predict(X_fold_val, batch_size=batch_size, verbose=0)
        val_preds = np.argmax(val_probs, axis=-1)

        macro_f1 = f1_score(y_fold_val_true, val_preds, average="macro")
        per_class_f1 = f1_score(y_fold_val_true, val_preds, average=None)

        fold_macro_f1_scores.append(macro_f1)
        fold_per_class_f1.append(per_class_f1)

        print(f"\n{'='*50}")
        print(f"  Fold {fold_idx + 1} — Validation Results")
        print(f"{'='*50}")
        print(f"  F1 Macro : {macro_f1:.4f}")
        print(classification_report(y_fold_val_true, val_preds, target_names=CLASS_NAMES))

    return fold_macro_f1_scores, fold_per_class_f1


def fine_tune_folds(checkpoint_dir, skf, X_train_3d, y_train_onehot, y_train, args):
    """
    Fine-tunes the last `args.unfrozen_layers` layers of each fold's best
    checkpoint for `args.finetune_epochs` epochs at a reduced learning rate.
    """
    for fold_idx, (train_indices, val_indices) in enumerate(skf.split(X_train_3d, y_train)):
        base_ckpt = os.path.join(checkpoint_dir, f"best_model_fold_{fold_idx + 1}.keras")
        finetuned_ckpt = os.path.join(checkpoint_dir, f"best_model_fold_{fold_idx + 1}_finetuned.keras")
        ft_history_path = os.path.join(checkpoint_dir, f"history_fold_{fold_idx + 1}_finetuned.json")

        if not os.path.exists(base_ckpt):
            print(f"  Base checkpoint for fold {fold_idx + 1} not found — skipping.")
            continue

        pretrained_model = tf.keras.models.load_model(
            base_ckpt,
            custom_objects={"focal_loss_fixed": focal_loss()},
        )

        # Freeze every layer except the trailing `unfrozen_layers`.
        for layer in pretrained_model.layers[: -args.unfrozen_layers]:
            layer.trainable = False

        pretrained_model.compile(
            optimizer=tf.keras.optimizers.AdamW(learning_rate=args.finetune_lr, weight_decay=0.01),
            loss=focal_loss(),
            metrics=["accuracy"],
        )

        X_fold_train = X_train_3d[train_indices]
        y_fold_train = y_train_onehot[train_indices]
        X_fold_val = X_train_3d[val_indices]
        y_fold_val = y_train_onehot[val_indices]

        pretrained_model.fit(
            X_fold_train, y_fold_train,
            validation_data=(X_fold_val, y_fold_val),
            epochs=args.finetune_epochs,
            batch_size=args.batch_size,
            callbacks=[
                F1MacroCallback(val_data=(X_fold_val, y_fold_val)),
                tf.keras.callbacks.ModelCheckpoint(
                    finetuned_ckpt, monitor="val_loss", save_best_only=True, verbose=1
                ),
                tf.keras.callbacks.EarlyStopping(
                    monitor="val_loss", patience=args.finetune_patience, restore_best_weights=True
                ),
                HistoryPersistenceCallback(history_path=ft_history_path),
            ],
            verbose=1,
        )
        print(f"  Fold {fold_idx + 1} fine-tuned model saved: {finetuned_ckpt}")


def build_oof_ensemble(checkpoint_dir, skf, X_train_3d, X_test_3d, y_train, num_classes, batch_size):
    """
    Reloads the fine-tuned checkpoint for each fold (falling back to the
    base checkpoint if fine-tuning was skipped or failed), and assembles
    the OOF probability matrix plus per-fold test-set probabilities.

    Returns
    -------
    oof_prob_matrix     : np.ndarray, shape (N_train, num_classes)
    oof_true_labels     : np.ndarray, shape (N_train,)
    ensemble_test_probs : np.ndarray, shape (n_folds, N_test, num_classes)
    fold_f1_scores      : list[float]
    fold_class_f1s      : list[np.ndarray]
    fold_conf_matrices  : list[np.ndarray]
    """
    oof_prob_matrix = np.zeros((X_train_3d.shape[0], num_classes), dtype=np.float32)
    oof_true_labels = np.zeros(X_train_3d.shape[0], dtype=np.int32)
    ensemble_test_probs = []

    fold_f1_scores = []
    fold_class_f1s = []
    fold_conf_matrices = []

    for fold_idx, (_, val_indices) in enumerate(skf.split(X_train_3d, y_train)):
        finetuned_ckpt = os.path.join(checkpoint_dir, f"best_model_fold_{fold_idx + 1}_finetuned.keras")
        base_ckpt = os.path.join(checkpoint_dir, f"best_model_fold_{fold_idx + 1}.keras")
        checkpoint_to_load = finetuned_ckpt if os.path.exists(finetuned_ckpt) else base_ckpt

        if not os.path.exists(checkpoint_to_load):
            print(f"  No checkpoint found for fold {fold_idx + 1} — skipping.")
            continue

        model = tf.keras.models.load_model(
            checkpoint_to_load,
            custom_objects={"focal_loss_fixed": focal_loss()},
        )

        X_fold_val = X_train_3d[val_indices]
        y_fold_val_true = y_train[val_indices]

        fold_val_probs = model.predict(X_fold_val, batch_size=batch_size, verbose=0)
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

        ensemble_test_probs.append(model.predict(X_test_3d, batch_size=batch_size, verbose=0))

    return (
        oof_prob_matrix,
        oof_true_labels,
        np.array(ensemble_test_probs),
        fold_f1_scores,
        fold_class_f1s,
        fold_conf_matrices,
    )


def plot_training_curves(checkpoint_dir, n_folds, suffix="", title="Training Curves per Fold — Loss · Accuracy · F1 Macro"):
    """
    Loads the persisted JSON history for every fold and renders a
    3-row x n_folds grid of Loss / Accuracy / F1-Macro curves.

    Parameters
    ----------
    suffix : str
        "" for the base training histories (history_fold_{i}.json), or
        "_finetuned" for the fine-tuning histories.
    """
    fig, axes = plt.subplots(3, n_folds, figsize=(5 * n_folds, 12))
    fig.suptitle(title, fontsize=14, fontweight="bold")

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
    filename = "training_curves.png" if suffix == "" else "finetuning_curves.png"
    save_path = os.path.join(checkpoint_dir, filename)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Curves saved → {save_path}")


def plot_performance_dashboard(checkpoint_dir, fold_f1_scores, fold_class_f1s, global_conf_mat, global_macro_f1):
    """
    Renders a 2-row performance dashboard: F1 macro / per-class F1 per
    fold, the global OOF confusion matrix, and per-fold normalised
    confusion matrices.
    """
    n_folds = len(fold_f1_scores)
    fold_labels = [f"Fold {i + 1}" for i in range(n_folds)]
    fold_class_f1_arr = np.array(fold_class_f1s)

    n_cols = max(3, n_folds)
    n_rows = 2 if n_folds <= 3 else 1 + int(np.ceil((n_folds + 1) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 6 * n_rows))
    axes = np.atleast_2d(axes)
    fig.suptitle(
        "ECG Arrhythmia Classifier — Cross-Validation Performance Dashboard",
        fontsize=14,
        fontweight="bold",
    )

    # Panel: F1 Macro per fold
    ax = axes[0, 0]
    bars = ax.bar(fold_labels, fold_f1_scores, color="#2c3e50", alpha=0.7)
    ax.axhline(np.mean(fold_f1_scores), color="#e74c3c", linestyle="--", label=f"Mean ({np.mean(fold_f1_scores):.4f})")
    ax.axhline(global_macro_f1, color="#27ae60", linestyle=":", label=f"Global OOF ({global_macro_f1:.4f})")
    for bar, val in zip(bars, fold_f1_scores):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005, f"{val:.3f}", ha="center", fontsize=9)
    ax.set_title("F1 Macro per Fold")
    ax.set_ylim([0.0, 1.0])
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # Panel: Per-class F1 per fold
    ax = axes[0, 1]
    x_pos = np.arange(n_folds)
    bar_width = 0.25
    ax.bar(x_pos - bar_width, fold_class_f1_arr[:, 0], bar_width, label="Normal", color="#2ecc71", alpha=0.8)
    ax.bar(x_pos, fold_class_f1_arr[:, 1], bar_width, label="Arrhythmia", color="#e67e22", alpha=0.8)
    ax.bar(x_pos + bar_width, fold_class_f1_arr[:, 2], bar_width, label="Noise", color="#3498db", alpha=0.8)
    ax.set_title("Per-Class F1 Score per Fold")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(fold_labels)
    ax.set_ylim([0.0, 1.0])
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # Panel: Global OOF confusion matrix
    ax = axes[0, 2]
    ConfusionMatrixDisplay(global_conf_mat, display_labels=CLASS_NAMES).plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title("Global OOF Confusion Matrix")

    # Hide any unused cells in row 0 beyond column 2
    for col in range(3, n_cols):
        axes[0, col].axis("off")

    # Remaining panels are unused in this simplified dashboard (kept for layout symmetry)
    for row in range(1, n_rows):
        for col in range(n_cols):
            axes[row, col].axis("off")

    plt.tight_layout()
    dashboard_path = os.path.join(checkpoint_dir, "performance_dashboard.png")
    plt.savefig(dashboard_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Dashboard saved: {dashboard_path}")


def main():
    args = parse_args()

    # ── Data loading & preprocessing ───────────────────────────────────────
    print("Loading data from HDF5 files …")
    X_train_raw, y_train, _patient_groups = load_train_data(args.train_data)
    X_test_raw = load_test_data(args.test_data)
    print(f"Dataset loaded — train: {X_train_raw.shape[0]} samples | test: {X_test_raw.shape[0]} samples")

    _X_train, X_train_3d, X_test_3d, y_train_onehot = prepare_datasets(
        X_train_raw, y_train, X_test_raw, args.num_classes
    )

    skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=args.random_state)

    # ── Step 1: reload base checkpoints and report held-out metrics ───────
    print("\nReloading best checkpoints for evaluation …")
    reload_and_evaluate_folds(
        args.checkpoint_dir, skf, X_train_3d, X_test_3d, y_train,
        gamma=3.0, batch_size=args.batch_size,
    )
    plot_training_curves(args.checkpoint_dir, args.n_folds, suffix="")

    # ── Step 2: fine-tune the classification head of each fold ────────────
    if not args.skip_finetune:
        print("\nFine-tuning best models (last {} layers unfrozen) …".format(args.unfrozen_layers))
        fine_tune_folds(args.checkpoint_dir, skf, X_train_3d, y_train_onehot, y_train, args)
        plot_training_curves(
            args.checkpoint_dir, args.n_folds, suffix="_finetuned",
            title="Fine-Tuning Curves per Fold — Loss · Accuracy · F1 Macro",
        )
    else:
        print("\n--skip-finetune set — using base checkpoints for the ensemble.")

    # ── Step 3: rebuild the OOF ensemble ───────────────────────────────────
    print("\nRebuilding OOF ensemble …")
    (
        oof_prob_matrix,
        oof_true_labels,
        ensemble_test_probs,
        fold_f1_scores,
        fold_class_f1s,
        fold_conf_matrices,
    ) = build_oof_ensemble(
        args.checkpoint_dir, skf, X_train_3d, X_test_3d, y_train,
        num_classes=args.num_classes, batch_size=args.batch_size,
    )

    oof_predictions = np.argmax(oof_prob_matrix, axis=-1)
    global_macro_f1 = f1_score(oof_true_labels, oof_predictions, average="macro")
    global_conf_mat = confusion_matrix(oof_true_labels, oof_predictions)

    print(f"\n{'='*60}")
    print("GLOBAL OOF METRICS")
    print(f"{'='*60}")
    print(f"  F1 Macro (global): {global_macro_f1:.4f}")
    print(classification_report(oof_true_labels, oof_predictions, target_names=CLASS_NAMES))

    # ── Step 4: persist ensemble artifacts for threshold_optimization.py ───
    np.save(os.path.join(args.checkpoint_dir, "oof_prob_matrix.npy"), oof_prob_matrix)
    np.save(os.path.join(args.checkpoint_dir, "oof_true_labels.npy"), oof_true_labels)
    np.save(os.path.join(args.checkpoint_dir, "ensemble_test_probs.npy"), ensemble_test_probs)
    print(f"\nSaved oof_prob_matrix.npy, oof_true_labels.npy, ensemble_test_probs.npy → {args.checkpoint_dir}")

    # ── Step 5: diagnostic dashboard ───────────────────────────────────────
    plot_performance_dashboard(args.checkpoint_dir, fold_f1_scores, fold_class_f1s, global_conf_mat, global_macro_f1)

    print("\nEvaluation complete. Run threshold_optimization.py next to obtain the final test predictions.")


if __name__ == "__main__":
    main()

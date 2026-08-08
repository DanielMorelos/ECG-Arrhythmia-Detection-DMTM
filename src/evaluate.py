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
       - oof_prob_matrix.npy        (N_train, num_classes)
       - oof_true_labels.npy        (N_train,)
       - ensemble_test_probs.npy    (n_folds, N_test, num_classes)
       - fold_f1_scores.npy         (n_folds,)
       - fold_class_f1_scores.npy   (n_folds, num_classes)
       - global_confusion_matrix.npy(num_classes, num_classes)
   The first three are the required inputs of threshold_optimization.py;
   the last three are consumed by its composite dashboard summary.
4. Render training-curve and performance-dashboard figures (PNG + PDF,
   row-normalized confusion matrices) to checkpoint_dir for inspection.

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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Reload fold checkpoints, fine-tune, and rebuild the OOF ensemble."
    )
    parser.add_argument("--train-data", type=str, required=True, help="Path to training HDF5 file.")
    parser.add_argument("--test-data", type=str, required=True, help="Path to test HDF5 file (unlabeled).")
    parser.add_argument("--checkpoint-dir", type=str, required=True, help="Directory containing fold checkpoints from train.py.")
    parser.add_argument("--input-length", type=int, default=2049, help="ECG segment length (time steps).")
    parser.add_argument("--num-classes", type=int, default=3, help="Number of target classes.")
    parser.add_argument("--class-names", type=str, default="Normal,Arrhythmia,Noise", help="Comma-separated class display names, in label order.")
    parser.add_argument("--n-folds", type=int, default=5, help="Number of stratified CV folds (must match train.py).")
    parser.add_argument("--batch-size", type=int, default=32, help="Inference / fine-tuning batch size.")
    parser.add_argument("--random-state", type=int, default=42, help="Random state for StratifiedKFold (must match train.py).")
    parser.add_argument("--finetune-epochs", type=int, default=15, help="Maximum fine-tuning epochs per fold.")
    parser.add_argument("--finetune-lr", type=float, default=1e-5, help="Learning rate for fine-tuning.")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay for AdamW during fine-tuning.")
    parser.add_argument("--finetune-patience", type=int, default=5, help="EarlyStopping patience during fine-tuning.")
    parser.add_argument("--unfrozen-layers", type=int, default=3, help="Number of trailing layers left trainable during fine-tuning.")
    parser.add_argument("--skip-finetune", action="store_true", help="Skip fine-tuning and evaluate base checkpoints only.")
    parser.add_argument("--focal-gamma", type=float, default=3.0, help="Focal loss focusing exponent (must match train.py).")
    parser.add_argument("--focal-alpha", type=float, default=1.0, help="Focal loss scaling factor (must match train.py).")
    parser.add_argument("--label-smoothing", type=float, default=0.05, help="Focal loss label smoothing (must match train.py).")
    parser.add_argument("--digits", type=int, default=4, help="Decimal precision used in reports, plots, and confusion-matrix cell labels.")
    parser.add_argument("--dpi", type=int, default=400, help="Resolution (DPI) used when saving PNG/PDF figures.")
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# CONFUSION-MATRIX UTILITIES
# ─────────────────────────────────────────────────────────────────────────────
def normalize_confusion_matrix(conf_matrix: np.ndarray) -> np.ndarray:
    """Row-normalises a confusion matrix so each row sums to 1.0, i.e.
    per-true-class recall proportions (no percentage scaling)."""
    conf_matrix = conf_matrix.astype(float)
    row_sums = conf_matrix.sum(axis=1, keepdims=True)
    return np.divide(
        conf_matrix, row_sums,
        out=np.zeros_like(conf_matrix),
        where=row_sums != 0,
    )


def plot_confusion_matrix_ratio(
    conf_matrix: np.ndarray,
    display_labels: list,
    ax: plt.Axes,
    title: str,
    cmap: str = "Blues",
    already_normalized: bool = False,
    colorbar: bool = False,
    digits: int = 4,
) -> ConfusionMatrixDisplay:
    """Plots a confusion matrix as row-normalised ratios formatted to
    `digits` decimal places, matching the standard reporting style used
    across every figure in this pipeline."""
    matrix_to_plot = conf_matrix if already_normalized else normalize_confusion_matrix(conf_matrix)
    disp = ConfusionMatrixDisplay(matrix_to_plot, display_labels=display_labels)
    disp.plot(ax=ax, colorbar=colorbar, cmap=cmap, values_format=f".{digits}f")
    ax.set_title(title)
    return disp


def save_figure(fig: plt.Figure, base_path: str, dpi: int) -> None:
    """Saves `fig` as both PNG and PDF at the given DPI. `base_path` may
    carry either extension; both variants are written alongside it."""
    root, _ext = os.path.splitext(base_path)
    fig.savefig(f"{root}.png", dpi=dpi, bbox_inches="tight")
    fig.savefig(f"{root}.pdf", dpi=dpi, bbox_inches="tight")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — RELOAD BASE CHECKPOINTS
# ─────────────────────────────────────────────────────────────────────────────
def reload_and_evaluate_folds(checkpoint_dir, skf, X_train_3d, y_train, class_names, gamma, alpha, label_smoothing, batch_size, digits):
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
            custom_objects={"focal_loss_fixed": focal_loss(gamma=gamma, alpha=alpha, label_smoothing=label_smoothing)},
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
        print(f"  F1 Macro : {macro_f1:.{digits}f}")
        print(classification_report(y_fold_val_true, val_preds, target_names=class_names, digits=digits))

    return fold_macro_f1_scores, fold_per_class_f1


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — FINE-TUNE
# ─────────────────────────────────────────────────────────────────────────────
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
            custom_objects={"focal_loss_fixed": focal_loss(gamma=args.focal_gamma, alpha=args.focal_alpha, label_smoothing=args.label_smoothing)},
        )

        # Freeze every layer except the trailing `unfrozen_layers`.
        for layer in pretrained_model.layers[: -args.unfrozen_layers]:
            layer.trainable = False

        pretrained_model.compile(
            optimizer=tf.keras.optimizers.AdamW(learning_rate=args.finetune_lr, weight_decay=args.weight_decay),
            loss=focal_loss(gamma=args.focal_gamma, alpha=args.focal_alpha, label_smoothing=args.label_smoothing),
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


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — OOF ENSEMBLE
# ─────────────────────────────────────────────────────────────────────────────
def build_oof_ensemble(checkpoint_dir, skf, X_train_3d, X_test_3d, y_train, class_names, num_classes, gamma, alpha, label_smoothing, batch_size, digits):
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
            custom_objects={"focal_loss_fixed": focal_loss(gamma=gamma, alpha=alpha, label_smoothing=label_smoothing)},
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

        print(f"\n  Fold {fold_idx + 1} — F1 Macro: {macro_f1:.{digits}f}")
        print(classification_report(y_fold_val_true, fold_val_preds, target_names=class_names, digits=digits))

        ensemble_test_probs.append(model.predict(X_test_3d, batch_size=batch_size, verbose=0))

    return (
        oof_prob_matrix,
        oof_true_labels,
        np.array(ensemble_test_probs),
        fold_f1_scores,
        fold_class_f1s,
        fold_conf_matrices,
    )


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — DIAGNOSTIC FIGURES
# ─────────────────────────────────────────────────────────────────────────────
def plot_training_curves(checkpoint_dir, n_folds, output_path, dpi, suffix="", title="Training Curves per Fold — Loss · Accuracy · F1 Macro"):
    """
    Loads the persisted JSON history for every fold and renders a
    3-row x n_folds grid of Loss / Accuracy / F1-Macro curves.
    Saved as PNG + PDF at `dpi` to `output_path`.

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
    save_figure(fig, output_path, dpi)
    plt.close(fig)
    print(f"Curves saved → {output_path}")


def plot_performance_dashboard(output_path, class_names, fold_f1_scores, fold_class_f1s, fold_conf_matrices, global_conf_mat, global_macro_f1, digits, dpi):
    """
    Renders a performance dashboard: F1 macro / per-class F1 per fold,
    the row-normalised global OOF confusion matrix, and one row-normalised
    confusion matrix per fold. Saved as PNG + PDF at `dpi`.
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
    ax.axhline(np.mean(fold_f1_scores), color="#e74c3c", linestyle="--", label=f"Mean ({np.mean(fold_f1_scores):.{digits}f})")
    ax.axhline(global_macro_f1, color="#27ae60", linestyle=":", label=f"Global OOF ({global_macro_f1:.{digits}f})")
    for bar, val in zip(bars, fold_f1_scores):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005, f"{val:.{digits}f}", ha="center", fontsize=9)
    ax.set_title("F1 Macro per Fold")
    ax.set_ylim([0.0, 1.0])
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # Panel: Per-class F1 per fold
    ax = axes[0, 1]
    x_pos = np.arange(n_folds)
    bar_width = 0.25
    bar_colors = ["#2ecc71", "#e67e22", "#3498db"]
    for class_idx, class_name in enumerate(class_names):
        offset = (class_idx - (len(class_names) - 1) / 2) * bar_width
        ax.bar(x_pos + offset, fold_class_f1_arr[:, class_idx], bar_width, label=class_name, color=bar_colors[class_idx % len(bar_colors)], alpha=0.8)
    ax.set_title("Per-Class F1 Score per Fold")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(fold_labels)
    ax.set_ylim([0.0, 1.0])
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # Panel: Global OOF confusion matrix (row-normalised)
    plot_confusion_matrix_ratio(
        global_conf_mat, class_names, axes[0, 2],
        title=f"Global OOF Confusion Matrix\nF1 Macro = {global_macro_f1:.{digits}f}",
        digits=digits,
    )

    # Hide any unused cells in row 0 beyond column 2
    for col in range(3, n_cols):
        axes[0, col].axis("off")

    # Panels: one row-normalised confusion matrix per fold, starting at row 1
    remaining_axes = [axes[row, col] for row in range(1, n_rows) for col in range(n_cols)]
    for fold_idx, conf_mat in enumerate(fold_conf_matrices):
        ax = remaining_axes[fold_idx]
        plot_confusion_matrix_ratio(
            conf_mat, class_names, ax,
            title=f"Fold {fold_idx + 1} — Confusion Matrix (normalised)",
            digits=digits,
        )

    # Hide any leftover unused panels
    for ax in remaining_axes[len(fold_conf_matrices):]:
        ax.axis("off")

    plt.tight_layout()
    save_figure(fig, output_path, dpi)
    plt.close(fig)
    print(f"Dashboard saved: {output_path}")


def main():
    args = parse_args()
    class_names = [name.strip() for name in args.class_names.split(",")]

    # ── Output paths (declared up front; per-fold paths stay inside loops) ──
    training_curves_path = os.path.join(args.checkpoint_dir, "training_curves.png")
    finetuning_curves_path = os.path.join(args.checkpoint_dir, "finetuning_curves.png")
    dashboard_path = os.path.join(args.checkpoint_dir, "performance_dashboard.png")
    oof_prob_path = os.path.join(args.checkpoint_dir, "oof_prob_matrix.npy")
    oof_labels_path = os.path.join(args.checkpoint_dir, "oof_true_labels.npy")
    ensemble_test_probs_path = os.path.join(args.checkpoint_dir, "ensemble_test_probs.npy")
    fold_f1_path = os.path.join(args.checkpoint_dir, "fold_f1_scores.npy")
    fold_class_f1_path = os.path.join(args.checkpoint_dir, "fold_class_f1_scores.npy")
    global_conf_mat_path = os.path.join(args.checkpoint_dir, "global_confusion_matrix.npy")

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
        args.checkpoint_dir, skf, X_train_3d, y_train, class_names,
        gamma=args.focal_gamma, alpha=args.focal_alpha, label_smoothing=args.label_smoothing,
        batch_size=args.batch_size, digits=args.digits,
    )
    plot_training_curves(args.checkpoint_dir, args.n_folds, training_curves_path, dpi=args.dpi)

    # ── Step 2: fine-tune the classification head of each fold ────────────
    if not args.skip_finetune:
        print(f"\nFine-tuning best models (last {args.unfrozen_layers} layers unfrozen) …")
        fine_tune_folds(args.checkpoint_dir, skf, X_train_3d, y_train_onehot, y_train, args)
        plot_training_curves(
            args.checkpoint_dir, args.n_folds, finetuning_curves_path, dpi=args.dpi,
            suffix="_finetuned",
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
        args.checkpoint_dir, skf, X_train_3d, X_test_3d, y_train, class_names,
        num_classes=args.num_classes, gamma=args.focal_gamma, alpha=args.focal_alpha,
        label_smoothing=args.label_smoothing, batch_size=args.batch_size, digits=args.digits,
    )

    oof_predictions = np.argmax(oof_prob_matrix, axis=-1)
    global_macro_f1 = f1_score(oof_true_labels, oof_predictions, average="macro")
    global_conf_mat = confusion_matrix(oof_true_labels, oof_predictions)

    print(f"\n{'='*60}")
    print("GLOBAL OOF METRICS")
    print(f"{'='*60}")
    print(f"  F1 Macro (global): {global_macro_f1:.{args.digits}f}")
    print(classification_report(oof_true_labels, oof_predictions, target_names=class_names, digits=args.digits))

    # ── Step 4: persist ensemble + fold-level artifacts ────────────────────
    # oof_prob_matrix / oof_true_labels / ensemble_test_probs feed threshold_optimization.py.
    # fold_f1_scores / fold_class_f1_scores / global_confusion_matrix feed its composite dashboard.
    np.save(oof_prob_path, oof_prob_matrix)
    np.save(oof_labels_path, oof_true_labels)
    np.save(ensemble_test_probs_path, ensemble_test_probs)
    np.save(fold_f1_path, np.array(fold_f1_scores))
    np.save(fold_class_f1_path, np.array(fold_class_f1s))
    np.save(global_conf_mat_path, global_conf_mat)
    print(f"\nSaved ensemble and fold-level metric artifacts → {args.checkpoint_dir}")

    # ── Step 5: diagnostic dashboard ───────────────────────────────────────
    plot_performance_dashboard(
        dashboard_path, class_names, fold_f1_scores, fold_class_f1s, fold_conf_matrices,
        global_conf_mat, global_macro_f1, digits=args.digits, dpi=args.dpi,
    )

    print("\nEvaluation complete. Run threshold_optimization.py next to obtain the final test predictions.")


if __name__ == "__main__":
    main()
"""
Threshold Optimization, Final Inference, and Composite Reporting
====================================================================
Loads the OOF probability matrix and ensemble test-set probabilities
produced by evaluate.py, performs a grid search over per-class decision
thresholds to maximize the OOF macro F1-score, and applies the optimized
thresholds to the mean ensemble test probabilities to produce the final
predictions.

Optionally (when --test-labels-path is given) evaluates those final
predictions against a labeled test set and renders a publication-ready
composite dashboard summarising the full pipeline (cross-validation,
threshold optimization, and external test performance).

Usage
-----
python src/threshold_optimization.py \
    --checkpoint-dir path/to/checkpoints/ \
    --output-path path/to/final_predictions.npy \
    --test-labels-path path/to/test_with_labels.h5
"""

import os
import argparse

import numpy as np
import h5py
import matplotlib.pyplot as plt
from sklearn.metrics import (
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Optimize per-class decision thresholds, produce final ensemble predictions, and report results.")
    parser.add_argument("--checkpoint-dir", type=str, required=True,
                         help="Directory containing oof_prob_matrix.npy, oof_true_labels.npy, and ensemble_test_probs.npy (produced by evaluate.py).")
    parser.add_argument("--output-path", type=str, required=True, help="Path to save the final one-hot predictions (.npy).")
    parser.add_argument("--test-labels-path", type=str, default=None,
                         help="Optional path to a labeled HDF5 test set (dataset 'labels'). If given, the final "
                              "predictions are evaluated against it and a composite dashboard summary is rendered.")
    parser.add_argument("--num-classes", type=int, default=3, help="Number of target classes.")
    parser.add_argument("--class-names", type=str, default="Normal,Arrhythmia,Noise", help="Comma-separated class display names, in label order.")
    parser.add_argument("--threshold-min", type=float, default=0.20, help="Lower bound of the threshold grid search.")
    parser.add_argument("--threshold-max", type=float, default=0.80, help="Upper bound of the threshold grid search.")
    parser.add_argument("--threshold-steps", type=int, default=15, help="Number of steps in the threshold grid search.")
    parser.add_argument("--digits", type=int, default=4, help="Decimal precision used in reports, plots, and confusion-matrix cell labels.")
    parser.add_argument("--dpi", type=int, default=400, help="Resolution (DPI) used when saving PNG/PDF figures.")
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# CONFUSION-MATRIX UTILITIES (kept local so this script stays TensorFlow-free)
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
    `digits` decimal places, matching evaluate.py's reporting style."""
    matrix_to_plot = conf_matrix if already_normalized else normalize_confusion_matrix(conf_matrix)
    disp = ConfusionMatrixDisplay(matrix_to_plot, display_labels=display_labels)
    disp.plot(ax=ax, colorbar=colorbar, cmap=cmap, values_format=f".{digits}f")
    ax.set_title(title)
    return disp


def save_figure(fig: plt.Figure, base_path: str, dpi: int) -> None:
    """Saves `fig` as both PNG and PDF at the given DPI."""
    root, _ext = os.path.splitext(base_path)
    fig.savefig(f"{root}.png", dpi=dpi, bbox_inches="tight")
    fig.savefig(f"{root}.pdf", dpi=dpi, bbox_inches="tight")


# ─────────────────────────────────────────────────────────────────────────────
# THRESHOLD DECODING AND GRID SEARCH
# ─────────────────────────────────────────────────────────────────────────────
def apply_threshold_decoding(probabilities: np.ndarray, thresholds: tuple) -> np.ndarray:
    """
    Decode class labels from a probability matrix using per-class thresholds.
    For samples where at least one class exceeds its threshold, the winning
    class is the one with the highest probability among those above threshold.
    Falls back to argmax for samples where no threshold is exceeded.
    """
    th0, th1, th2 = thresholds
    above_th0 = probabilities[:, 0] >= th0
    above_th1 = probabilities[:, 1] >= th1
    above_th2 = probabilities[:, 2] >= th2

    decoded_labels = np.argmax(probabilities, axis=-1).copy()
    any_above = above_th0 | above_th1 | above_th2

    masked_probs = probabilities.copy()
    masked_probs[~above_th0, 0] = -1
    masked_probs[~above_th1, 1] = -1
    masked_probs[~above_th2, 2] = -1
    decoded_labels[any_above] = np.argmax(masked_probs[any_above], axis=-1)

    return decoded_labels


def grid_search_thresholds(oof_prob_matrix: np.ndarray, oof_true_labels: np.ndarray, threshold_grid: np.ndarray):
    """
    Vectorised grid search over per-class decision thresholds (Normal,
    Arrhythmia, Noise) maximizing the OOF macro F1-score.
    """
    best_macro_f1 = 0.0
    best_thresholds = (0.33, 0.33, 0.33)

    prob_class0 = oof_prob_matrix[:, 0]
    prob_class1 = oof_prob_matrix[:, 1]
    prob_class2 = oof_prob_matrix[:, 2]

    for th0 in threshold_grid:
        for th1 in threshold_grid:
            for th2 in threshold_grid:
                above_th0 = prob_class0 >= th0
                above_th1 = prob_class1 >= th1
                above_th2 = prob_class2 >= th2

                candidate_preds = np.argmax(oof_prob_matrix, axis=-1).copy()
                any_above = above_th0 | above_th1 | above_th2

                masked_probs = oof_prob_matrix.copy()
                masked_probs[~above_th0, 0] = -1
                masked_probs[~above_th1, 1] = -1
                masked_probs[~above_th2, 2] = -1
                candidate_preds[any_above] = np.argmax(masked_probs[any_above], axis=-1)

                candidate_f1 = f1_score(oof_true_labels, candidate_preds, average="macro")
                if candidate_f1 > best_macro_f1:
                    best_macro_f1 = candidate_f1
                    best_thresholds = (th0, th1, th2)

    return best_thresholds, best_macro_f1


# ─────────────────────────────────────────────────────────────────────────────
# LABELED-TEST-SET EVALUATION
# ─────────────────────────────────────────────────────────────────────────────
def evaluate_against_test_labels(test_labels_path, final_test_labels, class_names, digits):
    """
    Loads ground-truth labels from a labeled HDF5 test set and scores the
    already-computed final predictions against them (weighted F1 + confusion
    matrix), matching the pipeline's external validation step.
    """
    with h5py.File(test_labels_path, "r") as h5_file:
        y_test_true = np.squeeze(np.array(h5_file["labels"], dtype=np.int32))

    weighted_f1 = f1_score(y_test_true, final_test_labels, average="weighted")
    conf_matrix = confusion_matrix(y_test_true, final_test_labels)

    print(f"\n{'='*60}")
    print("FINAL TEST-SET EVALUATION")
    print(f"{'='*60}")
    print(f"  F1-score (weighted): {weighted_f1:.{digits}f}")
    print(classification_report(y_test_true, final_test_labels, target_names=class_names, digits=digits))

    return weighted_f1, conf_matrix


def plot_test_confusion_matrix(output_path, conf_matrix, class_names, weighted_f1, digits, dpi):
    fig, ax = plt.subplots(figsize=(7, 6))
    fig.suptitle(f"F1-score weighted: {weighted_f1:.{digits}f}", fontsize=12, y=0.98)
    plot_confusion_matrix_ratio(
        conf_matrix, class_names, ax,
        title=f"Confusion Matrix\nWeighted F1 = {weighted_f1:.{digits}f}",
        cmap="viridis",
        colorbar=True,
        digits=digits,
    )
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    plt.tight_layout()
    save_figure(fig, output_path, dpi)
    plt.close(fig)
    print(f"Final test-set confusion matrix saved: {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# COMPOSITE DASHBOARD SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
def plot_composite_summary_dashboard(
    output_path,
    class_names,
    fold_f1_scores,
    fold_class_f1_arr,
    global_macro_f1,
    global_conf_mat,
    best_macro_f1,
    best_thresholds,
    test_weighted_f1,
    test_conf_matrix,
    digits,
    dpi,
):
    """
    Builds a single composite figure summarising the full experimental
    pipeline (cross-validation, threshold optimization, and, when available,
    external test-set evaluation) — intended as a graphical-abstract /
    results-summary figure.
    """
    has_test_eval = test_weighted_f1 is not None and test_conf_matrix is not None

    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1], hspace=0.45, wspace=0.4)
    fig.suptitle(
        "Composite Summary — Cross-Validation and Test Performance",
        fontsize=15, fontweight="bold", y=0.98,
    )

    fold_labels = [f"Fold {i + 1}" for i in range(len(fold_f1_scores))]

    # ── Panel A: F1 Macro per fold ──────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 0])
    bars = ax.bar(fold_labels, fold_f1_scores, color="#2c3e50", alpha=0.8)
    ax.axhline(np.mean(fold_f1_scores), color="#e74c3c", linestyle="--",
               label=f"Mean ({np.mean(fold_f1_scores):.{digits}f})")
    ax.axhline(global_macro_f1, color="#27ae60", linestyle=":",
               label=f"Global OOF ({global_macro_f1:.{digits}f})")
    for bar, val in zip(bars, fold_f1_scores):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{val:.{digits}f}", ha="center", fontsize=8)
    ax.set_title("A. F1 Macro per Fold", fontsize=11, fontweight="bold")
    ax.set_ylim([0.0, 1.0])
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(alpha=0.3)

    # ── Panel B: Per-class F1 (mean across folds) ───────────────────────────
    ax = fig.add_subplot(gs[0, 1])
    mean_class_f1 = fold_class_f1_arr.mean(axis=0)
    std_class_f1 = fold_class_f1_arr.std(axis=0)
    bar_colors = ["#2ecc71", "#e67e22", "#3498db"]
    bars = ax.bar(class_names, mean_class_f1, yerr=std_class_f1,
                   color=[bar_colors[i % len(bar_colors)] for i in range(len(class_names))],
                   alpha=0.85, capsize=5)
    for bar, val in zip(bars, mean_class_f1):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.{digits}f}", ha="center", fontsize=9)
    ax.set_title("B. Mean Per-Class F1 (± SD)", fontsize=11, fontweight="bold")
    ax.set_ylim([0.0, 1.0])
    ax.grid(alpha=0.3)

    # ── Panel C: Global OOF confusion matrix ────────────────────────────────
    plot_confusion_matrix_ratio(
        global_conf_mat, class_names, fig.add_subplot(gs[0, 2]),
        title="C. Global OOF Confusion Matrix",
        digits=digits,
    )

    # ── Panel D: Effect of threshold optimization ───────────────────────────
    ax = fig.add_subplot(gs[1, 0])
    stages = ["Argmax\n(OOF)", "Threshold-\ntuned (OOF)"]
    values = [global_macro_f1, best_macro_f1]
    bars = ax.bar(stages, values, color=["#95a5a6", "#8e44ad"], alpha=0.85)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{val:.{digits}f}", ha="center", fontsize=9)
    ax.set_title("D. Effect of Threshold Tuning", fontsize=11, fontweight="bold")
    ax.set_ylabel("F1 Macro")
    ax.set_ylim([0.0, 1.0])
    ax.grid(alpha=0.3)

    # ── Panel E: External test-set confusion matrix (if available) ─────────
    ax = fig.add_subplot(gs[1, 1])
    if has_test_eval:
        plot_confusion_matrix_ratio(
            test_conf_matrix, class_names, ax,
            title=f"E. External Test Confusion Matrix\nWeighted F1 = {test_weighted_f1:.{digits}f}",
            cmap="viridis",
            digits=digits,
        )
    else:
        ax.axis("off")
        ax.text(0.5, 0.5, "No labeled test set provided\n(--test-labels-path not set)",
                ha="center", va="center", fontsize=10, style="italic")
        ax.set_title("E. External Test Confusion Matrix", fontsize=11, fontweight="bold")

    # ── Panel F: Key-metrics summary box ────────────────────────────────────
    ax = fig.add_subplot(gs[1, 2])
    ax.axis("off")
    th0, th1, th2 = best_thresholds
    threshold_labels = " / ".join(class_names)
    summary_lines = [
        "Key Results Summary",
        "─────────────────────",
        f"CV Macro F1 (mean ± SD):\n  {np.mean(fold_f1_scores):.{digits}f} ± {np.std(fold_f1_scores):.{digits}f}",
        "",
        f"Global OOF Macro F1:\n  {global_macro_f1:.{digits}f}",
        "",
        f"OOF Macro F1 (threshold-tuned):\n  {best_macro_f1:.{digits}f}",
        "",
        f"Optimal thresholds ({threshold_labels}):\n  {th0:.{digits}f} / {th1:.{digits}f} / {th2:.{digits}f}",
    ]
    if has_test_eval:
        summary_lines += ["", f"External test Weighted F1:\n  {test_weighted_f1:.{digits}f}"]
    summary_text = "\n".join(summary_lines)
    ax.text(0.02, 0.98, summary_text, transform=ax.transAxes,
            fontsize=10, va="top", ha="left", family="monospace",
            bbox=dict(boxstyle="round,pad=0.6", facecolor="#f4f4f4", edgecolor="#bdbdbd"))
    ax.set_title("F. Summary Metrics", fontsize=11, fontweight="bold")

    save_figure(fig, output_path, dpi)
    plt.close(fig)
    print(f"Composite dashboard summary saved: {output_path}")


def main():
    args = parse_args()
    class_names = [name.strip() for name in args.class_names.split(",")]

    # ── Input / output paths (declared up front) ────────────────────────────
    oof_prob_path = os.path.join(args.checkpoint_dir, "oof_prob_matrix.npy")
    oof_labels_path = os.path.join(args.checkpoint_dir, "oof_true_labels.npy")
    ensemble_test_probs_path = os.path.join(args.checkpoint_dir, "ensemble_test_probs.npy")
    fold_f1_path = os.path.join(args.checkpoint_dir, "fold_f1_scores.npy")
    fold_class_f1_path = os.path.join(args.checkpoint_dir, "fold_class_f1_scores.npy")
    global_conf_mat_path = os.path.join(args.checkpoint_dir, "global_confusion_matrix.npy")
    final_test_confusion_matrix_path = os.path.join(args.checkpoint_dir, "final_test_confusion_matrix.png")
    dashboard_summary_path = os.path.join(args.checkpoint_dir, "composite_dashboard_summary.png")

    oof_prob_matrix = np.load(oof_prob_path)
    oof_true_labels = np.load(oof_labels_path)
    ensemble_test_probs = np.load(ensemble_test_probs_path)

    # ── Threshold grid search ────────────────────────────────────────────────
    print("\nSearching for optimal per-class decision thresholds …")

    mean_test_probs = np.mean(ensemble_test_probs, axis=0)
    threshold_grid = np.linspace(args.threshold_min, args.threshold_max, args.threshold_steps)

    best_thresholds, best_macro_f1 = grid_search_thresholds(oof_prob_matrix, oof_true_labels, threshold_grid)
    th0_opt, th1_opt, th2_opt = best_thresholds

    print("  Optimal thresholds:")
    for name, th in zip(class_names, best_thresholds):
        print(f"    {name}: {th:.{args.digits}f}")
    print(f"  OOF F1 Macro with threshold tuning: {best_macro_f1:.{args.digits}f}")

    # ── Final inference — soft-voting ensemble + threshold decoding ───────
    final_test_labels = apply_threshold_decoding(mean_test_probs, best_thresholds)
    final_test_onehot = np.eye(args.num_classes, dtype=np.int32)[final_test_labels]

    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    np.save(args.output_path, final_test_onehot)

    print(f"\n{'='*60}")
    print(f"Final predictions saved : {args.output_path}")
    print(f"  Shape  : {final_test_onehot.shape}  (samples x {args.num_classes})")
    print("\n  Test-set class distribution:")
    for class_idx, class_name in enumerate(class_names):
        count = np.sum(final_test_labels == class_idx)
        print(f"    {class_name}: {count} samples")

    # ── Optional: evaluate against labeled test set + composite dashboard ──
    test_weighted_f1, test_conf_matrix = None, None
    if args.test_labels_path:
        test_weighted_f1, test_conf_matrix = evaluate_against_test_labels(
            args.test_labels_path, final_test_labels, class_names, args.digits,
        )
        plot_test_confusion_matrix(
            final_test_confusion_matrix_path, test_conf_matrix, class_names,
            test_weighted_f1, args.digits, args.dpi,
        )

    fold_metrics_available = all(os.path.exists(p) for p in (fold_f1_path, fold_class_f1_path, global_conf_mat_path))
    if fold_metrics_available:
        fold_f1_scores = np.load(fold_f1_path)
        fold_class_f1_arr = np.load(fold_class_f1_path)
        global_conf_mat = np.load(global_conf_mat_path)
        oof_predictions = np.argmax(oof_prob_matrix, axis=-1)
        global_macro_f1 = f1_score(oof_true_labels, oof_predictions, average="macro")

        plot_composite_summary_dashboard(
            dashboard_summary_path, class_names,
            fold_f1_scores=fold_f1_scores,
            fold_class_f1_arr=fold_class_f1_arr,
            global_macro_f1=global_macro_f1,
            global_conf_mat=global_conf_mat,
            best_macro_f1=best_macro_f1,
            best_thresholds=best_thresholds,
            test_weighted_f1=test_weighted_f1,
            test_conf_matrix=test_conf_matrix,
            digits=args.digits,
            dpi=args.dpi,
        )
    else:
        print(
            "\nSkipping composite dashboard summary — fold_f1_scores.npy / "
            "fold_class_f1_scores.npy / global_confusion_matrix.npy not found "
            f"in {args.checkpoint_dir}. Re-run the updated evaluate.py to generate them."
        )


if __name__ == "__main__":
    main()
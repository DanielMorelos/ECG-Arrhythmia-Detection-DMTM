"""
Threshold Optimization and Final Inference
=============================================
Loads the OOF probability matrix and ensemble test-set probabilities
produced by evaluate.py, performs a grid search over per-class decision
thresholds to maximize the OOF macro F1-score, and applies the optimized
thresholds to the mean ensemble test probabilities to produce the final
predictions.

Usage
-----
python src/threshold_optimization.py \
    --checkpoint-dir path/to/checkpoints/ \
    --output-path path/to/final_predictions.npy
"""

import os
import argparse

import numpy as np
from sklearn.metrics import f1_score

NUM_CLASSES = 3
CLASS_NAMES = ["Normal", "Arrhythmia", "Noise"]


def parse_args():
    parser = argparse.ArgumentParser(description="Optimize per-class decision thresholds and produce final ensemble predictions.")
    parser.add_argument("--checkpoint-dir", type=str, required=True,
                         help="Directory containing oof_prob_matrix.npy, oof_true_labels.npy, and ensemble_test_probs.npy (produced by evaluate.py).")
    parser.add_argument("--output-path", type=str, required=True, help="Path to save the final one-hot predictions (.npy).")
    parser.add_argument("--threshold-min", type=float, default=0.20, help="Lower bound of the threshold grid search.")
    parser.add_argument("--threshold-max", type=float, default=0.80, help="Upper bound of the threshold grid search.")
    parser.add_argument("--threshold-steps", type=int, default=15, help="Number of steps in the threshold grid search.")
    return parser.parse_args()


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


def main():
    args = parse_args()

    oof_prob_matrix = np.load(os.path.join(args.checkpoint_dir, "oof_prob_matrix.npy"))
    oof_true_labels = np.load(os.path.join(args.checkpoint_dir, "oof_true_labels.npy"))
    ensemble_test_probs = np.load(os.path.join(args.checkpoint_dir, "ensemble_test_probs.npy"))

    print("\nSearching for optimal per-class decision thresholds …")

    mean_test_probs = np.mean(ensemble_test_probs, axis=0)
    threshold_grid = np.linspace(args.threshold_min, args.threshold_max, args.threshold_steps)

    best_thresholds, best_macro_f1 = grid_search_thresholds(oof_prob_matrix, oof_true_labels, threshold_grid)
    th0_opt, th1_opt, th2_opt = best_thresholds

    print("  Optimal thresholds:")
    print(f"    Normal     (class 0): {th0_opt:.3f}")
    print(f"    Arrhythmia (class 1): {th1_opt:.3f}")
    print(f"    Noise      (class 2): {th2_opt:.3f}")
    print(f"  OOF F1 Macro with threshold tuning: {best_macro_f1:.4f}")

    # ── Final inference — soft-voting ensemble + threshold decoding ───────
    final_test_labels = apply_threshold_decoding(mean_test_probs, best_thresholds)
    final_test_onehot = np.eye(NUM_CLASSES, dtype=np.int32)[final_test_labels]

    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    np.save(args.output_path, final_test_onehot)

    print(f"\n{'='*60}")
    print(f"Final predictions saved : {args.output_path}")
    print(f"  Shape  : {final_test_onehot.shape}  (samples x {NUM_CLASSES})")
    print("\n  Test-set class distribution:")
    for class_idx, class_name in enumerate(CLASS_NAMES):
        count = np.sum(final_test_labels == class_idx)
        print(f"    {class_name}: {count} samples")


if __name__ == "__main__":
    main()

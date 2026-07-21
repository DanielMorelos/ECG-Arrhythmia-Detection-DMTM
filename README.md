# ECGNet-SE-BiGRU: Robust Arrhythmia Detection from Single-Lead ECG Signals

A compact, attention-based deep learning model for automatic classification of single-lead ECG segments into **Normal rhythm**, **Arrhythmia**, and **Noise**, trained and evaluated on an intentionally limited subset of the [Icentia11k](https://physionet.org/content/icentia11k-continuous-ecg/1.0/) database to emulate realistic clinical development conditions (limited annotated data and computational resources).

## Overview

Atrial fibrillation (AF) is one of the most prevalent cardiac arrhythmias and a major risk factor for stroke, heart failure, and mortality. This project presents **ECGNet-SE-BiGRU**, a lightweight hybrid architecture combining:

- **1D Convolutional layers** for hierarchical morphological feature extraction
- **Squeeze-and-Excitation (SE) attention** for channel-wise feature recalibration
- **Bidirectional GRU** for temporal dependency modeling
- **Focal Loss with label smoothing** to address class imbalance
- **Post-training per-class threshold optimization** to improve minority-class (arrhythmia) detection

The model contains approximately **293,000 trainable parameters**, offering a favorable balance between predictive performance and computational efficiency — suitable for wearable and resource-constrained deployment scenarios.

## Key Results

| Metric | Value |
|---|---|
| OOF Macro F1-score (5-fold CV) | **0.9216 ± 0.0046** |
| F1 — Normal rhythm | 0.95 |
| F1 — Arrhythmia | 0.90 |
| F1 — Noise | 0.92 |
| External test set — Weighted F1-score | **0.9297** |

Results were obtained via stratified 5-fold cross-validation with out-of-fold (OOF) prediction aggregation, followed by evaluation on an independent, patient-level held-out test set (patients not seen during training or validation).

## Dataset

- **Source**: [Icentia11k](https://physionet.org/content/icentia11k-continuous-ecg/1.0/) — one of the largest public single-lead ambulatory ECG databases (~11,000 patients), available via PhysioNet.
- **Subset used**: 116,953 ECG segments (2,049 time steps each) derived from 500 patients, to emulate a realistic, data-constrained development scenario.
- **Preprocessing**: per-sample Z-score normalization along the time axis; `RandomOverSampler` applied to the training partition of each fold to correct class imbalance (never applied to validation/test data, to avoid leakage).

> **Note**: Raw ECG data is not included in this repository due to size and licensing. See [PhysioNet's Icentia11k page](https://physionet.org/content/icentia11k-continuous-ecg/1.0/) for access instructions.

## Architecture

```
Input (2049, 1)
 → Conv1D(32, k=7)  → BN → ReLU → MaxPool → Dropout(0.2)
 → Conv1D(64, k=5)  → BN → ReLU → MaxPool → Dropout(0.2)
 → Conv1D(128, k=3) → BN → ReLU → MaxPool → Dropout(0.2)
 → Conv1D(256, k=3) → BN → ReLU
 → Squeeze-and-Excitation block (reduction ratio = 8)
 → Bidirectional GRU(64)
 → Dense(128, ReLU) → Dropout(0.5)
 → Dense(3, Softmax)
```

## Repository Structure

```
ecg-arrhythmia-classifier/
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
├── src/
│   ├── data_preprocessing.py   # HDF5 loading, Z-score normalization
│   ├── model.py                # SE block + ECGNet-SE-BiGRU architecture
│   ├── losses.py                # Focal loss with label smoothing
│   ├── callbacks.py             # Custom Keras callbacks (checkpointing, F1 tracking, history)
│   ├── train.py                 # 5-fold stratified CV training loop
│   ├── evaluate.py              # OOF evaluation, confusion matrices, dashboard
│   └── threshold_optimization.py # Per-class decision threshold grid search
├── notebooks/
│   └── training_pipeline.ipynb  # End-to-end pipeline (Colab-ready)
├── results/
│   ├── figures/                 # Training curves, confusion matrices, dashboard
│   └── metrics.json             # Fold-level and global metrics
└── docs/
    └── paper_draft.md           # Manuscript draft (methods, results, discussion)
```

## Installation

```bash
git clone https://github.com/<your-username>/ecg-arrhythmia-classifier.git
cd ecg-arrhythmia-classifier
pip install -r requirements.txt
```

## Usage

```bash
python src/train.py \
  --train-data path/to/training_set.h5 \
  --test-data path/to/test_set.h5 \
  --checkpoint-dir path/to/checkpoints/

python src/evaluate.py --checkpoint-dir path/to/checkpoints/
```

> Paths in the original development notebook pointed to Google Drive (Colab environment). The modularized scripts in `src/` accept paths as command-line arguments instead.

## Methodology Summary

1. **Cross-validation**: Stratified 5-fold CV (random_state=42), preserving class proportions across folds.
2. **Optimization**: AdamW (lr=0.001, weight_decay=0.01), max 40 epochs, batch size 32, ReduceLROnPlateau (factor 0.5, patience 3), EarlyStopping on validation loss.
3. **Fine-tuning**: Last 3 layers unfrozen and fine-tuned for up to 15 epochs at a reduced learning rate (1e-5).
4. **Threshold optimization**: Grid search over per-class decision thresholds (0.20–0.80) on OOF probabilities to maximize macro F1-score.
5. **Final inference**: Soft-voting ensemble across the five fold-specific (fine-tuned) models, followed by optimized threshold decoding.

## Citation

If you use this work, please cite the associated manuscript (citation details to be added upon publication).

## References

1. Tuncer, T., Dogan, S., Pławiak, P., & Acharya, U. R. (2019). Automated arrhythmia detection using novel hexadecimal local pattern and multilevel wavelet transform with ECG signals. *Knowledge-Based Systems*, 186, 104923.
2. Guhdar, M., Mohammed, A. O., & Mstafa, R. J. (2025). Advanced deep learning framework for ECG arrhythmia classification using 1D-CNN with attention mechanism. *Knowledge-Based Systems*, 315, 113301.
3. Yao, G., Mao, X., Li, N., Xu, H., Xu, X., Jiao, Y., & Ni, J. (2021). Interpretation of electrocardiogram heartbeat by CNN and GRU. *Computational and Mathematical Methods in Medicine*, 2021, 1–10.
4. Kolhar, M., Kazi, R. N. A., Mohapatra, H., & Rajeh, A. M. A. (2024). AI-driven real-time classification of ECG signals for cardiac monitoring using i-AlexNet architecture. *Diagnostics*, 14(13), 1344.
5. Fajardo, C. A., Parra, A. S., & Castellanos-Parada, T. V. (2025). Lightweight deep learning for atrial fibrillation detection: Efficient models for wearable devices. *Ingeniería e Investigación*, 45(1), e114530.
6. Zou, Q., Xie, S., Lin, Z., Wu, M., & Ju, Y. (2016). Finding the best classification threshold in imbalanced classification. *Big Data Research*, 5, 2–8.

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

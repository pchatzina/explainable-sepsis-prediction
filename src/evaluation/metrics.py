"""
Pure mathematical/statistical functions for binary classifier evaluation.

Framework-agnostic module using only NumPy and scikit-learn.
Provides both threshold-independent (AUROC, AUPRC) and threshold-dependent
metrics, with strict validation-only threshold derivation.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    roc_auc_score,
    roc_curve,
)


def find_optimal_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """
    Find optimal classification threshold using Youden's J statistic.

    Youden's J = Sensitivity + Specificity - 1 = TPR - FPR.
    Must be computed **only on validation data** to prevent leakage.

    Returns:
        Optimal threshold (float between 0 and 1).
    """
    # Compute ROC curve across all possible thresholds
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)

    # Youden's J = Sensitivity + Specificity - 1 = tpr - fpr
    # Higher J indicates better balance between TPR and FPR
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)

    return float(thresholds[best_idx])


def compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    """
    Compute comprehensive binary classification metrics.

    Includes:
      - Threshold-independent: AUROC, AUPRC
      - Threshold-dependent: Accuracy, F1, Precision, Recall, Specificity
      - Calibration: Brier score, Expected Calibration Error (ECE)
      - Confusion matrix counts

    Args:
        y_true: Ground truth labels
        y_prob: Predicted probabilities
        threshold: Validation-derived threshold (Youden's J)

    Returns:
        Dictionary with all computed metrics.
    """
    # 1. Apply the strictly provided threshold to get hard predictions
    y_pred = (y_prob >= threshold).astype(int)

    # 2. Extract Confusion Matrix elements
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    # Handle edge cases to prevent division by zero
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    # 3. Calculate Expected Calibration Error (ECE)
    # ECE measures the gap between predicted probability and actual positive rate
    # Lower ECE indicates better calibration
    n_bins = 10
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_indices = np.digitize(y_prob, bins) - 1  # Assign each prediction to a bin
    ece = 0.0
    for i in range(n_bins):
        bin_mask = bin_indices == i
        if np.any(bin_mask):
            # Mean predicted probability in this bin
            bin_prob = np.mean(y_prob[bin_mask])
            # Actual positive rate in this bin
            bin_true = np.mean(y_true[bin_mask])
            # Fraction of samples in this bin
            bin_count = np.sum(bin_mask)
            # Weighted absolute error
            ece += (bin_count / len(y_prob)) * np.abs(bin_prob - bin_true)

    # 4. Compile the comprehensive metrics dictionary
    return {
        # Threshold-independent metrics (use full probability distribution)
        "auroc": float(roc_auc_score(y_true, y_prob)),
        "auprc": float(average_precision_score(y_true, y_prob)),
        # Threshold-dependent metrics (use hard predictions at threshold)
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(sensitivity),
        "specificity": float(specificity),
        # Calibration metrics
        "brier_score": float(brier_score_loss(y_true, y_prob)),
        "ece": float(ece),
        # Confusion matrix components
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        # Meta information
        "threshold": float(threshold),
        "n_samples": int(len(y_true)),
        "n_positive": int(y_true.sum()),
        "n_negative": int(len(y_true) - y_true.sum()),
    }

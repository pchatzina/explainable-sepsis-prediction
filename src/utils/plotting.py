import logging
import matplotlib.pyplot as plt
import textwrap
import numpy as np

from pathlib import Path
from typing import Dict, Tuple, Union
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

logger = logging.getLogger(__name__)


def _ensure_plot_path(save_path: Union[str, Path]) -> Path:
    """Helper to ensure the target directory exists and respects the file extension."""
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Default to .png only if no extension is provided
    if not path.suffix:
        return path.with_suffix(".png")
    return path


def plot_roc_curves(
    models_preds: Dict[str, Tuple[np.ndarray, np.ndarray]],
    save_path: Union[str, Path],
    title: str = "Receiver Operating Characteristic (ROC) Curve",
) -> None:
    """Plots a combined ROC curve for multiple models."""
    plt.figure(figsize=(8, 8))
    plt.plot([0, 1], [0, 1], "k--", label="Random Chance (AUROC = 0.500)")

    for name, (y_true, y_prob) in models_preds.items():
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auroc = roc_auc_score(y_true, y_prob)
        plt.plot(fpr, tpr, lw=2, label=f"{name} (AUROC = {auroc:.3f})")

    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate (Sensitivity)", fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, linestyle=":", alpha=0.6)

    final_path = _ensure_plot_path(save_path)
    plt.savefig(final_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("ROC Curve saved → %s", final_path)


def plot_pr_curves(
    models_preds: Dict[str, Tuple[np.ndarray, np.ndarray]],
    save_path: Union[str, Path],
    title: str = "Precision-Recall (PR) Curve",
) -> None:
    """Plots a combined Precision-Recall curve for multiple models."""
    plt.figure(figsize=(8, 8))

    first_y_true = next(iter(models_preds.values()))[0]
    prevalence = np.mean(first_y_true)
    plt.plot(
        [0, 1],
        [prevalence, prevalence],
        "k--",
        label=f"Baseline Prevalence ({prevalence:.3f})",
    )

    for name, (y_true, y_prob) in models_preds.items():
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        auprc = average_precision_score(y_true, y_prob)
        plt.plot(recall, precision, lw=2, label=f"{name} (AUPRC = {auprc:.3f})")

    plt.xlabel("Recall (Sensitivity)", fontsize=12)
    plt.ylabel("Precision (PPV)", fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(loc="upper right", fontsize=10)
    plt.grid(True, linestyle=":", alpha=0.6)

    final_path = _ensure_plot_path(save_path)
    plt.savefig(final_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("PR Curve saved → %s", final_path)


def plot_reliability_diagrams(
    models_preds: Dict[str, Tuple[np.ndarray, np.ndarray]],
    save_path: Union[str, Path],
    n_bins: int = 10,
    title: str = "Reliability Diagram (Calibration Curve)",
) -> None:
    """Plots calibration curves for multiple models on a single graph."""
    plt.figure(figsize=(8, 8))
    plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")

    for name, (y_true, y_prob) in models_preds.items():
        prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins)
        plt.plot(prob_pred, prob_true, "s-", label=name, alpha=0.8, linewidth=2)

    plt.xlabel("Mean predicted probability", fontsize=12)
    plt.ylabel("Fraction of positives (True probability)", fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.6)

    final_path = _ensure_plot_path(save_path)
    plt.savefig(final_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("Reliability diagram saved → %s", final_path)


def verify_and_plot_calibration(
    y_true: np.ndarray,
    probs_uncalibrated: np.ndarray,
    probs_calibrated: np.ndarray,
    model_name: str,
    save_path: Union[str, Path],
    n_bins: int = 10,
) -> None:
    """
    Plots a Reliability Diagram comparing uncalibrated vs calibrated probabilities.

    Args:
        y_true: Ground truth labels
        probs_uncalibrated: Raw model probabilities (before temperature scaling)
        probs_calibrated: Calibrated probabilities (after temperature scaling)
        model_name: Name to display (e.g., "EHR MLP", "4-Modality Fusion")
        save_path: Path to save the figure (can be directory or full path)
        n_bins: Number of bins for calibration curve
    """

    def _ece(y_true, y_prob, n_bins=10):
        """Expected Calibration Error: weighted mean |confidence - accuracy| across bins."""
        bin_edges = np.linspace(0, 1, n_bins + 1)
        bin_ids = np.digitize(y_prob, bin_edges[1:-1])
        ece = 0.0
        n = len(y_true)
        for b in range(n_bins):
            mask = bin_ids == b
            if not mask.any():
                continue
            conf = y_prob[mask].mean()
            acc = y_true[mask].mean()
            ece += (mask.sum() / n) * abs(conf - acc)
        return ece

    brier_uncal = brier_score_loss(y_true, probs_uncalibrated)
    brier_cal = brier_score_loss(y_true, probs_calibrated)
    ece_uncal = _ece(y_true, probs_uncalibrated, n_bins=n_bins)
    ece_cal = _ece(y_true, probs_calibrated, n_bins=n_bins)

    logger.info(f"--- Calibration Proof: {model_name} ---")
    logger.info(f"Uncalibrated: Brier={brier_uncal:.4f}, ECE={ece_uncal:.4f}")
    logger.info(f"Calibrated:   Brier={brier_cal:.4f}, ECE={ece_cal:.4f}")

    # Compute calibration curves
    prob_true_uncal, prob_pred_uncal = calibration_curve(
        y_true, probs_uncalibrated, n_bins=n_bins
    )
    prob_true_cal, prob_pred_cal = calibration_curve(
        y_true, probs_calibrated, n_bins=n_bins
    )

    plt.figure(figsize=(9, 8))
    plt.plot([0, 1], [0, 1], "k:", label="Perfectly Calibrated")

    plt.plot(
        prob_pred_uncal,
        prob_true_uncal,
        "s-",
        color="red",
        label=f"Uncalibrated (Brier: {brier_uncal:.4f}, ECE: {ece_uncal:.4f})",
        alpha=0.85,
        linewidth=2.2,
    )

    plt.plot(
        prob_pred_cal,
        prob_true_cal,
        "o-",
        color="blue",
        label=f"Calibrated (Brier: {brier_cal:.4f}, ECE: {ece_cal:.4f})",
        alpha=0.85,
        linewidth=2.2,
    )

    plt.ylabel("Actual Fraction of Positives", fontsize=12)
    plt.xlabel("Mean Predicted Probability", fontsize=12)
    plt.title(f"Calibration Proof — {model_name}", fontsize=14, pad=15)
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.6)

    final_path = _ensure_plot_path(save_path)
    if final_path.is_dir() or not final_path.suffix:
        final_path = (
            final_path / f"{model_name.lower().replace(' ', '_')}_calibration_proof.pdf"
        )

    plt.savefig(final_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Calibration proof saved → {final_path}")


def plot_xai_tornado(patient_data: dict, save_path: str, archetype: str):
    """
    Plots a diverging horizontal bar chart (Tornado Plot) for a single patient's XAI attributions.
    """
    pos_drivers = patient_data.get("top_positive", [])
    neg_drivers = patient_data.get("top_negative", [])

    if not pos_drivers and not neg_drivers:
        logger.warning("No drivers found for tornado plot.")
        return

    all_drivers = pos_drivers + neg_drivers
    # Sort by absolute score for true tornado layout
    all_drivers.sort(key=lambda x: abs(x["score"]), reverse=True)

    labels = []
    scores = []
    colors = []

    for item in all_drivers[:12]:  # Limit to top 12 for readability
        wrapped = textwrap.fill(item["token_string"], width=45)
        labels.append(wrapped)
        score = item["score"]
        scores.append(score)
        colors.append("#d62728" if score > 0 else "#1f77b4")

    fig, ax = plt.subplots(figsize=(11, 7), dpi=300)
    y_pos = np.arange(len(labels))

    ax.barh(y_pos, scores, color=colors, align="center", height=0.7)
    ax.axvline(0, color="black", linewidth=1.1)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()  # Highest impact at top

    sid = patient_data["subject_id"]
    risk = patient_data["calibrated_risk"] * 100
    label_str = f"True Label: {patient_data.get('true_label', '?')}"

    plt.title(
        f"{archetype.replace('_', ' ').title()} — XAI Tornado Plot\n"
        f"Subject {sid} | Risk: {risk:.1f}% | {label_str}",
        fontsize=13,
        pad=20,
    )
    plt.xlabel("Attribution Score (Gradient × Activation)", fontsize=11)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close()

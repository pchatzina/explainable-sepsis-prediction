"""Logistic Regression Training and Evaluation for Unimodal Sepsis Prediction.

Trains a standardized Logistic Regression model on normalized embeddings,
adjusts decision thresholds on validation set, and evaluates on test set.

Key steps:
1. Standardize embeddings using StandardScaler fit on training data
2. Train L-BFGS solver-based Logistic Regression
3. Extract probability predictions on val/test
4. Evaluate with locked thresholds via ModelEvaluator
"""

import logging
import joblib
import numpy as np
import torch
import json

from src.utils.config import Config
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from typing import List, Dict
from src.calibration.temperature_scaling import LBFGSCalibrator

logger = logging.getLogger(__name__)


def train_eval_lr(
    modality: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    test_ids: List[int],
) -> Dict[str, float]:
    """Train, calibrate, and evaluate Logistic Regression on normalized embeddings.

    Args:
        modality: Modality name (e.g., 'ehr', 'ecg') for logging and directory paths.
        X_train, y_train: Training embeddings and labels.
        X_val, y_val: Validation embeddings and labels.
        X_test, y_test: Test embeddings and labels.
        test_ids: List of subject IDs for test set (for result tracking).

    Outputs:
        Saves:
        - models/{modality}/lr/model.joblib: Trained LR model
        - models/{modality}/lr/scaler.joblib: StandardScaler used in training
        - results/unimodal/metrics/{modality}/: Evaluation metrics
        - results/unimodal/predictions/{modality}/: Test predictions
    """
    logger.info(f"--- Training Logistic Regression for {modality} ---")
    output_dir = Config.DIR_MODELS / modality / "lr"
    metrics_dir = Config.RESULTS_UNIMODAL_METRICS_DIR / modality
    predictions_dir = Config.RESULTS_UNIMODAL_PREDICTIONS_DIR / modality
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir.mkdir(parents=True, exist_ok=True)

    # Standardize features: fit on train, apply to all splits
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # Persist scaler for inference
    joblib.dump(scaler, output_dir / "scaler.joblib")

    # Train L-BFGS Logistic Regression (stable, converges reliably)
    model = LogisticRegression(solver="lbfgs", max_iter=2000, random_state=42)
    model.fit(X_train_scaled, y_train)
    joblib.dump(model, output_dir / "model.joblib")

    # --- Extract Logits (Decision Function) ---
    # For Scikit-Learn LR, decision_function returns the raw pre-activation log-odds
    val_logits_np = model.decision_function(X_val_scaled)
    test_logits_np = model.decision_function(X_test_scaled)

    # Convert to PyTorch tensors for the calibrator
    val_logits = torch.tensor(val_logits_np, dtype=torch.float32)
    val_targets = torch.tensor(y_val, dtype=torch.float32)
    test_logits = torch.tensor(test_logits_np, dtype=torch.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Temperature Calibration ---
    logger.info("--- Running LBFGS Temperature Calibration ---")
    logits_dict = {"final": val_logits}
    calibrator = LBFGSCalibrator(max_iter=50, lr=0.01)
    optimal_temps = calibrator.fit(logits_dict, val_targets, device=device)
    best_temp = optimal_temps["final"]

    # Store temperature in centralized tracking file
    temps_file = (
        Config.RESULTS_DIR / "unimodal" / "master_calibration_temperatures.json"
    )
    temps_file.parent.mkdir(parents=True, exist_ok=True)

    existing_temps = json.loads(temps_file.read_text()) if temps_file.exists() else {}
    existing_temps[f"{modality}_lr"] = best_temp
    temps_file.write_text(json.dumps(existing_temps, indent=4))
    logger.info(
        f"Saved {modality}_lr calibration temperature ({best_temp:.4f}) to {temps_file}"
    )

    # --- Apply Calibration for Inference ---
    # Scale logits by temperature and apply sigmoid to get final probabilities
    val_probs_calibrated = torch.sigmoid(val_logits / best_temp).cpu().numpy()
    test_probs_calibrated = torch.sigmoid(test_logits / best_temp).cpu().numpy()

    # --- Final Evaluation ---
    from src.evaluation.evaluator import ModelEvaluator

    evaluator = ModelEvaluator(
        run_name="lr",
        modality=modality,
        metrics_dir=metrics_dir,
        predictions_dir=predictions_dir,
    )

    evaluator.evaluate(
        y_true_val=y_val,
        y_prob_val=val_probs_calibrated,
        y_true_test=y_test,
        y_prob_test=test_probs_calibrated,
        subject_ids_test=test_ids,
    )
    logger.info(f"Finished LR Training, Calibration, and Evaluation for {modality}")

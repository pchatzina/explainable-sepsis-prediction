"""XGBoost Training and Evaluation for Unimodal Sepsis Prediction.

Trains an XGBoost classifier on normalized embeddings with GPU acceleration,
adjusts decision thresholds on validation set, and evaluates on test set.

Key steps:
1. Convert train/val/test arrays to DMatrix format (XGBoost internal format)
2. Configure GPU-accelerated histogram tree method
3. Train with early stopping on validation AUC+AUCPR
4. Extract probability predictions on val/test
5. Evaluate with locked thresholds via ModelEvaluator
"""

import logging
import xgboost as xgb
import torch
import json

from src.utils.config import Config
from src.calibration.temperature_scaling import LBFGSCalibrator

logger = logging.getLogger(__name__)


def train_eval_xgboost(
    modality, X_train, y_train, X_val, y_val, X_test, y_test, test_ids
):
    """Train, calibrate, and evaluate XGBoost on normalized embeddings.

    Args:
        modality: Modality name (e.g., 'ehr', 'ecg') for logging and directory paths.
        X_train, y_train: Training embeddings and labels.
        X_val, y_val: Validation embeddings and labels.
        X_test, y_test: Test embeddings and labels.
        test_ids: List of subject IDs for test set (for result tracking).

    Outputs:
        Saves:
        - models/{modality}/xgboost/model.json: Trained XGBoost model
        - results/unimodal/metrics/{modality}/: Evaluation metrics
        - results/unimodal/predictions/{modality}/: Test predictions
    """
    logger.info(f"--- Training XGBoost for {modality} ---")
    output_dir = Config.DIR_MODELS / modality / "xgboost"
    metrics_dir = Config.RESULTS_UNIMODAL_METRICS_DIR / modality
    predictions_dir = Config.RESULTS_UNIMODAL_PREDICTIONS_DIR / modality
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir.mkdir(parents=True, exist_ok=True)

    # Convert to XGBoost DMatrix
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    dtest = xgb.DMatrix(X_test, label=y_test)

    # XGBoost hyperparameters
    params = {
        "objective": "binary:logistic",
        "eval_metric": ["auc", "aucpr"],
        "tree_method": "hist",
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "learning_rate": 0.05,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "seed": 42,
        "verbosity": 0,
    }

    # Train with early stopping: stop if validation metric doesn't improve for 50 rounds
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=1000,
        evals=[(dtrain, "train"), (dval, "val")],
        early_stopping_rounds=50,
        verbose_eval=False,
    )

    # Persist model to disk
    model.save_model(str(output_dir / "model.json"))

    # --- Extract Logits (Margins) ---
    # output_margin=True forces XGBoost to return the raw pre-logistic scores (logits)
    val_logits_np = model.predict(dval, output_margin=True)
    test_logits_np = model.predict(dtest, output_margin=True)

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
    existing_temps[f"{modality}_xgb"] = best_temp
    temps_file.write_text(json.dumps(existing_temps, indent=4))
    logger.info(
        f"Saved {modality}_xgb calibration temperature ({best_temp:.4f}) to {temps_file}"
    )

    # --- Apply Calibration for Inference ---
    # Scale logits by temperature and apply sigmoid to get final probabilities
    val_probs_calibrated = torch.sigmoid(val_logits / best_temp).cpu().numpy()
    test_probs_calibrated = torch.sigmoid(test_logits / best_temp).cpu().numpy()

    # --- Final Evaluation ---
    from src.evaluation.evaluator import ModelEvaluator

    evaluator = ModelEvaluator(
        run_name="xgboost",
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
    logger.info(
        f"Finished XGBoost Training, Calibration, and Evaluation for {modality}"
    )

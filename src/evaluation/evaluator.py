"""Threshold-locked model evaluator used by all training scripts.

ModelEvaluator derives the classification threshold from the validation set (Youden's J)
and holds it fixed for test-set evaluation, preventing threshold leakage.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from src.evaluation.metrics import compute_metrics, find_optimal_threshold

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """
    Handles model evaluation with validation-derived threshold and artifact saving.

    Produces:
      - JSON metrics file
      - CSV predictions with subject IDs for traceability
    """

    def __init__(
        self,
        run_name: str,
        modality: str,
        metrics_dir: Path,
        predictions_dir: Path,
    ):
        """
        Initialize the ModelEvaluator.

        Args:
            run_name: Identifier for the model run
            modality: The primary modality being evaluated
            metrics_dir: Path where JSON metrics will be saved
            predictions_dir: Path where CSV predictions will be saved
        """
        self.run_name = run_name
        self.modality = modality
        self.metrics_dir = Path(metrics_dir)
        self.predictions_dir = Path(predictions_dir)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self.predictions_dir.mkdir(parents=True, exist_ok=True)

    def evaluate(
        self,
        y_true_val: np.ndarray,
        y_prob_val: np.ndarray,
        y_true_test: np.ndarray,
        y_prob_test: np.ndarray,
        subject_ids_test: List[int],
    ) -> Dict[str, float]:
        """
        Perform full evaluation pipeline.

        1. Compute optimal threshold on validation set
        2. Apply threshold to test set
        3. Compute metrics and save artifacts

        Returns:
            Test set metrics dictionary.
        """
        logger.info(f"--- Evaluating {self.run_name} ({self.modality}) ---")

        # 1. Derive threshold STRICTLY from the validation set
        # This ensures the threshold is independent of test data
        val_threshold = find_optimal_threshold(y_true_val, y_prob_val)
        logger.info(f"Validation-derived threshold: {val_threshold:.4f}")

        # 2. Compute Test Metrics using the validation-derived threshold
        # All threshold-dependent metrics (F1, precision, recall) use this fixed threshold
        test_metrics = compute_metrics(
            y_true_test, y_prob_test, threshold=val_threshold
        )

        # 3. Save Artifacts locally to the specific run folder
        self._save_local_artifacts(
            test_metrics, subject_ids_test, y_true_test, y_prob_test
        )

        return test_metrics

    def _save_local_artifacts(
        self,
        metrics: dict,
        subject_ids: List[int],
        y_true: np.ndarray,
        y_prob: np.ndarray,
    ):
        """
        Saves JSON metrics and CSV predictions to the specific run's folder.

        Creates two files:
          1. {run_name}_metrics.json: Dictionary with all metrics and confusion matrix
          2. {run_name}_predictions.csv: Tabular format for downstream analysis

        Args:
            metrics: Dictionary of computed metrics (must include 'threshold' key)
            subject_ids: List of subject IDs for predictions CSV
            y_true: Ground truth test labels
            y_prob: Predicted probabilities
        """
        # Save metrics as JSON for easy parsing and archival
        metrics_path = self.metrics_dir / f"{self.run_name}_metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)

        logger.info(f"JSON metrics saved to {self.metrics_dir}")

        # Save predictions in tabular format for XAI
        preds = (y_prob >= metrics["threshold"]).astype(int)
        df = pd.DataFrame(
            {
                "subject_id": subject_ids,
                "label": y_true,
                "probability": y_prob,
                "prediction": preds,
            }
        )
        df.to_csv(
            self.predictions_dir / f"{self.run_name}_predictions.csv", index=False
        )
        logger.info(f"Predictions saved to {self.predictions_dir}")

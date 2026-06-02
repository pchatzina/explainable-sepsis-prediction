"""
Run:
    pytest tests/test_evaluator.py -v
"""

import json
import pandas as pd
import numpy as np
from src.evaluation.evaluator import ModelEvaluator
import src.evaluation.evaluator as evaluator_module


def test_model_evaluator_pipeline(tmp_path):
    metrics_dir = tmp_path / "metrics"
    predictions_dir = tmp_path / "predictions"

    # Initialize Evaluator
    evaluator = ModelEvaluator(
        run_name="lr",
        modality="mock_modality",
        metrics_dir=metrics_dir,
        predictions_dir=predictions_dir,
    )

    # Dummy data
    y_true = np.array([0, 1, 0, 1])
    y_prob = np.array([0.1, 0.9, 0.4, 0.8])
    subject_ids = [101, 102, 103, 104]

    # 1. Run evaluation
    evaluator.evaluate(
        y_true_val=y_true,
        y_prob_val=y_prob,
        y_true_test=y_true,
        y_prob_test=y_prob,
        subject_ids_test=subject_ids,
    )

    # 2. Assert local artifacts were created in their respective directories
    metrics_file = metrics_dir / "lr_metrics.json"
    preds_file = predictions_dir / "lr_predictions.csv"

    assert metrics_file.exists(), "Metrics JSON was not saved"
    assert preds_file.exists(), "Predictions CSV was not saved"

    # 3. Verify JSON contents
    with open(metrics_file, "r") as f:
        saved_metrics = json.load(f)

    assert "auroc" in saved_metrics
    assert "threshold" in saved_metrics
    assert saved_metrics["n_samples"] == 4

    # 4. Verify CSV contents
    df = pd.read_csv(preds_file)
    assert len(df) == 4
    assert list(df["subject_id"]) == subject_ids
    assert "probability" in df.columns
    assert "prediction" in df.columns


def test_threshold_locked_from_validation(tmp_path, monkeypatch):
    metrics_dir = tmp_path / "metrics"
    predictions_dir = tmp_path / "predictions"

    evaluator = ModelEvaluator(
        run_name="lock_test",
        modality="mock_modality",
        metrics_dir=metrics_dir,
        predictions_dir=predictions_dir,
    )

    y_true_val = np.array([0, 0, 1, 1])
    y_prob_val = np.array([0.10, 0.40, 0.60, 0.90])
    y_true_test = np.array([0, 1, 0, 1])
    y_prob_test = np.array([0.30, 0.70, 0.20, 0.80])
    subject_ids = [201, 202, 203, 204]

    captured = {}

    def fake_find_optimal_threshold(y_true, y_prob):
        assert np.array_equal(y_true, y_true_val)
        assert np.array_equal(y_prob, y_prob_val)
        return 0.42

    def fake_compute_metrics(y_true, y_prob, threshold):
        # Ensure test arrays are evaluated with the validation-derived threshold.
        assert np.array_equal(y_true, y_true_test)
        assert np.array_equal(y_prob, y_prob_test)
        captured["threshold"] = threshold
        return {
            "auroc": 0.5,
            "auprc": 0.5,
            "accuracy": 0.5,
            "f1": 0.5,
            "precision": 0.5,
            "recall": 0.5,
            "specificity": 0.5,
            "brier_score": 0.25,
            "ece": 0.0,
            "tp": 1,
            "fp": 1,
            "tn": 1,
            "fn": 1,
            "threshold": threshold,
            "n_samples": int(len(y_true)),
            "n_positive": int(y_true.sum()),
            "n_negative": int(len(y_true) - y_true.sum()),
        }

    monkeypatch.setattr(
        evaluator_module, "find_optimal_threshold", fake_find_optimal_threshold
    )
    monkeypatch.setattr(evaluator_module, "compute_metrics", fake_compute_metrics)

    evaluator.evaluate(
        y_true_val=y_true_val,
        y_prob_val=y_prob_val,
        y_true_test=y_true_test,
        y_prob_test=y_prob_test,
        subject_ids_test=subject_ids,
    )

    assert captured["threshold"] == 0.42

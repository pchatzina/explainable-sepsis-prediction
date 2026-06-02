"""
Run:
    pytest tests/test_metrics.py -v
"""

import numpy as np
import pytest
from src.evaluation.metrics import compute_metrics, find_optimal_threshold


@pytest.fixture()
def random_predictions():
    """Realistic-sized random predictions."""
    rng = np.random.RandomState(42)
    y_true = rng.randint(0, 2, size=500)
    y_prob = rng.rand(500)
    return y_true, y_prob


def test_find_optimal_threshold(random_predictions):
    y_true, y_prob = random_predictions
    threshold = find_optimal_threshold(y_true, y_prob)
    assert 0.0 <= threshold <= 1.0
    assert isinstance(threshold, float)


def test_compute_metrics_structure_and_bounds(random_predictions):
    y_true, y_prob = random_predictions
    metrics = compute_metrics(y_true, y_prob, threshold=0.5)

    expected_keys = {
        "auroc",
        "auprc",
        "accuracy",
        "f1",
        "precision",
        "recall",
        "specificity",
        "brier_score",
        "ece",
        "tp",
        "fp",
        "tn",
        "fn",
        "threshold",
        "n_samples",
        "n_positive",
        "n_negative",
    }

    assert set(metrics.keys()) == expected_keys

    # Check bounds
    for key in (
        "auroc",
        "auprc",
        "accuracy",
        "f1",
        "precision",
        "recall",
        "specificity",
    ):
        assert 0.0 <= metrics[key] <= 1.0, f"{key} out of bounds"

#!/bin/bash
# ========================================================================
# PHASE 5 — Unimodal Baseline Training
# ========================================================================
# Purpose:
#   Train, calibrate, and evaluate standalone classifiers (LR, XGBoost, MLP)
#   for each modality, then generate a cross-classifier comparison report.
#
# High-level flow:
#   1. Train, calibrate, and evaluate unimodal classifiers for all four modalities
#   2. Generate cross-classifier comparison report
#
# Outputs:
#   - Per-modality calibrated model checkpoints
#   - Unimodal comparison report: results/unimodal/
#
# Usage:
#   ./scripts/05_run_unimodal_training.sh
#
# Strict mode:
#   -e: exit on error
#   -u: exit on undefined variable
#   -o pipefail: exit if any pipeline stage fails

set -euo pipefail

# === Stage 1/2 — Train, Calibrate, and Evaluate Unimodal Classifiers ===
echo "[Stage 1/2] Running unimodal train, calibrate, and evaluate (4 modalities)..."
python -m src.training.unimodal.run_unimodal modalities=4_modality

# === Stage 2/2 — Generate Comparison Report ===
echo "[Stage 2/2] Generating cross-classifier comparison report..."
python -m src.evaluation.report_generator --group unimodal

# === Validation ===
echo "[Tests] Validating metrics, MLP sanity, trainer factory, and evaluator..."
pytest tests/test_metrics.py tests/test_mlp_sanity.py tests/test_trainer_factory.py tests/test_evaluator.py -v

echo "[Phase 5] Complete."

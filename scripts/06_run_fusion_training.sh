#!/bin/bash
# ========================================================================
# PHASE 6 — Fusion Model Training
# ========================================================================
# Purpose:
#   Train, calibrate, and evaluate the late-fusion model in both pretrained-
#   heads and scratch configurations, then generate a fusion comparison report.
#
# High-level flow:
#   1. Train and calibrate the fusion model from pretrained unimodal heads and from scratch
#   2. Generate fusion experiment comparison report
#
# Outputs:
#   - Fusion model checkpoints (pretrained and scratch configurations)
#   - Fusion comparison report: results/fusion/
#
# Usage:
#   ./scripts/06_run_fusion_training.sh
#
# Strict mode:
#   -e: exit on error
#   -u: exit on undefined variable
#   -o pipefail: exit if any pipeline stage fails

set -euo pipefail

# === Stage 1/2 — Train and Calibrate Fusion Models ===
echo "[Stage 1/2] Running fusion train, calibrate, and evaluate (pretrained and scratch)..."
python -m src.training.fusion.run_fusion modalities=4_modality training=pretrained
python -m src.training.fusion.run_fusion modalities=4_modality training=scratch

# === Stage 2/2 — Generate Fusion Comparison Report ===
echo "[Stage 2/2] Generating fusion experiment comparison report..."
python -m src.evaluation.report_generator --group fusion

# === Validation ===
echo "[Tests] Validating fusion model training pipeline..."
pytest tests/test_fusion_model.py -v

echo "[Phase 6] Complete."

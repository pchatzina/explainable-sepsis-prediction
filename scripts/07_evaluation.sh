#!/bin/bash
# ========================================================================
# PHASE 7 — Evaluation
# ========================================================================
# Purpose:
#   Evaluate trained models on the held-out test set, quantify each modality's
#   incremental contribution, and extract learned gating network weights.
#
# High-level flow:
#   1. Run Incremental Value Analysis to measure per-modality contribution
#   2. Extract learned gating network weights from the late-fusion model
#   3. Generate evaluation reports and figures
#
# Outputs:
#   - IVA results and modality gating weights: results/evaluation/
#   - Comparison and calibration figures: results/figures/
#
# Usage:
#   ./scripts/07_evaluation.sh
#
# Strict mode:
#   -e: exit on error
#   -u: exit on undefined variable
#   -o pipefail: exit if any pipeline stage fails

set -euo pipefail

# === Stage 1/3 — Incremental Value Analysis ===
# Masks auxiliary modalities one at a time on the gold cohort (all 4 modalities present).
echo "[Stage 1/3] Running Incremental Value Analysis..."
python -m src.evaluation.incremental_value_analysis modalities=4_modality training=pretrained

# === Stage 2/3 — Extract Gating Weights ===
echo "[Stage 2/3] Extracting learned gating network weights..."
python -m src.explainability.extract_modality_weights

# === Stage 3/3 — Generate Reports and Figures ===
echo "[Stage 3/3] Generating evaluation reports and figures..."
python -m src.evaluation.report_generator --group iva
python -m src.evaluation.report_generator --group macro_xai
python -m src.evaluation.figure_generator --group comparison
python -m src.evaluation.figure_generator --group calibration

# === Validation ===
echo "[Tests] Validating IVA gold cohort masking and modality weights export..."
pytest tests/test_incremental_value_analysis.py tests/test_extract_modality_weights.py -v

echo "[Phase 7] Complete."

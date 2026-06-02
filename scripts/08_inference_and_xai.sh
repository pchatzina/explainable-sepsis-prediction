#!/bin/bash
# ========================================================================
# PHASE 8 — Inference & Explainability
# ========================================================================
# Purpose:
#   Run micro-level XAI and clinical archetype case studies on the test set,
#   producing attribution reports and tornado plots for the dissertation.
#
# High-level flow:
#   1. Generate FEMR test batches truncated at anchor time
#   2. Extract clinical archetypes (textbook, reassuring, smart mistake, tug-of-war)
#   3. Audit dataset noise via cohort-wide mass-fraction analysis
#   4. Generate dissertation case study Markdown reports
#   5. Generate XAI tornado plots for archetypes
#   6. Targeted per-patient inference (manual invocation only)
#
# Outputs:
#   - Attribution reports: results/archetypes/
#   - Tornado plots: results/figures/xai/
#
# Usage:
#   ./scripts/08_inference_and_xai.sh
#
# See also:
#   - src/explainability/README.md
#
# Strict mode:
#   -e: exit on error
#   -u: exit on undefined variable
#   -o pipefail: exit if any pipeline stage fails

set -euo pipefail

# === Stage 1/5 — Generate Test Batches ===
echo "[Stage 1/5] Generating FEMR test batches (anchor-time truncated)..."
python -m src.explainability.generate_test_batches

# === Stage 2/5 — Extract Clinical Archetypes ===
echo "[Stage 2/5] Extracting clinical archetypes..."
python -m src.explainability.extract_clinical_archetypes

# === Stage 3/5 — Dataset Noise Audit ===
echo "[Stage 3/5] Auditing dataset noise (cohort-wide mass-fraction analysis)..."
python -m src.explainability.find_dataset_noise

# === Stage 4/5 — Generate Case Study Reports ===
echo "[Stage 4/5] Generating dissertation case study Markdown reports..."
python -m src.evaluation.report_generator --group archetypes

# === Stage 5/5 — Generate Tornado Plots ===
echo "[Stage 5/5] Generating XAI tornado plots for archetypes..."
python -m src.evaluation.figure_generator --group xai

# Targeted per-patient inference is available outside the operative pipeline:
#   python -m src.explainability.clinical_inference_xai --subject_id <ID>
# <ID> is a MIMIC-IV subject identifier and must not be hardcoded.
# See src/explainability/README.md for invocation guidance and DUA implications.

# === Validation ===
echo "[Tests] Validating XAI pipeline outputs..."
pytest tests/test_xai.py -v

echo "[Phase 8] Complete."

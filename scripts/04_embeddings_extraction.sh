#!/bin/bash
# ========================================================================
# PHASE 4 — Embeddings Extraction
# ========================================================================
# Purpose:
#   Extract, L2-normalise, and validate embeddings from all four modalities
#   (EHR, ECG, CXR text, CXR image) using frozen foundation models.
#
# High-level flow:
#   1. Extract raw embeddings from each foundation model sequentially
#   2. L2-normalise all embeddings independently per split
#   3. Generate an embeddings health report
#
# Outputs:
#   - Raw embeddings: {modality}/embeddings/raw/{split}_embeddings_raw.pt
#   - Normalised embeddings: {modality}/embeddings/normalized/{split}_embeddings.pt
#   - Health report: results/embeddings/embeddings_health_report.md
#
# Usage:
#   ./scripts/04_embeddings_extraction.sh
#
# See also:
#   - src/embeddings/README.md
#
# Strict mode:
#   -e: exit on error
#   -u: exit on undefined variable
#   -o pipefail: exit if any pipeline stage fails

set -euo pipefail

# === Stage 1/3 — Extract Raw Embeddings ===
# Loads each foundation model sequentially with GPU memory cleanup between runs.
echo "[Stage 1/3] Extracting raw embeddings from foundation models..."
python -m src.embeddings.run_pipeline

# === Stage 2/3 — Normalise Embeddings ===
echo "[Stage 2/3] Applying L2-normalisation to all embeddings..."
python -m src.embeddings.normalize_embeddings

# === Stage 3/3 — Generate Health Report ===
echo "[Stage 3/3] Generating embeddings health report..."
python -m src.evaluation.report_generator --group embeddings

# === Validation ===
echo "[Tests] Validating structural integrity and absence of NaNs in normalised embeddings..."
pytest tests/test_embeddings.py -v

echo "[Phase 4] Complete."

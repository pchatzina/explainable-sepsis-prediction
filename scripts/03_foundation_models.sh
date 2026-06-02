#!/bin/bash
# ========================================================================
# PHASE 3 — Foundation Models Provisioning & Pretraining
# ========================================================================
# Purpose:
#   Download pre-trained foundation models from HuggingFace and pretrain
#   the EHR-specific MOTOR transformer on MIMIC-IV data.
#
# High-level flow:
#   1. Download and cache CXR-text (Bio_ClinicalBERT) and ECG (ecg-fm) models
#   2. Pretrain the EHR MOTOR transformer on MIMIC-IV data
#
# Outputs:
#   - CXR text model weights: Config.MODEL_CXR_TXT_PRETRAINED_DIR
#   - ECG model weights: Config.MODEL_ECG_PRETRAINED_DIR
#   - MOTOR inference bundle: Config.MODEL_EHR_MOTOR_WEIGHTS_DIR
#
# Usage:
#   ./scripts/03_foundation_models.sh
#
# See also:
#   - src/models/foundation/ehr/README.md
#
# Strict mode:
#   -e: exit on error
#   -u: exit on undefined variable
#   -o pipefail: exit if any pipeline stage fails

set -euo pipefail

# === Stage 1/2 — Download & Cache Foundation Models ===
echo "[Stage 1/2] Downloading and caching foundation models from HuggingFace..."
python -m src.models.foundation.setup_foundation_models

# === Stage 2/2 — EHR Foundation Model Pretraining ===
echo "[Stage 2/2] Pretraining EHR foundation model (MOTOR)..."
bash scripts/motor_foundation_model_pretraining.sh

# === Validation ===
echo "[Tests] Validating foundation model weights and MOTOR pretraining artifacts..."
pytest tests/test_motor_pipeline.py tests/test_foundation_weights.py -v

echo "[Phase 3] Complete."

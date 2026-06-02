#!/bin/bash
# ========================================================================
# MOTOR Foundation Model Pretraining
# ========================================================================
# Purpose:
#   Pretrain the MOTOR self-supervised time-to-event (TTE) transformer on
#   MIMIC-IV EHR data to generate patient representations for embedding extraction.
#
# High-level flow:
#   1. Prepare pretraining artifacts (ontology, tokenizer, train/val batches)
#   2. Pretrain the MOTOR transformer from scratch
#
# Outputs:
#   - Preparation artifacts (Config.MODEL_EHR_MOTOR_PRETRAINING_FILES_DIR):
#     ontology.pkl, motor_task.pkl, tokenizer/, train_batches/, val_batches/
#   - Inference bundle (Config.MODEL_EHR_MOTOR_WEIGHTS_DIR):
#     config.json, model.safetensors, dictionary.msgpack
#
# Usage:
#   ./scripts/motor_foundation_model_pretraining.sh
#
# Or via Phase 3 orchestrator:
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

# === Stage 1/2 — Prepare Pretraining Artifacts ===
# This stage is idempotent and can be safely rerun.
echo "[Stage 1/2] Preparing pretraining data artifacts..."
python -m src.models.foundation.ehr.prepare_motor

# === Stage 2/2 — Pretrain MOTOR ===
# Uses gradient accumulation and bf16 mixed precision for efficiency.
echo "[Stage 2/2] Pretraining MOTOR foundation model..."
python -m src.models.foundation.ehr.pretrain_motor

echo "[MOTOR Pretraining] Complete."

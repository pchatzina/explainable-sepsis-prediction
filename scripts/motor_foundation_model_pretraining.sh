#!/bin/bash
# motor_foundation_model_pretraining.sh
# Runs the MOTOR EHR foundation model pretraining pipeline as described in src/models/foundation/ehr/README.md

set -euo pipefail

# Set project root (assumes script is run from project root or scripts/)
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# Step 1: Data Preparation
# This step is idempotent and can be safely rerun.
echo "[MOTOR] Step 1: Preparing pretraining data artifacts..."
python -m src.models.foundation.ehr.prepare_motor

# Step 2: Pretraining
# Trains the MOTOR transformer model from scratch.
echo "[MOTOR] Step 2: Pretraining MOTOR foundation model..."
python -m src.models.foundation.ehr.pretrain_motor

echo "[MOTOR] Pretraining pipeline complete."

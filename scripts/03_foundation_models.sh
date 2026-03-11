#!/bin/bash
# 03_foundation_models.sh
# Sets up the foundation models required for embedding extraction
# Usage: ./scripts/03_foundation_models.sh

set -euo pipefail

# Set project root (assumes script is run from project root or scripts/)
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "[FOUNDATION MODELS] Downloading and caching foundation models..."
python -m src.models.foundation.setup_foundation_models

echo "[FOUNDATION MODELS] Running EHR foundation model pretraining..."
# Calls the MOTOR pretraining pipeline
bash scripts/motor_foundation_model_pretraining.sh

echo "[FOUNDATION MODELS] All steps complete."
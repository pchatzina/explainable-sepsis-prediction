#!/bin/bash
# 04_embeddings_extraction.sh
# Orchestrates the extraction, normalization, and inspection of multimodal embeddings.
# Usage: ./scripts/04_embeddings_extraction.sh

set -euo pipefail

# Set project root (assumes script is run from project root or scripts/)
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "========================================================"
echo " PHASE 3: MULTIMODAL EMBEDDING EXTRACTION "
echo "========================================================"

# Step 1: Run the Extraction Pipeline
echo -e "\n[Step 1/3] Extracting raw embeddings from Foundation Models..."
echo "This will load EHR, CXR (Text & Image), and ECG models sequentially."
echo "--------------------------------------------------------"
python -m src.embeddings.run_pipeline

# Step 2: Normalize the Embeddings
echo -e "\n[Step 2/3] Applying L2-Normalization to all embeddings..."
echo "Reading from /raw/ directories and saving to /normalized/ directories."
echo "--------------------------------------------------------"
python -m src.embeddings.normalize_embeddings

# Step 3: Generate the Health Report
echo -e "\n[Step 3/3] Generating Embeddings Health Report..."
echo "Inspecting the final normalized embeddings for NaNs, Infs, and shapes."
echo "--------------------------------------------------------"
python -m src.embeddings.inspect_embeddings --all

echo "========================================================"
echo "✅ EMBEDDING PHASE COMPLETED SUCCESSFULLY!"
echo "Check your configured REPORT_EMBEDDINGS_DIR for the embeddings_health_report.md summary."
echo "========================================================"
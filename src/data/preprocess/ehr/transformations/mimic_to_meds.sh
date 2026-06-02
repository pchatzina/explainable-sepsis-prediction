#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -euo pipefail

# Check if a dataset parameter is provided
if [ -z "${1:-}" ]; then
  echo "Error: No dataset specified."
  echo "Usage: $0 <dataset_name>"
  echo "Example: $0 pretraining"
  exit 1
fi

DATASET_NAME=$1

NUM_SHARDS="${NUM_SHARDS:-50}"
NUM_PROC="${NUM_PROC:-1}"
NUM_THREADS="${NUM_THREADS:-4}"

if [ -z "${RAW_BASE:-}" ] || [ -z "${PROCESSED_BASE:-}" ]; then
  echo "Error: Required environment variables RAW_BASE or PROCESSED_BASE are missing."
  exit 1
fi

# Construct specific paths
INPUT_PATH="${RAW_BASE}"
OUTPUT_PATH="${PROCESSED_BASE}/mimic-iv-meds"
READER_OUTPUT_PATH="${PROCESSED_BASE}/mimic-iv-meds-reader"

echo "=========================================="
echo "Starting pipeline for dataset: $DATASET_NAME"
echo "Input: $INPUT_PATH"
echo "Output: $OUTPUT_PATH"
echo "Shards: $NUM_SHARDS | Procs: $NUM_PROC | Threads: $NUM_THREADS"
echo "=========================================="

# 1. Run ETL
echo "Running meds_etl_mimic..."
meds_etl_mimic "$INPUT_PATH" "$OUTPUT_PATH" \
    --num_shards "$NUM_SHARDS" \
    --num_proc "$NUM_PROC" \
    --backend polars

# 2. Run Convert
echo "Running meds_reader_convert..."
meds_reader_convert "$OUTPUT_PATH" "$READER_OUTPUT_PATH" \
    --num_threads "$NUM_THREADS"

# 3. Run Verify
echo "Running patched meds_reader_verify..."
python -m src.data.preprocess.ehr.transformations.custom_meds_reader_verify "$OUTPUT_PATH" "$READER_OUTPUT_PATH"

echo "Pipeline finished successfully for $DATASET_NAME"

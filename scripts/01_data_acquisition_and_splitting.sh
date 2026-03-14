#!/bin/bash
# 01_data_acquisition_and_splitting.sh
# Pipeline for MIMIC-IV data acquisition, database setup, and cohort splitting
# Usage: ./scripts/01_data_acquisition_and_splitting.sh

# Strict mode: 
# -e: exit on error
# -u: exit on undefined variable
# -o pipefail: exit if any command in a pipeline fails
set -euo pipefail

# Define a standard PSQL command that strictly fails on errors
# -v ON_ERROR_STOP=1 ensures psql exits with a non-zero code if a query fails
PSQL_CMD="psql -d mimiciv -v ON_ERROR_STOP=1 -f"

# --- Phase 1: Base Cohort Definition ---
echo "[Phase 1] Creating generic EHR cohort table..."
$PSQL_CMD src/data/acquisition/db/cohort_creation/create_generic_ehr_cohort.sql

# --- Phase 2: Metadata Acquisition ---
echo "[Phase 2] Downloading metadata CSVs..."
python -m src.data.acquisition.downloads.download_metadata_files

echo "[Phase 2] Loading metadata into Postgres..."
python -m src.data.acquisition.db.setup.load_metadata

# --- Phase 3: Multimodal Cohort Construction ---
echo "[Phase 3] Creating generic CXR/ECG cohort tables..."
$PSQL_CMD src/data/acquisition/db/cohort_creation/create_generic_modalities_cohort.sql

echo "[Phase 3] Creating final cohort tables..."
$PSQL_CMD src/data/acquisition/db/cohort_creation/create_final_cohort.sql

# --- Phase 4: Raw Data Acquisition ---
echo "[Phase 4] Downloading CXR images and reports..."
python -m src.data.acquisition.downloads.download_cxr_files

echo "[Phase 4] Downloading ECG signals..."
python -m src.data.acquisition.downloads.download_ecg_files

# --- Phase 5: Stratified Splitting ---
echo "[Phase 5] Creating patient strata and dataset splits..."
$PSQL_CMD src/data/acquisition/db/data_splitting/create_training_split.sql

# --- Phase 6: Extract Master Cohort ---
echo "[Phase 6] Extracting master cohort and saving to parquet..."
python -m src.data.acquisition.extract_cohort_splits

echo "[Done] Data acquisition, splitting, and cohort extraction completed successfully."
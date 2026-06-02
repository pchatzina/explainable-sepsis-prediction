#!/bin/bash
# ========================================================================
# PHASE 1 — Data Acquisition & Cohort Splitting
# ========================================================================
# Purpose:
#   Extract the sepsis cohort from MIMIC-IV with temporal anchoring,
#   identify modality availability within a 66-hour pre-anchor window,
#   download raw files, and produce a deterministic train/validation/test split.
#
# High-level flow:
#   1. Define base EHR cohort with temporal anchoring (anchor = onset − 6 h)
#   2. Download and load CXR and ECG metadata schemas
#   3. Identify modality availability and finalise cohort
#   4. Download raw CXR images and ECG signal files
#   5. Stratified train/validation/test split (70%/15%/15%)
#   6. Serialise finalised cohort as master_cohort.parquet
#
# Outputs:
#   - master_cohort.parquet in the processed-data directory
#   - Postgres tables: mimiciv_ext.cohort, cohort_cxr, cohort_ecg
#
# Usage:
#   ./scripts/01_data_acquisition_and_splitting.sh
#
# Strict mode:
#   -e: exit on error
#   -u: exit on undefined variable
#   -o pipefail: exit if any pipeline stage fails

set -euo pipefail

# ON_ERROR_STOP=1 ensures psql exits with a non-zero code on SQL errors.
PSQL_CMD="psql -d mimiciv -v ON_ERROR_STOP=1 -f"

# === Stage 1/6 — Base Cohort Definition ===
# Positives: onset >= 24 h post-admission; anchor = onset - 6 h (prediction window end).
# Negatives: no sepsis history; anchor = ICU intime + median time-to-sepsis - 6 h.
# Prevents data leakage: modality availability is filtered to the pre-anchor window only.
echo "[Stage 1/6] Creating generic EHR cohort table..."
$PSQL_CMD src/data/acquisition/db/cohort_creation/create_generic_ehr_cohort.sql

# === Stage 2/6 — Metadata Acquisition ===
echo "[Stage 2/6] Downloading and loading CXR and ECG metadata..."
python -m src.data.acquisition.downloads.download_metadata_files
python -m src.data.acquisition.db.setup.load_metadata

# === Stage 3/6 — Multimodal Cohort Construction ===
# Deduplicates to 1 admission per patient using modality availability as the tie-breaker.
echo "[Stage 3/6] Creating modality-filtered and finalised cohort tables..."
$PSQL_CMD src/data/acquisition/db/cohort_creation/create_generic_modalities_cohort.sql
$PSQL_CMD src/data/acquisition/db/cohort_creation/create_final_cohort.sql

# === Stage 4/6 — Raw Data Acquisition ===
echo "[Stage 4/6] Downloading CXR images, reports, and ECG signals..."
python -m src.data.acquisition.downloads.download_cxr_files
python -m src.data.acquisition.downloads.download_ecg_files

# === Stage 5/6 — Stratified Splitting ===
# Stratification preserves sepsis label, modality signature, and recency tiers.
echo "[Stage 5/6] Creating patient strata and dataset splits..."
$PSQL_CMD src/data/acquisition/db/data_splitting/create_training_split.sql

# === Stage 6/6 — Master Cohort Extraction ===
# The output parquet file is the canonical input for all downstream phases.
echo "[Stage 6/6] Extracting master cohort to parquet..."
python -m src.data.acquisition.extract_cohort_splits

# === Validation ===
echo "[Tests] Validating database integrity, deterministic counts, and file accessibility..."
pytest tests/test_data_acquisition_pipeline.py tests/test_data_splitting.py -v

echo "[Phase 1] Complete."

#!/bin/bash
# ========================================================================
# PHASE 2 — Data Preprocessing
# ========================================================================
# Purpose:
#   Transform raw EHR and ECG data from Phase 1 into model-ready formats
#   for embedding extraction and training.
#
# High-level flow:
#   1. Export cohort EHR data from raw MIMIC-IV hosp and icu schemas
#   2. Export pretraining EHR data (test split excluded to prevent leakage)
#   3. Convert cohort CSVs to MEDS format via the femr pipeline
#   4. Convert pretraining CSVs to MEDS format via the femr pipeline
#   5. Generate binary labels and anchor times for the cohort
#   6. Extract and organise raw ECG signal records by subject
#   7. Preprocess ECG waveforms and serialise to .mat files
#
# Outputs:
#   - EHR raw CSV exports: RAW_EHR_COHORT_DIR, RAW_EHR_PRETRAINING_DIR
#   - EHR MEDS databases: PROCESSED_EHR_MEDS_COHORT_DIR, PROCESSED_EHR_MEDS_PRETRAINING_DIR
#   - EHR labels: PROCESSED_EHR_LABELS_DIR/labels.parquet
#   - ECG .mat files: PROCESSED_ECG_ROOT_DIR
#
# Usage:
#   ./scripts/02_data_preprocessing.sh
#
# See also:
#   - src/data/preprocess/ehr/README.md
#   - src/data/preprocess/ecg/README.md
#
# Strict mode:
#   -e: exit on error
#   -u: exit on undefined variable
#   -o pipefail: exit if any pipeline stage fails

set -euo pipefail

# === Stage 1/7 — Export Cohort EHR Data ===
echo "[Stage 1/7] Exporting cohort EHR data from raw MIMIC-IV schemas..."
python -m src.data.preprocess.ehr.run_meds_etl export-cohort

# === Stage 2/7 — Export Pretraining EHR Data ===
# The test split is excluded from this export to prevent data leakage into pretraining.
echo "[Stage 2/7] Exporting pretraining EHR data (test split excluded)..."
python -m src.data.preprocess.ehr.run_meds_etl export-pretraining

# === Stage 3/7 — Convert Cohort to MEDS Format ===
echo "[Stage 3/7] Converting cohort CSVs to MEDS format via femr pipeline..."
python -m src.data.preprocess.ehr.run_meds_etl meds-pipeline cohort

# === Stage 4/7 — Convert Pretraining Data to MEDS Format ===
echo "[Stage 4/7] Converting pretraining CSVs to MEDS format via femr pipeline..."
python -m src.data.preprocess.ehr.run_meds_etl meds-pipeline pretraining

# === Stage 5/7 — Generate EHR Labels ===
echo "[Stage 5/7] Generating binary labels and anchor times for the cohort..."
python -m src.data.preprocess.ehr.ehr_labels

# === Stage 6/7 — Extract ECG Signal Records ===
echo "[Stage 6/7] Extracting and organising raw ECG signal records by subject..."
python -m src.data.preprocess.ecg.records

# === Stage 7/7 — Preprocess ECG Waveforms ===
echo "[Stage 7/7] Preprocessing ECG waveforms to .mat format..."
python -m src.data.preprocess.ecg.signals

# === Validation ===
echo "[Tests] Validating EHR export integrity (no test leakage) and ECG preprocessing..."
pytest tests/test_ehr_preprocessing.py tests/test_ecg_preprocessing.py -v

echo "[Phase 2] Complete."

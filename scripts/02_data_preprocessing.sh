#!/bin/bash
# 02_data_preprocessing.sh
# Runs EHR and ECG preprocessing pipelines as described in src/data/preprocess/ehr/README.md and src/data/preprocess/ecg/README.md
# Usage: ./scripts/02_data_preprocessing.sh

set -euo pipefail

# Set project root (assumes script is run from project root or scripts/)
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# --- EHR Preprocessing ---
echo "[EHR] Step 1: Export cohort-only EHR data..."
python -m src.utils.bash_wrapper export-cohort

echo "[EHR] Step 2: Export pretraining EHR data..."
python -m src.utils.bash_wrapper export-pretraining

echo "[EHR] Step 3: Convert exported CSVs to MEDS format (cohort)..."
python -m src.utils.bash_wrapper meds-pipeline cohort

echo "[EHR] Step 4: Convert exported CSVs to MEDS format (pretraining)..."
python -m src.utils.bash_wrapper meds-pipeline pretraining

echo "[EHR] Step 5: Generate prediction labels and anchor times for the cohort..."
python -m src.data.preprocess.ehr.ehr_labels

# --- ECG Preprocessing ---
echo "[ECG] Step 1: Extract and organize raw ECG records..."
python -m src.data.preprocess.ecg.records

echo "[ECG] Step 2: Preprocess ECG signals..."
python -m src.data.preprocess.ecg.signals

echo "[ECG] Step 3: Generate prediction labels and anchor times for ECG cohort..."
python -m src.data.preprocess.ecg.ecg_labels

echo "[ECG] Step 4: Create fairseq-compatible manifest files..."
python -m src.data.preprocess.ecg.create_manifests

echo "[DATA PREPROCESSING] All steps complete."

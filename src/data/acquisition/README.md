# Data Acquisition & Database Setup

This folder contains the pipeline for acquiring the MIMIC-IV data, setting up the necessary Postgres schemas, and building the multimodal cohort for the thesis.

## ⚠️ Prerequisites
### Make sure the [Prerequisites & Setup](../../../README.md) is completed.

---

## Phase 1: Base Cohort Definition
Define the initial pool of Sepsis Positive and Negative patients based on clinical criteria (EHR data only).

```bash
# Creates table: mimiciv_ext.generic_ehr_cohort
psql -d mimiciv -f src/data/acquisition/db/cohort_creation/create_generic_ehr_cohort.sql
```

## Phase 2: Metadata
Download metadata files (CSVs) and initialize the secondary database schemas (mimiciv_cxr, mimiciv_ecg) required to query modality availability.

```bash
# 1. Download metadata CSVs (and initializes directories)
python -m src.data.acquisition.downloads.download_metadata_files

# 2. Create schemas and load metadata into Postgres
python -m src.data.acquisition.db.setup.load_metadata
```

## Phase 3: Multimodal Cohort Construction
Filter the base cohort to keep only patients with valid modalities and deduplicate admissions.

```bash
# 1. Identify all valid CXR/ECG studies within the time window
# Creates tables: mimiciv_ext.generic_cxr_cohort, mimiciv_ext.generic_ecg_cohort
psql -d mimiciv -f src/data/acquisition/db/cohort_creation/create_generic_modalities_cohort.sql

# 2. Finalize the cohort (1 admission per patient, 1 study per modality)
# Creates tables: mimiciv_ext.cohort, mimiciv_ext.cohort_cxr, mimiciv_ext.cohort_ecg
psql -d mimiciv -f src/data/acquisition/db/cohort_creation/create_final_cohort.sql
```

## Phase 4: Raw Data Acquisition
Download the specific raw data files (Images and Signals) for the finalized cohort only.

```bash
# Download CXR Images (JPG) and Reports
python -m src.data.acquisition.downloads.download_cxr_files

# Download ECG Signals (.dat/.hea)
python -m src.data.acquisition.downloads.download_ecg_files
```

## Phase 5: Stratified Splitting
Split the final cohort into Train (70%), Val (15%), and Test (15%).

```bash
# Creates tables: mimiciv_ext.patient_strata, mimiciv_ext.dataset_splits
psql -d mimiciv -f src/data/acquisition/db/data_splitting/create_training_split.sql
```
---

## 🚀 Run the Full Pipeline

To execute all phases sequentially with strict error handling, run the master script:

```bash
./scripts/01_data_acquisition_and_splitting.sh
```

## 🧪 Validation & Testing

After the pipeline completes, run the validation test suite to ensure the extracted cohort matches the exact deterministic thesis counts (Total: 15,513 | Positive: 8,436) and that no data leakage exists between splits.

```bash
# Database integrity, leakage checks, file corruption verification
pytest tests/test_data_splitting.py tests/test_data_acquisition_pipeline.py
```
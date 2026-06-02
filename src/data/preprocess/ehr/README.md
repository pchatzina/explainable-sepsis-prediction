# EHR Preprocessing

Converts raw MIMIC-IV `hosp` and `icu` event tables into MEDS-format patient databases via the `meds_reader` ETL pipeline, and generates binary sepsis labels with per-subject anchor times. Two separate databases are produced — a cohort database covering all 15,513 prediction subjects, and a pretraining database covering the broader MIMIC-IV population minus the cohort's test split — and they are kept disjoint to prevent data leakage into the MOTOR foundation model.

## Pipeline

The four commands below are orchestrated by [`scripts/02_data_preprocessing.sh`](../../../../scripts/02_data_preprocessing.sh).

**1. Export cohort data** — queries PostgreSQL and streams MIMIC-IV events for subjects in `mimiciv_ext.cohort` to CSV:

```bash
python -m src.data.preprocess.ehr.run_meds_etl export-cohort
```

**2. Export pretraining data** — streams events for all MIMIC-IV subjects *except* those in the held-out test split:

```bash
python -m src.data.preprocess.ehr.run_meds_etl export-pretraining
```

**3. Convert to MEDS format** — runs the `meds_reader` ingestion pipeline over both CSV exports:

```bash
python -m src.data.preprocess.ehr.run_meds_etl meds-pipeline cohort
python -m src.data.preprocess.ehr.run_meds_etl meds-pipeline pretraining
```

**4. Generate labels** — produces `labels.parquet` with per-subject binary sepsis labels and prediction-time anchor timestamps:

```bash
python -m src.data.preprocess.ehr.ehr_labels
```

The cohort/pretraining split is essential: MOTOR is pretrained on the pretraining database, and the cohort's test split is excluded from it so that test subjects contribute no signal before held-out evaluation.

## Outputs

| Artifact | Config attribute |
|---|---|
| Cohort MEDS database | `Config.PROCESSED_EHR_MEDS_COHORT_DIR` |
| Pretraining MEDS database | `Config.PROCESSED_EHR_MEDS_PRETRAINING_DIR` |
| Sepsis labels + anchor times | `Config.PROCESSED_EHR_LABELS_DIR` |

## See also

- [Main README](../../../../README.md)
- [scripts/02_data_preprocessing.sh](../../../../scripts/02_data_preprocessing.sh)
- [src/models/foundation/ehr/README.md](../../../models/foundation/ehr/README.md) — MOTOR pretraining consumes the pretraining MEDS database

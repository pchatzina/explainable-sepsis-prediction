# Tests

Validation for all pipeline phases, from raw data acquisition through evaluation and XAI. Tests split into two types: *integration tests* that run against a live PostgreSQL instance or precomputed artifacts (Phases 1–4 and parts of Phase 7), and *unit tests* that run on synthetic data with no external dependencies. `conftest.py` provides a module-scoped PostgreSQL engine fixture shared by the integration tests.

## Quick Start

```bash
# Run all tests
pytest tests/ -v

# Run a specific test file
pytest tests/test_ehr_preprocessing.py -v

# Run a specific test method
pytest tests/test_ehr_preprocessing.py::TestCohortExport::test_no_non_cohort_subjects -v
```

## Test Files Index

Each test file's own docstring enumerates the individual test methods and classes.

| File | Phase | Type | Requires |
|---|---|---|---|
| `test_data_acquisition_pipeline.py` | 1 — Data Acquisition | Integration | PostgreSQL + Phase 1 complete |
| `test_ehr_preprocessing.py` | 2 — EHR Preprocessing | Integration | PostgreSQL + EHR exports |
| `test_ecg_preprocessing.py` | 2 — ECG Preprocessing | Integration | ECG preprocessing complete |
| `test_foundation_weights.py` | 3 — Foundation Models | Integration | Downloaded model weights |
| `test_motor_pipeline.py` | 3 — MOTOR Pretraining | Integration | prepare_motor + pretrain_motor complete |
| `test_embeddings.py` | 4 — Embedding Extraction | Integration | Embedding extraction complete |
| `test_metrics.py` | 5 — Unimodal Baseline | Unit | — |
| `test_mlp_sanity.py` | 5 — Unimodal Baseline | Unit | — |
| `test_trainer_factory.py` | 5 — Unimodal Baseline | Unit | — |
| `test_fusion_model.py` | 6 — Multimodal Fusion | Unit | — |
| `test_evaluator.py` | 7 — Evaluation & XAI | Unit | — |
| `test_incremental_value_analysis.py` | 7 — Evaluation & XAI | Unit | — |
| `test_extract_modality_weights.py` | 7 — Evaluation & XAI | Unit | — |
| `test_xai.py` | 7 — Evaluation & XAI | Unit | — |
| `test_plotting.py` | Cross-phase | Unit | — |

## Expected Deterministic Counts

These values ensure reproducibility across runs. Any mismatch indicates cohort definition changes, data leakage, or database corruption:

| Entity | Count | Test |
|--------|-------|------|
| Total patients | 15,513 | test_final_cohort_exists_and_has_data |
| Sepsis+ cases | 8,436 | test_class_balance |
| CXR studies | 2,212 | test_cxr_cohort_integrity |
| ECG studies | 7,146 | test_ecg_cohort_integrity |
| Train split | 10,845 | test_stratification_ratios_exact |
| Validation split | 2,326 | test_stratification_ratios_exact |
| Test split | 2,342 | test_stratification_ratios_exact |

## Running Tests by Category

```bash
# Unit tests only (fast, no external dependencies)
pytest tests/test_metrics.py tests/test_mlp_sanity.py tests/test_trainer_factory.py \
       tests/test_fusion_model.py tests/test_evaluator.py tests/test_xai.py \
       tests/test_plotting.py tests/test_incremental_value_analysis.py \
       tests/test_extract_modality_weights.py -v

# Phase 1 (requires live DB + Phase 1 complete)
pytest tests/test_data_acquisition_pipeline.py -v

# Phase 2 (requires live DB + Phase 2 complete)
pytest tests/test_ehr_preprocessing.py tests/test_ecg_preprocessing.py -v

# Phase 3 (requires model weights + pretraining complete)
pytest tests/test_foundation_weights.py tests/test_motor_pipeline.py -v

# Phase 4 (requires embedding extraction complete)
pytest tests/test_embeddings.py -v

# All integration tests
pytest tests/test_data*.py tests/test_ehr*.py tests/test_ecg*.py \
       tests/test_motor*.py tests/test_foundation*.py tests/test_embeddings.py -v
```

## Skip Behaviour

Integration tests skip gracefully when their required artifacts are not yet present. A test that depends on the ECG preprocessed directory (`Config.PROCESSED_ECG_ROOT_DIR`), the MOTOR preparation artifacts, or the foundation model inference bundle will skip rather than fail if those outputs haven't been generated. This makes it safe to run the full test suite at any intermediate pipeline stage without triggering spurious failures.

## See also

- [Main README](../README.md)
- [src/data/preprocess/ehr/](../src/data/preprocess/ehr/README.md)
- [src/data/preprocess/ecg/](../src/data/preprocess/ecg/README.md)
- [src/embeddings/](../src/embeddings/README.md)
- [src/models/foundation/ehr/](../src/models/foundation/ehr/README.md)
- [src/explainability/](../src/explainability/README.md)

# Data Preprocessing

This folder contains the pipeline for EHR and ECG preprocessing.

## 🚀 Run the Full Pipeline

To execute all phases sequentially with strict error handling, run the master script:

```bash
./scripts/02_data_preprocessing.sh
```

## 🧪 Validation & Testing

After the pipeline completes, run the validation test suite.

```bash
# Database integrity, leakage checks, file corruption verification
pytest tests/test_ehr_preprocessing.py tests/test_ecg_preprocessing.py
```
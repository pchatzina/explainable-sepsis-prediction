# ECG Preprocessing

Organises raw MIMIC-IV ECG WFDB records from the PhysioNet download and converts them to preprocessed `.mat` files ready for embedding extraction. The two steps below are orchestrated by [`scripts/02_data_preprocessing.sh`](../../../../scripts/02_data_preprocessing.sh).

## Prerequisites

The custom `fairseq-signals` fork is required for ECG foundation model embedding extraction (Phase 4) and must be installed before running any ECG pipeline step:

```bash
git clone https://github.com/pchatzina/fairseq-signals.git
cd fairseq-signals
pip install -e .
```

## Pipeline

Run from the project root in sequence:

```bash
# 1. Extract and organise raw WFDB records
python -m src.data.preprocess.ecg.records

# 2. Preprocess ECG signals to .mat files
python -m src.data.preprocess.ecg.signals
```

## Outputs

Processed `.mat` files land under `Config.PROCESSED_ECG_ROOT_DIR`. Each file contains a 12-lead signal matrix with shape `(12, num_samples)`.

## See also

- [Main README](../../../../README.md)
- [scripts/02_data_preprocessing.sh](../../../../scripts/02_data_preprocessing.sh)
- [src/embeddings/README.md](../../../embeddings/README.md) — ECG-FM embedding extraction reads from `Config.PROCESSED_ECG_ROOT_DIR`

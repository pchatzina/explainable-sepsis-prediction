# ECG Preprocessing

This folder contains scripts and instructions for preparing ECG waveform data for downstream modeling.

## Prerequisites

- You must clone the custom fork of fairseq-signals (required for ECG foundation model embedding extraction):

```bash
git clone https://github.com/pchatzina/fairseq-signals.git
```

- Install the package (editable mode recommended):

```bash
cd fairseq-signals
pip install -e .
```

## Preprocessing Steps

Run the following scripts sequentially from the project root:

```bash
# 1. Extract and organize raw ECG records
python -m src.data.preprocess.ecg.records

# 2. Preprocess ECG signals (e.g., filtering, resampling)
python -m src.data.preprocess.ecg.signals

# 3. Generate prediction labels and anchor times for the ECG cohort
python -m src.data.preprocess.ecg.ecg_labels

# 4. Create fairseq-compatible manifest files for downstream embedding extraction
python -m src.data.preprocess.ecg.create_manifests
```

Each script will output intermediate files in the appropriate processed data directories as defined by your configuration.

See the main README for environment and configuration setup.

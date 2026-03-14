# Embedding Extraction Scripts

Extract per-subject embedding vectors from frozen foundation and pretrained models.
Each modality has a dedicated script that reads preprocessed data, runs
inference with the its model, and saves standardized `.pt` files.

## Output Format

All scripts produce the same `.pt` dict per split:

```python
{
    "embeddings": Tensor[N, D],   # float32, one row per subject/sample
    "labels":     List[int],      # 0 or 1
    "subject_ids": List[int],     # subject_id
}
```

Files produced per modality: `train_embeddings.pt`, `valid_embeddings.pt`, `test_embeddings.pt`

## Modality Scripts
**EHR** — `ehr_embeddings.py`
Runs the frozen MOTOR transformer over the MEDS cohort database.

- Input: Cohort MEDS database, pretrained MOTOR model, master_cohort.parquet

- Output: Config.PROCESSED_EHR_EMBEDDINGS_DIR/{split}_embeddings.pt

**ECG** — `ecg_embeddings.py`
Runs the frozen ECG-FM model over preprocessed ECG .mat files dynamically chunked into 5-second windows.

- Input: Preprocessed .mat files, pretrained ECG-FM model, master_cohort.parquet

- Output: Config.PROCESSED_ECG_EMBEDDINGS_DIR/{split}_embeddings.pt

**CXR Text (Reports)** — `cxr_txt_embeddings.py`
Runs the Bio_ClinicalBERT model over raw MIMIC-CXR report text.

- Input: Raw MIMIC-CXR ZIP file, offline ClinicalBERT weights, master_cohort.parquet

- Output: Config.PROCESSED_CXR_TXT_EMBEDDINGS_DIR/{split}_embeddings.pt

**CXR Images** — `cxr_img_embeddings.py`
Runs the DenseNet121 torchxrayvision model over frontal CXR .jpg files.

- Input: Raw CXR .jpg directory, offline DenseNet weights, master_cohort.parquet

- Output: Config.PROCESSED_CXR_IMG_EMBEDDINGS_DIR/{split}_embeddings.pt

## Orchestration — `run_pipeline.py`
Executes all modality extractors sequentially while safely managing GPU memory (gc.collect() and torch.cuda.empty_cache()) between models.

```bash
python -m src.embeddings.run_pipeline
```

## Normalization — `normalize_embeddings.py`
L2-normalize all raw embedding .pt files and save them to the normalized directories.

```bash
python -m src.embeddings.normalize_embeddings
```

## 🧪 Validation & Testing
### 1. Automated Testing (PyTest)
A comprehensive integration test suite is provided to validate the structural integrity, data types, and absence of data leakage in the generated .pt files.

```bash
pytest tests/test_embeddings.py -v
```

### 2. Inspect Embeddings — `inspect_embeddings.py`
Utility script to generate a statistical health report of the extracted tensors (verifying dimensionality, checking for NaNs/Infs, and printing label distribution).

```bash
# Inspect a specific modality in the console:
python -m src.embeddings.inspect_embeddings --modality ehr

# Inspect all modalities and generate a master Markdown report in REPORT_EMBEDDINGS_DIR:
python -m src.embeddings.inspect_embeddings --all
```

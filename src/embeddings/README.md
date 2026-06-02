# Embedding Extraction

Runs frozen foundation models over preprocessed data for each modality and saves fixed-size per-subject embedding vectors. These vectors are the primary input to all downstream classifiers (logistic regression, XGBoost, MLP, and late-fusion).

## Output Format

All extractors produce one `.pt` file per split (`train_embeddings.pt`, `valid_embeddings.pt`, `test_embeddings.pt`) with a shared dict structure:

```python
{
    "embeddings":  Tensor[N, D],  # one row per subject
    "labels":      List[int],     # 0 or 1
    "subject_ids": List[int],     # MIMIC-IV subject_id
}
```

The directories listed in the table below hold L2-normalized embeddings. Raw (pre-normalization) versions live in the corresponding `*_RAW_EMBEDDINGS_DIR` Config attributes.

## Modality Scripts

| Script | Modality | Foundation Model | Output Directory |
|---|---|---|---|
| `ehr_embeddings.py` | EHR | MOTOR (FEMRModel) | `Config.PROCESSED_EHR_EMBEDDINGS_DIR` |
| `ecg_embeddings.py` | ECG | ECG-FM (wanglab) | `Config.PROCESSED_ECG_EMBEDDINGS_DIR` |
| `cxr_txt_embeddings.py` | CXR Reports | Bio_ClinicalBERT | `Config.PROCESSED_CXR_TXT_EMBEDDINGS_DIR` |
| `cxr_img_embeddings.py` | CXR Images | DenseNet121 (torchxrayvision) | `Config.PROCESSED_CXR_IMG_EMBEDDINGS_DIR` |

ECG signals are chunked into 5-second windows (2,500 samples at 500 Hz) with padding, allowing variable-length recordings to be batched on the GPU. All other extractors process one subject at a time.

## Orchestration

`run_pipeline.py` executes all four extractors sequentially, explicitly releasing GPU memory between models:

```bash
python -m src.embeddings.run_pipeline
```

## Normalization

`normalize_embeddings.py` applies row-wise L2 normalization across all raw `.pt` files and writes results to the normalized directories:

```bash
python -m src.embeddings.normalize_embeddings
```

## Validation

```bash
pytest tests/test_embeddings.py -v
```

## See also

- [Main README](../../README.md)
- [src/data/preprocess/ehr/README.md](../data/preprocess/ehr/README.md)
- [src/data/preprocess/ecg/README.md](../data/preprocess/ecg/README.md)
- [src/models/foundation/ehr/README.md](../models/foundation/ehr/README.md)

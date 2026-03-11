# MOTOR Foundation Model — Pretraining Pipeline

This directory contains the two-stage pipeline for pretraining a **MOTOR** (Medical Outcome Time-to-event Representations) transformer on MIMIC-IV EHR data.  MOTOR learns temporal representations of patient medical histories via a self-supervised time-to-event prediction objective.

> **Reference:** Steinberg et al., *"MOTOR: A Time-To-Event Foundation Model For Structured Medical Records"*, 2023.  
> **Library:** [`femr`](https://github.com/som-shahlab/femr) by Stanford SHAH Lab.

## Prerequisites

| Dependency | Purpose |
|---|---|
| **MEDS-formatted data** | The EHR export + MEDS ETL must have been run first (`src/data/preprocess/ehr/`).  The pretraining database lives at `Config.PROCESSED_EHR_MEDS_PRETRAINING_DIR`. |
| **Athena vocabulary** | Download from [athena.ohdsi.org](https://athena.ohdsi.org/) and unpack at `Config.MODEL_EHR_MOTOR_VOCAB_DIR`.  Then run `java -Dumls-apikey=xxx -jar cpt4.jar 5`. Replace "xxx" with UMLS API KEY. |
| **UMLS API key** | Required by the Athena CPT4 tool — obtain a free account at [uts.nlm.nih.gov](https://uts.nlm.nih.gov/). |
| **Python packages** | `femr`, `meds_reader`, `transformers`, `datasets`, `xformers` (see `environment.yml`). |
| **GPU** | Training uses CUDA with bf16 precision. |

## Pipeline Overview

### Step 1: `prepare_motor.py` — Data Preparation

Builds intermediate artifacts from the MEDS pretraining database:

1. **Ontology** — Loads OHDSI concept graph from Athena vocabularies, prunes unused ontologies (SPL, HemOnc) to save memory.
2. **Split** — Deterministic 90/10 train/val hash-based split (seed 97).  This split is for pretraining only — downstream evaluation uses the cohort splits from PostgreSQL.
3. **Tokenizer** — Hierarchical vocabulary (16384 codes) over the medical ontology.
4. **MOTOR task** — Fits time-to-event distribution parameters for the pretraining head (8192 tasks, 8 bins, 512-dim final layer).
5. **Batches** — Tokenises the database into HuggingFace `Dataset` objects (train + val), saved to disk.

Every step is **idempotent**: existing artifacts are loaded from disk, so the script can safely resume after a crash.

```bash
python -m src.models.foundation.ehr.prepare_motor
```

**Output** (`Config.MODEL_EHR_MOTOR_PRETRAINING_FILES_DIR`):
```
pretraining_files/
├── ontology.pkl        # Pruned OHDSI ontology
├── split.csv           # Subject-level train/val assignment
├── tokenizer/          # HierarchicalTokenizer checkpoint
├── motor_task.pkl      # Pretraining task distribution params
├── train_batches/      # HF Dataset (tokenised, PyTorch format)
└── val_batches/        # HF Dataset (tokenised, PyTorch format)
```

### Step 2: `pretrain_motor.py` — Pretraining

Trains a `FEMRModel` from scratch using HuggingFace's `Trainer` with the MOTOR objective.

| Hyperparameter | Value | Rationale |
|---|---|---|
| Layers | 12 | "Base" model; use 6 for "small" |
| Learning rate | 1e-4 | Standard for transformer pretraining |
| Effective batch size | 32 | 1 × 32 gradient accumulation |
| Precision | bf16 | Saves VRAM without fp16 instability |
| Weight decay | 0.1 | Standard AdamW regularisation |
| Adam β₂ | 0.95 | Slightly lower than default for stability |
| Warmup | 500 steps | Gradual LR ramp-up |
| Epochs | 30 | Hard cap; early stopping typically triggers first |
| Early stopping patience | 5 | Stop after 5 evals (2 500 steps) with no val-loss improvement |
| Checkpoints kept | 2 | Saves disk; best model is reloaded at end |

```bash
python -m src.models.foundation.ehr.pretrain_motor
```

**Intermediate output** (`Config.MODEL_EHR_MOTOR_PRETRAINING_OUT_DIR`) — ephemeral, cleaned up automatically:
```
pretraining_output/
├── checkpoint-*/       # Trainer checkpoints (rolling, max 2)
├── runs/               # TensorBoard logs
└── final/              # Best model snapshot (copied, then removed)
```

**Final output** (`Config.MODEL_EHR_MOTOR_WEIGHTS_DIR`) — inference-ready bundle:
```
model/
├── config.json            # Model architecture config
├── model.safetensors      # Trained weights
└── dictionary.msgpack     # Tokenizer vocabulary
```

This is the self-contained directory that `femr.models.transformer.compute_features()` expects as `model_path`.

Monitor training:
```bash
tensorboard --logdir $MODEL_EHR_MOTOR_PRETRAINING_OUT_DIR/runs
```

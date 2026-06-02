# Evaluating the Clinical Utility of Multimodal Data for Early Sepsis Prediction: An Explainable Late-Fusion Approach

![Status](https://img.shields.io/badge/Status-Work_in_Progress-yellow)
![License](https://img.shields.io/badge/License-MIT-blue)
![MIMIC-IV](https://img.shields.io/badge/Dataset-MIMIC--IV-red)

---

## 📌 Project Overview

This repository implements a **multimodal neural network** for **proactive sepsis prediction** in ICU patients. The system predicts sepsis **6 hours before clinical onset** using a late-fusion architecture that intelligently integrates four complementary data modalities and naturally handles missing data through dynamic gating. Check [description](./docs/description.pdf) for details.

<!-- TODO: Add dissertation citation (DOI / institutional link) when available after submission. -->

---

## 📊 System Architecture

### Four Data Modalities

The system processes four clinical data streams in parallel:

1. **Tabular EHR** (Vitals, Lab Values)
   - Source: MIMIC-IV hosp and icu schema
   - Timeline: Continuous events from admission to anchor time

2. **Electrocardiograms (ECG)** 
   - Source: MIMIC-IV-ECG (12-lead waveforms)
   - Sampling: Latest ECG within 66-hour pre-anchor window

3. **Chest X-Ray Images**
   - Source: MIMIC-CXR-JPG dataset
   - Selection: Best capture position (PA > AP > Lateral), highest resolution
   - Sampling: Latest CXR within 66-hour pre-anchor window

4. **Chest X-Ray Reports** (Radiology Text)
   - Source: MIMIC-CXR structured reports
   - Extraction: Free-text clinical impressions from radiologists

---

## 🛠️ Pipeline Architecture

The project is organized as an **8-phase orchestrated pipeline** with strict temporal and data-leakage safeguards:

### Phase 1: Data Acquisition & Splitting
Extracts the sepsis cohort from MIMIC-IV with temporal anchoring (prediction window set to six hours before sepsis onset), identifies modality availability within a sixty-six-hour pre-anchor window, and downloads raw CXR images and ECG signal files for the finalised cohort. The phase produces a deterministic master cohort table partitioned into train/validation/test splits, with referential metadata linking each subject to its available modalities. Outputs land in the configured raw-data directory for modality files, and master_cohort.parquet in the processed-data directory.

- **Script**: `scripts/01_data_acquisition_and_splitting.sh`

### Phase 2: Data Preprocessing
Transforms raw EHR data from MIMIC-IV into MEDS format using the femr pipeline, generates binary labels and anchor times for the cohort, and preprocesses ECG waveforms into model-ready .mat files. The phase produces MEDS-format patient databases for both the cohort and the pretraining subset, a labels.parquet file, and subject-organised ECG signal files. EHR outputs land in the processed EHR directories for cohort and pretraining data; ECG outputs land in the processed ECG root directory.

- **Script**: `scripts/02_data_preprocessing.sh`

### Phase 3: Foundation Models
Downloads and caches pre-trained foundation models (Bio_ClinicalBERT for CXR text and ecg-fm for ECG) from HuggingFace, and pretrains the MOTOR EHR transformer on MIMIC-IV data. The phase produces cached model weights for the CXR-text and ECG models, and a MOTOR inference bundle containing the transformer weights, vocabulary, and architecture configuration. CXR-text and ECG model weights land in their respective config-specified directories; MOTOR artifacts land in the MOTOR weights directory.

- **Script**: `scripts/03_foundation_models.sh`

### Phase 4: Embedding Extraction
Uses each frozen foundation model to extract raw embeddings for every patient in each data split, then L2-normalises all embeddings and generates a health report verifying structural integrity. The phase produces raw and normalised embedding tensors per modality and per split, alongside a Markdown health report. Raw tensors land under {modality}/embeddings/raw/ and normalised tensors under {modality}/embeddings/normalized/; the health report lands in results/embeddings/.

- **Script**: `scripts/04_embeddings_extraction.sh`

### Phase 5: Unimodal Baseline Training
Trains, calibrates, and evaluates standalone classifiers — logistic regression, XGBoost, and MLP — independently for each of the four modalities, then generates a cross-classifier comparison report. The phase produces calibrated model checkpoints for each classifier–modality combination and a cross-classifier performance report. Model checkpoints and the comparison report land in the configured results directory under results/unimodal/.

- **Script**: `scripts/05_run_unimodal_training.sh`

### Phase 6: Fusion Training
Trains and calibrates the late-fusion model in two configurations — initialised from pretrained unimodal heads and trained from scratch — to assess the benefit of pretraining. The phase produces calibrated fusion model checkpoints for both configurations and a fusion experiment comparison report. Checkpoints and the comparison report land in results/fusion/.

- **Script**: `scripts/06_run_fusion_training.sh`

### Phase 7: Evaluation
Evaluates trained models on the held-out test set by running Incremental Value Analysis to quantify each modality's contribution and extracting the learned gating network weights from the late-fusion model. The phase produces IVA results, modality gating weight tables, evaluation reports, and calibration and comparison figures. Outputs land under results/incremental_value_analysis/, results/fusion/, and results/final_figures/.

- **Script**: `scripts/07_evaluation.sh`

### Phase 8: Inference & XAI
Runs micro-level explainability analysis on the test set by extracting clinical archetypes, auditing dataset noise, and generating attribution reports and tornado plots for the dissertation. The phase produces Markdown case study reports for each archetype, XAI tornado plot PDFs, and a noise audit summary. Reports land under results/explainability/reports/, figures under results/final_figures/xai_case_studies/; per-patient attribution dumps remain on the local machine and are not committed to the repository.

- **Script**: `scripts/08_inference_and_xai.sh`

---

## 🚀 Getting Started

### Prerequisites & Setup

**1. Data Access**

You must be a credentialed researcher on [PhysioNet](https://physionet.org/) with signed Data Use Agreements (DUA) for:
- [**MIMIC-IV (v2.2)**](https://physionet.org/content/mimiciv/2.2/) - Core EHR data
- [**MIMIC-IV-ECG**](https://physionet.org/content/mimic-iv-ecg/1.0/) - 12-lead waveforms
- [**MIMIC-CXR-JPG**](https://physionet.org/content/mimic-cxr-jpg/2.1.0/) - Chest X-Ray images
- [**MIMIC-CXR**](https://physionet.org/content/mimic-cxr/2.1.0/) - Radiologist reports

**2. Database Infrastructure**

A local PostgreSQL database with preloaded schemas:
- [**MIMIC-IV buildmimic (postgres)**](https://github.com/MIT-LCP/mimic-code/tree/main/mimic-iv/buildmimic/postgres) - Core EHR schema
- [**MIMIC-IV concepts (postgres)**](https://github.com/MIT-LCP/mimic-code/tree/main/mimic-iv/concepts_postgres) - Derived concepts (sepsis3, etc.)

**3. Python Environment**

This project requires **Python 3.10**. We recommend Conda for dependency management:

```bash
# Create environment from file
conda env create -f environment.yml

# Activate environment
conda activate thesis_py310
```

**4. Configuration**

Create a `.env` file at the project root:

```bash
# Copy template and fill in your PhysioNet login and database credentials
cp .env.example .env
```

**5. Athena Vocabulary & UMLS API**

- **UMLS API Key**: Required to generate the OMOP vocabularies. Obtain a free account at [uts.nlm.nih.gov](https://uts.nlm.nih.gov/).
- **Athena Vocabulary**: Downloaded from [athena.ohdsi.org](https://athena.ohdsi.org/).

For instructions check [MOTOR Pretraining Pipeline](./src/models/foundation/ehr/README.md).

### Running the Full Pipeline

Optuna policy: hyperparameter tuning is intentionally run in single-process mode (`n_jobs=1`) in the scope of this project.

```bash
# Phase 1: Data Acquisition
./scripts/01_data_acquisition_and_splitting.sh

# Phase 2: Data Preprocessing
./scripts/02_data_preprocessing.sh

# Phase 3: Foundation Models
./scripts/03_foundation_models.sh

# Phase 4: Embedding Extraction
./scripts/04_embeddings_extraction.sh

# Phase 5: Unimodal Training
./scripts/05_run_unimodal_training.sh

# Phase 6: Fusion Training
./scripts/06_run_fusion_training.sh

# Phase 7: Evaluation
./scripts/07_evaluation.sh

# Phase 8: Inference & XAI
./scripts/08_inference_and_xai.sh
```

---

## Results & Reproducibility

The pipeline produces aggregate metrics, predictions, and reports under the `results/` directory, which is gitignored. Per-patient outputs are not included in this repository, as MIMIC-IV is a restricted-access dataset under PhysioNet's Data Use Agreement and patient-level artifacts cannot be shared publicly.

The repository contains the full pipeline code, configuration, and tests required to reproduce the dissertation's findings on a credentialed MIMIC-IV installation. The dissertation document itself reports aggregate metrics, anonymised case studies for the explainability analysis, and the complete experimental setup. Reproducing the results requires:

- Credentialed access to MIMIC-IV, MIMIC-IV-ECG, MIMIC-CXR, and MIMIC-CXR-JPG (see the Prerequisites section).
- Running the eight pipeline phases in order on the local environment.
- The deterministic seeds and configurations in `src/utils/config.py` and `conf/`.

Patient-level artifacts produced by the pipeline (per-subject predictions, attribution dumps, tornado-plot PDFs) remain on the local machine and are not committed to the repository.

---

## ⚖️ License & Data Usage

### Code License

The source code is released under the **MIT License**. See [LICENSE.md](./docs/LICENSE.md) for details.

### Data License (Critical)

This project uses **MIMIC-IV**, a restricted-access dataset. The data is **NOT** included in this repository.

- You must be a credentialed researcher on [PhysioNet](https://physionet.org/)
- You must sign the Data Use Agreement for MIMIC-IV, MIMIC-CXR, and MIMIC-IV-ECG

---

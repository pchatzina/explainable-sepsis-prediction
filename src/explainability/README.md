# Explainability

Post-hoc XAI on the EHR branch of the multimodal pipeline. The module runs Captum LayerGradientXActivation attributions through the frozen MOTOR transformer to identify which clinical tokens drove each sepsis prediction, conducts a cohort-wide mass-fraction audit of attribution signal, and extracts learned gating weights from the late-fusion model for macro-level modality contribution analysis.

## Operative Pipeline

Scripts invoked by the automated bash scripts, in execution order:

| Script | Purpose | Invoked by |
|---|---|---|
| `generate_test_batches.py` | Tokenises the test split into FEMR batch format, truncated at each subject's sepsis anchor time, and saves the dataset to disk | `08_inference_and_xai.sh` |
| `extract_clinical_archetypes.py` | Two-pass selection of 25 patients across five clinical archetypes (Textbook Sepsis, Reassuring Normal, Smart Mistake, Missed Sepsis, Tug-of-War); runs Captum attribution only on the 5 top candidates per archetype | `08_inference_and_xai.sh` |
| `find_dataset_noise.py` | Cohort-wide accumulation of absolute Captum attribution scores per token; reports the fraction of total attribution mass carried by clinically-prefixed tokens and by LOINC Part (LP) hierarchy nodes | `08_inference_and_xai.sh` |
| `extract_modality_weights.py` | Inference pass over the test set using the trained 4-modality late-fusion model; extracts per-subject gating weights (w_ehr, w_ecg, w_cxr_img, w_cxr_txt) and synergy coefficient β = 1 − max(wᵢ) | `07_evaluation.sh` |

`clinical_inference_xai.py` is not invoked by any bash script; see [Targeted Inference](#targeted-inference) below.

## Utilities

`xai_utils.py` provides shared helpers used across all attribution scripts:

- **`EHREndToEndWrapper`** — bridges the MOTOR transformer and the trained EHR MLP into a single `nn.Module` compatible with Captum. A dummy input tensor satisfies Captum's API without entering the MOTOR computation graph; a `patient_pos` argument isolates a single subject's representation from the packed FEMR batch to prevent cross-patient gradient leakage. The forward pass runs MOTOR in `bfloat16` autocast and returns logits in `float32`.
- **`load_pipeline`** — loads the MOTOR weights and the calibrated EHR MLP checkpoint, extracts the temperature scaling factor, reads the optimal threshold from the metrics JSON, and returns a ready-to-use `EHREndToEndWrapper`.
- **`get_token_string_resilient`** — maps a MOTOR vocabulary token ID to a human-readable string via Athena CONCEPT.csv lookup (OHDSI vocabularies), MIMIC-IV `d_labitems`/`d_items` lookup, or falls back to the raw code string or a structured representation for numeric-range and text tokens.
- Supporting helpers: `prepare_batch`, `load_raw_msgpack_dictionary`, `load_athena_mapping`, `load_mimic_mapping`.

## Diagnostics

`build_leaf_concept_set.py` is a **negative-result diagnostic** preserved as evidence of methodology. It classifies every MOTOR vocabulary token as `leaf_code`, `internal_code`, `non_clinical_code`, `numeric`, `text`, or `other` by querying the FEMR ontology's children map, then checks whether a principled leaf-only attribution filter is feasible. The result was negative: specific LOINC/SNOMED/ATC/RxNorm concepts were also classified as `internal_code` via OMOP cross-vocabulary mapping edges, making a leaf filter infeasible. The operative pipeline therefore retains only LOINC Part (LP) categorical removal as the defensible filtering step. No bash script invokes this file; run it manually if re-evaluating the filtering approach.

```bash
python -m src.explainability.build_leaf_concept_set
```

## Targeted Inference

To run the end-to-end clinical inference and micro-XAI pipeline for a specific patient:

```bash
python -m src.explainability.clinical_inference_xai --subject_id <ID>
```

`<ID>` is a MIMIC-IV `subject_id`. MIMIC-IV is a restricted-access dataset governed by a Data Use Agreement; patient identifiers must not be committed to a public repository.

Two convenience flags are available for random patient selection:

```bash
# Select a random sepsis-positive patient from the test set
python -m src.explainability.clinical_inference_xai --random_positive

# Select a random control patient from the test set
python -m src.explainability.clinical_inference_xai --random_negative
```

Suggested practice: supply the ID per-run via an environment variable or a local gitignored file rather than hardcoding it in any script or configuration file.

## Outputs

All outputs land under `Config.RESULTS_DIR / "explainability"`:

| Artifact | Relative path | Produced by |
|---|---|---|
| FEMR test batches | `test_batches/` | `generate_test_batches.py` |
| Clinical archetype case studies | `reports/clinical_archetypes_case_studies.json` | `extract_clinical_archetypes.py` |
| Cohort-wide token attribution ranking | `top_dataset_tokens.csv` | `find_dataset_noise.py` |
| Per-subject modality gating weights | `modality_weights/4mod_architecture/test_set_modality_weights.csv` | `extract_modality_weights.py` |
| Per-patient micro-XAI JSON | `micro_xai/patient_<ID>_xai.json` | `clinical_inference_xai.py` |
| Clinical Markdown case reports | `reports/clinical_cases/` | `clinical_inference_xai.py` |
| Token classification table (diagnostic) | `token_classification.parquet` | `build_leaf_concept_set.py` |

## See also

- [Main README](../../README.md)
- [scripts/08_inference_and_xai.sh](../../scripts/08_inference_and_xai.sh)
- [scripts/07_evaluation.sh](../../scripts/07_evaluation.sh)

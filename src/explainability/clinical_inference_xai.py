"""
End-to-End Clinical Inference & Micro-XAI Pipeline.

Simulates a retrospective Clinical Decision Support (CDS) system:
  1. Loads FEMR tokens for a patient
  2. Computes calibrated sepsis risk
  3. Uses Captum LayerGradientXActivation for token-level attributions
  4. Maps tokens to human-readable clinical concepts
  5. Generates Markdown clinical case reports

Usage:
    python -m src.explainability.clinical_inference_xai --subject_id <SUBJECT_ID>
    python -m src.explainability.clinical_inference_xai --random_positive
"""

import argparse
import json
import logging
from pathlib import Path
import random

import datasets
import pandas as pd
import torch
from captum.attr import LayerGradientXActivation

from src.explainability.xai_utils import (
    EHREndToEndWrapper,
    load_pipeline,
    prepare_batch,
    load_raw_msgpack_dictionary,
    get_token_string_resilient,
    load_mimic_mapping,
    load_athena_mapping,
)
from src.utils.config import Config

logger = logging.getLogger(__name__)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def find_random_patient(label: int, test_batches) -> int:
    """
    Randomly samples a test patient matching a specific sepsis label.

    This function finds all patients in the test set with the desired sepsis label
    AND who appear in the FEMR test_batches dataset. This ensures the sampled patient
    has preprocessed FEMR tokens available for inference.

    Args:
        label (int): Sepsis label to filter by
        test_batches: Dataset of preprocessed test FEMR batches

    Returns:
        int: Randomly selected subject_id meeting the criteria
    """
    valid_sids = set()
    for batch in test_batches:
        valid_sids.update(int(sid) for sid in batch["subject_ids"])

    cohort_df = pd.read_parquet(Config.PROCESSED_COHORT_PARQUET_FILE)
    df = cohort_df[
        (cohort_df["dataset_split"] == "test") & (cohort_df["sepsis_label"] == label)
    ]

    available_sids = [
        sid for sid in df["subject_id"].astype(int).tolist() if sid in valid_sids
    ]
    if not available_sids:
        raise ValueError(f"No patients found with label {label} in the test batches.")

    return random.choice(available_sids)


def run_clinical_inference(
    wrapper: EHREndToEndWrapper,
    test_batches_path: Path,
    target_sid: int,
    t_val: float,
    optimal_threshold: float,
):
    """
    Run full inference + micro-XAI for a single patient.

    Workflow:
      1. Locate patient in test batches
      2. Forward pass through MOTOR + MLP → calibrated risk
      3. Captum gradient attribution on embedding layer
      4. Filter and aggregate clinical tokens
      5. Save JSON + Markdown clinical report

    Output:
        JSON and Markdown files in results/explainability/
    """
    logger.info(f"Initializing Inference & Captum XAI for Subject {target_sid}...")

    # Set up Captum attribution method: compute gradients w.r.t. MOTOR embedding layer
    embedding_layer = wrapper.motor.transformer.embed_bag
    attr_algo = LayerGradientXActivation(forward_func=wrapper, layer=embedding_layer)

    test_batches = datasets.Dataset.load_from_disk(test_batches_path)
    test_batches.set_format("pt")
    raw_vocab = load_raw_msgpack_dictionary()

    # Load clinical vocabularies for token-to-concept mapping
    athena_path = Config.MODEL_EHR_MOTOR_VOCAB_DIR / "CONCEPT.csv"
    athena_mapping = load_athena_mapping(athena_path)
    mimic_mapping = load_mimic_mapping()

    target_batch_idx, patient_pos = None, None

    # 1. Locate the batch and position containing the target patient
    for b_idx, batch in enumerate(test_batches):
        sids = batch["subject_ids"].numpy().tolist()
        if target_sid in sids:
            target_batch_idx = b_idx
            # Extract unique consecutive subject_ids (each EHR patient is a contiguous sequence)
            unique_sids = []
            for sid in sids:
                if not unique_sids or unique_sids[-1] != sid:
                    unique_sids.append(sid)
            patient_pos = unique_sids.index(target_sid)
            break

    if target_batch_idx is None:
        logger.error(
            f"Subject {target_sid} not found in test batches. Run generate_test_batches.py."
        )
        return

    # 2. Extract batch data and compute predictions
    batch = test_batches[target_batch_idx]
    batch_gpu = prepare_batch(batch)  # Add batch dimension [1, ...] and move to GPU

    dummy_input = torch.zeros(1, 1, requires_grad=True).to(DEVICE)

    # Get raw logit, apply temperature scaling, convert to risk probability via sigmoid
    with torch.no_grad():
        # Pass patient_pos so the wrapper isolates this specific patient
        raw_logit = wrapper(dummy_input, batch_gpu, patient_pos=patient_pos).item()
        calibrated_risk = torch.sigmoid(torch.tensor(raw_logit) / t_val).item()

    # 3. Run Captum XAI
    attributions = attr_algo.attribute(
        inputs=dummy_input, additional_forward_args=(batch_gpu, patient_pos), target=0
    )
    # Sum attributions across embedding dimension [batch, seq_len, embed_dim] -> [batch, seq_len]
    token_attributions = attributions.sum(dim=-1).squeeze().cpu().detach().numpy()
    tokens_flat = (
        batch_gpu["transformer"]["hierarchical_tokens"].squeeze().cpu().numpy()
    )

    # Extract exact patient's token sequence from the batch
    # Each patient has variable length; use subject_lengths to slice correctly
    lengths = batch["transformer"]["subject_lengths"].numpy()
    start_idx = int(lengths[:patient_pos].sum())
    end_idx = start_idx + int(lengths[patient_pos])

    patient_tokens = tokens_flat[start_idx:end_idx]
    patient_attrs = token_attributions[start_idx:end_idx]

    # Only keep tokens that belong to recognized clinical domains
    VALID_CLINICAL_PREFIXES = (
        "LOINC/",  # Labs and Vitals
        "SNOMED/",  # Clinical Findings and Diagnoses
        "NDC/",  # National Drug Codes
        "RxNorm/",  # Standardized Medications
        "ATC/",  # Anatomical Therapeutic Chemical (Drug classes)
        "ICD9",  # ICD-9 Diagnoses & Procedures
        "ICD10",  # ICD-10 Diagnoses & Procedures
        "HCPCS/",  # Healthcare Common Procedure Coding System
        "CVX/",  # Vaccines
        "MIMIC_IV_LABITEM/",  # Raw MIMIC Labs
        "MIMIC_IV_OMR/",  # Raw MIMIC Vitals
        "MIMIC_IV_ITEM/",  # Charted ICU Events
        "MIMIC_IV_Drug/",  # Raw MIMIC Prescriptions
        "MIMIC_IV_Microbiology",  # Microbiology Tests and Antibiotics
    )

    # Secondary filter to drop the abstract/administrative LOINC parent nodes found in the CSV
    NOISY_LOINC_ROLLUPS = (
        "LOINC/PANEL",
        "LOINC/CHEM",
        "LOINC/HEM",
        "LOINC/SPEC",
        "LOINC/DOC",
        "LOINC/ATTACH",
        "LOINC/ABXBACT",
        "LOINC/CLIN",
        "LOINC/DRUG",
        "LOINC/EYE",
        "LOINC/FUNCTION",
        "LOINC/MICRO",
        "LOINC/PULM",
        "LOINC/RESP",
        "LOINC/SERO",
        "LOINC/SURVEY",
        "LOINC/UA",
        "LOINC/COAG",
        "LOINC/BDYTMP",
        "LOINC/BDYHGT",
        "LOINC/BDYWGT",
        "LOINC/HRTRATE",
        "LOINC/BP",
    )

    # Build attribution list with human-readable token strings and filter structural tokens
    attributions_list = []
    for token_id, attr_score in zip(patient_tokens, patient_attrs):
        token_str = get_token_string_resilient(
            raw_vocab, token_id, athena_mapping, mimic_mapping
        )

        # 1. Must start with a recognized clinical domain prefix
        if not token_str.startswith(VALID_CLINICAL_PREFIXES):
            continue

        # 2. Must NOT be an abstract administrative LOINC rollup
        if token_str.startswith(NOISY_LOINC_ROLLUPS):
            continue

        attributions_list.append(
            {
                "token_string": token_str,
                "score": float(attr_score),
            }
        )

    # Aggregate attributions by token string (in case same concept appears multiple times)
    df = pd.DataFrame(attributions_list).groupby("token_string", as_index=False).sum()
    df = df.sort_values(by="score", ascending=False)

    # Extract and save top-5 positive (sepsis-promoting) and top-5 negative (sepsis-suppressing) drivers
    out_dir = Config.RESULTS_DIR / "explainability" / "micro_xai"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Extract top 5 positive drivers
    top_positive = df.head(5).to_dict(orient="records")

    # Extract top 5 negative drivers and sort them ascending so the most extreme negative is first
    top_negative = (
        df.tail(5).sort_values(by="score", ascending=True).to_dict(orient="records")
    )

    # Determine prediction class based on threshold
    prediction_class = (
        "Sepsis (Positive)"
        if calibrated_risk >= optimal_threshold
        else "Control (Negative)"
    )

    # Compile patient XAI report matching the data contract expected by report_generator.py
    patient_data = {
        "subject_id": target_sid,
        "calibrated_risk": calibrated_risk,
        "decision_threshold": optimal_threshold,
        "prediction_class": prediction_class,
        "top_positive": top_positive,
        "top_negative": top_negative,
    }

    # Save JSON report
    json_path = out_dir / f"patient_{target_sid}_xai.json"
    with open(json_path, "w") as f:
        json.dump(patient_data, f, indent=4)

    logger.info(f"Successfully saved micro-XAI data to {json_path}")

    from src.evaluation.report_generator import write_clinical_markdown

    report_dir = Config.RESULTS_DIR / "explainability" / "reports" / "clinical_cases"
    report_dir.mkdir(parents=True, exist_ok=True)

    report_path = write_clinical_markdown(patient_data, report_dir)
    logger.info(f"Successfully generated Clinical Markdown Report → {report_path}")


def main() -> None:
    """
    Entry point for clinical inference and micro-XAI.

    Supports explicit patient selection or random sampling.
    """
    Config.setup_logging()

    parser = argparse.ArgumentParser(
        description="Clinical Inference and Micro-XAI pipeline."
    )
    # Mutually exclusive group: user must specify exactly ONE patient selection method
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--subject_id", type=int, help="Explicit Patient Subject ID.")
    group.add_argument(
        "--random_positive", action="store_true", help="Pick a random Sepsis patient."
    )
    group.add_argument(
        "--random_negative", action="store_true", help="Pick a random Control patient."
    )

    args = parser.parse_args()
    test_batches_path = Config.RESULTS_DIR / "explainability" / "test_batches"

    # Sanity check: ensure test_batches have been preprocessed
    if not test_batches_path.exists():
        logger.error("Run generate_test_batches.py first.")
        return

    # Resolve target patient ID via specified method
    if args.subject_id:
        target_sid = args.subject_id
    else:
        # Random selection: load test_batches and sample uniformly from cohort
        test_batches = datasets.Dataset.load_from_disk(test_batches_path)
        label = 1 if args.random_positive else 0
        target_sid = find_random_patient(label, test_batches)
        logger.info(f"Automatically selected Subject ID {target_sid} (Label: {label})")

    # Execute clinical inference
    wrapper, t_val, optimal_threshold = load_pipeline()
    run_clinical_inference(
        wrapper, test_batches_path, target_sid, t_val, optimal_threshold
    )


if __name__ == "__main__":
    main()

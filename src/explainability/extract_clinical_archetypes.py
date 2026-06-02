"""
Extract Clinical Archetypes for Dissertation Case Studies.

Uses a highly optimized two-pass strategy:
1. Fast Forward Pass: Scans all test patients to compute calibrated risks and
   bins them into 5 distinct clinical archetypes (True Positive, True Negative,
   False Positive, False Negative, and Borderline).
2. Targeted XAI Pass: Runs Captum LayerGradientXActivation ONLY on the top 5
   candidates from each archetype to extract their clinical drivers.

Archetypes:
- Textbook Sepsis: High Risk, Label 1
- Reassuring Normal: Low Risk, Label 0
- Smart Mistake: High Risk, Label 0 (Clinical False Positives)
- Missed Sepsis: Low Risk, Label 1 (Clinical False Negatives)
- Tug-of-War: Risk hovering exactly at the decision threshold

Usage:
    python -m src.explainability.extract_clinical_archetypes
"""

import logging
import json
import torch
import datasets
import pandas as pd
from typing import Dict, Any

from captum.attr import LayerGradientXActivation
from src.utils.config import Config
from src.explainability.xai_utils import (
    load_pipeline,
    prepare_batch,
    load_raw_msgpack_dictionary,
    get_token_string_resilient,
    load_mimic_mapping,
    load_athena_mapping,
)

logger = logging.getLogger(__name__)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Clinical filters
VALID_CLINICAL_PREFIXES = (
    "LOINC/",
    "SNOMED/",
    "NDC/",
    "RxNorm/",
    "ATC/",
    "ICD9",
    "ICD10",
    "HCPCS/",
    "CVX/",
    "MIMIC_IV_LABITEM/",
    "MIMIC_IV_OMR/",
    "MIMIC_IV_ITEM/",
    "MIMIC_IV_Drug/",
    "MIMIC_IV_Microbiology",
)
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


def extract_attributions_for_patient(
    attr_algo,
    batch,
    batch_gpu,
    patient_pos,
    raw_vocab,
    athena_mapping,
    mimic_mapping,
) -> Dict[str, Any]:
    """Runs Captum XAI for a specific patient and extracts the top positive/negative drivers."""
    dummy_input = torch.zeros(1, 1, requires_grad=True).to(DEVICE)

    # 1. Run Captum
    with torch.no_grad():
        attributions = attr_algo.attribute(
            inputs=dummy_input,
            additional_forward_args=(batch_gpu, patient_pos),
            target=0,
        )

    token_attributions = attributions.sum(dim=-1).squeeze().cpu().detach().numpy()
    tokens_flat = (
        batch_gpu["transformer"]["hierarchical_tokens"].squeeze().cpu().numpy()
    )

    # 2. Slice patient tokens
    lengths = batch["transformer"]["subject_lengths"].numpy()
    start_idx = int(lengths[:patient_pos].sum())
    end_idx = start_idx + int(lengths[patient_pos])

    patient_tokens = tokens_flat[start_idx:end_idx]
    patient_attrs = token_attributions[start_idx:end_idx]

    # 3. Filter and map tokens to clinical concepts
    attributions_list = []
    for token_id, attr_score in zip(patient_tokens, patient_attrs):
        token_str = get_token_string_resilient(
            raw_vocab, token_id, athena_mapping, mimic_mapping
        )
        if not token_str.startswith(VALID_CLINICAL_PREFIXES):
            continue
        if token_str.startswith(NOISY_LOINC_ROLLUPS):
            continue
        # Remove LOINC Part (LP) nodes: hierarchy-building concepts identified
        # by the LP prefix, not orderable clinical observations.
        if token_str.startswith("LOINC/LP"):
            continue

        attributions_list.append(
            {"token_string": token_str, "score": float(attr_score)}
        )

    # Aggregate and sort
    df = pd.DataFrame(attributions_list)
    if df.empty:
        return {"top_positive": [], "top_negative": []}

    df = (
        df.groupby("token_string", as_index=False)
        .sum()
        .sort_values(by="score", ascending=False)
    )

    return {
        "top_positive": df.head(5).to_dict(orient="records"),
        "top_negative": df.tail(5)
        .sort_values(by="score", ascending=True)
        .to_dict(orient="records"),
    }


def main() -> None:
    Config.setup_logging()
    Config.set_seed()

    # --- Setup and Data Loading ---
    logger.info("Loading Pipeline and Metadata...")
    wrapper, t_val, optimal_threshold = load_pipeline()

    test_batches_path = Config.RESULTS_DIR / "explainability" / "test_batches"
    test_batches = datasets.Dataset.load_from_disk(test_batches_path)
    test_batches.set_format("pt")

    # Load Ground Truth Labels from Cohort
    cohort_df = pd.read_parquet(Config.PROCESSED_COHORT_PARQUET_FILE)
    test_cohort = cohort_df[cohort_df["dataset_split"] == "test"]
    sid_to_label = test_cohort.set_index("subject_id")["sepsis_label"].to_dict()

    # --- Pass 1: Fast Forward Pass (Predictions Only) ---
    logger.info("Pass 1: Scanning test batches to compute calibrated risks...")
    patient_pool = []

    for b_idx, batch in enumerate(test_batches):
        batch_gpu = prepare_batch(batch)
        sids = batch["subject_ids"].numpy().tolist()

        # Get unique subject IDs and their starting positions
        unique_sids = []
        for sid in sids:
            if not unique_sids or unique_sids[-1] != sid:
                unique_sids.append(sid)

        for patient_pos, sid in enumerate(unique_sids):
            if sid not in sid_to_label:
                continue

            dummy_input = torch.zeros(1, 1).to(DEVICE)  # No grad needed for pass 1
            with torch.no_grad():
                raw_logit = wrapper(
                    dummy_input, batch_gpu, patient_pos=patient_pos
                ).item()
                risk = torch.sigmoid(torch.tensor(raw_logit) / t_val).item()

            patient_pool.append(
                {
                    "subject_id": sid,
                    "label": sid_to_label[sid],
                    "risk": risk,
                    "batch_idx": b_idx,
                    "patient_pos": patient_pos,
                }
            )

    # --- Binning Patients into Archetypes ---
    logger.info("Sorting patients into Clinical Archetypes...")
    df_pool = pd.DataFrame(patient_pool)
    df_pool["risk_sort"] = df_pool["risk"].round(6)

    # 1. Textbook Sepsis (True Positives, highest risk)
    textbook = (
        df_pool[df_pool["label"] == 1]
        .sort_values(
            by=["risk_sort", "subject_id"], ascending=[False, True], kind="stable"
        )
        .head(5)
    )

    # 2. Reassuring Normal (True Negatives, lowest risk)
    reassuring = (
        df_pool[df_pool["label"] == 0]
        .sort_values(
            by=["risk_sort", "subject_id"], ascending=[True, True], kind="stable"
        )
        .head(5)
    )

    # 3. Smart Mistake (False Positives, highest risk despite negative label)
    smart_mistake = (
        df_pool[df_pool["label"] == 0]
        .sort_values(
            by=["risk_sort", "subject_id"], ascending=[False, True], kind="stable"
        )
        .head(5)
    )

    # 4. Missed Sepsis (False Negatives, lowest risk despite positive label)
    missed_sepsis = (
        df_pool[df_pool["label"] == 1]
        .sort_values(
            by=["risk_sort", "subject_id"], ascending=[True, True], kind="stable"
        )
        .head(5)
    )

    # 5. Tug-of-War (Closest to optimal threshold)
    df_pool["dist_to_thresh"] = (df_pool["risk_sort"] - optimal_threshold).abs()
    tug_of_war = df_pool.sort_values(
        by=["dist_to_thresh", "subject_id"], ascending=[True, True], kind="stable"
    ).head(5)

    archetypes = {
        "textbook_sepsis": textbook,
        "reassuring_normal": reassuring,
        "smart_mistake": smart_mistake,
        "missed_sepsis": missed_sepsis,
        "tug_of_war": tug_of_war,
    }

    # --- Pass 2: Targeted Captum XAI ---
    logger.info("Pass 2: Running heavy Captum XAI on 25 selected candidates...")
    attr_algo = LayerGradientXActivation(
        forward_func=wrapper, layer=wrapper.motor.transformer.embed_bag
    )

    raw_vocab = load_raw_msgpack_dictionary()
    athena_mapping = load_athena_mapping(
        str(Config.MODEL_EHR_MOTOR_VOCAB_DIR / "CONCEPT.csv")
    )
    mimic_mapping = load_mimic_mapping()

    final_report = {}

    for arch_name, arch_df in archetypes.items():
        logger.info(f"Extracting features for archetype: {arch_name}")
        final_report[arch_name] = []

        for _, row in arch_df.iterrows():
            b_idx = int(row["batch_idx"])
            patient_pos = int(row["patient_pos"])

            batch = test_batches[b_idx]
            batch_gpu = prepare_batch(batch)

            xai_data = extract_attributions_for_patient(
                attr_algo,
                batch,
                batch_gpu,
                patient_pos,
                raw_vocab,
                athena_mapping,
                mimic_mapping,
            )

            # Combine into final payload
            final_report[arch_name].append(
                {
                    "subject_id": int(row["subject_id"]),
                    "true_label": int(row["label"]),
                    "calibrated_risk": float(row["risk"]),
                    **xai_data,
                }
            )

    # --- Save JSON Report ---
    out_dir = Config.RESULTS_DIR / "explainability" / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "clinical_archetypes_case_studies.json"

    with open(out_path, "w") as f:
        json.dump(final_report, f, indent=4)

    logger.info(f"Done! 25 case studies saved to {out_path}")


if __name__ == "__main__":
    main()

"""
Global Token Attribution Audit.

This tool uses Captum's LayerGradientXActivation to identify which FEMR tokens
have the strongest influence on the model's final sepsis predictions across the
whole test set. Absolute attribution scores are accumulated per token, so both
strongly risk-raising and strongly risk-lowering tokens count as influential.

It additionally measures the share of total attribution mass carried by
clinically-prefixed tokens and, within those, by LOINC Part (LP) nodes — the
hierarchy-building concepts identified in the leaf-resolution diagnostic.

Workflow:
  1. Load the frozen MOTOR transformer and trained EHR MLP
  2. Iterate through test batches
  3. For each patient, compute embedding-layer attributions
  4. Accumulate absolute attribution scores per token across all patients
  5. Report attribution-mass shares and write the ranked token table

Usage:
    python -m src.explainability.find_dataset_noise

Output:
    CSV file: results/explainability/top_dataset_tokens.csv
"""

import argparse
import logging
from collections import defaultdict

import datasets
import pandas as pd
import torch
from captum.attr import LayerGradientXActivation
from src.explainability.xai_utils import (
    get_token_string_resilient,
    load_athena_mapping,
    load_mimic_mapping,
    load_pipeline,
    load_raw_msgpack_dictionary,
    prepare_batch,
)
from src.utils.config import Config

logger = logging.getLogger(__name__)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Clinical-domain prefixes (shared with the other attribution scripts).
# Used here ONLY to measure attribution-mass shares — not to filter the output.
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


def main() -> None:
    """
    Main orchestrator for global token attribution analysis.
    """
    Config.setup_logging()
    parser = argparse.ArgumentParser(
        description="Identify most influential tokens across the test set."
    )
    parser.add_argument(
        "--max_batches", type=int, default=None, help="Number of batches to scan."
    )
    args = parser.parse_args()

    test_batches_path = Config.RESULTS_DIR / "explainability" / "test_batches"
    if not test_batches_path.exists():
        logger.error("Test batches not found.")
        return

    logger.info("Loading Pipeline and Dictionary...")
    wrapper, _, _ = load_pipeline()
    test_batches = datasets.Dataset.load_from_disk(test_batches_path)
    test_batches.set_format("pt")
    raw_vocab = load_raw_msgpack_dictionary()

    # Clinical vocabularies for human-readable token names in the output table.
    athena_mapping = load_athena_mapping(
        str(Config.MODEL_EHR_MOTOR_VOCAB_DIR / "CONCEPT.csv")
    )
    mimic_mapping = load_mimic_mapping()

    total_dataset_batches = len(test_batches)
    scan_limit = (
        args.max_batches if args.max_batches is not None else total_dataset_batches
    )

    # Initialize Captum's gradient-based attribution at the MOTOR embedding layer
    attr_algo = LayerGradientXActivation(
        forward_func=wrapper, layer=wrapper.motor.transformer.embed_bag
    )

    # Accumulate the absolute attribution score to find the most "active" tokens.
    # Absolute value captures influence regardless of direction; both positive and
    # negative gradients indicate the token's effect on the model's decision.
    token_importance = defaultdict(float)

    # Attribution-mass accounting (for the Section 9.4 measurement).
    total_all_mass = 0.0
    total_clinical_mass = 0.0
    loinc_part_mass = 0.0

    logger.info(
        f"Scanning {scan_limit} out of {total_dataset_batches} batches to map global attributions..."
    )
    for b_idx, batch in enumerate(test_batches):
        if b_idx >= scan_limit:
            break

        batch_gpu = prepare_batch(batch)

        # 1. Count how many unique patients are packed into this FEMR batch
        sids = batch["subject_ids"].numpy().tolist()
        unique_sids = []
        for sid in sids:
            if not unique_sids or unique_sids[-1] != sid:
                unique_sids.append(sid)
        num_patients = len(unique_sids)

        # 2. Process each patient individually
        for patient_pos in range(num_patients):
            dummy_input = torch.zeros(1, 1, requires_grad=True).to(DEVICE)

            with torch.no_grad():
                attributions = attr_algo.attribute(
                    inputs=dummy_input,
                    additional_forward_args=(batch_gpu, patient_pos),
                    target=0,
                )

            token_attributions = (
                attributions.sum(dim=-1).squeeze().cpu().detach().numpy()
            )
            tokens_flat = (
                batch_gpu["transformer"]["hierarchical_tokens"].squeeze().cpu().numpy()
            )

            # 3. Slice out this specific patient's tokens from the batch
            lengths = batch["transformer"]["subject_lengths"].numpy()
            start_idx = int(lengths[:patient_pos].sum())
            end_idx = start_idx + int(lengths[patient_pos])

            patient_tokens = tokens_flat[start_idx:end_idx]
            patient_attrs = token_attributions[start_idx:end_idx]

            # 4. Accumulate absolute attributions and mass shares
            for token_id, attr_score in zip(patient_tokens, patient_attrs):
                token_str = get_token_string_resilient(
                    raw_vocab, token_id, athena_mapping, mimic_mapping
                )
                abs_score = abs(float(attr_score))

                total_all_mass += abs_score
                if token_str.startswith(VALID_CLINICAL_PREFIXES):
                    total_clinical_mass += abs_score
                    if token_str.startswith("LOINC/LP"):
                        loinc_part_mass += abs_score

                token_importance[token_str] += abs_score

    # Report attribution-mass shares (Section 9.4 measurement)
    if total_clinical_mass > 0:
        logger.info(
            "LOINC Part (LP) attribution mass: "
            f"{loinc_part_mass:.3e} of {total_clinical_mass:.3e} clinically-prefixed "
            f"mass ({loinc_part_mass / total_clinical_mass:.1%})"
        )
    if total_all_mass > 0:
        logger.info(
            "Clinically-prefixed attribution mass: "
            f"{total_clinical_mass:.3e} of {total_all_mass:.3e} total mass "
            f"({total_clinical_mass / total_all_mass:.1%}); "
            f"LOINC Part share of total: {loinc_part_mass / total_all_mass:.1%}"
        )

    # Sort and save results
    df = pd.DataFrame(
        [
            {"token": k, "absolute_attribution_sum": v}
            for k, v in token_importance.items()
        ]
    ).sort_values(by="absolute_attribution_sum", ascending=False)

    out_csv = Config.RESULTS_DIR / "explainability" / "top_dataset_tokens.csv"
    df.to_csv(out_csv, index=False)

    logger.info("Full list saved to: %s", out_csv)


if __name__ == "__main__":
    main()

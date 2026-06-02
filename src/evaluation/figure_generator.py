"""
Generates high-resolution figures comparing models:

Comparison Figures:
  1. ROC Curves: Compares EHR baseline vs. 4-modality fusion on sensitivity/specificity
  2. PR Curves: Precision-Recall tradeoff
  3. Reliability (Calibration) Diagrams: Assesses if predicted probabilities match actual rates

All figures use consistent styling, high DPI (300+), and are saved as PDFs.

Usage:
    python -m src.evaluation.figure_generator --group all
    python -m src.evaluation.figure_generator --group comparison
"""

import argparse
import logging
import pandas as pd
import json

from src.utils.plotting import (
    plot_pr_curves,
    plot_reliability_diagrams,
    plot_roc_curves,
    plot_xai_tornado,
    verify_and_plot_calibration,
)
from src.utils.config import Config

logger = logging.getLogger(__name__)


def generate_comparison_figures() -> None:
    """
    Generate main comparison figures: ROC, PR, and Reliability diagrams.

    Compares Standalone EHR MLP vs 4-Modality Late-Fusion (pretrained).
    """
    logger.info("=== Generating Main Comparison Figures ===")

    models_to_compare = {
        "EHR baseline (MLP)": Config.RESULTS_UNIMODAL_PREDICTIONS_DIR
        / "ehr"
        / "mlp_predictions.csv",
        "Late-Fusion (pretrained)": Config.RESULTS_FUSION_PREDICTIONS_DIR
        / f"{Config.FUSION_RUN_PRETRAINED}_predictions.csv",
        "Late-Fusion (from scratch)": Config.RESULTS_FUSION_PREDICTIONS_DIR
        / f"{Config.FUSION_RUN_SCRATCH}_predictions.csv",
    }

    models_data = {}

    # Load predictions from each model
    for name, csv_path in models_to_compare.items():
        if not csv_path.exists():
            logger.error(f"Missing prediction file for {name} at {csv_path}. Skipping.")
            continue

        # Load probabilities for the matplotlib curves
        df = pd.read_csv(csv_path)
        y_true = df["label"].values
        y_prob = df["probability"].values
        models_data[name] = (y_true, y_prob)

    if not models_data:
        logger.error("No valid models found. Exiting.")
        return

    plot_dir = Config.RESULTS_DIR / "final_figures"
    plot_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Generating high-resolution plots...")
    # Generate all three comparison plots
    plot_roc_curves(models_data, plot_dir / "ROC_Comparison.pdf")
    plot_pr_curves(models_data, plot_dir / "PR_Comparison.pdf")
    plot_reliability_diagrams(models_data, plot_dir / "Reliability_Comparison.pdf")

    logger.info(f"Saved final figures to {plot_dir}")


def generate_xai_figures() -> None:
    """
    Generates Tornado Plots for the 20 dissertation clinical archetypes.

    Reads the curated JSON and outputs individual PDFs showing the
    Tug-of-War between sepsis promoters and suppressors.

    Output:
        results/final_figures/xai_case_studies/
    """
    logger.info("=== Generating XAI Tornado Plots ===")

    json_path = (
        Config.RESULTS_DIR
        / "explainability"
        / "reports"
        / "clinical_archetypes_case_studies.json"
    )
    if not json_path.exists():
        logger.error(f"Cannot find archetype JSON at {json_path}. Skipping.")
        return

    output_dir = Config.RESULTS_DIR / "final_figures" / "xai_case_studies"
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(json_path, "r") as f:
        archetypes = json.load(f)

    count = 0
    for archetype_name, patients in archetypes.items():
        for patient_data in patients:
            sid = patient_data["subject_id"]
            save_path = output_dir / f"{archetype_name}_subject_{sid}_xai.pdf"

            plot_xai_tornado(patient_data, str(save_path), archetype_name)
            count += 1

    logger.info(f"Generated {count} XAI Tornado Plots in {output_dir}")


def generate_calibration_proofs() -> None:
    """
    Generates calibration before/after figures for the EHR baseline and both fusion modes.

    For each model, requires both a calibrated and an uncalibrated predictions CSV.
    If the uncalibrated CSV is missing, the model is skipped with a warning rather than
    silently rendering a misleading figure where both curves are identical.

    Uncalibrated CSVs can be recovered post-hoc from the calibrated CSVs and the fitted
    temperature (see scripts/recover_uncalibrated_predictions.py).
    """
    logger.info("=== Generating Calibration Proof Plots ===")

    plot_dir = Config.RESULTS_DIR / "final_figures" / "calibration_proofs"
    plot_dir.mkdir(parents=True, exist_ok=True)

    # Each entry: display_name -> (calibrated_csv, uncalibrated_csv, output_filename)
    proofs = {
        "EHR MLP": (
            Config.RESULTS_UNIMODAL_PREDICTIONS_DIR / "ehr" / "mlp_predictions.csv",
            Config.RESULTS_UNIMODAL_PREDICTIONS_DIR
            / "ehr"
            / "mlp_predictions_uncalibrated.csv",
            "ehr_mlp_calibration_proof.pdf",
        ),
        "4-Modality Late-Fusion (pretrained)": (
            Config.RESULTS_FUSION_PREDICTIONS_DIR
            / f"{Config.FUSION_RUN_PRETRAINED}_predictions.csv",
            Config.RESULTS_FUSION_PREDICTIONS_DIR
            / f"{Config.FUSION_RUN_PRETRAINED}_predictions_uncalibrated.csv",
            "4mod_fusion_pretrained_calibration_proof.pdf",
        ),
        "4-Modality Late-Fusion (from scratch)": (
            Config.RESULTS_FUSION_PREDICTIONS_DIR
            / f"{Config.FUSION_RUN_SCRATCH}_predictions.csv",
            Config.RESULTS_FUSION_PREDICTIONS_DIR
            / f"{Config.FUSION_RUN_SCRATCH}_predictions_uncalibrated.csv",
            "4mod_fusion_scratch_calibration_proof.pdf",
        ),
    }

    for model_name, (cal_path, uncal_path, out_filename) in proofs.items():
        if not cal_path.exists():
            logger.warning(
                f"{model_name}: missing calibrated CSV at {cal_path}; skipping."
            )
            continue
        if not uncal_path.exists():
            logger.warning(
                f"{model_name}: missing uncalibrated CSV at {uncal_path}; skipping. "
                f"Run scripts/recover_uncalibrated_predictions.py to generate it."
            )
            continue

        df_cal = pd.read_csv(cal_path)
        df_uncal = pd.read_csv(uncal_path)

        # Verify the two CSVs are aligned by subject. Catches a silent data-integrity bug
        # in the unlikely event that someone re-ran one CSV but not the other.
        if "subject_id" in df_cal.columns and "subject_id" in df_uncal.columns:
            if not df_cal["subject_id"].equals(df_uncal["subject_id"]):
                logger.error(
                    f"{model_name}: subject ID mismatch between calibrated and uncalibrated CSVs; "
                    "regenerate the uncalibrated file. Skipping."
                )
                continue

        verify_and_plot_calibration(
            y_true=df_cal["label"].values,
            probs_uncalibrated=df_uncal["probability"].values,
            probs_calibrated=df_cal["probability"].values,
            model_name=model_name,
            save_path=plot_dir / out_filename,
        )

    logger.info(f"Calibration proof plots saved to {plot_dir}")


def main() -> None:
    """CLI entry point. Dispatches to figure generators based on the --group argument."""
    Config.setup_logging()

    parser = argparse.ArgumentParser(description="Generate high-resolution figures.")
    parser.add_argument(
        "--group",
        choices=["comparison", "xai", "calibration", "all"],
        default="all",
        help="Which group of figures to generate.",
    )
    args = parser.parse_args()

    if args.group in ["comparison", "all"]:
        generate_comparison_figures()
    if args.group in ["xai", "all"]:
        generate_xai_figures()
    if args.group in ["calibration", "all"]:
        generate_calibration_proofs()


if __name__ == "__main__":
    main()

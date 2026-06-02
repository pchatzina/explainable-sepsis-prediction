"""
Dynamic Markdown Report Generator for Multimodal Sepsis Prediction.

This module creates structured Markdown reports from metrics, predictions,
and explainability artifacts across all pipeline phases.

Supported Report Groups:
    - unimodal     : Per-modality baseline comparisons
    - fusion       : Scratch vs Pretrained fusion comparison
    - iva          : Incremental Value Analysis (Gold Cohort)
    - macro_xai    : High-attention gating cases (EHR dominance proof)
    - micro_xai    : Per-patient clinical decision reports
    - archetypes   : Dissertation clinical case studies
    - embeddings   : Embedding health inspection
    - all          : Everything (default)

Usage:
    python -m src.evaluation.report_generator --group all
"""

import argparse
import json
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import torch

from src.utils.config import Config

logger = logging.getLogger(__name__)

# Metrics to display
DISPLAY_METRICS = [
    ("auroc", "AUROC"),
    ("auprc", "AUPRC"),
    ("accuracy", "Accuracy"),
    ("f1", "F1"),
    ("precision", "Precision"),
    ("recall", "Recall"),
    ("specificity", "Specificity"),
]

CONFUSION_KEYS = ["tp", "fp", "fn", "tn"]


def _load_metrics_from_dir(directory: Path) -> pd.DataFrame:
    """
    Recursively load all *_metrics.json files under directory into a DataFrame.

    Injects two metadata columns: 'model' (filename stem) and 'modality' (parent
    directory name). Returns an empty DataFrame if the directory does not exist
    or contains no matching files.
    """
    records = []
    if not directory.exists():
        return pd.DataFrame()

    # Recursively find all *_metrics.json files (e.g., lr_metrics.json, xgboost_metrics.json)
    for path in directory.rglob("*_metrics.json"):
        with open(path, "r") as f:
            data = json.load(f)
            # Inject metadata derived dynamically from the file path hierarchy
            # Example: path = "results/unimodal/metrics/ehr/lr_metrics.json"
            #          model = "lr", modality = "ehr"
            data["model"] = path.name.replace("_metrics.json", "")
            data["modality"] = path.parent.name
            records.append(data)

    return pd.DataFrame(records)


def generate_macro_xai_report() -> None:
    """
    Generate Macro-XAI report highlighting high-attention auxiliary modality cases.

    Identifies complete-case patients (all 4 modalities present) where the gating
    network assigned the highest weight to non-EHR modalities.

    Purpose:
      - Proves the model learned to rely almost exclusively on EHR (>98% weight)
      - Even in "max synergy" edge cases, auxiliary modalities contribute minimally

    Output Location:
        results/explainability/reports/high_attention_cases_report.md
    """
    # Construct path to modality weights CSV produced by explainability XAI extraction phase
    # Expected columns: subject_id, true_label, p_final, w_ehr, w_ecg, w_cxr_img, w_cxr_txt, beta
    weights_csv = (
        Config.RESULTS_DIR
        / "explainability"
        / "modality_weights"
        / "4mod_architecture"
        / "test_set_modality_weights.csv"
    )
    target_dir = Config.RESULTS_DIR / "explainability" / "reports"

    if not weights_csv.exists():
        logger.warning(f"Weights CSV not found at {weights_csv}. Skipping XAI report.")
        return

    target_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(weights_csv)

    # Filter for complete cases: auxiliary gates must have passed numerical thresholds
    # (1e-5 filters out near-zero values that MaskedSoftmax explicitly suppressed)
    # Ensures only patients with all modalities available during masking are analyzed.
    multimodal_df = df[
        (df["w_ecg"] > 1e-5) & (df["w_cxr_img"] > 1e-5) & (df["w_cxr_txt"] > 1e-5)
    ].copy()

    if multimodal_df.empty:
        logger.warning("No true 4-modality patients found in the weights CSV.")
        return

    # Calculate total auxiliary weight (sum of all non-EHR gates)
    # High aux_weight means the network routed more signal through auxiliary modalities,
    # but inspection will show that w_ehr still dominates even in these maximal cases
    multimodal_df["aux_weight"] = (
        multimodal_df["w_ecg"] + multimodal_df["w_cxr_img"] + multimodal_df["w_cxr_txt"]
    )

    # Sort by aux_weight descending to identify "peak multimodal" edge cases
    top_cases = multimodal_df.sort_values(by="aux_weight", ascending=False)

    lines = [
        "### Table: Top Complete-Case Patients by Auxiliary Modality Attention\n",
        "*Note: This table highlights the top 5 Sepsis Positive and top 5 Sepsis Negative patients (out of complete cases) where the Gating Network assigned the maximum possible weight to non-EHR modalities. Even in these maximal edge cases, the network routed >98% of the decision weight to the standalone EHR pathway.*\n",
        "| Patient ID | True Label | Predicted Risk | EHR Weight ($w_{ehr}$) | ECG Weight | CXR (Img) | CXR (Text) | Aux Total | Synergy ($\\beta$) |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
    ]

    def format_rows(target_df, label_str):
        """Helper to format and append 5 rows to the Markdown table."""
        for _, row in target_df.head(5).iterrows():
            sid_raw = str(row["subject_id"]).replace("tensor(", "").replace(")", "")
            sid = int(float(sid_raw))
            risk = row["p_final"] * 100
            w_ehr = row["w_ehr"] * 100
            w_ecg = row["w_ecg"] * 100
            w_img = row["w_cxr_img"] * 100
            w_txt = row["w_cxr_txt"] * 100
            aux = row["aux_weight"] * 100
            beta = row["beta"] * 100

            lines.append(
                f"| **{sid}** | {label_str} | {risk:.2f}% | **{w_ehr:.2f}%** | {w_ecg:.2f}% | {w_img:.2f}% | {w_txt:.2f}% | {aux:.2f}% | {beta:.2f}% |"
            )

    # Show top 5 of each class separately for balanced representation
    format_rows(top_cases[top_cases["true_label"] == 0], "Negative (0)")
    format_rows(top_cases[top_cases["true_label"] == 1], "Positive (1)")

    out_path = target_dir / "high_attention_cases_report.md"
    out_path.write_text("\n".join(lines))
    logger.info(f"XAI Report generated successfully → {out_path}")


def write_clinical_markdown(
    data: dict, target_dir: Path, archetype: str = None
) -> Path:
    """
    Core logic to format and write a single patient's CDS Markdown report.
    Safely handles both live inference (no true label) and retrospective analysis.
    """
    sid = data["subject_id"]
    risk = data["calibrated_risk"] * 100
    true_label = data.get("true_label")  # Will be None for live inference

    lines = [
        "# Clinical Decision Support System: Sepsis Early Warning\n",
        f"**Patient ID:** {sid}",
        f"**Calibrated Sepsis Risk:** {risk:.2f}%",
    ]

    # Inject retrospective metadata if available (for dissertation case studies)
    if true_label is not None:
        lines.append(f"**True Ground Truth Label:** {true_label}")
    if archetype is not None:
        lines.append(f"**Clinical Archetype:** {archetype.replace('_', ' ').title()}")

    lines.extend(["\n---\n", "### Top Factors Driving Risk UP (Sepsis Promoting)"])

    for item in data.get("top_positive", []):
        lines.append(f"* **+{item['score']:.2e}**: {item['token_string']}")

    lines.append("\n### Top Factors Driving Risk DOWN (Sepsis Suppressing)")

    for item in data.get("top_negative", []):
        lines.append(f"* **{item['score']:.2e}**: {item['token_string']}")

    # If this is an archetype report, save it in a specialized folder or with a specialized name
    prefix = f"{archetype}_" if archetype else ""
    out_path = target_dir / f"{prefix}patient_{sid}_clinical_report.md"

    out_path.write_text("\n".join(lines))
    return out_path


def generate_micro_xai_reports() -> None:
    """Batch generates CDS reports from all Captum XAI JSON artifacts."""
    micro_xai_dir = Config.RESULTS_DIR / "explainability" / "micro_xai"
    target_dir = Config.RESULTS_DIR / "explainability" / "reports" / "clinical_cases"

    if not micro_xai_dir.exists():
        logger.warning(f"Micro-XAI directory not found at {micro_xai_dir}. Skipping.")
        return

    target_dir.mkdir(parents=True, exist_ok=True)

    for json_file in micro_xai_dir.glob("*_xai.json"):
        with open(json_file, "r") as f:
            data = json.load(f)

        out_path = write_clinical_markdown(data, target_dir)
        logger.info(f"Clinical Micro-XAI Report generated → {out_path}")


def generate_unimodal_reports() -> None:
    """
    Generate per-modality baseline classifier comparison reports.

    For each modality (EHR, ECG, CXR Image, CXR Text), creates a dedicated
    Markdown report comparing LR, XGBoost, and MLP performance.

    Report Contents:
      - Test set statistics (N, positives, negatives)
      - Metrics table: AUROC, AUPRC, Accuracy, F1, Precision, Recall, Specificity
      - Confusion matrix table
      - Best values are bolded for easy visual comparison
      - Thresholds used (derived from validation set)

    Output Location:
        results/unimodal/reports/{modality}_report.md
    """
    # Load all unimodal metrics from the metrics directory
    # Expected hierarchy: results/unimodal/metrics/{modality}/{model}_metrics.json
    metrics_dir = Config.RESULTS_UNIMODAL_METRICS_DIR
    target_dir = Config.RESULTS_UNIMODAL_REPORTS_DIR

    # _load_metrics_from_dir() scans recursively and injects "modality" and "model" columns
    df = _load_metrics_from_dir(metrics_dir)
    if df.empty:
        logger.warning(f"No metrics found in {metrics_dir}. Skipping Unimodal.")
        return

    target_dir.mkdir(parents=True, exist_ok=True)

    # Process each modality separately, generating one report file per modality
    for modality, mod_df in df.groupby("modality"):
        # Deduplicate by model (keeping the last/most recent run)
        # Handles cases where multiple training runs exist for the same classifier
        mod_df = mod_df.drop_duplicates(subset=["model"], keep="last")

        model_names = mod_df["model"].tolist()
        # Extract test set statistics from first model's metrics (shared across all models for this modality)
        n_samples = mod_df.iloc[0].get("n_samples", "N/A")
        n_pos = mod_df.iloc[0].get("n_positive", "N/A")
        n_neg = mod_df.iloc[0].get("n_negative", "N/A")

        # Build threshold information string for reproducibility
        # Example: "lr=0.4920, xgboost=0.5123, mlp=0.4850"
        thresholds_str = ", ".join(
            f"{row['model']}={row['threshold']:.4f}"
            for _, row in mod_df.iterrows()
            if pd.notna(row.get("threshold"))
        )

        lines = [
            f"# {modality.upper()} - Classifier Comparison\n",
            f"**Test set:** N = {n_samples} (+{n_pos} / −{n_neg})\n",
            f"**Thresholds (derived from Validation):** {thresholds_str}\n\n",
        ]

        # --- Main Metrics Table ---
        # Header row: | Metric | Model1 | Model2 | Model3 |
        header = "| Metric | " + " | ".join(model_names) + " |"
        # Separator row with right-aligned columns (numbers naturally align right)
        sep = "|:---|:" + "|:".join("---:" for _ in model_names) + "|"
        lines.extend([header, sep])

        # For each metric, find maximum value and highlight with bold
        for key, display_name in DISPLAY_METRICS:
            if key not in mod_df.columns:
                continue

            series = mod_df[key]
            max_val = series.max()
            row_cells = []
            for val in series:
                if pd.isna(val):
                    # Missing metric value (e.g., specificity not computed)
                    row_cells.append("—")
                elif (
                    np.isclose(float(val), float(max_val), rtol=0.0, atol=1e-12)
                    and len(model_names) > 1
                ):
                    # Bold-format best value (only if multiple models to compare)
                    row_cells.append(f"**{val:.4f}**")
                else:
                    row_cells.append(f"{val:.4f}")

            lines.append(f"| **{display_name}** | " + " | ".join(row_cells) + " |")

        # --- Confusion Matrix ---
        # Shows TP, FP, FN, TN for each classifier
        lines.extend(["\n### Confusion Matrix\n", header, sep])
        for key in CONFUSION_KEYS:
            if key not in mod_df.columns:
                continue

            # Convert confusion values to integers for readability
            row_cells = [str(int(val)) if pd.notna(val) else "—" for val in mod_df[key]]
            lines.append(f"| **{key.upper()}** | " + " | ".join(row_cells) + " |")

        # Write report to modality-specific file
        out_path = target_dir / f"{modality}_report.md"
        out_path.write_text("\n".join(lines))
        logger.info(f"Unimodal Report generated successfully → {out_path}")


def generate_fusion_reports(num_mods=4) -> None:
    """
    Generate comparison report between Scratch and Pretrained fusion models.

    Compares two training regimes for the same 4-modality late-fusion architecture:
      - Scratch: Random initialization
      - Pretrained: Initialized from frozen foundation model weights

    Report includes:
      - All major metrics with best values bolded
      - Confusion matrices
      - Optimal thresholds and key hyperparameters

    Output:
        results/fusion/reports/late_fusion_{num_mods}mod_comparison.md
    """
    # Define directories for fusion metrics and predictions
    # Metrics stored as: {num_mods}mod_scratch_final_metrics.json, {num_mods}mod_pretrained_final_metrics.json
    # Predictions stored as: {num_mods}mod_scratch_final_predictions.csv, etc.
    metrics_dir = Config.RESULTS_FUSION_METRICS_DIR
    predictions_dir = Config.RESULTS_FUSION_PREDICTIONS_DIR
    target_dir = Config.RESULTS_FUSION_REPORTS_DIR
    target_dir.mkdir(parents=True, exist_ok=True)

    # Define the two training experiments to compare
    # tuple: (display_name, metrics_filename_pattern)
    experiments = [
        ("Scratch", f"{num_mods}mod_scratch_final_metrics.json"),
        ("Pretrained", f"{num_mods}mod_pretrained_final_metrics.json"),
    ]

    all_test = []  # Holds parsed metrics dicts in same order as experiments
    names = []  # Holds experiment display names
    models_preds = {}  # Dict mapping name → (labels, probabilities) for optional plotting

    # Load metrics and predictions for each experiment
    for name, filename in experiments:
        path = metrics_dir / filename
        if not path.exists():
            logger.warning(f"Skipping '{name}' — metrics not found at {path}.")
            continue

        # Parse metrics JSON
        with open(path) as f:
            all_test.append(json.load(f))
        names.append(name)

        # Try to load predictions for this experiment (used for optional reliability diagrams)
        pred_path = predictions_dir / filename.replace(
            "_metrics.json", "_predictions.csv"
        )

        if pred_path.exists():
            df = pd.read_csv(pred_path)
            models_preds[name] = (df["label"].values, df["probability"].values)
        else:
            logger.warning(
                f"Prediction file missing for '{name}' at {pred_path}. Skipping in plot."
            )

    if not names:
        logger.error(f"No results found in {metrics_dir} to generate a fusion report.")
        return

    def find_best(values: list) -> int | None:
        """Helper: Find index of maximum non-None value in list."""
        valid = [(i, v) for i, v in enumerate(values) if v is not None]
        if not valid:
            return None
        return max(valid, key=lambda x: x[1])[0]

    # Extract test set statistics from first experiment (same test set for both)
    n_samples = all_test[0].get("n_samples", "?")
    n_pos = all_test[0].get("n_positive", "?")
    n_neg = all_test[0].get("n_negative", "?")

    # Build threshold information string for reproducibility
    # Example: "Scratch=0.4850, Pretrained=0.5102"
    thresholds_str = ", ".join(
        f"{n}={m.get('threshold', '?'):.4f}"
        if isinstance(m.get("threshold"), (int, float))
        else f"{n}=?"
        for n, m in zip(names, all_test)
    )

    # Extract optimal EHR dropout rates if available (result of Optuna hyperparameter tuning)
    # Lower values indicate EHR dominance; higher values would indicate auxiliary reliance
    # Optuna typically converges to low dropout rates for EHR-dominant datasets
    dropout_str = ", ".join(
        f"{n}: ehr_dropout={m.get('ehr_dropout_rate', 'N/A')}"
        for n, m in zip(names, all_test)
        if m.get("ehr_dropout_rate") is not None
    )

    lines = [
        f"# Late-Fusion Sepsis Model ({num_mods} Modalities) - Experiment Comparison\n",
        f"**Test set:** N = {n_samples} (+{n_pos} / −{n_neg}), thresholds: {thresholds_str}\n",
    ]

    # Include optimal dropout rates if available
    if dropout_str:
        lines.append(f"**Optimal EHR Dropout (tuned by Optuna):** {dropout_str}\n")

    lines.append("")

    # --- Main metrics table ---
    # Header: | Metric | Scratch | Pretrained |
    header = "| Metric |" + "|".join(f" {n} " for n in names) + "|"
    # Separator with center-aligned metric column and right-aligned numbers
    sep = "|---|" + "|".join(":---:" for _ in names) + "|"
    lines.extend([header, sep])

    # For each metric, find best value and highlight with bold
    for key, display_name in DISPLAY_METRICS:
        values = [m.get(key) for m in all_test]
        best_idx = find_best(values)
        cells = []
        for i, v in enumerate(values):
            if v is None:
                cells.append("—")
            elif i == best_idx and len(names) > 1:
                # Bold-format the best value across experiments
                cells.append(f"**{v:.4f}**")
            else:
                cells.append(f"{v:.4f}")
        lines.append(f"| {display_name} |" + "|".join(f" {c} " for c in cells) + "|")

    # --- Confusion matrix ---
    # Shows diagnostic counts: TP, FP, FN, TN
    lines.extend(["", "### Confusion Matrix\n", header, sep])
    for key in CONFUSION_KEYS:
        values = [str(m.get(key, "—")) for m in all_test]
        lines.append(f"| {key.upper()} |" + "|".join(f" {v} " for v in values) + "|")

    lines.append("")
    # Write final report to file
    out_path = target_dir / f"late_fusion_{num_mods}mod_comparison.md"
    out_path.write_text("\n".join(lines))
    logger.info(f"Fusion Report successfully written to -> {out_path}")


def generate_iva_reports() -> None:
    """
    Generate Incremental Value Analysis (IVA) reports on the Gold Cohort.

    Evaluates whether auxiliary modalities (ECG, CXR Image, CXR Text) provide
    incremental predictive value beyond EHR alone.

    Scope:
      - Restricted to "Gold Cohort" patients who have ALL 4 modalities
      - Uses 8 progressive masking combinations (EHR-only → Full Fusion)
      - Compares AUROC and AUPRC against the unimodal EHR baseline

    Key Output:
      Markdown table showing metric differences vs baseline with smart
      formatting for near-zero changes ("identical", "< +1e-4", etc.)

    Output Location:
        results/incremental_value_analysis/reports/{run_name}_iva_report.md
    """
    # Define base directory containing incremental value metrics
    # Expected hierarchy: results/incremental_value_analysis/metrics/{run_name}/{combo}_metrics.json
    metrics_dir_base = Config.RESULTS_INCREMENTAL_VALUE_METRICS_DIR
    # Ensure a robust fallback if REPORTS_DIR isn't explicitly defined in Config
    target_dir = getattr(
        Config,
        "RESULTS_INCREMENTAL_VALUE_REPORTS_DIR",
        Config.RESULTS_DIR / "incremental_value_analysis" / "reports",
    )

    if not metrics_dir_base.exists():
        logger.warning(
            f"IVA metrics directory not found at {metrics_dir_base}. Skipping IVA reports."
        )
        return

    target_dir.mkdir(parents=True, exist_ok=True)

    # Define the 8-step masking hierarchy evaluated on Gold Cohort
    combinations = [
        "0_Unimodal_Baseline_EHR",
        "1_Fusion_EHR_Only",
        "2_Fusion_EHR_ECG",
        "3_Fusion_EHR_IMG",
        "4_Fusion_EHR_TXT",
        "5_Fusion_EHR_IMG_TXT",
        "6_Fusion_EHR_ECG_IMG",
        "7_Fusion_All_Modalities",
    ]

    def format_diff(diff_val: float) -> str:
        """
        Helper to format metric differences cleanly, avoiding floating-point display artifacts.

        Handles three cases:
        1. diff_val ≈ 0 exactly → "identical"
        2. 0 < diff_val < 5e-5 (rounds to +0.0000 in .4f) → "< +1e-4"
        3. -5e-5 < diff_val < 0 (rounds to -0.0000 in .4f) → "> -1e-4"
        Otherwise: Standard +/- format with 4 decimal places
        """
        if abs(diff_val) < 1e-8:
            return "identical"
        elif 0 < diff_val < 5e-5:
            # +0.0000 in .4f rounds to zero; represented as < +1e-4 instead.
            return "< +1e-4"
        elif -5e-5 < diff_val < 0:
            # -0.0000 in .4f rounds to zero; represented as > -1e-4 instead.
            return "> -1e-4"
        else:
            return f"{diff_val:+.4f}"

    # Iterate over each specific training run inside the IVA metrics folder
    # Each run subdirectory contains 8 *_metrics.json files (one per masking combination)
    for run_dir in metrics_dir_base.iterdir():
        if not run_dir.is_dir():
            continue

        run_name = run_dir.name
        results = {}  # Maps combination name → parsed metrics dict
        max_auroc = 0.0  # Track best AUROC across all combinations
        max_auprc = 0.0  # Track best AUPRC across all combinations

        # Load metrics for each masking combination
        for combo in combinations:
            file_path = run_dir / f"{combo}_metrics.json"
            if file_path.exists():
                with open(file_path, "r") as f:
                    data = json.load(f)
                    results[combo] = data
                    # Update running maxima for bold-formatting in table
                    if data["auroc"] > max_auroc:
                        max_auroc = data["auroc"]
                    if data["auprc"] > max_auprc:
                        max_auprc = data["auprc"]

        # Baseline (EHR-only) must exist for IVA to be meaningful
        if "0_Unimodal_Baseline_EHR" not in results:
            logger.error(
                f"Baseline EHR metrics missing for run '{run_name}'. Cannot generate IVA table."
            )
            continue

        # Extract baseline metrics for difference calculation
        base_auroc = results["0_Unimodal_Baseline_EHR"]["auroc"]
        base_auprc = results["0_Unimodal_Baseline_EHR"]["auprc"]

        # Build Markdown table rows
        lines = [
            f"### Incremental Value Analysis: {run_name}\n",
            "| Combination | AUROC | vs Baseline | AUPRC | vs Baseline |",
            "| :--- | :--- | :--- | :--- | :--- |",
        ]

        # Populate one row per masking combination
        for combo in combinations:
            if combo not in results:
                # Missing metrics for this combination (data file not generated)
                lines.append(f"| {combo} | MISSING | - | MISSING | - |")
                continue

            data = results[combo]
            auroc = data["auroc"]
            auprc = data["auprc"]

            # Calculate differences vs Baseline EHR
            diff_auroc = auroc - base_auroc
            diff_auprc = auprc - base_auprc

            # Format metric values with bold for best performance
            auroc_str = f"{auroc:.4f}"
            auprc_str = f"{auprc:.4f}"

            # Bold-format if this is the best AUROC/AUPRC across all combinations
            if abs(auroc - max_auroc) < 1e-6:
                auroc_str = f"**{auroc_str}**"
            if abs(auprc - max_auprc) < 1e-6:
                auprc_str = f"**{auprc_str}**"

            # Baseline row uses "—" for difference columns (no delta vs itself)
            if combo == "0_Unimodal_Baseline_EHR":
                diff_auroc_str = "—"
                diff_auprc_str = "—"
            else:
                diff_auroc_str = format_diff(diff_auroc)
                diff_auprc_str = format_diff(diff_auprc)

            lines.append(
                f"| {combo} | {auroc_str} | {diff_auroc_str} | {auprc_str} | {diff_auprc_str} |"
            )

        # Write final IVA report to file
        out_path = target_dir / f"{run_name}_iva_report.md"
        out_path.write_text("\n".join(lines))
        logger.info(f"IVA Report generated successfully → {out_path}")


def generate_embeddings_report() -> None:
    """
    Inspect the statistical distributions of extracted embeddings for any modality.
    Outputs a consolidated Markdown report in tabular format.
    """
    MODALITY_DIRS = {
        "ehr": Config.PROCESSED_EHR_EMBEDDINGS_DIR,
        "ecg": Config.PROCESSED_ECG_EMBEDDINGS_DIR,
        "cxr_img": Config.PROCESSED_CXR_IMG_EMBEDDINGS_DIR,
        "cxr_txt": Config.PROCESSED_CXR_TXT_EMBEDDINGS_DIR,
    }

    def get_split_data(file_path: Path, split: str):
        split_cap = split.capitalize()
        if not file_path.exists():
            stats = {
                "Split": split_cap,
                "Embeddings Shape": "**[!] File not found**",
                "Number of Samples": "N/A",
                "Embedding Dimension": "N/A",
                "Mean": "N/A",
                "Std": "N/A",
                "Min": "N/A",
                "Max": "N/A",
                "Contains NaN": "**[!] Warning:** File missing",
                "Contains Inf": "**[!] Warning:** File missing",
            }
            return stats, None

        data = torch.load(file_path, map_location="cpu", weights_only=False)
        embeddings = data["embeddings"]
        labels = data["labels"]
        subject_ids = data["subject_ids"]

        emb_shape = list(embeddings.shape)
        shape_str = " × ".join(str(x) for x in emb_shape)
        embedding_dim = (
            embeddings.shape[1] if len(embeddings.shape) > 1 else embeddings.shape[0]
        )

        nan_str = (
            "**YES (CRITICAL ERROR)**" if torch.isnan(embeddings).any().item() else "No"
        )
        inf_str = (
            "**YES (CRITICAL ERROR)**" if torch.isinf(embeddings).any().item() else "No"
        )

        stats = {
            "Split": split_cap,
            "Embeddings Shape": shape_str,
            "Number of Samples": len(subject_ids),
            "Embedding Dimension": embedding_dim,
            "Mean": f"{embeddings.mean().item():.6f}",
            "Std": f"{embeddings.std().item():.6f}",
            "Min": f"{embeddings.min().item():.6f}",
            "Max": f"{embeddings.max().item():.6f}",
            "Contains NaN": nan_str,
            "Contains Inf": inf_str,
        }

        label_info = None
        if labels is not None and len(labels) > 0:
            int_labels = [int(l) for l in labels]
            unique_labels = sorted(set(int_labels))
            label_counts = {label: int_labels.count(label) for label in unique_labels}
            label_info = {
                "split": split_cap,
                "counts": label_counts,
            }

        return stats, label_info

    modalities_to_run = MODALITY_DIRS.keys()
    splits = ["train", "valid", "test"]

    Config.REPORT_EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = Config.REPORT_EMBEDDINGS_DIR / "embeddings_health_report.md"

    with open(report_path, "w") as out_file:
        out_file.write("# Embedding Inspection Report\n\n")

        for mod in modalities_to_run:
            out_file.write(f"## Modality: {mod.upper()}\n\n")
            stats_rows = []
            label_infos = []
            for split in splits:
                file_path = MODALITY_DIRS[mod] / f"{split}_embeddings.pt"
                stats, label_info = get_split_data(file_path, split)
                stats_rows.append(stats)
                if label_info is not None:
                    label_infos.append(label_info)

            out_file.write("### Embedding Statistics\n\n")
            if stats_rows:
                columns = list(stats_rows[0].keys())
                header = "| " + " | ".join(columns) + " |\n"
                separator = "| " + " | ".join(["---"] * len(columns)) + " |\n"
                out_file.write(header)
                out_file.write(separator)

                for row in stats_rows:
                    row_values = [str(row[col]) for col in columns]
                    out_file.write("| " + " | ".join(row_values) + " |\n")
                out_file.write("\n")

            if label_infos:
                out_file.write("### Label Distributions\n\n")
                for label_info in label_infos:
                    split_name = label_info["split"]
                    counts = label_info["counts"]
                    total = sum(counts.values())

                    out_file.write(f"#### Split: {split_name}\n\n")
                    out_file.write("| Class | Count | Percentage (%) |\n")
                    out_file.write("|-------|-------|----------------|\n")

                    for label in sorted(counts.keys()):
                        count = counts[label]
                        perc = (count / total * 100) if total > 0 else 0.0
                        out_file.write(f"| {label} | {count} | {perc:.1f} |\n")
                    out_file.write("\n")

            out_file.write("---\n\n")

    logger.info("Full Markdown report successfully generated at: %s", report_path)


def generate_archetype_reports() -> None:
    """
    Generate clean Markdown reports for the dissertation clinical archetypes.

    Reads the curated 20-patient JSON generated by extract_clinical_archetypes.py
    and formats them into individual clinical case studies for thesis inclusion.

    Output: results/explainability/reports/dissertation_cases/{archetype}_patient_{sid}_clinical_report.md
    """
    json_path = (
        Config.RESULTS_DIR
        / "explainability"
        / "reports"
        / "clinical_archetypes_case_studies.json"
    )

    if not json_path.exists():
        logger.warning(
            f"Archetype JSON not found at {json_path}. Run extract_clinical_archetypes.py first. Skipping."
        )
        return

    # Create a dedicated directory for the dissertation cases
    target_dir = (
        Config.RESULTS_DIR / "explainability" / "reports" / "dissertation_cases"
    )
    target_dir.mkdir(parents=True, exist_ok=True)

    with open(json_path, "r") as f:
        archetypes = json.load(f)

    count = 0
    for archetype_name, patients in archetypes.items():
        for patient_data in patients:
            # Uses the write_clinical_markdown helper
            out_path = write_clinical_markdown(
                patient_data, target_dir, archetype=archetype_name
            )
            logger.info(f"Archetype Report generated → {out_path.name}")
            count += 1

    logger.info(
        f"Successfully generated {count} Markdown case studies for the dissertation."
    )


def main() -> None:
    """CLI entry point. Dispatches to report generators based on the --group argument."""
    Config.setup_logging()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--group",
        choices=[
            "unimodal",
            "fusion",
            "iva",
            "macro_xai",
            "micro_xai",
            "embeddings",
            "archetypes",
            "all",
        ],
        default="all",
        help="Which report group to generate.",
    )
    args = parser.parse_args()

    if args.group in ["embeddings", "all"]:
        generate_embeddings_report()
    if args.group in ["unimodal", "all"]:
        generate_unimodal_reports()
    if args.group in ["fusion", "all"]:
        generate_fusion_reports()
    if args.group in ["iva", "all"]:
        generate_iva_reports()
    if args.group in ["macro_xai", "all"]:
        generate_macro_xai_report()
    if args.group in ["micro_xai", "all"]:
        generate_micro_xai_reports()
    if args.group in ["archetypes", "all"]:
        generate_archetype_reports()


if __name__ == "__main__":
    main()

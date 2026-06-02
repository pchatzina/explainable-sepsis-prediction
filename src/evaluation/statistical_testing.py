"""
Unified Statistical Significance Testing for Multimodal Sepsis Prediction.

This module implements paired bootstrap statistical testing to evaluate whether auxiliary
modalities (ECG waveforms, chest X-rays, radiology reports) provide a statistically significant
incremental improvement over the EHR-only baseline for 6-hour sepsis prediction on the Gold Cohort.

BOOTSTRAP METHODOLOGY:
    - Paired bootstrapping preserves the sample structure by resampling the same subject indices
      from both baseline and comparison models, ensuring that paired differences are computed on
      identical patient subsets across bootstrap iterations.
    - Threshold-independent metrics (AUROC, AUPRC) are used to avoid threshold selection bias.
    - Empirical p-values are computed as the proportion of bootstrap iterations where the
      comparison model does NOT outperform the baseline (diff <= 0), testing the null hypothesis
      that auxiliary modalities provide no incremental value.
    - 95% Confidence Intervals (CIs) are constructed from the 2.5th and 97.5th percentiles of
      the bootstrap distribution of comparison model metrics.

WHY PAIRED BOOTSTRAPPING:
    - Patient-level correlation: The same subject's features and outcomes are evaluated by both
      models. Unpaired resampling would ignore this dependency and inflate variance estimates.
    - Gold Cohort: This test assumes the "Gold Cohort" (patients with all 4 modalities) are used.
      Paired sampling ensures fair, unbiased comparison under identical data availability.
    - Statistical Power: Paired tests have higher power because they account for inter-subject
      variability, making it easier to detect true incremental improvements.

OUTPUT:
    - CSV report with AUROC/AUPRC point estimates, 95% CIs, and empirical p-values for each
      comparison (e.g., EHR+ECG, EHR+IMG, etc.).
    - Interpretation: p-value < 0.05 indicates statistically significant improvement in that metric.
      No adjustment for multiple comparisons is applied; the results reflect individual tests.

Usage:
    python -m src.evaluation.statistical_testing --run_name 4mod_pretrained_final
"""

import argparse
import json
import logging

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from tqdm import tqdm

from src.utils.config import Config

logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION: Baseline Selection
# ============================================================================
# USE_STANDALONE_EHR_BASELINE:
#   - If True: Compare against the Standalone Unimodal EHR MLP model (independent training).
#             This isolates the contribution of auxiliary modalities in the late-fusion setting.
#   - If False: Compare against the "1_Fusion_EHR_Only" masked fusion model (all modalities
#               masked except EHR). This serves as an internal baseline within the fusion pipeline.
# The choice affects interpretation: Standalone EHR is more conservative (measures absolute
# improvement); masked EHR within fusion is more optimistic (measures relative improvement
# in architecture, not training procedure).
# ============================================================================
USE_STANDALONE_EHR_BASELINE = True

# ============================================================================
# COMPARISON LADDER: Incremental Modality Addition
# ============================================================================
# This ordered list defines the "ladder" of multimodal combinations to test against the
# baseline. Each entry tests the hypothesis: "Does adding [modality/combination] to EHR
# provide a statistically significant improvement?"
#
#   1. "2_Fusion_EHR_ECG"       -> Tests cardiac signal value
#   2. "3_Fusion_EHR_IMG"       -> Tests radiographic imaging value
#   3. "4_Fusion_EHR_TXT"       -> Tests clinical text value
#   4. "5_Fusion_EHR_IMG_TXT"   -> Tests synergy between imaging and text
#   5. "6_Fusion_EHR_ECG_IMG"   -> Tests synergy between cardiac and imaging
#   6. "7_Fusion_All_Modalities"-> Tests full 4-modality synergy
# ============================================================================
COMPARISONS = [
    "2_Fusion_EHR_ECG",
    "3_Fusion_EHR_IMG",
    "4_Fusion_EHR_TXT",
    "5_Fusion_EHR_IMG_TXT",
    "6_Fusion_EHR_ECG_IMG",
    "7_Fusion_All_Modalities",
]

# Bootstrap Configuration
N_BOOTSTRAP = 1000  # Number of bootstrap resamples for CI and p-value estimation
ALPHA = 0.05  # Significance level (two-tailed); 95% CI uses ALPHA/2 on each tail


def paired_bootstrap_test(
    y_true: np.ndarray,
    y_prob_base: np.ndarray,
    y_prob_comp: np.ndarray,
) -> dict:
    """
    Perform paired bootstrap resampling to compute 95% CIs and empirical p-values.

    BOOTSTRAP PROCEDURE:
        1. For each of N_BOOTSTRAP iterations:
           a. Resample n_samples subject indices uniformly with replacement.
           b. Check that the bootstrap sample contains at least both classes (to compute AUC).
           c. Slice predictions for both baseline and comparison models using the same indices.
           d. Compute AUROC and AUPRC for both models on the bootstrap sample.
           e. Record: (i) absolute metric values for the comparison model (for CI),
                      (ii) metric differences (comparison - baseline) for p-value.
        2. Compute 95% CI as [2.5th, 97.5th] percentiles of bootstrap comparison metrics.
        3. Compute empirical p-value = fraction of iterations where diff <= 0 (one-tailed test).

    Args:
        y_true: Binary outcome labels, shape (n_samples,).
        y_prob_base: Predicted probabilities from the baseline model, shape (n_samples,).
        y_prob_comp: Predicted probabilities from the comparison model, shape (n_samples,).

    Returns:
        Dict keyed by metric name ("auroc", "auprc"). Each value is a dict with
        "mean", "ci_lower", "ci_upper", and "p_value" fields.
        Metrics that cannot be computed (e.g. single class in all bootstrap samples) are omitted.
    """
    n_samples = len(y_true)
    metric_keys = ["auroc", "auprc"]

    # Initialize storage for bootstrap results
    # abs_comp: Stores absolute metric values of the comparison model (used to build CI)
    # diffs: Stores metric differences (comparison - baseline) for p-value computation
    abs_comp = {k: [] for k in metric_keys}
    diffs = {k: [] for k in metric_keys}

    # ========================================================================
    # BOOTSTRAP LOOP: Resample with replacement and compute paired metrics
    # ========================================================================
    for _ in range(N_BOOTSTRAP):
        # Step 1: Resample subject indices with replacement (paired resampling)
        indices = np.random.choice(n_samples, n_samples, replace=True)
        y_t_b = y_true[indices]

        # Step 2: Validity check - skip if bootstrap doesn't contain both classes
        # (AUC/AUPRC require both positive and negative labels; otherwise undefined)
        if len(np.unique(y_t_b)) < 2:
            continue

        # Step 3: Extract paired predictions using identical indices
        y_p_base_b = y_prob_base[indices]
        y_p_comp_b = y_prob_comp[indices]

        # Step 4: Compute threshold-independent metrics for this bootstrap sample
        # These metrics do not depend on a classification threshold, only on ranking
        base_auroc = roc_auc_score(y_t_b, y_p_base_b)
        base_auprc = average_precision_score(y_t_b, y_p_base_b)

        comp_auroc = roc_auc_score(y_t_b, y_p_comp_b)
        comp_auprc = average_precision_score(y_t_b, y_p_comp_b)

        # Step 5: Accumulate results
        # For AUROC
        abs_comp["auroc"].append(comp_auroc)
        diffs["auroc"].append(comp_auroc - base_auroc)

        # For AUPRC
        abs_comp["auprc"].append(comp_auprc)
        diffs["auprc"].append(comp_auprc - base_auprc)

    # ========================================================================
    # COMPUTE STATISTICS: CI and p-values from bootstrap distributions
    # ========================================================================
    results = {}
    for metric in metric_keys:
        diff_array = np.array(diffs[metric])
        comp_array = np.array(abs_comp[metric])

        # Edge case: if all bootstrap iterations failed (e.g., severe label imbalance)
        if len(comp_array) == 0:
            logger.error(f"Failed to bootstrap {metric} - check label distributions.")
            continue

        # Compute empirical 95% Confidence Interval from percentiles of comparison metric
        ci_lower = np.percentile(comp_array, (ALPHA / 2) * 100)  # 2.5th percentile
        ci_upper = np.percentile(comp_array, (1 - ALPHA / 2) * 100)  # 97.5th percentile

        # Compute empirical one-tailed p-value
        # H0: Comparison model is NOT better (mean difference <= 0)
        # p_value = fraction of bootstrap iterations where diff <= 0
        p_value = np.mean(diff_array <= 0.0)

        results[metric] = {
            "mean": np.mean(comp_array),
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "p_value": p_value,
        }

    return results


def main(run_name: str) -> None:
    """
    Run paired bootstrap significance testing for all combinations in COMPARISONS.

    Baseline selection is controlled by USE_STANDALONE_EHR_BASELINE (see module-level
    constant). Comparisons are defined by the COMPARISONS constant. Merges baseline and
    comparison predictions on subject_id, runs paired_bootstrap_test() for each combination,
    and writes results to results/statistical_analysis/{run_name}/significance_results.csv.

    Args:
        run_name: Fusion training run identifier (e.g. "4mod_pretrained_final").
            Points to results/incremental_value_analysis/{run_name}/ for comparison CSVs.
    """
    Config.setup_logging()

    # ========================================================================
    # STEP 1: BASELINE SELECTION AND LOADING
    # ========================================================================
    # Determine baseline path based on configuration flag
    if USE_STANDALONE_EHR_BASELINE:
        base_csv = (
            Config.RESULTS_UNIMODAL_PREDICTIONS_DIR / "ehr" / "mlp_predictions.csv"
        )
        baseline_name = "Standalone_EHR_MLP"
        logger.info(
            f"Using STANDALONE EHR baseline: {base_csv} | "
            "Tests absolute improvement of fusion vs. best-in-class unimodal."
        )
    else:
        base_csv = (
            Config.RESULTS_INCREMENTAL_VALUE_PREDICTIONS_DIR
            / run_name
            / "1_Fusion_EHR_Only_predictions.csv"
        )
        baseline_name = "1_Fusion_EHR_Only (Masked Fusion)"
        logger.info(
            f"Using MASKED FUSION EHR baseline: {base_csv} | "
            "Tests relative improvement within fusion architecture."
        )

    # Verify baseline file exists
    if not base_csv.exists():
        logger.error(f"Cannot find baseline predictions at {base_csv}. Aborting.")
        return

    df_base = pd.read_csv(base_csv)
    logger.info(
        f"Loaded baseline predictions: {len(df_base)} subjects, "
        f"columns={df_base.columns.tolist()}"
    )

    # ========================================================================
    # STEP 2: ITERATE THROUGH COMPARISON LADDER
    # ========================================================================
    preds_dir = Config.RESULTS_INCREMENTAL_VALUE_PREDICTIONS_DIR / run_name
    results_summary = []

    logger.info(
        f"Running {N_BOOTSTRAP} bootstrap iterations for significance testing across "
        f"{len(COMPARISONS)} comparison combinations..."
    )
    np.random.seed(42)  # Seed for reproducibility of bootstrap samples

    for combo in tqdm(COMPARISONS, desc="Evaluating Combinations"):
        comp_csv = preds_dir / f"{combo}_predictions.csv"

        # Verify comparison file exists
        if not comp_csv.exists():
            logger.warning(f"Missing prediction file for {combo}. Skipping.")
            continue

        df_comp = pd.read_csv(comp_csv)

        # ====================================================================
        # PAIRING: Merge baseline and comparison on subject_id to ensure both
        # models are evaluated on identical patient subsets
        # ====================================================================
        merged = pd.merge(
            df_base, df_comp, on="subject_id", suffixes=("_base", "_comp")
        )

        if len(merged) == 0:
            logger.error(
                f"No matching subject IDs found between baseline and {combo}. Skipping."
            )
            continue

        # Extract paired arrays (same indices for both models)
        y_true_aligned = merged["label_base"].values
        y_prob_base_aligned = merged["probability_base"].values
        y_prob_comp_aligned = merged["probability_comp"].values

        logger.info(
            f"Testing {combo}: {len(y_true_aligned)} paired predictions, "
            f"positive rate = {y_true_aligned.mean():.3f}"
        )

        # ====================================================================
        # BOOTSTRAP TESTING: Compute 95% CIs and p-values for this comparison
        # ====================================================================
        stats = paired_bootstrap_test(
            y_true_aligned,
            y_prob_base_aligned,
            y_prob_comp_aligned,
        )

        # Format results for display and CSV
        row = {
            "Model": combo,
            "AUROC": (
                f"{stats['auroc']['mean']:.4f} "
                f"[{stats['auroc']['ci_lower']:.4f}-{stats['auroc']['ci_upper']:.4f}]"
            ),
            "AUROC p-val": f"{stats['auroc']['p_value']:.4f}",
            "AUPRC": (
                f"{stats['auprc']['mean']:.4f} "
                f"[{stats['auprc']['ci_lower']:.4f}-{stats['auprc']['ci_upper']:.4f}]"
            ),
            "AUPRC p-val": f"{stats['auprc']['p_value']:.4f}",
        }

        results_summary.append(row)

    # ========================================================================
    # STEP 3: OUTPUT REPORTING
    # ========================================================================
    if not results_summary:
        logger.error("No valid comparisons were processed.")
        return

    # Create results DataFrame and display
    df_results = pd.DataFrame(results_summary)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)

    logger.info("\n%s", "=" * 100)
    logger.info("STATISTICAL SIGNIFICANCE REPORT (vs. %s)", baseline_name)
    logger.info(
        "Legend: AUROC/AUPRC = point estimate [95% CI lower - 95% CI upper], "
        "p-val = empirical one-tailed p-value"
    )
    logger.info(
        "Interpretation: p < 0.05 indicates significant improvement. Bootstrap N=%d.",
        N_BOOTSTRAP,
    )
    logger.info("%s", "=" * 100)
    logger.info("\n%s", df_results.to_string(index=False))
    logger.info("%s\n", "=" * 100)

    # Save results to CSV
    out_dir = Config.RESULTS_DIR / "statistical_analysis" / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "significance_results.csv"
    df_results.to_csv(out_path, index=False)
    logger.info(f"Saved full statistical report to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run statistical testing on fusion models."
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default=Config.FUSION_RUN_PRETRAINED,
        help="The specific run configuration to evaluate.",
    )
    args = parser.parse_args()

    main(args.run_name)

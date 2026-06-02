"""
Incremental Value Analysis (IVA) via Inference Masking.

This module evaluates whether auxiliary modalities (ECG, CXR images, CXR text)
provide any incremental value beyond EHR alone.

Key Principles:
  1. GOLD COHORT: Evaluation is restricted to patients possessing ALL 4 modalities
  2. MASKING SIMULATION: Auxiliary modalities are artificially ablated while keeping
     EHR active
  3. LOCKED THRESHOLDS: Both EHR and fusion metrics use globally locked thresholds,
     ensuring consistency across comparisons
  4. COMPARISON LADDER: Incrementally add modalities to isolate the contribution
     of each modality

Usage:
    python -m src.evaluation.incremental_value_analysis modalities=4_modality training=pretrained
"""

import json
import logging
from typing import Dict, List, Tuple, Union

import hydra
from omegaconf import DictConfig, OmegaConf
import torch
from torch.utils.data import DataLoader
import numpy as np

from src.utils.config import Config
from src.data.loaders.fusion_datamodule import FusionDataModule
from src.models.fusion.late_fusion_module import LateFusionModule
from src.models.unimodal.mlp import UnimodalModule
from src.evaluation.metrics import compute_metrics
from src.evaluation.evaluator import ModelEvaluator

logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _extract_subject_ids(
    raw_subject_ids: torch.Tensor,
    is_gold_np: np.ndarray,
) -> List[int]:
    """Extract subject IDs aligned to the Gold Cohort mask."""
    if not isinstance(raw_subject_ids, torch.Tensor):
        raise TypeError("subject_id must be a torch.Tensor")

    subject_array = raw_subject_ids.detach().cpu().numpy()

    if subject_array.shape[0] != is_gold_np.shape[0]:
        raise ValueError(
            "subject_id length does not match batch size: "
            f"{subject_array.shape[0]} vs {is_gold_np.shape[0]}"
        )

    return [int(sid) for sid in subject_array[is_gold_np].tolist()]


def get_masked_predictions(
    model: torch.nn.Module,
    dataloader: DataLoader,
    active_mods: Dict[str, Union[int, float, bool]],
    is_unimodal: bool = False,
) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """
    Runs inference on the Gold Cohort, either with fusion masking or true unimodal EHR.

    This function implements the core IVA mechanism:
      1. Identifies the "Gold Cohort": patients where ALL 4 modalities are present
      2. For is_unimodal=True: Runs the standalone EHR MLP directly
      3. For is_unimodal=False: Runs the fusion model with artificial masking
         (setting selected modality masks to 0 in the batch)

    The MaskedSoftmax gating network normalizes across active modalities only,
    ensuring that masking a modality removes its contribution (not sets it to 0 weight).

    Args:
        model: Either UnimodalModule (EHR MLP) or LateFusionModule (fusion network)
        dataloader: DataLoader for the test split
        active_mods: Dict mapping modality names to ablation flags (0 or 1)
                    Ignored if is_unimodal=True
        is_unimodal: If True, use the standalone EHR MLP; else, use fusion model

    Returns:
        Tuple of (y_true, y_prob_calibrated, subject_ids)
            - y_true: Ground truth labels for Gold Cohort only
            - y_prob_calibrated: Calibrated predicted probabilities [0, 1]
            - subject_ids: MIMIC-IV subject IDs for traceability
    """
    model.eval()
    model.to(DEVICE)

    all_p_calibrated = []
    all_targets = []
    all_subject_ids = []

    with torch.no_grad():
        for batch in dataloader:
            # 1. Isolate the "Gold Cohort" (Patients who naturally have ALL modalities)
            # Use masks to confirm all modalities are present (mask == 1.0)
            is_gold = torch.ones(len(batch["label"]), dtype=torch.bool)
            for mod in batch["masks"].keys():
                is_gold = is_gold & (batch["masks"][mod].squeeze(dim=-1) == 1.0)

            # Skip batch if no gold cohort patients
            if not is_gold.any():
                continue

            targets = batch["label"][is_gold].to(DEVICE).float()

            # 2. Safely filter subject_ids to match is_gold mask
            is_gold_np = is_gold.cpu().numpy()
            if "subject_id" not in batch:
                raise ValueError("subject_id missing from batch")
            subject_ids = _extract_subject_ids(batch["subject_id"], is_gold_np)

            # --- Unimodal EHR Evaluation ---
            if is_unimodal:
                # Pass EHR embeddings directly to the standalone MLP
                inputs = batch["embeddings"]["ehr"][is_gold].to(DEVICE)
                logits = model(inputs).squeeze(dim=-1)
                # Calibrate with the temperature stored in the checkpoint
                p_calibrated = torch.sigmoid(logits / float(model.temperature))

            # --- Fusion Model with Artificial Ablation Masking ---
            else:
                # 1. Safely push the required tensors to the GPU
                batch_gpu = {
                    "embeddings": {
                        k: v.to(DEVICE) for k, v in batch["embeddings"].items()
                    },
                    "masks": {k: v.to(DEVICE) for k, v in batch["masks"].items()},
                    "label": batch["label"].to(DEVICE),
                    "subject_id": batch.get("subject_id"),
                }

                # 2. Apply the artificial ablation mask to the GPU batch
                # active_mods specifies which modalities to "turn on" (multiply masks by 0 or 1)
                for mod in model.active_modalities:
                    mask_val = float(active_mods.get(mod, 0.0))
                    batch_gpu["masks"][mod] = batch_gpu["masks"][mod] * mask_val

                # 3. Use the built-in predict_step for consistent inference
                outputs = model.predict_step(batch_gpu, batch_idx=0)

                # 4. Extract only the Gold Cohort predictions
                # (is_gold must be moved to DEVICE to index the GPU output tensor)
                p_calibrated = outputs["p_calibrated"][is_gold.to(DEVICE)]

            all_p_calibrated.append(p_calibrated.cpu())
            all_targets.append(targets.cpu())
            all_subject_ids.extend(subject_ids)

    if len(all_targets) == 0:
        logger.error("No Gold Cohort patients found in this dataloader!")
        return np.array([]), np.array([]), []

    y_true = torch.cat(all_targets, dim=0).numpy().flatten()
    p_calibrated_np = torch.cat(all_p_calibrated, dim=0).numpy().flatten()

    return y_true, p_calibrated_np, all_subject_ids


@hydra.main(config_path=str(Config.HYDRA_CONFIG_DIR), config_name="config")
def main(cfg: DictConfig):
    """
    Run Incremental Value Analysis (IVA) via modality masking.

    Evaluates 8 progressive masking combinations on the Gold Cohort to quantify the
    incremental value of each auxiliary modality over the EHR baseline.
    """
    Config.setup_logging()
    Config.set_seed(cfg.seed)

    logger.info(
        f"Running Incremental Value Analysis with config:\n{OmegaConf.to_yaml(cfg)}"
    )

    active_modalities = cfg.modalities.active
    num_mods = len(active_modalities)

    use_weights = cfg.training.get("use_pretrained_unimodal_weights", False)
    study_name = f"{num_mods}mod_{'pretrained' if use_weights else 'scratch'}"
    run_name = f"{study_name}_final"

    # 1. Load Data
    datamodule = FusionDataModule(
        active_modalities=active_modalities,
        batch_size=256,
        num_workers=cfg.training.get("num_workers", 4),
        ehr_dropout_rate=0.0,
    )
    datamodule.setup(stage="test")
    test_loader = datamodule.test_dataloader()

    # 2. Load Best Fusion Model
    checkpoint_dir = Config.DIR_MODELS / "fusion" / run_name
    model_path = checkpoint_dir / f"best_{run_name}_calibrated.ckpt"
    logger.info(f"Loading Fusion Model from {model_path.name}...")
    fusion_model = LateFusionModule.load_from_checkpoint(model_path, weights_only=False)

    # 3. Load Unimodal Baseline (EHR MLP)
    ehr_checkpoint_dir = Config.DIR_MODELS / "ehr" / "mlp"
    ehr_model_path = ehr_checkpoint_dir / "best_ehr_mlp_calibrated.ckpt"
    logger.info(f"Loading Baseline EHR Model from {ehr_model_path.name}...")
    ehr_model = UnimodalModule.load_from_checkpoint(ehr_model_path, weights_only=False)

    # 4. Load Global Thresholds from previous runs
    ehr_metrics_path = Config.RESULTS_UNIMODAL_METRICS_DIR / "ehr" / "mlp_metrics.json"
    if not ehr_metrics_path.exists():
        logger.error(
            f"EHR metrics file missing at {ehr_metrics_path}. Cannot read threshold."
        )
        return
    with open(ehr_metrics_path, "r") as f:
        ehr_threshold = json.load(f)["threshold"]

    fusion_metrics_path = Config.RESULTS_FUSION_METRICS_DIR / f"{run_name}_metrics.json"
    if not fusion_metrics_path.exists():
        logger.error(
            f"Fusion metrics file missing at {fusion_metrics_path}. Cannot read threshold."
        )
        return
    with open(fusion_metrics_path, "r") as f:
        fusion_threshold = json.load(f)["threshold"]

    logger.info(
        f"Loaded global thresholds -> Fusion: {fusion_threshold:.4f}, EHR: {ehr_threshold:.4f}"
    )

    # 5. Build Mask Configurations
    modality_combinations = {
        "0_Unimodal_Baseline_EHR": {"is_unimodal": True},
        "1_Fusion_EHR_Only": {"ehr": 1, "ecg": 0, "cxr_img": 0, "cxr_txt": 0},
        "2_Fusion_EHR_ECG": {"ehr": 1, "ecg": 1, "cxr_img": 0, "cxr_txt": 0},
        "3_Fusion_EHR_IMG": {"ehr": 1, "ecg": 0, "cxr_img": 1, "cxr_txt": 0},
        "4_Fusion_EHR_TXT": {"ehr": 1, "ecg": 0, "cxr_img": 0, "cxr_txt": 1},
        "5_Fusion_EHR_IMG_TXT": {"ehr": 1, "ecg": 0, "cxr_img": 1, "cxr_txt": 1},
        "6_Fusion_EHR_ECG_IMG": {"ehr": 1, "ecg": 1, "cxr_img": 1, "cxr_txt": 0},
        "7_Fusion_All_Modalities": {"ehr": 1, "ecg": 1, "cxr_img": 1, "cxr_txt": 1},
    }

    base_metrics_dir = Config.RESULTS_INCREMENTAL_VALUE_METRICS_DIR / run_name
    base_preds_dir = Config.RESULTS_INCREMENTAL_VALUE_PREDICTIONS_DIR / run_name

    for combo_name, active_mods in modality_combinations.items():
        logger.info(f"Evaluating Masked Combination: {combo_name}")

        is_unimodal = active_mods.get("is_unimodal", False)
        current_model = ehr_model if is_unimodal else fusion_model
        # Use the respective pre-calculated global threshold
        global_threshold = ehr_threshold if is_unimodal else fusion_threshold

        # Test Inference: Apply strictly to the Gold Cohort
        y_true_gold, p_cal_gold, subject_ids_gold = get_masked_predictions(
            current_model, test_loader, active_mods, is_unimodal
        )

        if len(y_true_gold) == 0:
            logger.warning(f"Skipping {combo_name} due to empty Gold Cohort.")
            continue

        evaluator = ModelEvaluator(
            run_name=combo_name,
            modality="incremental_value",
            metrics_dir=base_metrics_dir,
            predictions_dir=base_preds_dir,
        )

        # Apply the global threshold
        metrics = compute_metrics(y_true_gold, p_cal_gold, threshold=global_threshold)
        evaluator._save_local_artifacts(
            metrics, subject_ids_gold, y_true_gold, p_cal_gold
        )

        logger.info(
            f"Applied Threshold: {global_threshold:.4f} | AUPRC: {metrics['auprc']:.4f} | F1: {metrics['f1']:.4f}"
        )

    logger.info(f"Incremental value artifacts successfully saved.")


if __name__ == "__main__":
    main()

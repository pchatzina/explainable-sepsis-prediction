"""
Fusion Train, Calibrate, Evaluate.

This orchestrator implements the complete fusion training workflow:
  1. Config Extraction: Parse Hydra config and load unimodal architectures if pretrained.
  2. Data Setup: Initialize FusionDataModule with embeddings and optional EHR dropout.
  3. Tuning: Run Optuna-based hyperparameter search to find optimal
     gating/synergy architectures and hyperparameters.
  4. Training: Train final model with best-found hyperparameters using val/total_loss
     as checkpoint monitor.
  5. Calibration: Fit LBFGS temperature scaling on validation logits to calibrate
     predicted probabilities.
  6. Validation & Test Inference: Generate predictions on val/test splits with
     calibrated temperatures.
  7. Evaluation: Compute and persist comprehensive metrics using the ModelEvaluator.

Usage:
    python -m src.training.fusion.run_fusion modalities=4_modality training=pretrained
    python -m src.training.fusion.run_fusion modalities=4_modality training=scratch
"""

import json
import logging
import gc
from pathlib import Path

import hydra
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf

from src.utils.config import Config
from src.data.loaders.fusion_datamodule import FusionDataModule
from src.models.fusion.late_fusion_module import LateFusionModule
from src.optimization.fusion.fusion_tuner import run_hydra_tuner
from src.training.trainer_factory import setup_reproducibility, build_trainer
from src.calibration.temperature_scaling import LBFGSCalibrator
from src.evaluation.evaluator import ModelEvaluator

logger = logging.getLogger(__name__)


def load_unimodal_configs(active_modalities: list) -> dict:
    """Load optimal unimodal architecture configs for pretrained weight compatibility.

    When training fusion models in pretrained mode, each modality's standalone MLP
    must be reconstructed with the exact same architecture as was used during unimodal
    training. This function reads the saved `best_hyperparameters.json` from each
    modality's tuning directory and extracts the architecture configuration needed.

    Args:
        active_modalities: List of modality keys

    Returns:
        Dictionary mapping modality key -> architecture config dict with keys:
        'hidden_dim_1', 'hidden_dim_2', 'dropout_rate', 'activation'
    """
    configs = {}
    for mod in active_modalities:
        path = Config.DIR_MODELS / mod / "tuning" / "best_hyperparameters.json"
        if not path.exists():
            raise FileNotFoundError(
                f"Unimodal config not found for '{mod}' at {path}. "
                "Run unimodal tuning first."
            )

        with open(path, "r") as f:
            params = json.load(f)["params"]
        mod_config = params.get("config", params)

        required_keys = ["activation", "hidden_dim_1", "hidden_dim_2", "dropout_rate"]
        missing = [k for k in required_keys if k not in mod_config]
        if missing:
            raise KeyError(
                f"Unimodal config for '{mod}' at {path} is missing required "
                f"architecture keys: {missing}."
            )

        logger.info(
            f"Loaded unimodal config for {mod.upper()}: "
            f"hidden_dims=[{mod_config['hidden_dim_1']}, {mod_config['hidden_dim_2']}], "
            f"dropout_rate={mod_config['dropout_rate']}, "
            f"activation={mod_config['activation']}"
        )
        configs[mod] = mod_config
    return configs


@hydra.main(config_path=str(Config.HYDRA_CONFIG_DIR), config_name="config")
def main(cfg: DictConfig) -> None:
    """Main entry point for fusion training pipeline.

    Handles both pretrained and from-scratch initialization modes.

    Args:
        cfg: OmegaConf DictConfig from Hydra. Non-obvious keys:
          - training.use_pretrained_unimodal_weights: If True, load frozen pretrained
            unimodal weights; False trains all unimodal MLPs from scratch.
          - training.tune_ehr_dropout: If True, include EHR dropout rate in the
            Optuna search space (regularises the dominant EHR modality).
    """
    Config.setup_logging()
    setup_reproducibility(cfg.seed)

    logger.info(
        f"Running Unified Fusion Pipeline with config:\n{OmegaConf.to_yaml(cfg)}"
    )

    # --- 1. Config Extraction ---
    # Extract active modalities and training mode (pretrained vs. scratch)
    active_modalities = cfg.modalities.active
    num_mods = len(active_modalities)

    current_input_dims = {mod: Config.EMBEDDING_DIMS[mod] for mod in active_modalities}
    use_weights = cfg.training.get("use_pretrained_unimodal_weights", False)
    tune_ehr_dropout = cfg.training.get("tune_ehr_dropout", True)

    study_name = f"{num_mods}mod_{'pretrained' if use_weights else 'scratch'}"
    run_name = f"{study_name}_final"

    logger.info("=== FUSION PIPELINE CONFIGURATION ===")
    logger.info(f"Active Modalities : {active_modalities}")
    logger.info(f"Pretrained Weights: {use_weights}")
    logger.info(f"Tune EHR Dropout  : {tune_ehr_dropout}")
    logger.info(f"Study Name        : {study_name}")
    logger.info("=====================================")

    # --- 2. Data Setup ---
    # Initialize FusionDataModule with embeddings. Start with ehr_dropout_rate=0.0.
    # The tuner will dynamically inject different dropout rates per trial by updating
    # datamodule.ehr_dropout_rate before each trainer.fit call. This ensures that
    # Lightning's internal setup() re-invocations always use the trial-specific dropout value.
    # Validation and test datasets are always created with ehr_dropout_rate=0.0 to ensure
    # a consistent evaluation signal across all trials.
    logger.info("Initializing FusionDataModule...")
    datamodule = FusionDataModule(
        active_modalities=active_modalities,
        batch_size=cfg.training.batch_size,
        num_workers=4,
        ehr_dropout_rate=0.0,
    )
    datamodule.setup(stage="fit")

    # --- 3. Tuning (30 trials of Optuna) ---
    # Optuna searches over:
    #   - Gating network architecture: gate_hidden_1, gate_hidden_2
    #   - Synergy head architecture: syn_hidden_1, syn_hidden_2
    #   - Regularization: dropout_rate, weight_decay
    #   - Gating temperature
    #   - Learning rate
    #   - Lambda weight (auxiliary loss coefficient)
    #   - EHR dropout if tune_ehr_dropout=True
    # Objective metric: val/auprc (maximized). Uses median pruning with early stopping.
    output_tuning_dir = Config.DIR_MODELS / "fusion" / "tuning" / run_name
    output_tuning_dir.mkdir(parents=True, exist_ok=True)

    # Load unimodal architectures if in pretrained mode
    unimodal_configs = load_unimodal_configs(active_modalities) if use_weights else None

    logger.info("Starting Hyperparameter Tuning (30 trials)...")
    best_params = run_hydra_tuner(
        active_modalities=active_modalities,
        input_dims=current_input_dims,
        use_pretrained=use_weights,
        unimodal_configs=unimodal_configs,
        datamodule=datamodule,
        epochs=cfg.training.epochs,
        output_dir=output_tuning_dir,
        study_name=study_name,
        n_trials=30,
        seed=cfg.seed,
        tune_ehr_dropout=tune_ehr_dropout,
    )

    # --- 4. Training (Final Model) ---
    # Initialize the final model using the best hyperparameters found by Optuna.
    # Re-apply the optimal EHR dropout rate to the datamodule; any subsequent setup() calls
    # will use this value for creating the training dataset.
    optimal_dropout = best_params.get("ehr_dropout_rate", 0.0)
    datamodule.ehr_dropout_rate = optimal_dropout
    datamodule.setup(stage="fit")
    logger.info(f"Final Training EHR Dropout Rate: {optimal_dropout:.2f}")

    logger.info("Initializing Final LateFusion Model...")
    final_config = best_params["config"]

    if use_weights:
        model = LateFusionModule.from_pretrained(
            input_dims=current_input_dims,
            active_modalities=active_modalities,
            learning_rate=best_params["learning_rate"],
            config=final_config,
            unimodal_configs=unimodal_configs,
            lambda_weight=best_params["lambda_weight"],
        )
    else:
        model = LateFusionModule.from_scratch(
            input_dims=current_input_dims,
            active_modalities=active_modalities,
            learning_rate=best_params["learning_rate"],
            config=final_config,
            lambda_weight=best_params["lambda_weight"],
        )

    # Monitor val/total_loss for early stopping and checkpoint selection (not AUPRC).
    # Rationale: During tuning, lambda is fixed, so total_loss provides an unbiased
    # composite signal of both main and auxiliary objectives. Loss is also more stable
    # epoch-to-epoch than AUPRC, improving convergence detection.
    trainer, checkpoint_cb = build_trainer(
        max_epochs=cfg.training.epochs,
        checkpoint_dir=Config.DIR_MODELS / "fusion" / run_name,
        experiment_name=run_name,
        monitor_metric="val/total_loss",
        patience=10,
        use_tensorboard=True,
    )

    logger.info(f"Starting Training for configuration: {run_name}")
    trainer.fit(model, datamodule=datamodule)

    # --- 5. Load Best Checkpoint & Temperature Calibration ---
    # Load the best checkpoint saved during training, then fit LBFGS-based
    # temperature scaling on the validation set to improve probability calibration.
    # Temperature scaling computes per-component temperatures (for each modality's logits
    # and the final joint prediction) that minimize cross-entropy on validation:
    #   p_calibrated = sigmoid(logit / T).
    logger.info("--- Running LBFGS Temperature Calibration ---")
    best_model = LateFusionModule.load_from_checkpoint(
        checkpoint_cb.best_model_path, weights_only=False
    )

    # Note: Validation and test datasets always use ehr_dropout_rate=0.0 (enforced by
    # FusionDataModule.setup()), ensuring consistent evaluation regardless of whether
    # training used EHR dropout.
    val_loader = datamodule.val_dataloader()

    # Predict with temperature=1.0 (uncalibrated) to extract raw logits for calibration fitting
    val_preds_uncal = trainer.predict(best_model, dataloaders=val_loader)

    # Reconstruct the logits_dict and targets from the batched outputs
    logits_dict = {}
    keys = val_preds_uncal[0]["logits"].keys()
    for k in keys:
        logits_dict[k] = torch.cat([batch["logits"][k] for batch in val_preds_uncal])

    val_targets = torch.cat([batch["targets"] for batch in val_preds_uncal])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    calibrator = LBFGSCalibrator(max_iter=50, lr=0.01)

    optimal_temps = calibrator.fit(logits_dict, val_targets, device=device)
    best_model.temperatures = optimal_temps
    if (
        abs(float(best_model.temperatures["final"]) - float(optimal_temps["final"]))
        > 1e-8
    ):
        raise RuntimeError("Final calibration temperature mismatch after assignment.")
    logger.info(f"Calibration temperatures: {optimal_temps}")

    # Persist calibration temperatures
    temps_file = (
        Config.RESULTS_DIR
        / "fusion"
        / f"master_calibration_temperatures_{num_mods}mod.json"
    )
    temps_file.parent.mkdir(parents=True, exist_ok=True)
    existing = json.loads(temps_file.read_text()) if temps_file.exists() else {}
    existing[run_name] = optimal_temps
    temps_file.write_text(json.dumps(existing, indent=4))
    logger.info(f"Saved calibration temperatures to {temps_file}")

    # Persist a calibrated checkpoint so downstream evaluation can load both weights and
    # in-memory calibration state from a single artifact.
    calibrated_ckpt_path = (
        Config.DIR_MODELS / "fusion" / run_name / f"best_{run_name}_calibrated.ckpt"
    )
    calibrated_ckpt = torch.load(
        checkpoint_cb.best_model_path, map_location="cpu", weights_only=False
    )
    calibrated_ckpt["state_dict"] = best_model.state_dict()
    calibrated_ckpt["temperatures"] = {
        k: float(v) for k, v in best_model.temperatures.items()
    }
    torch.save(calibrated_ckpt, calibrated_ckpt_path)
    logger.info(f"Saved calibrated checkpoint to {calibrated_ckpt_path}")

    # Prefer calibrated checkpoint for downstream inference if available.
    # This keeps inference behavior consistent even when this stage is resumed/re-run.
    inference_ckpt_path = (
        calibrated_ckpt_path
        if calibrated_ckpt_path.exists()
        else Path(checkpoint_cb.best_model_path)
    )
    logger.info(f"Loading inference model from {inference_ckpt_path}")
    best_model = LateFusionModule.load_from_checkpoint(
        inference_ckpt_path, weights_only=False
    )

    # --- 6. Validation Inference with Calibrated Temperatures ---
    # Re-run validation predictions using the fitted calibration temperatures embedded
    # in the model. These calibrated probabilities are used for metric computation.
    logger.info("--- Re-running Validation Inference with Calibrated Temperature ---")
    val_preds_cal = trainer.predict(best_model, dataloaders=val_loader)
    val_probs_calibrated = (
        torch.cat([x["p_calibrated"] for x in val_preds_cal]).cpu().numpy()
    )
    val_targets_np = torch.cat([x["targets"] for x in val_preds_cal]).cpu().numpy()

    # --- 7. Test Inference with Calibrated Temperatures ---
    # Generate predictions on the held-out test set using the final calibrated model.
    logger.info("--- Running Inference on Test Set ---")
    datamodule.setup(stage="test")
    test_loader = datamodule.test_dataloader()
    test_preds = trainer.predict(best_model, dataloaders=test_loader)

    test_probs_calibrated = (
        torch.cat([x["p_calibrated"] for x in test_preds]).cpu().numpy()
    )
    test_targets_np = torch.cat([x["targets"] for x in test_preds]).cpu().numpy()
    test_ids = []
    for out in test_preds:
        if out.get("subject_ids") is not None:
            test_ids.extend(out["subject_ids"].cpu().numpy().tolist())

    # --- 8. Evaluate ---
    logger.info("--- Saving Unified Evaluation Metrics ---")
    metrics_dir = Config.RESULTS_FUSION_METRICS_DIR
    predictions_dir = Config.RESULTS_FUSION_PREDICTIONS_DIR

    evaluator = ModelEvaluator(
        run_name=run_name,
        modality="fusion",
        metrics_dir=metrics_dir,
        predictions_dir=predictions_dir,
    )

    evaluator.evaluate(
        y_true_val=val_targets_np,
        y_prob_val=val_probs_calibrated,
        y_true_test=test_targets_np,
        y_prob_test=test_probs_calibrated,
        subject_ids_test=test_ids,
    )

    logger.info(f"=== Pipeline Completed Successfully for {run_name} ===")

    # --- Stage 8a: Persist Uncalibrated Test Predictions ---
    # predict_step applies only self.temperatures["final"] to compute p_calibrated
    # (verified in LateFusionModule.predict_step — no other temperature keys participate).
    # Overriding that key to 1.0 yields sigmoid(logit_final), the uncalibrated probability.
    logger.info("--- Persisting Uncalibrated Test Predictions ---")
    saved_temp_final = best_model.temperatures["final"]
    best_model.temperatures["final"] = 1.0
    test_preds_uncal = trainer.predict(best_model, dataloaders=test_loader)
    best_model.temperatures["final"] = saved_temp_final

    test_probs_uncalibrated = (
        torch.cat([x["p_calibrated"] for x in test_preds_uncal]).cpu().numpy()
    )
    uncal_path = predictions_dir / f"{run_name}_predictions_uncalibrated.csv"
    pd.DataFrame(
        {
            "subject_id": test_ids,
            "label": test_targets_np,
            "probability": test_probs_uncalibrated,
        }
    ).to_csv(uncal_path, index=False)
    logger.info(f"Saved uncalibrated test predictions to {uncal_path}")

    # --- Cleanup ---
    del trainer, model, best_model, datamodule, val_loader, test_loader
    torch.cuda.empty_cache()
    gc.collect()


if __name__ == "__main__":
    main()

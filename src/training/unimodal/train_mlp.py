"""MLP Training, Hyperparameter Tuning, Calibration, and Evaluation.

This module orchestrates the complete MLP training pipeline:
1. Optuna-based hyperparameter tuning (20 trials on AUPRC)
2. Instantiation and training of the best model
3. Temperature scaling calibration on validation set
4. Final evaluation on test set with calibrated probabilities

Key design decisions:
- Tuning optimizes val/auprc (clinically meaningful, lambda-independent)
- Checkpoint saves on val/loss (smoother epoch-to-epoch signal)
- Calibration uses LBFGS to minimize BCE on validation logits
- GPU memory is explicitly freed after training
"""

import logging
import torch
import json
import pandas as pd

from pathlib import Path
from src.calibration.temperature_scaling import LBFGSCalibrator
from src.models.unimodal.mlp import UnimodalModule
from src.optimization.unimodal.unimodal_tuner import run_unimodal_tuner
from src.utils.config import Config
from src.data.loaders.helpers import get_unimodal_dataloader
from src.training.trainer_factory import build_trainer

logger = logging.getLogger(__name__)


def train_eval_mlp(
    modality,
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test,
    train_ids,
    val_ids,
    test_ids,
    cfg,
):
    """Complete unimodal MLP training, calibration, and evaluation pipeline.

    Steps:
    1. Create dataloaders for train/val/test
    2. Run Optuna hyperparameter tuning (20 trials) → save best params
    3. Instantiate best model and train with early stopping (monitor: val/loss)
    4. Load best checkpoint and apply temperature calibration on validation set
    5. Evaluate on test set with calibrated probabilities

    Args:
        modality: Modality name (e.g., 'ehr', 'ecg').
        X_train, y_train: Training embeddings and labels.
        X_val, y_val: Validation embeddings and labels.
        X_test, y_test: Test embeddings and labels.
        train_ids, val_ids, test_ids: Subject IDs for each split.
        cfg: Hydra config object with seed and other parameters.

    Outputs:
        Saves:
        - models/{modality}/tuning/best_hyperparameters.json: Best Optuna hyperparameters
        - models/{modality}/mlp/best_{modality}_mlp.ckpt: Best checkpoint
        - models/{modality}/mlp/best_{modality}_mlp_weights.pt: Model weights
        - results/unimodal/master_calibration_temperatures.json: Optimal temperature
        - results/unimodal/metrics/{modality}/: Evaluation metrics
        - results/unimodal/predictions/{modality}/: Test predictions
    """
    logger.info(f"--- Training Lightning MLP for {modality} ---")

    # Setup output directories
    output_dir = Config.DIR_MODELS / modality / "mlp"
    tuning_dir = Path(Config.DIR_MODELS) / modality / "tuning"
    metrics_dir = Config.RESULTS_UNIMODAL_METRICS_DIR / modality
    predictions_dir = Config.RESULTS_UNIMODAL_PREDICTIONS_DIR / modality
    output_dir.mkdir(parents=True, exist_ok=True)
    tuning_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir.mkdir(parents=True, exist_ok=True)

    input_dim = X_train.shape[1]

    # === Stage 1: Create dataloaders ===
    train_loader = get_unimodal_dataloader(X_train, y_train, train_ids, shuffle=True)
    val_loader = get_unimodal_dataloader(X_val, y_val, val_ids, shuffle=False)
    test_loader = get_unimodal_dataloader(X_test, y_test, test_ids, shuffle=False)

    # === Stage 2: Hyperparameter Tuning ===
    logger.info(f"Starting Optuna hyperparameter tuning for {modality}...")
    best_params = run_unimodal_tuner(
        modality=modality,
        input_dim=input_dim,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=30,
        output_dir=tuning_dir,
        n_trials=20,
        seed=cfg.seed,
    )

    # === Stage 3: Model Training ===
    logger.info(f"Instantiating model with best hyperparameters...")
    model = UnimodalModule(
        input_dim=input_dim,
        learning_rate=best_params["learning_rate"],
        config=best_params["config"],
        weight_decay=best_params["weight_decay"],
    )

    # Note: Intentional mismatch between Optuna's tuning metric and the checkpoint metric.
    # Optuna tunes on val/auprc — the primary clinical objective — to select the best
    # architecture and learning rate. Within a fixed training run, val/loss is used for
    # checkpointing (consistent with the Late-Fusion pipeline) because it is a smoother,
    # more stable epoch-to-epoch signal than AUPRC, reducing the risk of saving
    # a checkpoint that happens to score well on a noisy epoch.
    trainer, checkpoint_cb = build_trainer(
        max_epochs=100,
        checkpoint_dir=output_dir,
        experiment_name=f"{modality}_mlp",
        monitor_metric="val/loss",
        patience=10,
    )

    logger.info(f"Training model...")
    trainer.fit(model, train_loader, val_loader)

    # === Stage 4: Load Best Checkpoint and Prepare for Calibration ===
    best_model = UnimodalModule.load_from_checkpoint(
        checkpoint_cb.best_model_path, weights_only=False
    )

    # Save the base state_dict for downstream use
    torch.save(
        best_model.model.state_dict(), output_dir / f"best_{modality}_mlp_weights.pt"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # === Stage 5: Temperature Calibration ===
    logger.info("--- Extracting Uncalibrated Predictions for Calibration ---")
    val_preds_uncal = trainer.predict(best_model, dataloaders=val_loader)

    # Extract logits and targets for calibration
    val_logits = torch.cat([x["logits"] for x in val_preds_uncal])
    val_targets = torch.cat([x["targets"] for x in val_preds_uncal])
    val_targets_np = val_targets.cpu().numpy()

    # Run LBFGS optimization for temperature scaling
    logger.info("--- Running LBFGS Temperature Calibration ---")
    logits_dict = {"final": val_logits}
    calibrator = LBFGSCalibrator(max_iter=50, lr=0.01)
    optimal_temps = calibrator.fit(logits_dict, val_targets, device=device)

    best_model.temperature = optimal_temps["final"]
    if abs(float(best_model.temperature) - float(optimal_temps["final"])) > 1e-8:
        raise RuntimeError("Final calibration temperature mismatch after assignment.")
    logger.info(f"Optimal temperature applied: {optimal_temps['final']:.4f}")

    # Store temperature in centralized tracking file
    temps_file = (
        Config.RESULTS_DIR / "unimodal" / "master_calibration_temperatures.json"
    )
    temps_file.parent.mkdir(parents=True, exist_ok=True)

    existing_temps = json.loads(temps_file.read_text()) if temps_file.exists() else {}
    existing_temps[f"{modality}_mlp"] = optimal_temps["final"]
    temps_file.write_text(json.dumps(existing_temps, indent=4))
    logger.info(f"Saved {modality}_mlp calibration temperature to {temps_file}")

    # Persist a calibrated checkpoint so downstream evaluation can load both weights and
    # in-memory calibration state from a single artifact.
    calibrated_ckpt_path = output_dir / f"best_{modality}_mlp_calibrated.ckpt"
    calibrated_ckpt = torch.load(
        checkpoint_cb.best_model_path, map_location="cpu", weights_only=False
    )
    calibrated_ckpt["state_dict"] = best_model.state_dict()
    calibrated_ckpt["temperature"] = float(best_model.temperature)
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
    best_model = UnimodalModule.load_from_checkpoint(
        inference_ckpt_path, weights_only=False
    )

    # === Stage 6: Re-run Inference on Calibrated Model ===
    logger.info("--- Extracting Calibrated Predictions ---")
    val_preds_cal = trainer.predict(best_model, dataloaders=val_loader)
    test_preds_cal = trainer.predict(best_model, dataloaders=test_loader)

    val_probs_calibrated = (
        torch.cat([x["p_calibrated"] for x in val_preds_cal]).cpu().numpy()
    )
    test_probs_calibrated = (
        torch.cat([x["p_calibrated"] for x in test_preds_cal]).cpu().numpy()
    )

    test_targets_np = torch.cat([x["targets"] for x in test_preds_cal]).cpu().numpy()

    # === Stage 7: Evaluate ===
    logger.info("--- Running Final Evaluation ---")
    from src.evaluation.evaluator import ModelEvaluator

    evaluator = ModelEvaluator(
        run_name="mlp",
        modality=modality,
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
    logger.info(f"Finished MLP Training and Evaluation for {modality}")

    # === Stage 8: Persist Uncalibrated Test Predictions ===
    logger.info("--- Persisting Uncalibrated Test Predictions ---")
    saved_temperature = best_model.temperature
    best_model.temperature = 1.0
    test_preds_uncal = trainer.predict(best_model, dataloaders=test_loader)
    best_model.temperature = saved_temperature

    test_probs_uncalibrated = (
        torch.cat([x["p_calibrated"] for x in test_preds_uncal]).cpu().numpy()
    )
    uncal_path = predictions_dir / "mlp_predictions_uncalibrated.csv"
    pd.DataFrame(
        {
            "subject_id": test_ids,
            "label": test_targets_np,
            "probability": test_probs_uncalibrated,
        }
    ).to_csv(uncal_path, index=False)
    logger.info(f"Saved uncalibrated test predictions to {uncal_path}")

    # === Cleanup ===
    del trainer, model, best_model, train_loader, val_loader, test_loader
    torch.cuda.empty_cache()
    import gc

    gc.collect()

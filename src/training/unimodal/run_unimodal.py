"""
Unimodal Train, Calibrate, Evaluate.

Trains, calibrates, and evaluates baseline unimodal classifiers (LR, XGBoost, and MLP)
for all active modalities specified in the Hydra configuration.

Usage:
    python -m src.training.unimodal.run_unimodal modalities=4_modality
"""

import logging

import hydra
from omegaconf import DictConfig
from src.utils.config import Config
from src.training.trainer_factory import setup_reproducibility

from src.data.loaders.helpers import load_embeddings

from src.training.unimodal.train_lr import train_eval_lr
from src.training.unimodal.train_mlp import train_eval_mlp
from src.training.unimodal.train_xgb import train_eval_xgboost

logger = logging.getLogger(__name__)


@hydra.main(config_path=str(Config.HYDRA_CONFIG_DIR), config_name="config")
def main(cfg: DictConfig):
    """Main entry point for unimodal training pipeline.

    Args:
        cfg: Hydra DictConfig with keys:
            - modalities: Config specifying active modalities (ehr, ecg, cxr_img, cxr_txt)
            - seed: Random seed for reproducibility
    """
    Config.setup_logging()
    setup_reproducibility(cfg.seed)

    logger.info(f"Running Unimodal Pipeline for: {cfg.modalities.name}")

    # Iterate over all active modalities from the config
    for mod_name in cfg.modalities.active:
        logger.info(
            f"\n{'=' * 50}\nSTARTING PIPELINE FOR MODALITY: {mod_name.upper()}\n{'=' * 50}"
        )

        # Construct path to normalized embeddings for this modality
        emb_dir = Config.DIR_PROCESSED / mod_name / "embeddings" / "normalized"

        # Load train/val/test splits with labels and subject IDs
        X_train, y_train, train_ids = load_embeddings(emb_dir / "train_embeddings.pt")
        X_val, y_val, val_ids = load_embeddings(emb_dir / "valid_embeddings.pt")
        X_test, y_test, test_ids = load_embeddings(emb_dir / "test_embeddings.pt")

        # Train Logistic Regression baseline
        train_eval_lr(
            modality=mod_name,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            X_test=X_test,
            y_test=y_test,
            test_ids=test_ids,
        )

        # Train XGBoost baseline
        train_eval_xgboost(
            modality=mod_name,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            X_test=X_test,
            y_test=y_test,
            test_ids=test_ids,
        )

        # Train MLP with Optuna tuning, temperature calibration, and evaluation
        train_eval_mlp(
            modality=mod_name,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            X_test=X_test,
            y_test=y_test,
            train_ids=train_ids,
            val_ids=val_ids,
            test_ids=test_ids,
            cfg=cfg,
        )

    logger.info("=== All Unimodal Models Trained and Evaluated ===")


if __name__ == "__main__":
    main()

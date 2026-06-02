"""Factory for PyTorch Lightning Trainers and Reproducibility Setup."""

import logging
from pathlib import Path
from typing import Tuple

from src.utils.config import Config
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

logger = logging.getLogger(__name__)


def setup_reproducibility(seed: int = 42):
    """Set up strict reproducibility across PyTorch, NumPy, and CUDA.

    This function must be called before initializing models or dataloaders
    to ensure deterministic behavior and experiment reproducibility.

    Args:
        seed: Random seed value (default 42).
    """
    Config.set_seed(seed)
    pl.seed_everything(seed, workers=True)
    logger.info(f"Reproducibility enforced.")


def build_trainer(
    max_epochs: int,
    checkpoint_dir: Path | str,
    experiment_name: str,
    monitor_metric: str = "val/total_loss",
    mode: str = "min",
    patience: int = 10,
    deterministic: bool = True,
    use_tensorboard: bool = False,
) -> Tuple[pl.Trainer, ModelCheckpoint]:
    """Construct a PyTorch Lightning Trainer with standard callbacks.

    This factory function creates a consistent Trainer configuration used across
    unimodal and fusion training pipelines. It automatically sets up:
    - ModelCheckpoint: Saves best model based on monitored metric
    - EarlyStopping: Stops training if metric doesn't improve for `patience` epochs
    - Deterministic behavior (for reproducibility)

    Args:
        max_epochs: Maximum number of training epochs.
        checkpoint_dir: Directory to save checkpoint files. Created if not exists.
        experiment_name: Name used in checkpoint filename.
        monitor_metric: Validation metric to monitor (e.g., "val/loss", "val/auprc").
            Must match a metric logged by the LightningModule.
        mode: "min" if monitoring loss, "max" if monitoring accuracy/AUPRC/etc.
        patience: Number of epochs without improvement before early stopping.
        deterministic: If True, enforces deterministic algorithms (slightly slower).
        use_tensorboard: If True, enables TensorBoard logging for visualization.

    Returns:
        Tuple of:
            - pl.Trainer: Configured PyTorch Lightning Trainer instance
            - ModelCheckpoint: The checkpoint callback (useful for retrieving best_model_path)
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Checkpoint callback: save best model based on monitor_metric
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename=f"best_{experiment_name}",
        monitor=monitor_metric,
        mode=mode,
        save_top_k=1,  # Only keep the single best checkpoint
    )

    # Early stopping: prevent overfitting by stopping training if metric plateaus
    early_stop_callback = EarlyStopping(
        monitor=monitor_metric, patience=patience, mode=mode
    )

    # Optional TensorBoard logging
    if use_tensorboard:
        # Use Config's TensorBoard directory for organization
        tb_logger = TensorBoardLogger(
            save_dir=Config.TENSORBOARD_DIR,
            name=experiment_name,
            version="",  # Leaves versioning to default
        )
    else:
        tb_logger = False

    # Build Trainer
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="auto",  # Auto-detect GPU/TPU availability
        devices=1,  # Use single GPU/CPU
        logger=tb_logger,
        callbacks=[checkpoint_callback, early_stop_callback],
        deterministic=deterministic,
    )

    return trainer, checkpoint_callback

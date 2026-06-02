import logging
import gc
from abc import ABC, abstractmethod

import optuna
import pytorch_lightning as pl
import torch
from optuna.integration import PyTorchLightningPruningCallback
from pytorch_lightning.callbacks import EarlyStopping

from src.models.fusion.late_fusion_module import LateFusionModule

logger = logging.getLogger(__name__)


class _BaseObjective(ABC):
    """Abstract base for fusion Optuna objectives.

    Subclasses implement ``_suggest_config`` to construct the full config
    dict that ``LateFusionModule`` consumes, and ``_build_model`` to
    instantiate the Lightning module. This base class owns the training
    loop and metric extraction.

    If a subclass includes ``ehr_dropout_rate`` in the config returned by
    ``_suggest_config``, it is automatically extracted and applied to the
    datamodule before each trial's ``trainer.fit`` call, ensuring any
    internal Lightning ``setup()`` re-calls use the correct value.
    Validation and test datasets always use ``ehr_dropout_rate=0.0``,
    as enforced by ``FusionDataModule.setup()``.
    """

    def __init__(
        self,
        active_modalities: list,
        input_dims: dict,
        datamodule: pl.LightningDataModule,
        epochs: int = 50,
    ):
        self.active_modalities = active_modalities
        self.input_dims = input_dims
        self.datamodule = datamodule
        self.epochs = epochs

    @abstractmethod
    def _suggest_config(self, trial: optuna.trial.Trial) -> dict:
        """Return the full config dict consumed by LateFusionModule."""

    @abstractmethod
    def _build_model(
        self,
        trial: optuna.trial.Trial,
        config: dict,
        learning_rate: float,
        lambda_weight: float,
    ) -> LateFusionModule:
        """Construct and return the Lightning module for this trial."""

    def __call__(self, trial: optuna.trial.Trial) -> float:
        """Execute a single trial of the fusion model tuning loop.

        This method orchestrates one complete trial:
          1. Suggest hyperparameters (architecture, dropout, regularization)
          2. Extract EHR dropout rate and inject into DataModule
          3. Build the model (pretrained or scratch)
          4. Train for max_epochs with early stopping on val/auprc
          5. Extract best validation AUPRC score and return it

        Args:
            trial: Optuna trial object with suggest_categorical, suggest_float methods.

        Returns:
            Best validation AUPRC achieved during training.
        """
        config = self._suggest_config(trial)
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
        lambda_weight = trial.suggest_float("lambda_weight", 0.05, 0.6, step=0.05)

        # --- EHR DROPOUT INJECTION ---
        # Extract ehr_dropout_rate from config and set on datamodule.
        # This ensures any internal setup() re-call by Lightning recreates train_dataset
        # with the correct dropout rate. Validation and test datasets always use
        # ehr_dropout_rate=0.0 as enforced by FusionDataModule.setup().
        ehr_dropout_rate = config.pop("ehr_dropout_rate", 0.0)
        self.datamodule.ehr_dropout_rate = ehr_dropout_rate
        self.datamodule.setup(stage="fit")
        logger.info(
            f"Trial {trial.number}: ehr_dropout_rate set to {ehr_dropout_rate:.2f}"
        )

        # Build the model using the suggested hyperparameters
        model = self._build_model(trial, config, learning_rate, lambda_weight)

        # Create a lightweight trainer for this trial
        trainer = pl.Trainer(
            max_epochs=self.epochs,
            accelerator="auto",
            logger=False,
            deterministic=True,
            enable_checkpointing=False,
            callbacks=[
                # Early stopping: stops if val/auprc doesn't improve for 10 epochs
                EarlyStopping(monitor="val/auprc", patience=10, mode="max"),
                # Optuna pruning: automatically prunes poor trials mid-training
                PyTorchLightningPruningCallback(trial, monitor="val/auprc"),
            ],
        )

        # Train the model
        trainer.fit(model, datamodule=self.datamodule)

        # Extract the best validation AUPRC from the early stopping callback
        early_stop_cb = next(
            cb for cb in trainer.callbacks if isinstance(cb, EarlyStopping)
        )
        best_metric = early_stop_cb.best_score.item()

        # --- Memory Cleanup ---
        del trainer, model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return best_metric

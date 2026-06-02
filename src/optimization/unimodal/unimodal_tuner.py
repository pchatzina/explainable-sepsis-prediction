"""Hyperparameter Tuning for Unimodal Classifiers.

This module provides an Optuna-based hyperparameter tuning pipeline for unimodal MLP
classifiers. It optimizes both architecture (hidden dimensions, activation, dropout) and
training hyperparameters (learning rate, weight decay) on the validation AUPRC metric.

Key design decisions:
- Optimization metric: val/auprc (clinically interpretable, lambda-independent)
- Pruner: MedianPruner with startup trials and warmup steps to avoid early termination
- GPU memory: Aggressively freed after each trial to enable many trials in sequence
"""

import json
import logging
from pathlib import Path

import optuna
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from optuna.integration import PyTorchLightningPruningCallback
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


class UnimodalObjective:
    """Optuna objective function for tuning unimodal MLP hyperparameters.

    This class encapsulates a trial-based optimization target that Optuna samples and
    evaluates. Each trial trains a fresh MLP with proposed hyperparameters and returns
    the best validation AUPRC achieved (with early stopping).

    Attributes:
        input_dim: Embedding dimension from foundation model.
        train_loader: PyTorch DataLoader for training set.
        val_loader: PyTorch DataLoader for validation set.
        epochs: Maximum number of training epochs per trial.
    """

    def __init__(
        self,
        input_dim: int,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 50,
    ):
        """Initialize the Optuna objective.

        Args:
            input_dim: Embedding dimension (e.g., 768).
            train_loader: DataLoader for training data.
            val_loader: DataLoader for validation data.
            epochs: Max epochs per trial (default 50).
        """
        self.input_dim = input_dim
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.epochs = epochs

        # Warn if input must be projected to 768-dim standard size
        if self.input_dim != 768:
            logger.warning(
                f"Input dimension is {self.input_dim} (!= 768). "
                f"A linear projection will be applied. These projection weights are randomly "
                f"initialised (no pretraining), so this modality may require more epochs to converge."
            )

    def __call__(self, trial: optuna.trial.Trial) -> float:
        """Run a single trial: sample hyperparameters, train, and return best validation AUPRC.

        This method is called once per Optuna trial. It:
        1. Samples architecture/training hyperparameters from Optuna's search space
        2. Instantiates a UnimodalModule with those hyperparameters
        3. Trains with early stopping (monitor: val/auprc, patience: 10)
        4. Returns the best AUPRC achieved on validation set
        5. Explicitly frees GPU memory to allow subsequent trials

        Args:
            trial: Optuna Trial object for sampling hyperparameters.

        Returns:
            float: Best validation AUPRC achieved in this trial.
        """
        # Sample hyperparameters from search space
        config = {
            "activation": trial.suggest_categorical("activation", ["ReLU", "GELU"]),
            "hidden_dim_1": trial.suggest_categorical("hidden_dim_1", [128, 256, 512]),
            "hidden_dim_2": trial.suggest_categorical("hidden_dim_2", [32, 64, 128]),
            "dropout_rate": trial.suggest_categorical(
                "dropout_rate", [0.0, 0.1, 0.2, 0.3]
            ),
        }
        learning_rate = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True)

        from src.models.unimodal.mlp import UnimodalModule

        # Instantiate model with sampled hyperparameters
        model = UnimodalModule(
            input_dim=self.input_dim,
            learning_rate=learning_rate,
            config=config,
            weight_decay=weight_decay,
        )

        # Build trainer with AUPRC as the optimization target.
        # Note: AUPRC is lambda-independent and clinically meaningful, enabling fair comparison
        # across all trials. Early stopping on AUPRC ensures consistent pruning behavior.
        trainer = pl.Trainer(
            max_epochs=self.epochs,
            accelerator="auto",
            logger=False,
            deterministic=True,
            enable_checkpointing=False,
            callbacks=[
                EarlyStopping(monitor="val/auprc", patience=10, mode="max"),
                PyTorchLightningPruningCallback(trial, monitor="val/auprc"),
            ],
        )

        # Train and validate
        trainer.fit(
            model, train_dataloaders=self.train_loader, val_dataloaders=self.val_loader
        )

        # Extract best metric from early stopping callback
        early_stop_cb = next(
            cb for cb in trainer.callbacks if isinstance(cb, EarlyStopping)
        )
        best_metric = early_stop_cb.best_score.item()

        # Explicitly free GPU memory to enable subsequent trials to fit in VRAM
        del trainer, model
        import gc

        gc.collect()

        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return best_metric


def run_unimodal_tuner(
    modality: str,
    input_dim: int,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    output_dir: Path,
    n_trials: int = 30,
    seed: int = 42,
) -> dict:
    """Run Optuna hyperparameter tuning for a unimodal MLP classifier.

        If cached results exist for this modality, they are loaded and returned immediately.
        Otherwise, a new Optuna study is created and optimized over n_trials.

        Optuna execution policy:
            - Tuning is intentionally single-process (n_jobs=1) for deterministic
                behavior and consistent resource usage.

    Args:
        modality: Name of the modality (e.g., 'ehr', 'ecg') for logging and caching.
        input_dim: Embedding dimension from foundation model.
        train_loader: DataLoader for training set.
        val_loader: DataLoader for validation set.
        epochs: Maximum training epochs per trial.
        output_dir: Directory to save best_hyperparameters.json.
        n_trials: Number of Optuna trials to run (default 30).
        seed: Random seed for Optuna sampler (default 42).

    Returns:
        dict: Best hyperparameters in format {"config": {...}, "learning_rate": ..., "weight_decay": ...}
    """
    best_params_path = output_dir / "best_hyperparameters.json"

    # Cache check: a previously tuned modality reuses its cached hyperparameters.
    if best_params_path.exists():
        logger.info(
            f"Using cached hyperparameters explicitly found at: {best_params_path}"
        )
        with open(best_params_path, "r") as f:
            return json.load(f)["params"]

    # No cache: begin fresh Optuna study
    logger.info(
        f"Optimal parameters not found for {modality}. Initiating Optuna Tuning for {n_trials} trials..."
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    # Configure pruner: MedianPruner is robust; startup trials allow initial exploration
    # before aggressive pruning. Warmup steps prevent early termination.
    pruner = optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=15)

    # TPESampler: Tree-structured Parzen Estimator for efficient hyperparameter search
    sampler = optuna.samplers.TPESampler(seed=seed)

    study = optuna.create_study(
        direction="maximize",
        pruner=pruner,
        sampler=sampler,
        study_name=f"tune_{modality}",
    )

    # Create the objective function and optimize
    objective = UnimodalObjective(
        input_dim=input_dim,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
    )
    study.optimize(objective, n_trials=n_trials, n_jobs=1)

    # Extract best parameters and reconstruct in the required format
    best_params = study.best_params

    config_dict = {
        "activation": best_params["activation"],
        "hidden_dim_1": best_params["hidden_dim_1"],
        "hidden_dim_2": best_params["hidden_dim_2"],
        "dropout_rate": best_params["dropout_rate"],
    }

    reconstructed_config = {
        "params": {
            "config": config_dict,
            "learning_rate": best_params["lr"],
            "weight_decay": best_params["weight_decay"],
        }
    }

    # Persist hyperparameters to disk for future runs
    with open(best_params_path, "w") as f:
        json.dump(reconstructed_config, f, indent=4)

    logger.info(f"Completed! Cached hyperparams to {best_params_path}")
    return reconstructed_config["params"]

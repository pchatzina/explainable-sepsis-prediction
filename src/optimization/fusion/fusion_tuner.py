import json
import logging
from pathlib import Path
import optuna
import pytorch_lightning as pl

from src.models.fusion.late_fusion_module import LateFusionModule
from src.optimization.fusion.base_objective import _BaseObjective

logger = logging.getLogger(__name__)


class PretrainedObjective(_BaseObjective):
    """Optuna objective function for pretrained-weight fusion model tuning.

    This objective tunes the fusion-specific hyperparameters while keeping the
    pretrained unimodal weights frozen. Since unimodal MLPs are frozen,
    only the gating network, synergy head, and optional EHR dropout are
    subject to optimization.

    Objective Metric: Validates on val/auprc (maximized).
    Early Stopping: 10 trials of patience.
    """

    def __init__(
        self,
        active_modalities: list,
        input_dims: dict,
        unimodal_configs: dict,
        datamodule: pl.LightningDataModule,
        epochs: int = 50,
        tune_ehr_dropout: bool = True,
    ):
        super().__init__(active_modalities, input_dims, datamodule, epochs)
        self.unimodal_configs = unimodal_configs
        self.tune_ehr_dropout = tune_ehr_dropout

    def _suggest_config(self, trial: optuna.trial.Trial) -> dict:
        """Suggest hyperparameters for gating and synergy networks (pretrained mode).

        Args:
            trial: Optuna trial object for suggesting hyperparameter values.

        Returns:
            Configuration dict with keys:
              - gate_hidden_1, gate_hidden_2: Hidden dimensions for gating MLP
              - syn_hidden_1, syn_hidden_2: Hidden dimensions for synergy MLP
              - dropout_rate: Dropout applied to fusion components
              - weight_decay: L2 regularization coefficient
              - gating_temperature: Softmax temperature for gating network
              - ehr_dropout_rate: Probability of masking EHR (if tune_ehr_dropout=True)
        """
        config = {
            "gate_hidden_1": trial.suggest_categorical("gate_hidden_1", [256, 512]),
            "gate_hidden_2": trial.suggest_categorical("gate_hidden_2", [64, 128]),
            "syn_hidden_1": trial.suggest_categorical("syn_hidden_1", [256, 512]),
            "syn_hidden_2": trial.suggest_categorical("syn_hidden_2", [64, 128]),
            "dropout_rate": trial.suggest_categorical(
                "dropout_rate", [0.0, 0.1, 0.2, 0.3]
            ),
            "weight_decay": trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True),
            "gating_temperature": trial.suggest_float(
                "gating_temperature", 0.5, 2.0, step=0.25
            ),
        }

        # If true, Optuna tests values from 0.0 up to 0.5
        if self.tune_ehr_dropout:
            config["ehr_dropout_rate"] = trial.suggest_float(
                "ehr_dropout_rate", 0.0, 0.5, step=0.1
            )
        else:
            config["ehr_dropout_rate"] = 0.0

        return config

    def _build_model(
        self,
        trial: optuna.trial.Trial,
        config: dict,
        learning_rate: float,
        lambda_weight: float,
    ) -> LateFusionModule:
        """Build a LateFusionModule with pretrained unimodal weights.

        Args:
            trial: Optuna trial object.
            config: Hyperparameter config dict from _suggest_config.
            learning_rate: Learning rate for the optimizer.
            lambda_weight: Weight for the auxiliary unimodal loss term.

        Returns:
            LateFusionModule instance with frozen pretrained unimodal weights
            and trainable gating/synergy components.
        """
        return LateFusionModule.from_pretrained(
            input_dims=self.input_dims,
            active_modalities=self.active_modalities,
            learning_rate=learning_rate,
            config=config,
            unimodal_configs=self.unimodal_configs,
            lambda_weight=lambda_weight,
        )


class ScratchObjective(_BaseObjective):
    """Optuna objective function for from-scratch fusion model tuning.

    This objective tunes all fusion-level hyperparameters (including the unimodal
    MLP dimensions) starting from random initialization. All components (unimodal MLPs,
    gating network, synergy head) are trainable.

    Objective Metric: Validates on val/auprc (maximized).
    Early Stopping: 10 trials of patience.
    """

    def __init__(
        self,
        active_modalities: list,
        input_dims: dict,
        datamodule: pl.LightningDataModule,
        epochs: int = 50,
        tune_ehr_dropout: bool = True,
    ):
        super().__init__(active_modalities, input_dims, datamodule, epochs)
        self.tune_ehr_dropout = tune_ehr_dropout

    def _suggest_config(self, trial: optuna.trial.Trial) -> dict:
        """Suggest hyperparameters for all fusion components (scratch mode).

        Args:
            trial: Optuna trial object for suggesting hyperparameter values.

        Returns:
            Configuration dict with keys:
              - uni_hidden_1, uni_hidden_2: Hidden dimensions for unimodal MLPs
              - gate_hidden_1, gate_hidden_2: Hidden dimensions for gating MLP
              - syn_hidden_1, syn_hidden_2: Hidden dimensions for synergy MLP
              - dropout_rate: Dropout applied to all components
              - weight_decay: L2 regularization coefficient
              - gating_temperature: Softmax temperature for gating network
              - ehr_dropout_rate: Probability of masking EHR (if tune_ehr_dropout=True)
        """
        config = {
            "uni_hidden_1": trial.suggest_categorical("uni_hidden_1", [128, 256, 512]),
            "uni_hidden_2": trial.suggest_categorical("uni_hidden_2", [32, 64, 128]),
            "gate_hidden_1": trial.suggest_categorical("gate_hidden_1", [256, 512]),
            "gate_hidden_2": trial.suggest_categorical("gate_hidden_2", [64, 128]),
            "syn_hidden_1": trial.suggest_categorical("syn_hidden_1", [256, 512]),
            "syn_hidden_2": trial.suggest_categorical("syn_hidden_2", [64, 128]),
            "dropout_rate": trial.suggest_categorical(
                "dropout_rate", [0.0, 0.1, 0.2, 0.3]
            ),
            "weight_decay": trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True),
            "gating_temperature": trial.suggest_float(
                "gating_temperature", 0.5, 2.0, step=0.25
            ),
        }

        if self.tune_ehr_dropout:
            config["ehr_dropout_rate"] = trial.suggest_float(
                "ehr_dropout_rate", 0.0, 0.5, step=0.1
            )
        else:
            config["ehr_dropout_rate"] = 0.0

        return config

    def _build_model(
        self,
        trial: optuna.trial.Trial,
        config: dict,
        learning_rate: float,
        lambda_weight: float,
    ) -> LateFusionModule:
        """Build a LateFusionModule with randomly initialized components.

        Args:
            trial: Optuna trial object.
            config: Hyperparameter config dict from _suggest_config.
            learning_rate: Learning rate for the optimizer.
            lambda_weight: Weight for the auxiliary unimodal loss term.

        Returns:
            LateFusionModule instance with all components randomly initialized
            and fully trainable (no frozen weights).
        """
        return LateFusionModule.from_scratch(
            input_dims=self.input_dims,
            active_modalities=self.active_modalities,
            learning_rate=learning_rate,
            config=config,
            lambda_weight=lambda_weight,
        )


def run_hydra_tuner(
    active_modalities: list,
    input_dims: dict,
    use_pretrained: bool,
    unimodal_configs: dict,
    datamodule: pl.LightningDataModule,
    epochs: int,
    output_dir: Path,
    study_name: str,
    n_trials: int = 30,
    seed: int = 42,
    tune_ehr_dropout: bool = True,
) -> dict:
    """Run Optuna hyperparameter search for a late-fusion sepsis prediction model.

        Implements tree-structured parzen estimator (TPE) sampling with median pruning
        to efficiently explore the hyperparameter space over n_trials trials.

        Optuna execution policy:
            - This project enforces single-process tuning (n_jobs=1) for reproducibility
                and to avoid thread-safety issues caused by mutable datamodule trial state.

    Args:
        active_modalities: List of modality keys.
        input_dims: Dictionary mapping modality key -> raw embedding dimension.
        use_pretrained: If True, use PretrainedObjective; else use ScratchObjective.
        unimodal_configs: Per-modality architecture dicts (required if use_pretrained=True).
        datamodule: FusionDataModule instance with train/val dataloaders available.
        epochs: Max epochs per trial (with early stopping at patience=10).
        output_dir: Directory to save best_hyperparameters_{study_name}.json.
        study_name: Name for the Optuna study.
        n_trials: Number of trials to run (default 30).
        seed: Random seed for reproducibility.
        tune_ehr_dropout: If True, include ehr_dropout_rate in the search space.

    Returns:
        Dictionary with best hyperparameters:
          - 'config': Fusion-level architecture config
          - 'learning_rate': Optimal learning rate
          - 'lambda_weight': Optimal auxiliary loss weight
          - 'ehr_dropout_rate': Optimal EHR dropout (if tune_ehr_dropout=True)
    """
    # Check for cached results first to avoid redundant tuning runs
    best_params_path = output_dir / f"best_hyperparameters_{study_name}.json"

    if best_params_path.exists():
        logger.info(f"Using cached hyperparameters from: {best_params_path}")
        with open(best_params_path, "r") as f:
            return json.load(f)["params"]

    logger.info(
        f"Optimal parameters not found. Initiating Optuna Tuning for {n_trials} trials..."
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create Optuna study with TPE sampler and median pruner for efficient search
    pruner = optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=15)
    sampler = optuna.samplers.TPESampler(seed=seed)

    study = optuna.create_study(
        direction="maximize", pruner=pruner, sampler=sampler, study_name=study_name
    )

    # Instantiate appropriate objective based on training mode
    if use_pretrained:
        objective = PretrainedObjective(
            active_modalities=active_modalities,
            input_dims=input_dims,
            unimodal_configs=unimodal_configs,
            datamodule=datamodule,
            epochs=epochs,
            tune_ehr_dropout=tune_ehr_dropout,
        )
    else:
        objective = ScratchObjective(
            active_modalities=active_modalities,
            input_dims=input_dims,
            datamodule=datamodule,
            epochs=epochs,
            tune_ehr_dropout=tune_ehr_dropout,
        )

    # Run optimization in single-process mode by design.
    study.optimize(objective, n_trials=n_trials, n_jobs=1)

    # Extract best parameters and restructure
    best_params = study.best_params

    # Separate model config parameters from top-level hyperparameters
    config_block = {
        k: v
        for k, v in best_params.items()
        if k not in ["learning_rate", "lambda_weight", "ehr_dropout_rate"]
    }

    reconstructed_config = {
        "params": {
            "config": config_block,
            "learning_rate": best_params.get("learning_rate", 1e-4),
            "lambda_weight": best_params.get("lambda_weight", 0.4),
            "ehr_dropout_rate": best_params.get("ehr_dropout_rate", 0.0),
        }
    }

    # Cache results to disk for reproducibility
    with open(best_params_path, "w") as f:
        json.dump(reconstructed_config, f, indent=4)

    logger.info(f"Tuning completed! Cached hyperparams to {best_params_path}")
    return reconstructed_config["params"]

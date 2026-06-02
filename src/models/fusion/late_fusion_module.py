"""PyTorch Lightning wrapper for the late-fusion sepsis prediction model.

LateFusionModule is the primary training and inference entry point, imported by
run_fusion.py, incremental_value_analysis.py, and extract_modality_weights.py.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from typing import Dict, Any, List, Tuple, Optional
from torchmetrics.classification import BinaryAveragePrecision
import logging

from src.models.fusion.architecture import LateFusionSepsisModel
from src.models.fusion.loss import composite_sepsis_loss
from src.utils.config import Config

logger = logging.getLogger(__name__)


class LateFusionModule(pl.LightningModule):
    """PyTorch Lightning wrapper for ``LateFusionSepsisModel``.

    Handles dynamic modalities, composite loss with unimodal accountability,
    and post-hoc temperature calibration. Construct via the ``from_pretrained``
    or ``from_scratch`` class methods for explicit mode selection.
    """

    def __init__(
        self,
        input_dims: Dict[str, int],
        active_modalities: List[str],
        learning_rate: float,
        config: Dict[str, Any],
        unimodal_configs: Optional[Dict[str, Any]] = None,
        lambda_weight: float = 0.4,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.active_modalities = active_modalities

        self.model = LateFusionSepsisModel(
            input_dims=input_dims,
            config=config,
            unimodal_configs=unimodal_configs,
            active_modalities=active_modalities,
        )

        self.learning_rate = learning_rate
        self.lambda_weight = lambda_weight
        self.bce_loss = nn.BCELoss()
        self.val_auprc = BinaryAveragePrecision()

        self.temperatures = {mod: 1.0 for mod in active_modalities}
        self.temperatures["add"] = 1.0
        self.temperatures["final"] = 1.0
        self.temperatures["syn"] = 1.0

    # ------------------------------------------------------------------
    # Class-method constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_pretrained(
        cls,
        input_dims: Dict[str, int],
        active_modalities: List[str],
        learning_rate: float,
        config: Dict[str, Any],
        unimodal_configs: Dict[str, Any],
        lambda_weight: float = 0.4,
    ) -> "LateFusionModule":
        """Construct a model and immediately load pretrained unimodal weights."""
        instance = cls(
            input_dims=input_dims,
            active_modalities=active_modalities,
            learning_rate=learning_rate,
            config=config,
            unimodal_configs=unimodal_configs,
            lambda_weight=lambda_weight,
        )
        loaded, failed = instance._load_pretrained_unimodal_weights()
        if failed:
            raise RuntimeError(
                f"Pretrained weights failed to load for modalities: {failed}. "
                "Re-run unimodal training first or switch to scratch config."
            )
        logger.info(
            f"from_pretrained: all weights loaded successfully "
            f"({', '.join(m.upper() for m in loaded)})"
        )
        return instance

    @classmethod
    def from_scratch(
        cls,
        input_dims: Dict[str, int],
        active_modalities: List[str],
        learning_rate: float,
        config: Dict[str, Any],
        lambda_weight: float = 0.4,
    ) -> "LateFusionModule":
        """Construct a model with randomly initialised unimodal MLPs."""
        return cls(
            input_dims=input_dims,
            active_modalities=active_modalities,
            learning_rate=learning_rate,
            config=config,
            unimodal_configs=None,
            lambda_weight=lambda_weight,
        )

    # ------------------------------------------------------------------
    # Pretrained weight loader
    # ------------------------------------------------------------------

    def _load_pretrained_unimodal_weights(self) -> Tuple[List[str], List[str]]:
        """Load and freeze pretrained projector and MLP weights for each active modality.

        Key rewriting: state dict keys prefixed with "projection." are mapped to the
        projector layer; keys prefixed with "network." are stripped and loaded into the
        unimodal MLP's .network sub-module via strict=True. If "projection.weight" is
        absent, an identity projector is initialised instead.

        Returns:
            (loaded, failed) — lists of modality keys that succeeded and failed respectively.
        """
        models_env = Config.DIR_MODELS
        model_paths = {
            "ehr": models_env / "ehr" / "mlp" / "best_ehr_mlp_weights.pt",
            "ecg": models_env / "ecg" / "mlp" / "best_ecg_mlp_weights.pt",
            "cxr_img": models_env / "cxr_img" / "mlp" / "best_cxr_img_mlp_weights.pt",
            "cxr_txt": models_env / "cxr_txt" / "mlp" / "best_cxr_txt_mlp_weights.pt",
        }

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        loaded: List[str] = []
        failed: List[str] = []

        for mod in self.active_modalities:
            path = model_paths.get(mod)
            if not path or not path.exists():
                failed.append(mod)
                logger.warning(f"Pre-trained weights MISSING for {mod} at {path}")
                continue

            state_dict = torch.load(path, map_location=device, weights_only=False)

            if "projection.weight" in state_dict:
                self.model.projectors[mod].load_state_dict(
                    {
                        "weight": state_dict["projection.weight"],
                        "bias": state_dict["projection.bias"],
                    },
                    strict=True,
                )
                logger.info(f"  -> Loaded projection for {mod.upper()}")
            else:
                torch.nn.init.eye_(self.model.projectors[mod].weight)
                torch.nn.init.zeros_(self.model.projectors[mod].bias)
                logger.info(f"  -> Identity projection for {mod.upper()}")

            network_state_dict = {
                k.replace("network.", ""): v
                for k, v in state_dict.items()
                if k.startswith("network.")
            }
            try:
                self.model.unimodal_mlps[mod].network.load_state_dict(
                    network_state_dict, strict=True
                )
                logger.info(f"  -> Loaded MLP weights for {mod.upper()}")
                loaded.append(mod)

                # Freeze pretrained weights
                self.model.projectors[mod].requires_grad_(False)
                self.model.unimodal_mlps[mod].requires_grad_(False)
                logger.info(f"  -> Frozen projector and MLP for {mod.upper()}")
            except Exception as e:
                failed.append(mod)
                logger.error(f"  -> FAILED to load MLP for {mod.upper()}: {e}")

        return loaded, failed

    def forward(
        self,
        embeddings: Dict[str, torch.Tensor],
        masks: Dict[str, torch.Tensor],
    ) -> Dict[str, Any]:
        return self.model(embeddings, masks)

    def _shared_step(
        self,
        batch: Dict[str, Any],
        batch_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """Shared computation for training/validation steps.

        Computes forward pass and composite loss for a batch. Used by both
        training_step and validation_step to avoid duplication.

        Args:
            batch: Dictionary with keys:
              - 'embeddings': Dict[modality -> embedding tensor]
              - 'masks': Dict[modality -> binary mask tensor]
              - 'label': Target sepsis label
            batch_idx: Batch index.

        Returns:
            Tuple of (total_loss, main_loss, aux_loss, outputs)
              - main_loss: Sum of BCE on final fusion prediction
              - aux_loss: Weighted sum of per-modality BCE predictions
              - outputs: Model forward pass output dict
        """
        embeddings = batch["embeddings"]
        masks = batch["masks"]
        targets = batch["label"]

        outputs = self(embeddings, masks)

        total_loss, main_loss, aux_loss = composite_sepsis_loss(
            p_final=outputs["p_final"],
            p_unimodal=outputs["p_unimodal"],
            masks=masks,
            targets=targets,
            lambda_weight=self.lambda_weight,
        )

        return total_loss, main_loss, aux_loss, outputs

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        """Lightning training step.

        Computes loss and logs training metrics.

        Args:
            batch: Batch from the train DataLoader.
            batch_idx: Batch index.

        Returns:
            total_loss tensor.
        """
        total_loss, main_loss, aux_loss, _ = self._shared_step(batch, batch_idx)

        self.log("train/total_loss", total_loss, on_step=False, on_epoch=True)
        self.log("train/main_loss", main_loss, on_step=False, on_epoch=True)
        self.log("train/aux_loss", aux_loss, on_step=False, on_epoch=True)

        return total_loss

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        """Lightning validation step.

        Computes loss and tracks AUPRC metric for early stopping.

        Args:
            batch: Batch from the val DataLoader.
            batch_idx: Batch index.

        Returns:
            total_loss tensor.
        """
        total_loss, _, _, outputs = self._shared_step(batch, batch_idx)

        self.val_auprc.update(outputs["p_final"].squeeze(dim=-1), batch["label"].long())

        self.log(
            "val/total_loss", total_loss, on_step=False, on_epoch=True, prog_bar=True
        )

        return total_loss

    def on_validation_epoch_end(self) -> None:
        """Compute and log AUPRC at the end of each validation epoch."""
        auprc = self.val_auprc.compute()
        self.log("val/auprc", auprc, prog_bar=True)
        self.val_auprc.reset()

    def predict_step(self, batch: Dict[str, Any], batch_idx: int) -> Dict[str, Any]:
        """Lightning predict step for generating calibrated predictions.

        Runs forward pass and applies temperature scaling to the final fused logit
        only. Returned logits are uncalibrated; calibration is applied to the
        final probability via self.temperatures["final"].

        Args:
            batch: Batch from the test DataLoader.
            batch_idx: Batch index.

        Returns:
            Dictionary with keys:
              - 'p_calibrated': Calibrated probabilities
              - 'logits': Dict of calibrated logits (per modality + add + final)
              - 'targets': Ground truth labels
              - 'subject_ids': MIMIC-IV subject IDs for linking predictions to patients
        """
        embeddings = batch["embeddings"]
        masks = batch["masks"]
        targets = batch["label"].squeeze(dim=-1)
        outputs = self(embeddings, masks)
        p_final = outputs["p_final"]
        eps = 1e-7
        p_clipped = torch.clamp(p_final, eps, 1 - eps)
        logit_final = torch.log(p_clipped / (1 - p_clipped))
        p_calibrated = torch.sigmoid(logit_final / self.temperatures["final"]).squeeze(
            dim=-1
        )

        batch_logits = {k: v.squeeze(dim=-1) for k, v in outputs["logits"].items()}
        batch_logits["add"] = torch.logit(outputs["p_add"], eps=eps).squeeze(dim=-1)
        batch_logits["final"] = logit_final.squeeze(dim=-1)

        return {
            "p_calibrated": p_calibrated,
            "logits": batch_logits,
            "targets": targets,
            "subject_ids": batch.get("subject_id"),
        }

    def configure_optimizers(self) -> Dict[str, Any]:
        weight_decay = self.hparams.config.get("weight_decay", 1e-4)

        # Filter parameters: apply weight decay only to weights > 1D
        decay_params = []
        no_decay_params = []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            # ndim < 2 covers: biases (1D), scalars like gating_temperature (0D)
            if param.ndim < 2 or "bias" in name or "LayerNorm" in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]

        optimizer = optim.AdamW(optim_groups, lr=self.learning_rate)

        # Monitor 'val/total_loss' since that's what the validation_step logs.
        # Note: patience=5 here paired with EarlyStopping patience=10 creates an
        # intentional "ladder". The LR will decay twice before the run is stopped,
        # allowing the optimizer to settle into narrower local minima.
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val/total_loss",
                "frequency": 1,
            },
        }

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """Persist calibration temperatures alongside model weights."""
        checkpoint["temperatures"] = {k: float(v) for k, v in self.temperatures.items()}

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """Restore calibration temperatures from checkpoint when available."""
        saved_temps = checkpoint.get("temperatures")
        if isinstance(saved_temps, dict):
            self.temperatures = {k: float(v) for k, v in saved_temps.items()}

"""Unimodal MLP Classifier for Binary Sepsis Prediction.

This module contains:
- UnimodalMLP: A PyTorch neural network that projects embeddings to 768-dim, applies
  LayerNorm + Dense layers with dropout, and outputs raw logits (not probabilities).
- UnimodalModule: A PyTorch Lightning wrapper providing training, validation, and
  inference with external temperature scaling support.

Key design decisions:
- Raw logits are output to maintain numerical stability via BCEWithLogitsLoss
- Temperature scaling (T) is applied externally and injected post-training
- LayerNorm is applied after projection and after each dense layer (improves stability)
- Weight decay is NOT applied to LayerNorm or bias parameters
"""

import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from typing import Dict, Any
from torchmetrics.classification import BinaryAveragePrecision


class UnimodalMLP(nn.Module):
    """Unimodal MLP for binary classification from normalized embeddings.

    Architecture:
    1. Projection layer (if input_dim != 768) to standardize to 768 dimensions
    2. LayerNorm + Dense(768 -> hidden_dim_1) + Activation + Dropout
    3. LayerNorm + Dense(hidden_dim_1 -> hidden_dim_2) + Activation + Dropout
    4. Dense(hidden_dim_2 -> 1) [outputs raw logits]

    All dense layers are preceded by LayerNorm for improved training stability.
    Activation can be ReLU or GELU (determined by config).

    Args:
        input_dim: Embedding dimension (e.g., 768 for MOTOR/BERT).
        config: Dict with keys:
            - hidden_dim_1: Size of first hidden layer (e.g., 256)
            - hidden_dim_2: Size of second hidden layer (e.g., 64)
            - dropout_rate: Dropout probability (e.g., 0.2)
            - activation: 'ReLU' or 'GELU'
    """

    def __init__(self, input_dim: int, config: dict):
        super().__init__()

        hidden_1 = config.get("hidden_dim_1", 256)
        hidden_2 = config.get("hidden_dim_2", 64)
        dropout_rate = config.get("dropout_rate", 0.0)
        activation_name = config.get("activation", "GELU")

        # Align input to 768 if necessary
        if input_dim != 768:
            self.projection = nn.Linear(input_dim, 768)
        else:
            self.projection = nn.Identity()

        current_dim = 768

        def _make_activation(name):
            """Create activation layer by name."""
            return nn.ReLU() if name == "ReLU" else nn.GELU()

        # Build sequential layer stack with LayerNorm for stability
        layers = []

        # Input: [N, 768] after projection
        layers.append(nn.LayerNorm(current_dim))
        layers.append(nn.Linear(current_dim, hidden_1))
        layers.append(nn.LayerNorm(hidden_1))
        layers.append(_make_activation(activation_name))
        layers.append(nn.Dropout(dropout_rate))

        # Hidden: [N, hidden_1]
        layers.append(nn.Linear(hidden_1, hidden_2))
        layers.append(nn.LayerNorm(hidden_2))
        layers.append(_make_activation(activation_name))
        layers.append(nn.Dropout(dropout_rate))

        # Output: [N, hidden_2] -> [N, 1] (raw logits)
        layers.append(nn.Linear(hidden_2, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """Forward pass through the MLP.

        Args:
            x: Tensor of shape [batch_size, input_dim].

        Returns:
            Tensor of shape [batch_size, 1] containing raw logits (NOT probabilities).
            Apply torch.sigmoid() externally to obtain probabilities if needed.
        """
        x = self.projection(x)
        return self.network(x)


class UnimodalModule(pl.LightningModule):
    """PyTorch Lightning wrapper for UnimodalMLP with AUPRC monitoring and calibration support.

    Key design:
    - Outputs raw logits during training for numerical stability (BCEWithLogitsLoss)
    - Temperature scaling is applied externally (not part of this module) and injected
      as self.temperature before prediction/inference
    - Monitors val/auprc as the primary validation metric (clinically meaningful)
    - Uses standard AdamW with per-parameter group weight decay (no decay on LayerNorm/bias)

    Attributes:
        model: UnimodalMLP instance.
        learning_rate: Learning rate for AdamW optimizer.
        weight_decay: L2 regularization strength.
        criterion: Loss function (BCEWithLogitsLoss).
        val_auprc: Metric tracker for validation AUPRC (BinaryAveragePrecision).
        temperature: Temperature scaling parameter (default 1.0, updated by LBFGSCalibrator).
    """

    def __init__(
        self,
        input_dim: int,
        learning_rate: float,
        config: Dict[str, Any],
        weight_decay: float = 1e-4,
    ):
        """Initialize the Lightning module.

        Args:
            input_dim: Embedding dimension.
            learning_rate: AdamW learning rate.
            config: Architecture config dict (passed to UnimodalMLP).
            weight_decay: L2 regularization strength for decayable parameters.
        """
        super().__init__()
        self.save_hyperparameters()

        self.model = UnimodalMLP(input_dim, config)
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.criterion = nn.BCEWithLogitsLoss()
        self.val_auprc = BinaryAveragePrecision()

        # Default temperature (no scaling). Will be overwritten by LBFGSCalibrator
        self.temperature = 1.0

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        """Single training step.

        Args:
            batch: Tuple of (inputs, targets, subject_ids).
            batch_idx: Batch index (unused).

        Returns:
            Loss value for this batch.
        """
        inputs, targets, _ = batch
        logits = self(inputs).squeeze(dim=-1)
        loss = self.criterion(logits, targets.float())

        self.log("train/loss", loss, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """Single validation step.

        Args:
            batch: Tuple of (inputs, targets, subject_ids).
            batch_idx: Batch index (unused).

        Returns:
            Loss value for this batch.
        """
        inputs, targets, _ = batch
        logits = self(inputs).squeeze(dim=-1)
        loss = self.criterion(logits, targets.float())

        # Compute probabilities for AUPRC metric (requires sigmoid of logits)
        probs = torch.sigmoid(logits)
        self.val_auprc.update(probs, targets.long())

        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def on_validation_epoch_end(self):
        """Compute and log AUPRC from accumulated validation predictions."""
        auprc = self.val_auprc.compute()
        self.log("val/auprc", auprc, prog_bar=True)
        self.val_auprc.reset()

    def predict_step(self, batch, batch_idx):
        """Generate predictions with temperature scaling applied.

        Args:
            batch: Tuple of (inputs, targets, subject_ids).
            batch_idx: Batch index (unused).

        Returns:
            Dict with keys:
                - 'logits': Raw logits from the model
                - 'p_calibrated': Temperature-scaled probabilities
                - 'targets': Ground truth labels
        """
        inputs, targets, _ = batch
        logits = self(inputs).squeeze(dim=-1)

        # Apply temperature scaling formula: p_cal = sigmoid(logits / T)
        p_calibrated = torch.sigmoid(logits / self.temperature)

        # Return a dictionary for flexible access to intermediate values
        return {"logits": logits, "p_calibrated": p_calibrated, "targets": targets}

    def configure_optimizers(self):
        """Configure AdamW optimizer with per-parameter group weight decay.

        Weight decay is applied only to multi-dimensional parameters (weights).
        Bias terms, LayerNorm parameters, and 1D tensors are not regularized.

        Returns:
            Dict with optimizer and learning rate scheduler configuration.
        """
        # Split parameters: weight decay should NOT be applied to LayerNorm parameters,
        # biases, or scalar/1D tensors.
        decay_params = []
        no_decay_params = []
        for name, param in self.model.named_parameters():
            if param.ndim < 2 or "bias" in name or "LayerNorm" in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        optim_groups = [
            {"params": decay_params, "weight_decay": self.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]

        optimizer = optim.AdamW(optim_groups, lr=self.learning_rate)

        # ReduceLROnPlateau: reduce LR if validation loss plateaus
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val/loss",
                "frequency": 1,
            },
        }

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """Persist calibration temperature alongside model weights."""
        checkpoint["temperature"] = float(self.temperature)

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """Restore calibration temperature from checkpoint when available."""
        if "temperature" in checkpoint:
            self.temperature = float(checkpoint["temperature"])

"""Temperature Scaling Calibration for Binary Classification.

Implements temperature scaling to calibrate model probability
estimates. This post-hoc technique is applied after training to improve the
alignment between predicted probabilities and true likelihoods.

Formula: p_calibrated = sigmoid(logits / T), where T > 0 is optimized on validation set.

Key components:
- TemperatureScaler: Holds and applies temperature parameters using softplus for stability
- LBFGSCalibrator: Optimizes temperatures using LBFGS with BCE loss

Usage:
    1. Extract validation logits from trained model
    2. Instantiate LBFGSCalibrator()
    3. Call calibrator.fit(logits_dict, targets, device) to get optimal T values
    4. Store optimal T and apply at inference: p_cal = sigmoid(logit / T)
"""

import logging
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.optim as optim

logger = logging.getLogger(__name__)


class TemperatureScaler(nn.Module):
    """Holds and applies temperature parameters for one or multiple prediction branches.

    Each branch has its own temperature parameter T trained via LBFGS.

    Design:
    - Temperatures are stored as free parameters (no constraints)
    - softplus(T) + epsilon ensures T > 0.001 for numerical stability
    - Forward pass applies temperature scaling to logits

    Args:
        keys: List of branch names
    """

    def __init__(self, keys: list[str]):
        super().__init__()
        # Initialize temperature to 1.5 (slight over-confidence correction as starting point)
        self.temperatures = nn.ParameterDict(
            {key: nn.Parameter(torch.ones(1) * 1.5) for key in keys}
        )

    def _effective_temp(self, key: str) -> torch.Tensor:
        """Compute effective temperature with softplus constraint.

        Args:
            key: Branch/modality name.

        Returns:
            Effective temperature T = softplus(param) + 1e-3, ensuring T > 1e-3.
        """
        return F.softplus(self.temperatures[key]) + 1e-3

    def forward(
        self, logits_dict: dict[str, torch.Tensor], return_logits: bool = False
    ) -> dict[str, torch.Tensor]:
        """Apply temperature scaling to logits from multiple branches.

        Temperature scaling formula: logit_scaled = logit / T

        Args:
            logits_dict: Dict mapping branch names to logit tensors [batch_size].
            return_logits: If True, return temperature-scaled logits (for stable BCE training).
                          If False, return probabilities (for inference).

        Returns:
            Dict with same keys as logits_dict:
                - If return_logits=True: scaled logits [batch_size]
                - If return_logits=False: probabilities in [0, 1] [batch_size]
        """
        outputs = {}
        for key, logit in logits_dict.items():
            t = self._effective_temp(key)
            scaled_logit = logit / t

            if return_logits:
                outputs[key] = scaled_logit
            else:
                outputs[key] = torch.sigmoid(scaled_logit)
        return outputs


class LBFGSCalibrator:
    """Optimizes Temperature Scaling using the LBFGS algorithm.

    LBFGS is chosen for calibration because:
    - It's a 2nd-order quasi-Newton optimizer (better than 1st-order SGD for small problems)
    - Temperatures are typically few parameters
    - No line search tuning needed; automatically finds step sizes

    Process:
    1. Initialize TemperatureScaler with given branch names
    2. Feed validation logits and targets to the scaler
    3. Minimize BCE loss with respect to temperature parameters
    4. Return optimal temps

    Args:
        max_iter: Max LBFGS iterations (default 50).
        lr: Effective learning rate for LBFGS (default 0.01).
    """

    def __init__(self, max_iter: int = 50, lr: float = 0.01):
        self.max_iter = max_iter
        self.lr = lr
        self.bce_logits_loss = nn.BCEWithLogitsLoss()

    def fit(
        self,
        logits_dict: dict[str, torch.Tensor],
        targets: torch.Tensor,
        device: torch.device,
    ) -> dict[str, float]:
        """Optimize temperature scaling on validation set logits and targets.

        Args:
            logits_dict: Dict mapping branch names to raw logit tensors from model.
            targets: Ground truth binary labels [0, 1], shape [N].
            device: Torch device to use ("cuda" or "cpu").

        Returns:
            Dict mapping branch names to optimal temperature values.
        """
        logger.info("Starting LBFGS optimization for Temperature Scaling...")

        keys = list(logits_dict.keys())
        scaler = TemperatureScaler(keys).to(device)

        # Move all tensors to device
        logits_dict = {k: v.to(device) for k, v in logits_dict.items()}
        targets = targets.to(device).float()

        # LBFGS optimizer for temperature parameters
        optimizer = optim.LBFGS(scaler.parameters(), lr=self.lr, max_iter=self.max_iter)

        # Define closure for LBFGS (required for full batch optimization)
        def eval_closure():
            optimizer.zero_grad()
            # Use raw scaled logits for numerical stability
            scaled_logits = scaler(logits_dict, return_logits=True)

            # Sum BCE loss across all branches
            loss = 0.0
            for key in scaled_logits.keys():
                loss = loss + self.bce_logits_loss(scaled_logits[key], targets)

            loss.backward()
            return loss

        # Run LBFGS optimization
        optimizer.step(eval_closure)

        # Extract and log optimal temperatures
        optimal_temps = {}
        with torch.no_grad():
            for key, _ in scaler.temperatures.items():
                # Apply the exact same transformation used in forward()
                effective_temp = scaler._effective_temp(key).item()
                optimal_temps[key] = effective_temp
                logger.info(f"Optimal T_{key}: {effective_temp:.4f}")

        return optimal_temps

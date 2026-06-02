"""Composite loss for the late-fusion sepsis model.

composite_sepsis_loss combines a main BCE loss on the fused prediction with auxiliary
per-modality BCE losses, enforcing that each active modality maintains standalone
predictive accuracy during joint training (unimodal accountability).
"""

import torch
import torch.nn.functional as F
from typing import Dict, Tuple


def composite_sepsis_loss(
    p_final: torch.Tensor,
    p_unimodal: Dict[str, torch.Tensor],
    masks: Dict[str, torch.Tensor],
    targets: torch.Tensor,
    lambda_weight: float = 0.4,
    clamp_eps: float = 1e-7,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Computes the composite loss for the late-fusion sepsis prediction network.

    Total Loss = BCE(p_final, y) + λ × (Sum of unimodal BCE losses over active modalities)

    The auxiliary loss enforces strong "unimodal accountability": it prevents a
    dominant modality (like EHR) from taking over the learning process. By requiring
    each active modality to maintain an accurate standalone prediction, it ensures
    weaker modalities do not become "lazy" or ignored during joint training.

    Design rationale:
    - Auxiliary losses are SUMMED (not averaged) over active modalities per patient.
    - Because EHR is present for 100% of patients while ECG/CXR are minority
      modalities, this naturally up-weights the auxiliary gradient for patients
      who have richer multimodal data — exactly the patients that drive learning
      of good gating and synergy behaviour.

    Args:
        p_final:       Final fusion probabilities, shape (B, 1)
        p_unimodal:    Dictionary of unimodal probabilities {mod: tensor}, each (B, 1)
        masks:         Dictionary of binary masks {mod: tensor}, each (B, 1)
        targets:       Ground truth labels, shape (B,)
        lambda_weight: Weight for the auxiliary unimodal loss (tuned by Optuna)
        clamp_eps:     Small constant to avoid log(0) when probabilities saturate

    Returns:
        total_loss, main_loss, aux_loss
    """
    targets_float = targets.float()

    # 1. Main loss on the final fused prediction
    p_final_clamped = torch.clamp(p_final.squeeze(dim=-1), clamp_eps, 1.0 - clamp_eps)
    main_loss = F.binary_cross_entropy(p_final_clamped, targets_float)

    # 2. Auxiliary loss: sum over active modalities only
    mod_losses = []
    mod_masks = []

    for mod, p_uni in p_unimodal.items():
        p_uni_clamped = torch.clamp(p_uni.squeeze(dim=-1), clamp_eps, 1.0 - clamp_eps)
        loss = F.binary_cross_entropy(
            p_uni_clamped, targets_float, reduction="none"
        )  # shape: (B,)
        mask = masks[mod].float().squeeze(dim=-1)  # shape: (B,)
        mod_losses.append(loss)
        mod_masks.append(mask)

    if mod_losses:
        loss_matrix = torch.stack(mod_losses, dim=1)  # (B, M)
        mask_matrix = torch.stack(mod_masks, dim=1)  # (B, M)

        # Sum masked losses per patient, then average over the batch
        per_patient_aux = (loss_matrix * mask_matrix).sum(dim=1)  # (B,)
        aux_loss = per_patient_aux.mean()
    else:
        aux_loss = torch.tensor(0.0, device=p_final.device)

    # 3. Total composite loss
    total_loss = main_loss + (lambda_weight * aux_loss)

    return total_loss, main_loss, aux_loss

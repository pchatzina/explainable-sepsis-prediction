"""Core fusion architecture for multimodal sepsis prediction.

Exports three names consumed by late_fusion_module.py:
  - LateFusionSepsisModel: gated additive + synergy late-fusion model
  - MLPBlock: shared MLP building block for gating, synergy, and scratch-mode unimodal heads
  - build_unimodal_mlps: factory constructing per-modality MLPs in pretrained or scratch mode

In pretrained mode (unimodal_configs provided), each unimodal MLP mirrors the architecture
used during standalone training to guarantee state_dict compatibility. In scratch mode, all
MLPs are initialised from the fusion-level config.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional

from src.models.unimodal.mlp import UnimodalMLP


class MLPBlock(nn.Module):
    """Standard MLP block used by the gating network, synergy head, and
    scratch-mode unimodal heads.

    Architecture (10 layers, indices 0-9):
        LayerNorm -> Linear -> LayerNorm -> GELU -> Dropout ->
        Linear -> LayerNorm -> GELU -> Dropout -> Linear
    """

    def __init__(
        self,
        input_dim: int,
        hidden_1: int,
        hidden_2: int,
        output_dim: int,
        dropout_rate: float = 0.0,
    ):
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_1),
            nn.LayerNorm(hidden_1),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_1, hidden_2),
            nn.LayerNorm(hidden_2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_2, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the sequential MLP transformation.

        Args:
            x: Input tensor of shape (batch_size, input_dim).

        Returns:
            Output tensor of shape (batch_size, output_dim).
        """
        return self.network(x)


def build_unimodal_mlps(
    modalities: list,
    common_dim: int,
    unimodal_configs: Optional[Dict[str, dict]] = None,
    config: Optional[dict] = None,
) -> nn.ModuleDict:
    """Factory that constructs one MLP per modality.

    In **pretrained mode** (``unimodal_configs`` provided), each MLP is a
    ``UnimodalMLP`` whose architecture exactly mirrors what was used during
    unimodal training. This guarantees ``state_dict`` compatibility when
    loading pretrained weights via ``strict=True``.

    In **scratch mode**, each MLP is an ``MLPBlock`` parameterised by the
    fusion-level config (``uni_hidden_1``, ``uni_hidden_2``, ``dropout_rate``).

    Args:
        modalities: Active modality keys.
        common_dim: Input dimension for each MLP (shared projection dim).
        unimodal_configs: Per-modality architecture dicts from saved
            ``best_hyperparameters.json`` (pretrained mode).
        config: Fusion-level config dict (scratch mode).

    Returns:
        ``nn.ModuleDict`` mapping modality key to an ``nn.Module`` with a
        ``.network`` attribute.
    """
    mlps = nn.ModuleDict()
    for mod in modalities:
        if unimodal_configs is not None and mod in unimodal_configs:
            mod_cfg = unimodal_configs[mod]
            mlps[mod] = UnimodalMLP(input_dim=common_dim, config=mod_cfg)
        else:
            cfg = config or {}
            mlps[mod] = MLPBlock(
                input_dim=common_dim,
                hidden_1=cfg.get("uni_hidden_1", 256),
                hidden_2=cfg.get("uni_hidden_2", 128),
                output_dim=1,
                dropout_rate=cfg.get("dropout_rate", 0.1),
            )
    return mlps


class LateFusionSepsisModel(nn.Module):
    """Late-fusion sepsis prediction model with gated additive + synergy head.

    Args:
        input_dims: Raw embedding dimensions per modality.
        config: Hyperparameters for gating, synergy, and (scratch-mode) unimodal MLPs.
        unimodal_configs: Per-modality architecture dicts for pretrained mode.
            ``None`` triggers scratch-mode construction via *config*.
        common_dim: Shared projection dimension.
        active_modalities: Ordered list of modality keys to use.
    """

    def __init__(
        self,
        input_dims: dict,
        config: dict,
        unimodal_configs: dict = None,
        common_dim: int = 768,
        active_modalities: list = None,
    ):
        super().__init__()
        self.common_dim = common_dim
        self.modalities = (
            active_modalities if active_modalities else list(input_dims.keys())
        )
        num_mods = len(self.modalities)

        # 1. Linear Projectors
        self.projectors = nn.ModuleDict(
            {mod: nn.Linear(input_dims[mod], common_dim) for mod in self.modalities}
        )

        # 2. Unimodal MLPs via factory
        self.unimodal_mlps = build_unimodal_mlps(
            modalities=self.modalities,
            common_dim=common_dim,
            unimodal_configs=unimodal_configs,
            config=config,
        )

        # 3. Gating Network
        gate_input_dim = (common_dim * num_mods) + num_mods
        self.gating_network = MLPBlock(
            input_dim=gate_input_dim,
            hidden_1=config.get("gate_hidden_1", 512),
            hidden_2=config.get("gate_hidden_2", 128),
            output_dim=num_mods,
            dropout_rate=config.get("dropout_rate", 0.1),
        )

        self.gating_temperature = nn.Parameter(
            torch.tensor(config.get("gating_temperature", 1.0))
        )

        # 4. Synergy Head
        syn_input_dim = (common_dim * num_mods) + num_mods
        self.synergy_head = MLPBlock(
            input_dim=syn_input_dim,
            hidden_1=config.get("syn_hidden_1", 512),
            hidden_2=config.get("syn_hidden_2", 128),
            output_dim=1,
            dropout_rate=config.get("dropout_rate", 0.1),
        )

    def forward(self, embeddings: dict, masks: dict) -> dict:
        """Late-fusion forward pass with gated additive and synergy components.

        Mathematical Flow (Steps A-F):
          A. Project: z_i = Linear(embedding_i)
          B. Mask: z_i *= M_i (binary mask)
          C. Unimodal: p_i = sigmoid(MLP(z_i))
          D. Gating: w = softmax(MLP_gate([z_1,...,z_4, M_1,...,M_4]) / T_gate)
          E. Fusion:
            - Additive: p_add = sum(w_i * p_i)
            - Synergy: p_syn = sigmoid(MLP_syn([z_1,...,z_4, M_1,...,M_4]))
          F. Final: p_final = (1 - beta) * p_add + beta * p_syn
            where beta = 1 - max(w_i)

        When a modality is missing (M_i=0), it's masked out via:
          1. Pre-projection masking (z_i *= M_i)
          2. Large negative logit in gating softmax (effectively w_i -> 0)
          3. If only one modality is active, beta -> 0 -> p_final -> p_add

        Args:
            embeddings: Dict[modality -> L2-normalized embedding tensor]
            masks: Dict[modality -> binary mask {0, 1}]

        Returns:
            Dictionary with keys:
              - 'p_final': Final fusion probability
              - 'p_add': Additive component probability
              - 'p_unimodal': Dict of per-modality probabilities
              - 'logits': Dict of per-modality logits
              - 'weights': Gating weights w
              - 'beta': Interpolation coefficient
        """
        batch_size = embeddings["ehr"].size(0)
        device = embeddings["ehr"].device

        z_dict = {}
        p_dict = {}
        logit_dict = {}

        # --- Projection, Masking, and Unimodal Prediction ---
        for mod in self.modalities:
            # Mask out missing modalities before projection
            mask_tensor = masks[mod]  # Shape: [B, 1]
            raw_emb = embeddings[mod] * mask_tensor

            # Linear projection: z_i = Linear(z_raw_i)
            z_i = self.projectors[mod](raw_emb)
            # Re-apply mask to ensure missing modalities remain exactly zero vectors
            z_i = z_i * mask_tensor
            z_dict[mod] = z_i

            # Unimodal prediction: p_i = sigmoid(MLP(z_i))
            logit_i = self.unimodal_mlps[mod](z_i)
            logit_dict[mod] = logit_i
            p_dict[mod] = torch.sigmoid(logit_i)

        # --- Gating Network Mechanism ---
        # Concatenate [z_ehr, z_ecg, z_img, z_txt, M_ehr, M_ecg, M_img, M_txt]
        z_list = [z_dict[mod] for mod in self.modalities]
        m_list = [masks[mod] for mod in self.modalities]

        v_gate = torch.cat(z_list + m_list, dim=1)
        logits_gate = self.gating_network(v_gate)  # Shape: [B, 4]

        effective_temp = F.softplus(self.gating_temperature) + 1e-3
        logits_gate = logits_gate / effective_temp

        # Masked Softmax: Apply large negative value where mask == 0 so exp(logits) becomes 0
        mask_concat = torch.cat(m_list, dim=1)  # Shape: [B, 4]
        masked_logits_gate = logits_gate.masked_fill(mask_concat == 0, -1e9)
        w = F.softmax(masked_logits_gate, dim=-1)  # Shape: [B, 4]

        # --- Additive Head ---
        p_add = torch.zeros((batch_size, 1), device=device)
        for i, mod in enumerate(self.modalities):
            # w[:, i:i+1] extracts the specific weight column keeping shape [B, 1]
            p_add += w[:, i : i + 1] * p_dict[mod]

        # --- Synergy Head ---
        # Synergy head sees all modality embeddings and masks, learns their joint interaction
        # This allows the model to discover modality-specific synergies beyond simple weighted averaging
        z_joint = torch.cat(z_list + m_list, dim=1)
        logit_syn = self.synergy_head(z_joint)
        p_syn = torch.sigmoid(logit_syn)

        # --- Final Prediction ---
        # Calculate beta = 1 - max(w_i)
        # When EHR dominates (max weight near 1.0), beta -> 0, so p_final -> p_add
        # When weights are balanced (max weight near 0.5), beta -> 0.5, allowing synergy component
        max_w, _ = torch.max(w, dim=1, keepdim=True)
        beta = 1.0 - max_w

        p_final = (1 - beta) * p_add + beta * p_syn

        return {
            "p_final": p_final,
            "p_add": p_add,
            "p_unimodal": p_dict,
            "weights": w,
            "beta": beta,
            "logits": {**logit_dict, "syn": logit_syn},
        }

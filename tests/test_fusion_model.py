"""
Fusion Model Unit Tests
Run:
    pytest tests/test_fusion_model.py -v
"""

import torch
import pytest

from src.models.fusion.architecture import (
    LateFusionSepsisModel,
    build_unimodal_mlps,
    MLPBlock,
)
from src.models.unimodal.mlp import UnimodalMLP
from src.models.fusion.late_fusion_module import LateFusionModule


@pytest.fixture
def dummy_config():
    return {
        "gate_hidden_1": 256,
        "gate_hidden_2": 64,
        "syn_hidden_1": 256,
        "syn_hidden_2": 64,
        "dropout_rate": 0.0,
        "gating_temperature": 1.0,
        "activation": "GELU",
    }


@pytest.fixture
def dummy_input_dims():
    return {
        "ehr": 1024,
        "ecg": 768,
        "cxr_img": 1024,
        "cxr_txt": 768,
    }


@pytest.fixture
def dummy_batch():
    """Small batch matching FusionDataModule dict structure."""
    batch_size = 8
    embeddings = {
        "ehr": torch.randn(batch_size, 1024),
        "ecg": torch.randn(batch_size, 768),
        "cxr_img": torch.randn(batch_size, 1024),
        "cxr_txt": torch.randn(batch_size, 768),
    }
    masks = {mod: torch.ones(batch_size, 1) for mod in embeddings}

    return {
        "embeddings": embeddings,
        "masks": masks,
        "label": torch.randint(0, 2, (batch_size,)).float(),
        "subject_id": torch.arange(batch_size),
    }


def test_late_fusion_forward_shapes_and_masking(
    dummy_config, dummy_input_dims, dummy_batch
):
    """Core test: shapes, masking, and gating behavior."""
    embeddings = dummy_batch["embeddings"]
    masks = dummy_batch["masks"]

    model = LateFusionSepsisModel(
        input_dims=dummy_input_dims,
        config=dummy_config,
        active_modalities=list(dummy_input_dims.keys()),
    )
    model.eval()

    # All modalities present
    out = model(embeddings, masks)
    assert out["p_final"].shape == (8, 1)
    assert out["weights"].shape == (8, 4)
    assert torch.allclose(out["weights"].sum(dim=1), torch.ones(8), atol=1e-6)

    # Mask out one modality (ECG)
    masks["ecg"] = torch.zeros(8, 1)
    out_masked = model(embeddings, masks)

    assert (out_masked["weights"][:, 1] < 1e-5).all()  # ECG weight must be ~0
    assert torch.allclose(out_masked["weights"].sum(dim=1), torch.ones(8), atol=1e-5)
    assert 0.0 <= out_masked["beta"].min() <= out_masked["beta"].max() <= 1.0


def test_build_unimodal_mlps_pretrained_vs_scratch(dummy_config, dummy_input_dims):
    """Ensures pretrained mode uses real UnimodalMLP and scratch mode uses MLPBlock."""
    unimodal_cfgs = {
        "ehr": {
            "hidden_dim_1": 256,
            "hidden_dim_2": 64,
            "dropout_rate": 0.0,
            "activation": "ReLU",
        },
        "ecg": {
            "hidden_dim_1": 128,
            "hidden_dim_2": 32,
            "dropout_rate": 0.1,
            "activation": "GELU",
        },
    }

    # Pretrained mode
    mlps_pretrained = build_unimodal_mlps(
        modalities=["ehr", "ecg"],
        common_dim=768,
        unimodal_configs=unimodal_cfgs,
    )
    assert isinstance(mlps_pretrained["ehr"], UnimodalMLP)
    assert isinstance(mlps_pretrained["ecg"], UnimodalMLP)

    # Scratch mode
    mlps_scratch = build_unimodal_mlps(
        modalities=["ehr", "ecg"],
        common_dim=768,
        config=dummy_config,
    )
    assert isinstance(mlps_scratch["ehr"], MLPBlock)
    assert isinstance(mlps_scratch["ecg"], MLPBlock)


def test_fusion_predict_step(dummy_config, dummy_input_dims, dummy_batch):
    """Verifies predict_step returns the exact dict structure used in run_fusion.py."""
    lightning_model = LateFusionModule.from_scratch(
        input_dims=dummy_input_dims,
        active_modalities=list(dummy_input_dims.keys()),
        learning_rate=1e-3,
        config=dummy_config,
        lambda_weight=0.5,
    )
    lightning_model.eval()

    # Pass the dictionary directly
    output = lightning_model.predict_step(dummy_batch, batch_idx=0)

    required_keys = {"p_calibrated", "logits", "targets", "subject_ids"}
    assert set(output.keys()) >= required_keys

    assert output["p_calibrated"].shape == (8,)


def test_gating_zeros_missing_modalities(dummy_config, dummy_input_dims, dummy_batch):
    """When a modality is completely masked, its gating weight must be exactly zero."""
    embeddings = dummy_batch["embeddings"]
    masks = dummy_batch["masks"]

    model = LateFusionSepsisModel(
        input_dims=dummy_input_dims,
        config=dummy_config,
        active_modalities=list(dummy_input_dims.keys()),
    )
    model.eval()

    # Mask out two modalities
    masks["ecg"] = torch.zeros(8, 1)
    masks["cxr_txt"] = torch.zeros(8, 1)

    out = model(embeddings, masks)
    weights = out["weights"]

    assert (weights[:, 1] < 1e-6).all()  # ECG
    assert (weights[:, 3] < 1e-6).all()  # CXR_TXT
    # Remaining weights should still sum to 1
    assert torch.allclose(weights[:, [0, 2]].sum(dim=1), torch.ones(8), atol=1e-5)


def test_beta_collapse_when_only_ehr_active(
    dummy_config, dummy_input_dims, dummy_batch
):
    """When only EHR is active, beta should collapse to ~0 and p_final ~= p_add."""
    embeddings = dummy_batch["embeddings"]
    masks = {mod: torch.zeros(8, 1) for mod in dummy_batch["masks"]}
    masks["ehr"] = torch.ones(8, 1)

    model = LateFusionSepsisModel(
        input_dims=dummy_input_dims,
        config=dummy_config,
        active_modalities=list(dummy_input_dims.keys()),
    )
    model.eval()

    out = model(embeddings, masks)
    assert torch.all(out["beta"] < 1e-6)
    assert torch.allclose(out["p_final"], out["p_add"], atol=1e-6)


def test_all_masked_modalities_are_finite(dummy_config, dummy_input_dims, dummy_batch):
    """Extreme mask edge case should remain numerically stable (no NaN/Inf)."""
    embeddings = dummy_batch["embeddings"]
    masks = {mod: torch.zeros(8, 1) for mod in dummy_batch["masks"]}

    model = LateFusionSepsisModel(
        input_dims=dummy_input_dims,
        config=dummy_config,
        active_modalities=list(dummy_input_dims.keys()),
    )
    model.eval()

    out = model(embeddings, masks)

    assert torch.isfinite(out["p_final"]).all()
    assert torch.isfinite(out["p_add"]).all()
    assert torch.isfinite(out["beta"]).all()
    assert torch.isfinite(out["weights"]).all()
    assert torch.allclose(out["weights"].sum(dim=1), torch.ones(8), atol=1e-6)


def test_temperatures_checkpoint_roundtrip(dummy_config, dummy_input_dims):
    """Calibration temperatures should persist via checkpoint hooks."""
    model = LateFusionModule.from_scratch(
        input_dims=dummy_input_dims,
        active_modalities=list(dummy_input_dims.keys()),
        learning_rate=1e-3,
        config=dummy_config,
        lambda_weight=0.5,
    )
    model.temperatures["final"] = 1.73
    model.temperatures["add"] = 1.11

    checkpoint = {}
    model.on_save_checkpoint(checkpoint)

    restored = LateFusionModule.from_scratch(
        input_dims=dummy_input_dims,
        active_modalities=list(dummy_input_dims.keys()),
        learning_rate=1e-3,
        config=dummy_config,
        lambda_weight=0.5,
    )
    restored.on_load_checkpoint(checkpoint)

    assert restored.temperatures["final"] == pytest.approx(1.73)
    assert restored.temperatures["add"] == pytest.approx(1.11)


def test_temperatures_checkpoint_persisted_to_disk(
    dummy_config, dummy_input_dims, tmp_path
):
    """Temperatures should survive a checkpoint save/load cycle on disk."""
    model = LateFusionModule.from_scratch(
        input_dims=dummy_input_dims,
        active_modalities=list(dummy_input_dims.keys()),
        learning_rate=1e-3,
        config=dummy_config,
        lambda_weight=0.5,
    )
    model.temperatures["final"] = 2.01
    model.temperatures["add"] = 0.91

    checkpoint = {}
    model.on_save_checkpoint(checkpoint)

    ckpt_path = tmp_path / "fusion_calibrated.ckpt"
    torch.save(checkpoint, ckpt_path)

    reloaded = LateFusionModule.from_scratch(
        input_dims=dummy_input_dims,
        active_modalities=list(dummy_input_dims.keys()),
        learning_rate=1e-3,
        config=dummy_config,
        lambda_weight=0.5,
    )
    reloaded.on_load_checkpoint(torch.load(ckpt_path))

    assert reloaded.temperatures["final"] == pytest.approx(2.01)
    assert reloaded.temperatures["add"] == pytest.approx(0.91)

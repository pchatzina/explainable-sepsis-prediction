"""
Run:
    pytest tests/test_mlp_sanity.py -v
"""

import torch
from torch.utils.data import DataLoader, TensorDataset
import pytorch_lightning as pl
from src.models.unimodal.mlp import UnimodalModule


def test_mlp_can_overfit_single_batch():
    # 1. Create a dummy dataset (1 batch of 16 samples)
    input_dim = 128
    X = torch.randn(16, input_dim)
    y = torch.randint(0, 2, (16,)).float()
    ids = torch.arange(16)

    train_loader = DataLoader(TensorDataset(X, y, ids), batch_size=16)

    # 2. Initialize Model
    config = {
        "hidden_dim_1": 64,
        "hidden_dim_2": 32,
        "dropout_rate": 0.0,
        "activation": "ReLU",
    }
    model = UnimodalModule(input_dim=input_dim, learning_rate=0.01, config=config)

    # 3. Get initial loss manually
    with torch.no_grad():
        initial_logits = model(X).squeeze()
        initial_loss = model.criterion(initial_logits, y).item()

    # 4. Train for a short burst
    trainer = pl.Trainer(
        max_epochs=30,
        accelerator="cpu",
        logger=False,
        enable_checkpointing=False,
    )
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=train_loader)

    # 5. Get final loss manually
    with torch.no_grad():
        final_logits = model(X).squeeze()
        final_loss = model.criterion(final_logits, y).item()

    # 6. Assert the model learned successfully
    assert final_loss < initial_loss
    assert final_loss < 0.1


def test_predict_step_returns_correct_dict_and_shapes():
    """
    Verifies that predict_step (used by trainer.predict) returns exactly
    the dictionary structure expected by train_mlp.py, evaluator, and
    calibration/plotting code.
    """
    # 1. Tiny dummy batch
    input_dim = 128
    X = torch.randn(16, input_dim)
    y = torch.randint(0, 2, (16,)).float()
    ids = torch.arange(16)
    batch = (X, y, ids)

    # 2. Create model with default temperature=1.0
    config = {
        "hidden_dim_1": 64,
        "hidden_dim_2": 32,
        "dropout_rate": 0.0,
        "activation": "ReLU",
    }
    model = UnimodalModule(
        input_dim=input_dim,
        learning_rate=0.01,
        config=config,
    )
    model.eval()

    # 3. Call predict_step directly
    output = model.predict_step(batch, batch_idx=0)

    # 4. Core assertions
    assert isinstance(output, dict), "predict_step must return a dict"
    assert set(output.keys()) == {"logits", "p_calibrated", "targets"}, (
        f"Missing keys. Got: {list(output.keys())}"
    )

    # Shape checks
    assert output["logits"].shape == (16,), (
        f"Logits shape wrong: {output['logits'].shape}"
    )
    assert output["p_calibrated"].shape == (16,), (
        f"Calibrated probs shape wrong: {output['p_calibrated'].shape}"
    )
    assert output["targets"].shape == (16,)

    # Mathematical correctness when T=1.0
    expected_probs = torch.sigmoid(output["logits"])
    assert torch.allclose(output["p_calibrated"], expected_probs, atol=1e-6), (
        "Probs must be sigmoid(logits) when temperature=1.0"
    )

    # 5. Test temperature scaling actually affects probs
    model.temperature = 2.0
    output2 = model.predict_step(batch, batch_idx=0)
    expected_probs_t2 = torch.sigmoid(output2["logits"] / 2.0)
    assert torch.allclose(output2["p_calibrated"], expected_probs_t2, atol=1e-6), (
        "Temperature scaling not applied correctly in predict_step"
    )


def test_temperature_checkpoint_roundtrip():
    """Temperature should persist via checkpoint hooks."""
    config = {
        "hidden_dim_1": 64,
        "hidden_dim_2": 32,
        "dropout_rate": 0.0,
        "activation": "ReLU",
    }
    model = UnimodalModule(
        input_dim=128,
        learning_rate=0.01,
        config=config,
    )
    model.temperature = 1.77

    checkpoint = {}
    model.on_save_checkpoint(checkpoint)

    restored = UnimodalModule(
        input_dim=128,
        learning_rate=0.01,
        config=config,
    )
    restored.on_load_checkpoint(checkpoint)

    assert restored.temperature == 1.77


def test_temperature_checkpoint_persisted_to_disk(tmp_path):
    """Temperature should survive a checkpoint save/load cycle on disk."""
    config = {
        "hidden_dim_1": 64,
        "hidden_dim_2": 32,
        "dropout_rate": 0.0,
        "activation": "ReLU",
    }
    model = UnimodalModule(
        input_dim=128,
        learning_rate=0.01,
        config=config,
    )
    model.temperature = 2.25

    checkpoint = {}
    model.on_save_checkpoint(checkpoint)

    ckpt_path = tmp_path / "unimodal_calibrated.ckpt"
    torch.save(checkpoint, ckpt_path)

    reloaded = UnimodalModule(
        input_dim=128,
        learning_rate=0.01,
        config=config,
    )
    reloaded.on_load_checkpoint(torch.load(ckpt_path))

    assert reloaded.temperature == 2.25

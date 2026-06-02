"""
Run:
    pytest tests/test_trainer_factory.py -v
"""

from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from src.training.trainer_factory import build_trainer


def test_build_trainer_has_exact_callbacks(tmp_path):
    """
    Ensures that build_trainer correctly instantiates a PyTorch Lightning
    Trainer with at least two callbacks: ModelCheckpoint and EarlyStopping.
    """
    # Set up dummy parameters using a temporary directory
    max_epochs = 5
    checkpoint_dir = tmp_path / "mock_models"
    experiment_name = "test_run"

    trainer, checkpoint_cb = build_trainer(
        max_epochs=max_epochs,
        checkpoint_dir=checkpoint_dir,
        experiment_name=experiment_name,
    )

    assert len(trainer.callbacks) >= 2, "Expected at least 2 callbacks"

    # Extract the types of the callbacks attached to the trainer
    callback_types = [type(cb) for cb in trainer.callbacks]

    # Verify that the correct callback types are present
    assert ModelCheckpoint in callback_types, "ModelCheckpoint was not attached"
    assert EarlyStopping in callback_types, "EarlyStopping was not attached"
    assert isinstance(checkpoint_cb, ModelCheckpoint), (
        "Returned callback is not a ModelCheckpoint"
    )

"""Tests to verify the presence of downloaded and pretrained Foundation Model weights.

Run:
    pytest tests/test_foundation_weights.py -v
"""

import pytest

from src.utils.config import Config


class TestFoundationModelWeights:
    """Verify that all foundation model directories are populated with weights."""

    def test_cxr_model_weights_exist(self):
        """Check that the Bio_ClinicalBERT CXR model files exist."""
        dir_path = Config.MODEL_CXR_TXT_PRETRAINED_DIR
        assert dir_path is not None, "CXR model path is not defined in Config"

        if not dir_path.exists():
            pytest.skip(f"CXR model dir does not exist yet: {dir_path}")

        assert any(dir_path.iterdir()), f"CXR model directory is empty: {dir_path}"

    def test_ecg_model_weights_exist(self):
        """Check that the ecg-fm model files exist."""
        dir_path = Config.MODEL_ECG_PRETRAINED_DIR
        assert dir_path is not None, "ECG model path is not defined in Config"

        if not dir_path.exists():
            pytest.skip(f"ECG model dir does not exist yet: {dir_path}")

        assert any(dir_path.iterdir()), f"ECG model directory is empty: {dir_path}"

    def test_motor_model_weights_exist(self):
        """Check that the final inference-ready MOTOR bundle exists."""
        dir_path = Config.MODEL_EHR_MOTOR_WEIGHTS_DIR
        assert dir_path is not None, "MOTOR weights path is not defined in Config"

        if not dir_path.exists():
            pytest.skip(f"MODEL_EHR_MOTOR_WEIGHTS_DIR does not exist yet: {dir_path}")

        expected_files = [
            "config.json",
            "model.safetensors",
            "dictionary.msgpack",
        ]

        for filename in expected_files:
            file_path = dir_path / filename
            assert file_path.exists(), f"Missing MOTOR inference file: {file_path}"
            assert file_path.stat().st_size > 0, (
                f"MOTOR inference file is empty: {file_path}"
            )

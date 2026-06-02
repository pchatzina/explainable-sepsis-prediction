"""
Unit tests for Incremental Value Analysis.
Run:
    pytest tests/test_incremental_value_analysis.py -v
"""

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.evaluation.incremental_value_analysis import get_masked_predictions


class DummyUnimodal(torch.nn.Module):
    def __init__(self, temperature: float = 1.0) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class DummyFusion(torch.nn.Module):
    def __init__(self, active_modalities: list[str]) -> None:
        super().__init__()
        self.active_modalities = active_modalities

    def predict_step(self, batch: dict, batch_idx: int) -> dict:
        batch_size = batch["label"].shape[0]
        device = batch["label"].device
        return {
            "p_calibrated": torch.full((batch_size,), 0.5, device=device),
            "logits": {},
            "targets": batch["label"].squeeze(dim=-1),
            "subject_ids": batch.get("subject_id"),
        }


class DummyDataset(Dataset):
    def __init__(self) -> None:
        self.subject_ids = [101, 102, 103]
        self.labels = [0, 1, 0]
        self.ehr_vals = [0.1, 0.2, 0.3]
        self.ecg_mask = [1.0, 0.0, 1.0]

    def __len__(self) -> int:
        return len(self.subject_ids)

    def __getitem__(self, idx: int) -> dict:
        return {
            "subject_id": self.subject_ids[idx],
            "label": torch.tensor(self.labels[idx], dtype=torch.float32),
            "embeddings": {
                "ehr": torch.tensor([self.ehr_vals[idx]], dtype=torch.float32),
                "ecg": torch.tensor([0.0], dtype=torch.float32),
            },
            "masks": {
                "ehr": torch.tensor([1.0], dtype=torch.float32),
                "ecg": torch.tensor([self.ecg_mask[idx]], dtype=torch.float32),
            },
        }


def test_iva_filters_gold_cohort_unimodal():
    dataloader = DataLoader(DummyDataset(), batch_size=3)
    model = DummyUnimodal(temperature=1.0)

    y_true, p_cal, subject_ids = get_masked_predictions(
        model,
        dataloader,
        active_mods={"ehr": 1, "ecg": 1},
        is_unimodal=True,
    )

    assert list(subject_ids) == [101, 103]
    assert len(y_true) == 2
    assert len(p_cal) == 2


def test_iva_filters_gold_cohort_fusion():
    dataloader = DataLoader(DummyDataset(), batch_size=3)
    model = DummyFusion(active_modalities=["ehr", "ecg"])

    y_true, p_cal, subject_ids = get_masked_predictions(
        model,
        dataloader,
        active_mods={"ehr": 1, "ecg": 0},
        is_unimodal=False,
    )

    assert list(subject_ids) == [101, 103]
    assert np.allclose(p_cal, np.array([0.5, 0.5]))
    assert len(y_true) == 2

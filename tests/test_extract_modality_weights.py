"""Unit tests for modality weight extraction.
Run:
    pytest tests/test_extract_modality_weights.py -v
"""

from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

import src.explainability.extract_modality_weights as weights_module
from src.utils.config import Config


class DummyWeightsDataset(Dataset):
    def __init__(self) -> None:
        self.subject_ids = [1, 2]
        self.labels = [0.0, 1.0]

    def __len__(self) -> int:
        return len(self.subject_ids)

    def __getitem__(self, idx: int) -> dict:
        return {
            "subject_id": self.subject_ids[idx],
            "label": torch.tensor(self.labels[idx], dtype=torch.float32),
            "embeddings": {
                "ehr": torch.zeros(1),
                "ecg": torch.zeros(1),
            },
            "masks": {
                "ehr": torch.ones(1),
                "ecg": torch.ones(1),
            },
        }


class DummyDataModule:
    def __init__(self, *args, **kwargs) -> None:
        self._loader = DataLoader(DummyWeightsDataset(), batch_size=2)

    def setup(self, *args, **kwargs) -> None:
        return None

    def test_dataloader(self) -> DataLoader:
        return self._loader


class DummyFusionModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.active_modalities = ["ehr", "ecg"]

    def forward(self, embeddings: dict, masks: dict) -> dict:
        batch_size = next(iter(embeddings.values())).shape[0]
        weights = torch.zeros(batch_size, len(self.active_modalities))
        weights[:, 0] = 1.0
        return {
            "weights": weights,
            "beta": torch.zeros(batch_size),
            "p_final": torch.full((batch_size, 1), 0.5),
        }


def test_extract_modality_weights_writes_csv(tmp_path, monkeypatch):
    monkeypatch.setattr(Config, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(weights_module, "FusionDataModule", DummyDataModule)
    monkeypatch.setattr(
        weights_module.LateFusionModule,
        "load_from_checkpoint",
        lambda *args, **kwargs: DummyFusionModel(),
    )

    weights_module.main()

    csv_path = (
        Path(tmp_path)
        / "explainability"
        / "modality_weights"
        / "4mod_architecture"
        / "test_set_modality_weights.csv"
    )
    assert csv_path.exists()

    df = pd.read_csv(csv_path)
    assert set(df.columns) >= {"subject_id", "true_label", "p_final", "beta"}
    assert set(df.columns) >= {"w_ehr", "w_ecg"}
    assert (df["w_ehr"] == 1.0).all()
    assert (df["beta"] == 0.0).all()

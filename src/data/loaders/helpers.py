"""Data Loading Helpers for Unimodal and Fusion Training.

Provides utilities for:
- Loading normalized embeddings from .pt files in standardized format
- Creating PyTorch DataLoaders for unimodal training
"""

import logging
from pathlib import Path
from typing import List, Tuple, Union
from torch.utils.data import DataLoader, TensorDataset

import numpy as np
import torch

logger = logging.getLogger(__name__)


def load_embeddings(
    filepath: Union[str, Path],
) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """Load normalized embeddings from a .pt file.

    Args:
        filepath: Path to the .pt embedding file.

    Returns:
        Tuple of:
            - X (np.ndarray): Embeddings of shape (N, D), dtype float32
            - y (np.ndarray): Labels of shape (N,), dtype int (0/1)
            - subject_ids (List[int]): MIMIC-IV subject IDs for tracking and debugging
    """
    filepath = Path(filepath)
    logger.info("Loading embeddings from %s", filepath)
    data = torch.load(filepath, map_location="cpu", weights_only=False)

    X = data["embeddings"].cpu().numpy().astype(np.float32)
    y = np.array([int(label) for label in data["labels"]], dtype=int)

    subject_ids = [int(sid) for sid in data["subject_ids"]]

    logger.info("  → %d samples, dim %d", X.shape[0], X.shape[1])
    return X, y, subject_ids


def get_unimodal_dataloader(X, y, ids, batch_size=256, shuffle=False):
    """Create a PyTorch DataLoader for unimodal training/inference.

    Wraps embeddings, labels, and subject IDs in a TensorDataset and returns
    a DataLoader ready for use with PyTorch Lightning or manual training loops.

    Args:
        X: Embeddings array of shape (N, D), dtype float32.
        y: Labels array of shape (N,), dtype int (0/1).
        ids: Subject IDs list of length N.
        batch_size: Batch size (default 256).
        shuffle: Whether to shuffle data (typically True for training, False for val/test).

    Returns:
        DataLoader yielding batches of (X_batch, y_batch, ids_batch).
    """
    dataset = TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32),
        torch.tensor(ids, dtype=torch.long),
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        persistent_workers=False,
    )

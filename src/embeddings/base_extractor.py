"""Abstract base class shared by EHRExtractor, ECGExtractor, CXRImageExtractor, and CXRTextExtractor.

Provides device setup, output directory creation, and the canonical save_split() method,
which persists embeddings as {split}_embeddings_raw.pt files containing embeddings,
labels, and subject_ids tensors.
"""

import logging
import torch
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List

from src.utils.config import Config

logger = logging.getLogger(__name__)


class BaseEmbeddingExtractor(ABC):
    """Abstract base class for all multimodal embedding extraction pipelines."""

    def __init__(self, output_dir: Path):
        """Initializes standard configurations shared across all modalities.

        Args:
            output_dir (Path): Root directory for saving extracted embeddings.
        """
        Config.setup_logging()
        Config.check_dirs()
        Config.set_seed()

        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Initialized {self.__class__.__name__} on {self.device}")

    @abstractmethod
    def extract_and_save(self) -> None:
        """The core extraction logic to be implemented by child classes."""
        pass

    def save_split(
        self,
        split_name: str,
        embeddings: torch.Tensor,
        labels: List[int],
        subject_ids: List[int],
    ) -> None:
        """Standardizes the saving of embedding tensors to disk in canonical format.

        Args:
            split_name (str): Dataset split identifier ("train", "valid", or "test").
            embeddings (torch.Tensor): Embedding vectors.
            labels (List[int]): Binary sepsis labels (0 or 1) for each subject.
            subject_ids (List[int]): MIMIC-IV subject identifiers for traceability.

        Raises:
            Logs warning if split_name has no data; save is skipped.
        """
        if not subject_ids:
            logger.warning(f"No data found for split '{split_name}'. Skipping save.")
            return

        # === Check Embeddings Type ===
        if not isinstance(embeddings, torch.Tensor):
            raise ValueError(
                f"Embeddings must be a PyTorch Tensor, got {type(embeddings)}"
            )

        # === Save to Canonical .pt Format ===
        # File naming follows convention: {split}_embeddings_raw.pt
        # (Raw = before L2 normalization; normalized copies saved by normalize_embeddings.py)
        out_path = self.output_dir / f"{split_name}_embeddings_raw.pt"

        torch.save(
            {
                "embeddings": embeddings,  # Shape: [N, D] where N=samples, D=embedding_dim
                "labels": labels,  # List of int (0 or 1)
                "subject_ids": subject_ids,  # List of int; enables subject-level analysis
            },
            out_path,
        )

        logger.info(
            f"Saved {split_name} split: {embeddings.shape[0]} samples, "
            f"dim={embeddings.shape[1]} -> {out_path.name}"
        )

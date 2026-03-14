import logging
import torch
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Union
import numpy as np

from src.utils.config import Config

logger = logging.getLogger(__name__)


class BaseEmbeddingExtractor(ABC):
    def __init__(self, output_dir: Path):
        """Initializes standard configurations shared across all modalities."""
        Config.setup_logging()
        Config.check_dirs()
        Config.set_seed()

        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Initialized {self.__class__.__name__} on {self.device}")

    @abstractmethod
    def extract_and_save(self):
        """The core extraction logic to be implemented by child classes."""
        pass

    def save_split(
        self,
        split_name: str,
        embeddings: Union[torch.Tensor, np.ndarray, List],
        labels: List[int],
        subject_ids: List[int],
    ):
        """Standardizes the saving of embedding tensors to disk."""
        if not subject_ids:
            logger.warning(f"No data found for split '{split_name}'. Skipping save.")
            return

        # Ensure embeddings are a Tensor
        if isinstance(embeddings, list):
            # Stack 1D tensors or convert numpy arrays
            embeddings = (
                torch.stack(embeddings)
                if isinstance(embeddings[0], torch.Tensor)
                else torch.tensor(np.array(embeddings, dtype=np.float32))
            )
        elif isinstance(embeddings, np.ndarray):
            embeddings = torch.from_numpy(embeddings.astype(np.float32))

        out_path = self.output_dir / f"{split_name}_embeddings_raw.pt"

        torch.save(
            {
                "embeddings": embeddings,
                "labels": labels,
                "subject_ids": subject_ids,
            },
            out_path,
        )

        logger.info(
            f"Saved {split_name} split: {embeddings.shape[0]} samples, "
            f"dim={embeddings.shape[1]} -> {out_path.name}"
        )

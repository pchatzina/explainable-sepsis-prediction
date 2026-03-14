"""
L2-normalize all raw embedding .pt files and save them to the normalized directories.

Usage:
    python -m src.embeddings.normalize_embeddings
"""

import logging
import torch
from pathlib import Path

from src.utils.config import Config

logger = logging.getLogger(__name__)


def main():
    Config.setup_logging()
    logger.info("Starting L2-normalization of all embeddings...")

    # Map each modality to its corresponding (raw_directory, normalized_directory)
    MODALITIES = {
        "ehr": (
            Config.PROCESSED_EHR_RAW_EMBEDDINGS_DIR,
            Config.PROCESSED_EHR_EMBEDDINGS_DIR,
        ),
        "ecg": (
            Config.PROCESSED_ECG_RAW_EMBEDDINGS_DIR,
            Config.PROCESSED_ECG_EMBEDDINGS_DIR,
        ),
        "cxr_img": (
            Config.PROCESSED_CXR_IMG_RAW_EMBEDDINGS_DIR,
            Config.PROCESSED_CXR_IMG_EMBEDDINGS_DIR,
        ),
        "cxr_txt": (
            Config.PROCESSED_CXR_TXT_RAW_EMBEDDINGS_DIR,
            Config.PROCESSED_CXR_TXT_EMBEDDINGS_DIR,
        ),
    }

    splits = ["train", "valid", "test"]

    for mod_name, (raw_dir, norm_dir) in MODALITIES.items():
        logger.info(f"--- Processing Modality: {mod_name.upper()} ---")

        norm_dir.mkdir(parents=True, exist_ok=True)

        for split in splits:
            in_path = raw_dir / f"{split}_embeddings_raw.pt"
            out_path = norm_dir / f"{split}_embeddings.pt"

            if not in_path.exists():
                logger.warning(f"Skipping missing raw file: {in_path}")
                continue

            logger.info(f"Normalizing {in_path.name}...")

            # Load the raw tensors
            data = torch.load(in_path, map_location="cpu", weights_only=False)
            emb = data["embeddings"].float()

            # Apply L2 Normalization (with epsilon to prevent division by zero)
            norms = torch.norm(emb, p=2, dim=1, keepdim=True) + 1e-8
            data["embeddings"] = emb / norms

            # Save to the normalized directory
            torch.save(data, out_path)
            logger.info(f"→ Saved to {out_path.name}")

    logger.info("✅ All embeddings normalized! Ready for fusion.")


if __name__ == "__main__":
    main()

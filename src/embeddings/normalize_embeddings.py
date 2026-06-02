"""
L2-normalize all raw embedding .pt files and save them to the normalized directories.

Usage:
    python -m src.embeddings.normalize_embeddings
"""

import logging
import torch

from src.utils.config import Config

logger = logging.getLogger(__name__)


def main() -> None:
    """Orchestrates L2 normalization across all modalities and splits.

    Workflow:
        1. Define mapping of modalities to (raw, normalized) directory pairs.
        2. For each modality and split, load raw embeddings from disk.
        3. Apply L2 normalization (row-wise) to ensure ||v|| = 1.
        4. Preserve labels and subject_ids metadata unchanged.
        5. Save normalized embeddings in canonical format.
    """
    Config.setup_logging()
    logger.info("Starting L2-normalization of all embeddings...")

    # === Modality Directory Mapping ===
    # Links each modality to its raw (pre-normalize) and normalized output directories.
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

    # === Iterate Over Modalities ===
    for mod_name, (raw_dir, norm_dir) in MODALITIES.items():
        logger.info(f"--- Processing Modality: {mod_name.upper()} ---")

        # Create output directory for normalized embeddings
        norm_dir.mkdir(parents=True, exist_ok=True)

        # === Iterate Over Splits ===
        # Process train/valid/test independently to enable evaluation on any split.
        for split in splits:
            in_path = raw_dir / f"{split}_embeddings_raw.pt"
            out_path = norm_dir / f"{split}_embeddings.pt"

            # === Check File Existence ===
            # Skip splits where no raw embeddings were produced (e.g., empty modality).
            if not in_path.exists():
                logger.warning(f"Skipping missing raw file: {in_path}")
                continue

            logger.info(f"Normalizing {in_path.name}...")

            # === Load Raw Embeddings ===
            # Loaded to CPU to preserve GPU memory for downstream tasks.
            data = torch.load(in_path, map_location="cpu", weights_only=False)
            emb = data["embeddings"].float()

            # === L2 Normalization with Numerical Stability ===
            # Compute L2 norm (Euclidean norm) per row: ||v||_2 = sqrt(sum(v_i^2))
            # Add epsilon (1e-8) to prevent division by zero for zero vectors.
            norms = torch.norm(emb, p=2, dim=1, keepdim=True) + 1e-8
            # Normalize: v_normalized = v / ||v||
            data["embeddings"] = emb / norms

            # === Save Normalized Embeddings ===
            # Preserves original {labels, subject_ids} metadata; only embeddings are normalized.
            torch.save(data, out_path)
            logger.info(f"→ Saved to {out_path.name}")

    logger.info("All embeddings normalized! Ready for fusion.")


if __name__ == "__main__":
    main()

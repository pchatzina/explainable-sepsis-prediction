"""
Orchestrates the sequential extraction of all multimodal embeddings.
Ensures GPU memory is cleared between loading different Foundation Models.

Usage:
    python -m src.embeddings.run_pipeline
"""

import gc
import logging
import sys
import torch

from src.embeddings.ehr_embeddings import EHRExtractor
from src.embeddings.ecg_embeddings import ECGExtractor
from src.embeddings.cxr_txt_embeddings import CXRTextExtractor
from src.embeddings.cxr_img_embeddings import CXRImageExtractor
from src.utils.config import Config

logger = logging.getLogger(__name__)


def clean_memory():
    """Forces Python and PyTorch to release unreferenced GPU memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> None:
    Config.setup_logging()
    logger.info("Starting the Multi-Modal Embedding Pipeline...")

    # === Define Extraction Pipeline Order ===
    extractors_to_run = [
        ("EHR", EHRExtractor),
        ("CXR_Text", CXRTextExtractor),
        ("CXR_Image", CXRImageExtractor),
        ("ECG", ECGExtractor),
    ]

    has_failures = False

    # === Sequential Modality Processing ===
    for name, ExtractorClass in extractors_to_run:
        logger.info("=" * 50)
        logger.info(f"--- Starting {name} Extraction ---")

        try:
            # === Instantiate Extractor ===
            # This loads the foundation model into GPU.
            extractor = ExtractorClass()

            # === Run Extraction and Persistence ===
            # Processes all splits (train/valid/test) and saves raw embeddings.
            extractor.extract_and_save()

            logger.info(f"{name} extraction completed successfully.")

        except Exception as e:
            has_failures = True
            logger.error(f"Pipeline failed during {name} extraction: {str(e)}")
            logger.error("Continuing to the next modality...")

        finally:
            # === Aggressive GPU Memory Cleanup ===
            # Delete the extractor instance (which holds the foundation model).
            # Then garbage collect and empty PyTorch cache before next modality.
            # This is critical to prevent OOM on small GPUs.
            if "extractor" in locals():
                del extractor
            clean_memory()
            logger.info(f"Cleared GPU memory after {name}.")

    logger.info("=" * 50)

    # === Exit Status Reporting ===
    # Critical: fail the python process if any modality failed.
    # The bash orchestrator (04_embeddings_extraction.sh) checks exit code.
    if has_failures:
        logger.error(
            "Pipeline finished, but one or more modalities FAILED. Check logs."
        )
        sys.exit(1)
    else:
        logger.info("All embedding extractions have finished successfully!")


if __name__ == "__main__":
    main()

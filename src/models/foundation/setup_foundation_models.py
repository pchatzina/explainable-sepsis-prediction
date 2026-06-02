"""Entry point for Phase 3, Stage 1. See scripts/03_foundation_models.sh for orchestration context."""

import logging
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer, AutoModel

from src.utils.config import Config

logger = logging.getLogger(__name__)


def download_cxr_txt_model():
    """
    Downloads and caches Bio_ClinicalBERT for CXR report text encoding.
    """
    logger.info("Downloading CXR Text model (Bio_ClinicalBERT)...")
    model_name = "emilyalsentzer/Bio_ClinicalBERT"
    model_dir = Config.MODEL_CXR_TXT_PRETRAINED_DIR
    model_dir.mkdir(parents=True, exist_ok=True)

    # Calling from_pretrained automatically downloads and caches if not present
    AutoTokenizer.from_pretrained(model_name, cache_dir=model_dir)
    AutoModel.from_pretrained(model_name, cache_dir=model_dir)
    logger.info("Bio_ClinicalBERT cached successfully.")


def download_ecg_model():
    """
    Downloads and caches ECG Foundation Model (ECG-FM).
    """
    logger.info("Downloading ECG Foundation Model (ecg-fm)...")
    checkpoint_dir = Config.MODEL_ECG_PRETRAINED_DIR
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    hf_hub_download(
        repo_id="wanglab/ecg-fm",
        filename="mimic_iv_ecg_physionet_pretrained.pt",
        local_dir=checkpoint_dir,
        local_dir_use_symlinks=False,
    )
    logger.info("ecg-fm cached successfully.")


def main() -> None:
    Config.setup_logging()
    logger.info("Starting Foundation Model provisioning...")

    try:
        download_cxr_txt_model()
        download_ecg_model()
        logger.info("ECG and CXR TXT models successfully downloaded and cached!")
    except Exception as e:
        logger.error(f"Setup failed: {e}")
        raise


if __name__ == "__main__":
    main()

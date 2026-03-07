"""Download CXR and ECG metadata CSV files from PhysioNet."""

import logging
import subprocess
import sys

from src.utils.config import Config
from src.utils.download import download_with_wget

logger = logging.getLogger(__name__)


def main():
    # First python script that touches the file system, initialize folders
    Config.check_dirs()

    logger.info("Starting Metadata Downloads...")

    # ---------------------------------------------------------
    # 1. Download CXR Metadata
    # ---------------------------------------------------------
    logger.info("[1/2] Downloading CXR metadata CSVs...")
    for filename, base_url in Config.CXR_METADATA_FILES:
        local_path = Config.RAW_CXR_IMG_DIR / filename
        url = f"{base_url}{filename}"

        try:
            download_with_wget(
                url, local_path, Config.PHYSIONET_USER, Config.PHYSIONET_PASS
            )
            logger.info("Downloaded: %s", local_path.name)
        except subprocess.CalledProcessError:
            logger.critical("Failed to download %s", url)
            sys.exit(1)

    # ---------------------------------------------------------
    # 2. Download ECG Metadata
    # ---------------------------------------------------------
    logger.info("[2/2] Downloading ECG metadata CSVs...")
    for filename in Config.ECG_METADATA_FILES:
        local_path = Config.RAW_ECG_DIR / filename
        url = f"{Config.URL_ECG_BASE}{filename}"

        try:
            download_with_wget(
                url, local_path, Config.PHYSIONET_USER, Config.PHYSIONET_PASS
            )
            logger.info("Downloaded: %s", local_path.name)
        except subprocess.CalledProcessError:
            logger.critical("Failed to download %s", url)
            sys.exit(1)

    logger.info("Metadata download complete.")


if __name__ == "__main__":
    Config.setup_logging()
    main()

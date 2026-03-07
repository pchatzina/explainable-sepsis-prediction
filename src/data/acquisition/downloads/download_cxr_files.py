"""Download CXR image files and reports from PhysioNet for cohort patients."""

import logging
import subprocess

from sqlalchemy import text

from src.utils.config import Config
from src.utils.database import get_engine
from src.utils.download import download_with_wget

logger = logging.getLogger(__name__)


def download_reports():
    """Download the CXR reports zip from PhysioNet into RAW_CXR_TXT_DIR."""
    logger.info("Starting CXR Reports Download...")

    url = f"{Config.URL_CXR_BASE}{Config.CXR_REPORTS_FILE}"
    local_file = Config.RAW_CXR_TXT_DIR / Config.CXR_REPORTS_FILE

    try:
        download_with_wget(
            url, local_file, Config.PHYSIONET_USER, Config.PHYSIONET_PASS
        )
        logger.info("Downloaded CXR reports to %s", local_file)
    except subprocess.CalledProcessError:
        logger.error("Failed to download CXR reports from %s", url)

    logger.info("CXR Reports Download Complete.")


def download_images():
    """Download individual CXR JPG images for each cohort patient."""
    logger.info("Starting CXR Image Download for Cohort...")
    engine = get_engine()

    query = text("SELECT subject_id, study_id, study_path FROM mimiciv_ext.cohort_cxr")

    with engine.connect() as conn:
        rows = conn.execute(query).fetchall()

    total = len(rows)
    logger.info("Found %d CXR images to download.", total)

    for i, (sub_id, _, db_path) in enumerate(rows, 1):
        url = f"{Config.URL_CXR_JPG_BASE}{db_path}"
        local_file = Config.RAW_CXR_IMG_DIR / db_path

        if local_file.exists():
            continue

        try:
            download_with_wget(
                url, local_file, Config.PHYSIONET_USER, Config.PHYSIONET_PASS
            )
        except subprocess.CalledProcessError:
            logger.error("Failed to download CXR for Subject %s (%s)", sub_id, url)

        if i % 50 == 0:
            logger.info("[%d/%d] Downloaded CXR for Subject %s", i, total, sub_id)

    logger.info("CXR Image Download Complete.")


def main():
    download_reports()
    download_images()


if __name__ == "__main__":
    Config.setup_logging()
    main()

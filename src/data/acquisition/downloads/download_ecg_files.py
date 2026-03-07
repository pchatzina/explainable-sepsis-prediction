"""Download ECG signal files (.dat/.hea) from PhysioNet for cohort patients."""

import logging
import subprocess

from sqlalchemy import text

from src.utils.config import Config
from src.utils.database import get_engine
from src.utils.download import download_with_wget

logger = logging.getLogger(__name__)


def main():
    logger.info("Starting ECG Record Download for Cohort...")
    engine = get_engine()

    query = text("SELECT subject_id, study_id, study_path FROM mimiciv_ext.cohort_ecg")

    with engine.connect() as conn:
        rows = conn.execute(query).fetchall()

    total = len(rows)
    logger.info("Found %d ECG records to download.", total)

    for i, (sub_id, _, db_path) in enumerate(rows, 1):
        # Download both .dat and .hea files
        for ext in [".dat", ".hea"]:
            url = f"{Config.URL_ECG_BASE}{db_path}{ext}"
            local_file = Config.RAW_ECG_DIR / f"{db_path}{ext}"

            if local_file.exists():
                continue

            try:
                download_with_wget(
                    url, local_file, Config.PHYSIONET_USER, Config.PHYSIONET_PASS
                )
            except subprocess.CalledProcessError:
                logger.error("Failed to download ECG %s for Subject %s", ext, sub_id)

        if i % 50 == 0:
            logger.info("[%d/%d] Processed ECG for Subject %s", i, total, sub_id)

    logger.info("ECG Download Complete.")


if __name__ == "__main__":
    Config.setup_logging()
    main()

"""Download CXR and ECG metadata CSV files from PhysioNet.

Stage 2 of data acquisition orchestrator (scripts/01_data_acquisition_and_splitting.sh).

This module downloads CXR and ECG study metadata required to query modality availability
in Stage 3 (create_generic_modalities_cohort.sql).

Files are downloaded from PhysioNet using wget (requires credentials).
"""

import logging
import subprocess
import sys

from src.utils.config import Config
from src.utils.download import download_with_wget

logger = logging.getLogger(__name__)


def main() -> None:
    """Download metadata CSVs from PhysioNet."""
    # First python script that touches the file system; initialize folders
    Config.check_dirs()

    logger.info("Starting Metadata Downloads (Stage 2 of data acquisition)...")

    # ---------------------------------------------------------
    # STEP 1: Download CXR Metadata (3 files)
    # ---------------------------------------------------------
    # Files:
    #   - cxr-study-list.csv.gz: study_id, subject_id, path per CXR study
    #   - cxr-record-list.csv.gz: study_id, dicom_id, subject_id per DICOM record
    #   - mimic-cxr-2.0.0-metadata.csv.gz: DICOM metadata
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
    # STEP 2: Download ECG Metadata (2 files)
    # ---------------------------------------------------------
    # Files:
    #   - record_list.csv: study_id, subject_id, ecg_time, path per ECG record
    #   - machine_measurements.csv: study_id, subject_id, measurements
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

    logger.info("Metadata download complete. Proceeding to loading DB tables.")


if __name__ == "__main__":
    Config.setup_logging()
    main()

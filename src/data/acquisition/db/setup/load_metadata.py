"""Create DB schemas and load CXR/ECG metadata into PostgreSQL.

Stage 2 of data acquisition orchestrator (scripts/01_data_acquisition_and_splitting.sh).

This module performs 3 sequential steps:

  STEP 1: Create Schemas (DDL)
    - mimiciv_cxr: Tables for DICOM metadata (study_list, record_list, metadata)
    - mimiciv_ecg: Tables for ECG metadata (record_list, machine_measurements)

  STEP 2: Load CXR Metadata (3 files from download_metadata_files.py)
    - cxr-study-list.csv.gz
    - cxr-record-list.csv.gz
    - mimic-cxr-2.0.0-metadata.csv.gz
    Populates: mimiciv_cxr.study_list, mimiciv_cxr.record_list, mimiciv_cxr.metadata

  STEP 3: Load ECG Metadata (2 files from download_metadata_files.py)
    - record_list.csv
    - machine_measurements.csv
    Populates: mimiciv_ecg.record_list, mimiciv_ecg.machine_measurements
"""

import logging

from src.utils.config import Config
from src.utils.database import get_engine, load_table_from_csv, run_ddl_script

logger = logging.getLogger(__name__)


def main() -> None:
    engine = get_engine()
    script_root = Config.SRC_DB_SETUP_DIR

    # ==========================================
    # PHASE 1: CREATE SCHEMAS (DDL)
    # ==========================================
    logger.info("Phase 1: Creating Schemas...")
    run_ddl_script(engine, script_root / "create_cxr.sql")
    run_ddl_script(engine, script_root / "create_ecg.sql")

    # ==========================================
    # PHASE 2: LOAD CXR DATA
    # ==========================================
    logger.info("Phase 2: Loading CXR Data...")
    cxr_tasks = [
        ("mimiciv_cxr.record_list", Config.RAW_CXR_IMG_DIR / "cxr-record-list.csv.gz"),
        ("mimiciv_cxr.study_list", Config.RAW_CXR_IMG_DIR / "cxr-study-list.csv.gz"),
        (
            "mimiciv_cxr.metadata",
            Config.RAW_CXR_IMG_DIR / "mimic-cxr-2.0.0-metadata.csv.gz",
        ),
    ]

    for table, fpath in cxr_tasks:
        load_table_from_csv(engine, table, fpath, compressed=True)

    # ==========================================
    # PHASE 3: LOAD ECG DATA
    # ==========================================
    logger.info("Phase 3: Loading ECG Data...")
    ecg_tasks = [
        ("mimiciv_ecg.record_list", Config.RAW_ECG_DIR / "record_list.csv"),
        (
            "mimiciv_ecg.machine_measurements",
            Config.RAW_ECG_DIR / "machine_measurements.csv",
        ),
    ]

    for table, fpath in ecg_tasks:
        load_table_from_csv(engine, table, fpath, compressed=False)

    logger.info("Database setup complete.")


if __name__ == "__main__":
    Config.setup_logging()
    main()

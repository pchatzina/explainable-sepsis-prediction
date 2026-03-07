"""Create DB schemas and load CXR/ECG metadata into PostgreSQL."""

import logging

from src.utils.config import Config
from src.utils.database import get_engine, load_table_from_csv, run_ddl_script

logger = logging.getLogger(__name__)


def main():
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

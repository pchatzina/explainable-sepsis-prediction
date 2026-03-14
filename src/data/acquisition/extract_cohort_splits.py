"""
Queries the live database to build a static master cohort file.
This ensures reproducibility and standardizes dataset splits.

Usage:
    python -m src.data.acquisition.extract_cohort_splits
"""

import logging
from pathlib import Path
import pandas as pd

from src.utils.config import Config
from src.utils.database import query_to_df

logger = logging.getLogger(__name__)


def create_master_cohort():
    Config.setup_logging()

    logger.info("Fetching master cohort and split definitions from database...")

    query = """
        SELECT 
            s.subject_id,
            s.dataset_split,
            s.sepsis_label,
            s.modality_signature,
            cxr.study_id AS cxr_study_id,
            cxr.study_path AS cxr_study_path,
            ecg.study_id AS ecg_study_id,
            ecg.study_path AS ecg_study_path
        FROM mimiciv_ext.dataset_splits s
        LEFT JOIN mimiciv_ext.cohort_cxr cxr ON s.subject_id = cxr.subject_id
		LEFT JOIN mimiciv_ext.cohort_ecg ecg ON s.subject_id = ecg.subject_id
        WHERE s.dataset_split IS NOT NULL;
    """

    df = query_to_df(query)

    logger.info("Standardizing dataset splits...")
    df["dataset_split"] = df["dataset_split"].str.lower().replace({"validate": "valid"})

    output_path = Config.PROCESSED_COHORT_PARQUET_FILE
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_parquet(output_path, index=False)

    logger.info(f"Successfully saved {len(df)} records to {output_path}")
    logger.info(f"Final Split Distribution:\n{df['dataset_split'].value_counts()}")


if __name__ == "__main__":
    create_master_cohort()

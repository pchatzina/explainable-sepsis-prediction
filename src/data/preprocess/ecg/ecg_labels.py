"""
Create ECG labels for sepsis prediction.

Queries the cohort from PostgreSQL and saves a labels CSV consumed by
`create_manifests.py` to build fairseq TSV/LBL manifests.

Usage:
    python -m src.data.preprocess.ecg.ecg_labels
"""

import logging

from src.utils.config import Config
from src.utils.database import query_to_df

logger = logging.getLogger(__name__)

LABELS_QUERY = """
    SELECT
        ds.dataset_split AS split,
        ROW_NUMBER() OVER (
            PARTITION BY ds.dataset_split ORDER BY ds.subject_id
        ) - 1 AS id,
        ds.subject_id AS sample_id,
        ds.sepsis_label AS label
    FROM mimiciv_ext.dataset_splits ds
    INNER JOIN mimiciv_ext.cohort_ecg e
        ON ds.subject_id = e.subject_id
        AND ds.hadm_id = e.hadm_id
    ORDER BY split, id
"""


def main():
    """Fetch ECG cohort labels from the database and save to CSV."""
    Config.setup_logging()
    logger.info("Querying ECG cohort labels from database...")

    df = query_to_df(LABELS_QUERY)
    logger.info("Found %d labelled ECGs.", len(df))

    output_dir = Config.PROCESSED_ECG_LABELS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "labels.csv"

    df.to_csv(out_path, index=False)
    logger.info("Labels saved -> %s", out_path)


if __name__ == "__main__":
    main()

"""
Master Cohort Extraction & Serialization Module.

Queries the finalized cohort tables (created in Stage 3 of the data pipeline) and
serializes the result to master_cohort.parquet, the canonical input for all downstream
phases (embedding extraction, model training, evaluation, and explainability analysis).

Output file: master_cohort.parquet
Row count: 15,513 (one row per unique patient).

Schema:
    subject_id          Patient unique identifier.
    dataset_split       One of {train, valid, test}; stratified and locked.
    sepsis_label        0 = negative control, 1 = sepsis-positive.
    modality_signature  Available modalities: EHR, EHR_CXR, EHR_ECG, EHR_CXR_ECG.
    admittime           Admission timestamp.
    anchor_time         Prediction window boundary (sepsis onset minus 6 hours for
                        positives; pseudo-onset minus 6 hours for negatives).
    cxr_study_id        CXR study ID (NULL if unavailable).
    cxr_study_path      Path to CXR JPG file (NULL if unavailable).
    ecg_study_id        ECG study ID (NULL if unavailable).
    ecg_study_path      Path to ECG signal file (.dat/.hea, NULL if unavailable).

Usage:
    python -m src.data.acquisition.extract_cohort_splits
"""

import logging

from src.utils.config import Config
from src.utils.database import query_to_df

logger = logging.getLogger(__name__)


def create_master_cohort() -> None:
    """
    Extract finalized cohort from database and serialize to Parquet.

    Joins four database tables (cohort, cohort_cxr, cohort_ecg, dataset_splits)
    and normalises split keys to lowercase.
    Writes to Config.PROCESSED_COHORT_PARQUET_FILE. The schema is documented in
    the module docstring above.
    """
    Config.setup_logging()

    logger.info("Fetching master cohort and split definitions from database...")

    # Query retrieves patients with their modality availability and split assignment.
    query = """
        SELECT 
            s.subject_id,
            s.dataset_split,
            s.sepsis_label,
            s.modality_signature,
            c.admittime,
            c.anchor_time,
            cxr.study_id AS cxr_study_id,
            cxr.study_path AS cxr_study_path,
            ecg.study_id AS ecg_study_id,
            ecg.study_path AS ecg_study_path
        FROM mimiciv_ext.dataset_splits s
        JOIN mimiciv_ext.cohort c ON s.subject_id = c.subject_id
        LEFT JOIN mimiciv_ext.cohort_cxr cxr ON s.subject_id = cxr.subject_id
		LEFT JOIN mimiciv_ext.cohort_ecg ecg ON s.subject_id = ecg.subject_id
        WHERE s.dataset_split IS NOT NULL;
    """

    df = query_to_df(query)
    logger.info(f"Query returned {len(df)} rows")

    # Ensures consistent split keys downstream
    df["dataset_split"] = df["dataset_split"].str.lower().replace({"validate": "valid"})

    # Ensure output directory exists and serialize to Parquet
    output_path = Config.PROCESSED_COHORT_PARQUET_FILE
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_parquet(output_path, index=False)

    logger.info(f"Successfully saved {len(df)} records to {output_path}")
    logger.info(f"Output shape: {df.shape}")
    logger.info(f"Final Split Distribution:\n{df['dataset_split'].value_counts()}")


if __name__ == "__main__":
    create_master_cohort()

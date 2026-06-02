"""
Module: ehr_labels.py
======================
Purpose: Generate sepsis ground-truth labels for MOTOR.

Schema Expected by FEMR:
  - subject_id (int): MIMIC patient identifier
  - prediction_time (datetime): Anchor time (6 hours before sepsis onset)
  - boolean_value (bool): Sepsis label (true=sepsis positive, false=sepsis negative)
"""

import logging
import pandas as pd
import meds_reader
import femr.labelers

from src.utils.config import Config

logger = logging.getLogger(__name__)


class SepsisCohortLabeler(femr.labelers.Labeler):
    """
    Assign sepsis labels to FEMR subjects.
    """

    def __init__(self, cohort_df):
        """
        Initialize labeler with cohort data.

        Args:
            cohort_df: DataFrame with columns [subject_id, admittime, anchor_time, sepsis_label]
        """
        super().__init__()
        self.cohort_dict = cohort_df.set_index(["subject_id", "admittime"]).to_dict(
            orient="index"
        )

    def label(self, subject):
        """
        Assign labels to all admissions in a subject's FEMR record.

        Args:
            subject: FEMR subject object

        Returns:
            List of label dicts: {subject_id, prediction_time, boolean_value}
            One entry per admission in cohort for this subject.

        Process:
            1. Extract admission times from MEDS events (events starting with "MIMIC_IV_Admission/")
            2. For each admission, lookup cohort row by (subject_id, admittime)
            3. If found in cohort: emit label with prediction_time = anchor_time
        """
        admission_starts = set()
        for event in subject.events:
            if event.code.startswith("MIMIC_IV_Admission/"):
                admission_starts.add(event.time)

        labels = []
        for admission_start in admission_starts:
            key = (subject.subject_id, admission_start)

            if key in self.cohort_dict:
                row_data = self.cohort_dict[key]
                labels.append(
                    {
                        "subject_id": subject.subject_id,
                        "prediction_time": row_data["anchor_time"],
                        "boolean_value": row_data["sepsis_label"] == 1,
                    }
                )
        return labels


def main() -> None:
    """
    Generate labels.parquet for MOTOR embedding extraction.

    Output:
        labels.parquet: {subject_id, prediction_time, boolean_value}
        Record count: One per admission in cohort (15,513 rows)
    """
    Config.setup_logging()

    logger.info("Reading cohort from parquet file")
    cohort_file = Config.PROCESSED_COHORT_PARQUET_FILE
    cohort_df = pd.read_parquet(cohort_file)
    logger.info("Cohort rows: %d", len(cohort_df))

    output_dir = Config.PROCESSED_EHR_LABELS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Opening MEDS database: %s", Config.PROCESSED_EHR_MEDS_COHORT_DIR)
    with meds_reader.SubjectDatabase(
        str(Config.PROCESSED_EHR_MEDS_COHORT_DIR), num_threads=6
    ) as database:
        labeler = SepsisCohortLabeler(cohort_df=cohort_df)
        labels_df = labeler.apply(database)

    out_path = output_dir / "labels.parquet"
    labels_df.to_parquet(out_path, index=False)
    logger.info("Saved %d labels → %s", len(labels_df), out_path)


if __name__ == "__main__":
    main()

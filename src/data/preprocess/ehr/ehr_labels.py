"""Create EHR labels for MOTOR embedding extraction.

Produces a ``labels.parquet`` file with the schema that femr expects::

    subject_id:      int
    prediction_time: datetime
    boolean_value:   bool

This file is consumed by ``extract_ehr_embeddings.py`` which passes
the labels to ``femr.models.transformer.compute_features``.

Usage:
    python -m src.data.preprocess.ehr.ehr_labels
"""

import logging

import meds_reader

import femr.labelers

from src.utils.config import Config
from src.utils.database import query_to_df

logger = logging.getLogger(__name__)

COHORT_QUERY = """
    SELECT subject_id, admittime, anchor_time, sepsis_label
    FROM mimiciv_ext.cohort
"""


class SepsisCohortLabeler(femr.labelers.Labeler):
    """Assign sepsis labels using cohort admission times as prediction points."""

    def __init__(self, cohort_df):
        super().__init__()
        self.cohort_dict = cohort_df.set_index(["subject_id", "admittime"]).to_dict(
            orient="index"
        )

    def label(self, subject):
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


def main():
    """Generate labels.parquet for MOTOR embedding extraction."""
    Config.setup_logging()

    logger.info("Querying cohort from database")
    cohort_df = query_to_df(COHORT_QUERY)
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

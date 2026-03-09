"""
Extracts and processes tabular metadata for the MIMIC-IV-ECG v1.0 dataset.

Merges patient demographics, cardiac marker lab events, and
machine-generated ECG reports, while handling MIMIC-IV privacy-preserving
temporal offsets (anchor years) to align timelines.

Outputs are saved as CSV files for downstream ECG waveform processing.

Usage:
    python -m src.data.preprocess.ecg.records
"""

import logging
import pandas as pd

from src.utils.config import Config
from fairseq_signals_scripts.preprocess.ecg.preprocess import FEMALE_VALUE, MALE_VALUE
from src.utils.database import query_to_df

logger = logging.getLogger(__name__)


class MockArgs:
    """A simple class to mimic the structure of argparse results."""

    def __init__(self):
        self.raw_root = Config.RAW_ECG_DIR
        self.processed_root = Config.PROCESSED_ECG_ROOT_DIR
        self.mimic_iv_root = Config.RAW_EHR_COHORT_DIR


def process_patients(path: str) -> pd.DataFrame:
    """
    Processes the patient dataset and performs data manipulations.
    Handles MIMIC-IV date shifting to align timelines.
    """
    logger.info("Loading patients data from: %s", path)
    patients = pd.read_csv(path)
    logger.info("Initial patient count: %d", len(patients))

    # Map gender chars to numeric constants defined in fairseq_signals
    patients.rename({"gender": "sex"}, axis=1, inplace=True)
    patients["sex"] = (
        patients["sex"].map({"F": FEMALE_VALUE, "M": MALE_VALUE}).astype("Int64")
    )

    # MIMIC-IV Privacy Logic Validation
    assert (
        (patients["anchor_year_group"].str.slice(stop=4).astype(int) + 2)
        == patients["anchor_year_group"].str.slice(start=-4).astype(int)
    ).all()

    # Calculate offsets
    patients["anchor_year_group_middle"] = (
        patients["anchor_year_group"].str.slice(stop=4).astype(int) + 1
    )
    patients["anchor_year_offset"] = (
        patients["anchor_year_group_middle"] - patients["anchor_year"]
    )
    patients["anchor_day_offset"] = (patients["anchor_year_offset"] * 365.25).astype(
        "timedelta64[D]"
    )

    # Apply offset to Date of Death (dod)
    patients["dod_anchored"] = (
        pd.to_datetime(patients["dod"]) + patients["anchor_day_offset"]
    )

    # Clean up and rename
    patients.drop(
        ["anchor_year", "anchor_year_group", "anchor_year_offset", "dod"],
        axis=1,
        inplace=True,
    )
    patients.rename(
        {"anchor_year_group_middle": "anchor_year", "dod_anchored": "dod"},
        axis=1,
        inplace=True,
    )

    # Anchor year datetime
    patients["anchor_year_dt"] = pd.to_datetime(
        {"year": patients["anchor_year"], "month": 1, "day": 1}
    )

    logger.info("Patient data processing complete.")
    return patients


def process_cardiac_markers(path: str) -> pd.DataFrame:
    """
    Processes the cardiac marker data from a CSV file.
    Filters for Troponin T and CK-MB in chunks to prevent OOM.
    """
    logger.info("Processing cardiac markers from: %s", path)

    cardiac_marker_chunks = []
    chunk_count = 0

    for chunk in pd.read_csv(path, chunksize=1e5, low_memory=False):
        filtered_chunk = chunk[chunk["itemid"].isin([51003, 50911])]
        if not filtered_chunk.empty:
            cardiac_marker_chunks.append(filtered_chunk)

        chunk_count += 1
        if chunk_count % 50 == 0:
            logger.debug("Processed %d chunks...", chunk_count)

    if cardiac_marker_chunks:
        cardiac_markers = pd.concat(cardiac_marker_chunks)
        cardiac_markers["label"] = cardiac_markers["itemid"].map(
            {51003: "Troponin T", 50911: "Creatine Kinase, MB Isoenzyme"}
        )
        logger.info("Cardiac markers extracted. Total rows: %d", len(cardiac_markers))
    else:
        logger.warning("No matching cardiac markers found.")
        cardiac_markers = pd.DataFrame()

    return cardiac_markers


def main():
    Config.setup_logging()
    logger.info("Starting MIMIC-IV-ECG Data Extraction")

    args = MockArgs()
    records = query_to_df("SELECT * FROM mimiciv_ecg.record_list")
    logger.info("Raw records found: %d", len(records))

    logger.info("Fetching valid cohort from PostgreSQL...")
    cohort_query = "SELECT subject_id, study_id FROM mimiciv_ext.cohort_ecg"
    cohort_df = query_to_df(cohort_query)

    records = records.merge(cohort_df, how="inner", on=["subject_id", "study_id"])
    logger.info("Records remaining after cohort filtering: %d", len(records))

    results = {}

    if args.mimic_iv_root:
        patients_path = args.mimic_iv_root / "hosp" / "patients.csv.gz"
        results["patients"] = process_patients(patients_path)

        logger.info("Merging ECG records with patient data...")
        records = records.merge(results["patients"], how="left", on="subject_id")

        logger.info("Adjusting ECG timestamps using anchor offsets...")
        records["ecg_time"] = (
            pd.to_datetime(records["ecg_time"]) + records["anchor_day_offset"]
        )
        records["age"] = (
            records["anchor_age"]
            + (records["ecg_time"] - records["anchor_year_dt"]).dt.days / 365.25
        )

        lab_events_path = args.mimic_iv_root / "hosp" / "labevents.csv.gz"
        results["cardiac_markers"] = process_cardiac_markers(lab_events_path)

    logger.info("Processing machine measurements and aggregating reports...")
    measurements = query_to_df("SELECT * FROM mimiciv_ecg.machine_measurements")

    report_cols = measurements.columns[measurements.columns.str.startswith("report_")]
    measurements["machine_report"] = (
        measurements[report_cols]
        .fillna("")
        .agg("; ".join, axis=1)
        .str.replace(r"(;\s*)+", "; ", regex=True)
        .str.strip("; ")
    )

    # Keep only the keys and the new aggregated column
    measurements = measurements[["subject_id", "study_id", "machine_report"]]

    # Safely merge into records
    records = records.merge(measurements, how="left", on=["subject_id", "study_id"])
    results["records"] = records

    logger.info("Saving processed files to: %s", args.processed_root)
    args.processed_root.mkdir(parents=True, exist_ok=True)

    for filename, data in results.items():
        save_path = args.processed_root / f"{filename}.csv"
        logger.info("Saving %s.csv (%d rows)...", filename, len(data))
        data.to_csv(save_path, index=False)

    logger.info("Done!")


if __name__ == "__main__":
    main()

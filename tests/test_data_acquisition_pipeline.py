"""
Run:
pytest tests/test_data_acquisition_pipeline.py -v
"""

import pytest
from pathlib import Path
from sqlalchemy import text
from src.utils.config import Config

import zipfile
from PIL import Image
import wfdb
import pandas as pd

# --- EXPECTED DETERMINISTIC COUNTS ---
EXPECTED_TOTAL_PATIENTS = 15513
EXPECTED_CXR_STUDIES = 2212
EXPECTED_ECG_STUDIES = 7146

EXPECTED_POSITIVES = 8436

# --- FIXTURES ---


@pytest.fixture(scope="module")
def cohort_stats(db_engine):
    """Fetch basic stats once to use in multiple tests."""
    with db_engine.connect() as conn:
        result = conn.execute(
            text("""
            SELECT 
                COUNT(*) as total,
                SUM(sepsis_label) as positives
            FROM mimiciv_ext.cohort
            """)
        ).fetchone()
    return result


# --- DATABASE INTEGRITY TESTS ---


def test_db_connection(db_engine):
    """Can we connect to the database?"""
    with db_engine.connect() as conn:
        result = conn.execute(text("SELECT 1")).scalar()
    assert result == 1


def test_final_cohort_exists_and_has_data(cohort_stats):
    """
    Does the final cohort table have the exact deterministic number of patients?
    """
    total = cohort_stats.total
    assert total == EXPECTED_TOTAL_PATIENTS, (
        f"Expected exactly {EXPECTED_TOTAL_PATIENTS} patients, got {total}."
    )


def test_class_balance(cohort_stats):
    """
    Is the exact number of positive sepsis cases correct?
    """
    positives = cohort_stats.positives
    assert positives == EXPECTED_POSITIVES, (
        f"Expected exactly {EXPECTED_POSITIVES} positive cases, got {positives}."
    )


def test_one_admission_per_patient(db_engine):
    """
    Verify strictly that there is only one admission per patient.
    """
    with db_engine.connect() as conn:
        result = conn.execute(
            text("""
            SELECT subject_id, COUNT(hadm_id) 
            FROM mimiciv_ext.cohort 
            GROUP BY subject_id 
            HAVING COUNT(hadm_id) > 1
            """)
        ).fetchall()
    assert len(result) == 0, f"Found {len(result)} patients with multiple admissions!"


# --- MODALITY TESTS ---


def test_cxr_cohort_integrity(db_engine):
    """
    Verify exact CXR counts, uniqueness, and no orphans.
    """
    with db_engine.connect() as conn:
        total_cxr = conn.execute(
            text("SELECT COUNT(*) FROM mimiciv_ext.cohort_cxr")
        ).scalar()
        assert total_cxr == EXPECTED_CXR_STUDIES, (
            f"Expected {EXPECTED_CXR_STUDIES} CXRs, got {total_cxr}."
        )

        orphans = conn.execute(
            text("""
            SELECT COUNT(*) FROM mimiciv_ext.cohort_cxr c
            LEFT JOIN mimiciv_ext.cohort p ON c.subject_id = p.subject_id
            WHERE p.subject_id IS NULL
            """)
        ).scalar()
        assert orphans == 0, "Found CXR records linked to non-existent patients!"

        duplicates = conn.execute(
            text("""
            SELECT subject_id FROM mimiciv_ext.cohort_cxr
            GROUP BY subject_id HAVING COUNT(*) > 1
            """)
        ).fetchall()
        assert len(duplicates) == 0, "Found patients with multiple CXR entries!"


def test_ecg_cohort_integrity(db_engine):
    """
    Verify exact ECG counts and no orphans.
    """
    with db_engine.connect() as conn:
        total_ecg = conn.execute(
            text("SELECT COUNT(*) FROM mimiciv_ext.cohort_ecg")
        ).scalar()
        assert total_ecg == EXPECTED_ECG_STUDIES, (
            f"Expected {EXPECTED_ECG_STUDIES} ECGs, got {total_ecg}."
        )

        orphans = conn.execute(
            text("""
            SELECT COUNT(*) FROM mimiciv_ext.cohort_ecg e
            LEFT JOIN mimiciv_ext.cohort p ON e.subject_id = p.subject_id
            WHERE p.subject_id IS NULL
            """)
        ).scalar()
        assert orphans == 0, "Found ECG records linked to non-existent patients!"


# --- FILE SYSTEM TESTS ---


@pytest.fixture(scope="module")
def cxr_paths(db_engine):
    with db_engine.connect() as conn:
        rows = conn.execute(
            text("SELECT study_path FROM mimiciv_ext.cohort_cxr")
        ).fetchall()
    return [row[0] for row in rows]


@pytest.fixture(scope="module")
def ecg_paths(db_engine):
    with db_engine.connect() as conn:
        rows = conn.execute(
            text("SELECT study_path FROM mimiciv_ext.cohort_ecg")
        ).fetchall()
    return [row[0] for row in rows]


def test_cxr_dir_exists():
    assert Config.RAW_CXR_IMG_DIR.is_dir(), (
        f"CXR directory not found: {Config.RAW_CXR_IMG_DIR}"
    )


def test_ecg_dir_exists():
    assert Config.RAW_ECG_DIR.is_dir(), f"ECG directory not found: {Config.RAW_ECG_DIR}"


def test_cxr_files_on_disk(cxr_paths):
    assert len(cxr_paths) == EXPECTED_CXR_STUDIES
    missing, empty = [], []
    for db_path in cxr_paths:
        full_path = Config.RAW_CXR_IMG_DIR / db_path
        if not full_path.exists():
            missing.append(str(full_path))
        elif full_path.stat().st_size == 0:
            empty.append(str(full_path))
    assert len(missing) == 0, f"{len(missing)} missing: {missing[:3]}..."
    assert len(empty) == 0, f"{len(empty)} are 0-byte: {empty[:3]}..."


def test_ecg_files_on_disk(ecg_paths):
    assert len(ecg_paths) == EXPECTED_ECG_STUDIES
    missing, empty = [], []
    for db_path in ecg_paths:
        for ext in (".hea", ".dat"):
            fpath = Config.RAW_ECG_DIR / f"{db_path}{ext}"
            if not fpath.exists():
                missing.append(str(fpath))
            elif fpath.stat().st_size == 0:
                empty.append(str(fpath))
    assert len(missing) == 0, f"{len(missing)} missing: {missing[:3]}..."
    assert len(empty) == 0, f"{len(empty)} are 0-byte: {empty[:3]}..."


def test_cxr_files_not_corrupted(cxr_paths):
    corrupted = []
    for db_path in cxr_paths:
        full_path = Config.RAW_CXR_IMG_DIR / db_path
        if full_path.exists() and full_path.stat().st_size > 0:
            try:
                with Image.open(full_path) as img:
                    img.verify()
            except Exception as e:
                corrupted.append((str(full_path), str(e)))
    assert len(corrupted) == 0, f"Found {len(corrupted)} corrupted CXR files."


def test_ecg_files_not_corrupted(ecg_paths):
    corrupted = []
    for db_path in ecg_paths:
        record_path = Config.RAW_ECG_DIR / db_path
        hea_path = record_path.with_suffix(".hea")
        dat_path = record_path.with_suffix(".dat")
        if (
            hea_path.exists()
            and dat_path.exists()
            and hea_path.stat().st_size > 0
            and dat_path.stat().st_size > 0
        ):
            try:
                wfdb.rdrecord(str(record_path))
            except Exception as e:
                corrupted.append((str(record_path), str(e)))
    assert len(corrupted) == 0, f"Found {len(corrupted)} corrupted ECG records."


def test_cxr_reports_zip_not_corrupted():
    reports_path = Config.RAW_CXR_TXT_DIR / Config.CXR_REPORTS_FILE
    if not reports_path.exists():
        pytest.skip("CXR reports file not found — skipping corruption check.")
    assert zipfile.is_zipfile(reports_path), f"{reports_path} is not a valid ZIP file"
    with zipfile.ZipFile(reports_path, "r") as zf:
        bad_file = zf.testzip()
        assert bad_file is None, f"Corrupted entry in CXR reports ZIP: {bad_file}"


# --- EXTRACT COHORT SPLITS TESTS ---


def test_extract_cohort_splits_creates_file(db_engine, tmp_path):
    """
    Test that extract_cohort_splits creates the master cohort file.
    """
    from src.data.acquisition.extract_cohort_splits import create_master_cohort

    # Override the output path to a temporary directory
    original_path = Config.PROCESSED_COHORT_PARQUET_FILE
    temp_output_path = tmp_path / "master_cohort.parquet"
    Config.PROCESSED_COHORT_PARQUET_FILE = temp_output_path

    try:
        create_master_cohort()

        # Check if the file is created
        assert temp_output_path.exists(), "Master cohort file was not created."

        # Validate the file content
        df = pd.read_parquet(temp_output_path)
        assert not df.empty, "Master cohort file is empty."
        assert set(df.columns) >= {"subject_id", "dataset_split", "sepsis_label"}, (
            "Master cohort file is missing required columns."
        )
    finally:
        # Restore the original path
        Config.PROCESSED_COHORT_PARQUET_FILE = original_path


def test_extract_cohort_splits_split_distribution(db_engine, tmp_path):
    """
    Test that the dataset splits are correctly standardized.
    """
    from src.data.acquisition.extract_cohort_splits import create_master_cohort

    # Override the output path to a temporary directory
    original_path = Config.PROCESSED_COHORT_PARQUET_FILE
    temp_output_path = tmp_path / "master_cohort.parquet"
    Config.PROCESSED_COHORT_PARQUET_FILE = temp_output_path

    try:
        create_master_cohort()

        # Load the generated file
        df = pd.read_parquet(temp_output_path)

        # Check split distribution
        splits = df["dataset_split"].value_counts()
        assert "train" in splits, "Train split is missing."
        assert "valid" in splits, "Validation split is missing."
        assert "test" in splits, "Test split is missing."
    finally:
        # Restore the original path
        Config.PROCESSED_COHORT_PARQUET_FILE = original_path

"""Tests for EHR data export integrity.

Validates that exported CSV files match database expectations:
- Cohort export contains ONLY cohort subjects (no leakage).
- Pretraining export excludes ALL test-split subjects (no leakage).
- Row counts match between DB and exported files.
"""

import pandas as pd
import pytest
from sqlalchemy import text

from src.utils.config import Config

# --- Constants ---

ADMISSIONS_TABLE = "mimiciv_hosp.admissions"
COHORT_TABLE = "mimiciv_ext.cohort"
SPLIT_TABLE = "mimiciv_ext.dataset_splits"
CHUNK_SIZE = 100_000


# --- Fixtures ---


@pytest.fixture(scope="module")
def cohort_subject_ids(db_engine):
    """Fetch the set of subject_ids in the cohort."""
    with db_engine.connect() as conn:
        rows = conn.execute(text(f"SELECT subject_id FROM {COHORT_TABLE}")).fetchall()
    return {row[0] for row in rows}


@pytest.fixture(scope="module")
def test_split_subject_ids(db_engine):
    """Fetch the set of subject_ids in the test split."""
    with db_engine.connect() as conn:
        rows = conn.execute(
            text(f"SELECT subject_id FROM {SPLIT_TABLE} WHERE dataset_split = 'test'")
        ).fetchall()
    return {row[0] for row in rows}


# --- Cohort Export Tests ---


class TestCohortExport:
    """Verify the cohort export contains only cohort subjects."""

    FILE_PATH = Config.RAW_EHR_COHORT_DIR / "hosp" / "admissions.csv.gz"

    @pytest.fixture(scope="class")
    def scan_results(self, cohort_subject_ids):
        """Scan the exported CSV once; return row count and non-cohort count."""
        if not self.FILE_PATH.is_file():
            pytest.skip(f"Export file not found: {self.FILE_PATH}")

        row_count = 0
        non_cohort_count = 0

        for chunk in pd.read_csv(
            self.FILE_PATH, compression="gzip", chunksize=CHUNK_SIZE
        ):
            row_count += len(chunk)
            assert "subject_id" in chunk.columns, "CSV missing 'subject_id' column"
            non_cohort_count += len(
                chunk[~chunk["subject_id"].isin(cohort_subject_ids)]
            )

        return {"row_count": row_count, "non_cohort_count": non_cohort_count}

    def test_no_non_cohort_subjects(self, scan_results):
        """All rows in the export must belong to cohort subjects."""
        assert scan_results["non_cohort_count"] == 0, (
            f"Found {scan_results['non_cohort_count']} rows from "
            f"non-cohort subjects in cohort export"
        )

    def test_row_count_matches_db(self, db_engine, scan_results):
        """Exported row count must match the DB query result."""
        with db_engine.connect() as conn:
            expected = conn.execute(
                text(
                    f"SELECT count(*) FROM {ADMISSIONS_TABLE} t "
                    f"WHERE EXISTS ("
                    f"  SELECT 1 FROM {COHORT_TABLE} c "
                    f"  WHERE c.subject_id = t.subject_id"
                    f")"
                )
            ).scalar()

        actual = scan_results["row_count"]
        assert actual == expected, (
            f"Row count mismatch: expected {expected}, got {actual} "
            f"(diff={abs(actual - expected)})"
        )


# --- Pretraining Export Tests ---


class TestPretrainingExport:
    """Verify the pretraining export excludes all test-split subjects."""

    FILE_PATH = Config.RAW_EHR_PRETRAINING_DIR / "hosp" / "admissions.csv.gz"

    @pytest.fixture(scope="class")
    def scan_results(self, test_split_subject_ids):
        """Scan the exported CSV once; return row count and leakage count."""
        if not self.FILE_PATH.is_file():
            pytest.skip(f"Export file not found: {self.FILE_PATH}")

        row_count = 0
        leakage_count = 0

        for chunk in pd.read_csv(
            self.FILE_PATH, compression="gzip", chunksize=CHUNK_SIZE
        ):
            row_count += len(chunk)
            assert "subject_id" in chunk.columns, "CSV missing 'subject_id' column"
            leakage_count += len(
                chunk[chunk["subject_id"].isin(test_split_subject_ids)]
            )

        return {"row_count": row_count, "leakage_count": leakage_count}

    def test_no_test_split_leakage(self, scan_results):
        """No test-split subjects may appear in the pretraining export."""
        assert scan_results["leakage_count"] == 0, (
            f"DATA LEAKAGE: Found {scan_results['leakage_count']} rows "
            f"from test-split subjects in pretraining export"
        )

    def test_row_count_matches_db(self, db_engine, scan_results):
        """Exported row count must match the DB query result."""
        with db_engine.connect() as conn:
            expected = conn.execute(
                text(
                    f"SELECT count(*) FROM {ADMISSIONS_TABLE} t "
                    f"WHERE NOT EXISTS ("
                    f"  SELECT 1 FROM {SPLIT_TABLE} s "
                    f"  WHERE s.subject_id = t.subject_id "
                    f"  AND s.dataset_split = 'test'"
                    f")"
                )
            ).scalar()

        actual = scan_results["row_count"]
        assert actual == expected, (
            f"Row count mismatch: expected {expected}, got {actual} "
            f"(diff={abs(actual - expected)})"
        )

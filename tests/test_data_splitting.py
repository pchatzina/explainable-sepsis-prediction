import pytest
from sqlalchemy import text

# --- EXPECTED DETERMINISTIC COUNTS ---
EXPECTED_TOTAL_PATIENTS = 15513
EXPECTED_TRAIN_COUNT = 10845
EXPECTED_VAL_COUNT = 2326
EXPECTED_TEST_COUNT = 2342


def test_split_completeness(db_engine):
    """
    Verify that EVERY patient in the cohort has been assigned a split,
    matching our exact expected total.
    """
    with db_engine.connect() as conn:
        total_cohort = conn.execute(
            text("SELECT COUNT(*) FROM mimiciv_ext.cohort")
        ).scalar()
        total_splits = conn.execute(
            text("SELECT COUNT(*) FROM mimiciv_ext.dataset_splits")
        ).scalar()

    assert total_cohort == EXPECTED_TOTAL_PATIENTS, (
        f"Cohort size mismatch: {total_cohort}"
    )
    assert total_splits == EXPECTED_TOTAL_PATIENTS, (
        f"Split table size mismatch: {total_splits}"
    )
    assert total_cohort == total_splits, (
        "Mismatch between cohort size and assigned splits!"
    )


def test_no_data_leakage(db_engine):
    """
    CRITICAL: Verify no patient appears in multiple splits.
    """
    with db_engine.connect() as conn:
        duplicates = conn.execute(
            text("""
            SELECT subject_id, COUNT(DISTINCT dataset_split) 
            FROM mimiciv_ext.dataset_splits 
            GROUP BY subject_id 
            HAVING COUNT(DISTINCT dataset_split) > 1
            """)
        ).fetchall()

    assert len(duplicates) == 0, (
        f"DATA LEAKAGE DETECTED: {len(duplicates)} patients are in multiple splits!"
    )


def test_stratification_ratios_exact(db_engine):
    """
    Check if the splits exactly match our deterministic target counts.
    """
    with db_engine.connect() as conn:
        stats = conn.execute(
            text("""
            SELECT dataset_split, COUNT(*) 
            FROM mimiciv_ext.dataset_splits 
            GROUP BY dataset_split
            """)
        ).fetchall()

    counts = {row[0]: row[1] for row in stats}

    train_count = counts.get("train", 0)
    val_count = counts.get("validate", 0)
    test_count = counts.get("test", 0)

    assert train_count == EXPECTED_TRAIN_COUNT, (
        f"Expected {EXPECTED_TRAIN_COUNT} train cases, got {train_count}"
    )
    assert val_count == EXPECTED_VAL_COUNT, (
        f"Expected {EXPECTED_VAL_COUNT} validate cases, got {val_count}"
    )
    assert test_count == EXPECTED_TEST_COUNT, (
        f"Expected {EXPECTED_TEST_COUNT} test cases, got {test_count}"
    )


def test_cxr_freshness_tier_exists_and_valid(db_engine):
    """
    Verify the newly added CXR freshness tier contains no nulls and only expected values.
    """
    with db_engine.connect() as conn:
        null_tiers = conn.execute(
            text(
                "SELECT COUNT(*) FROM mimiciv_ext.dataset_splits WHERE cxr_freshness_tier IS NULL"
            )
        ).scalar()

        invalid_tiers = conn.execute(
            text("""
            SELECT COUNT(*) FROM mimiciv_ext.dataset_splits 
            WHERE cxr_freshness_tier NOT IN ('Fresh (0-12h)', 'Recent (12-24h)', 'Older (>24h)', 'N/A')
            """)
        ).scalar()

    assert null_tiers == 0, "Found NULL values in cxr_freshness_tier!"
    assert invalid_tiers == 0, "Found unrecognized values in cxr_freshness_tier!"


def test_ecg_freshness_tier_exists_and_valid(db_engine):
    """
    Verify the newly added ECG freshness tier contains no nulls and only expected values.
    """
    with db_engine.connect() as conn:
        null_tiers = conn.execute(
            text(
                "SELECT COUNT(*) FROM mimiciv_ext.dataset_splits WHERE ecg_freshness_tier IS NULL"
            )
        ).scalar()

        invalid_tiers = conn.execute(
            text("""
            SELECT COUNT(*) FROM mimiciv_ext.dataset_splits 
            WHERE ecg_freshness_tier NOT IN ('Fresh (0-12h)', 'Recent (12-24h)', 'Older (>24h)', 'N/A')
            """)
        ).scalar()

    assert null_tiers == 0, "Found NULL values in ecg_freshness_tier!"
    assert invalid_tiers == 0, "Found unrecognized values in ecg_freshness_tier!"

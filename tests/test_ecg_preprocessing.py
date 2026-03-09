"""
Run:
pytest tests/test_ecg_preprocessing.py -v -s
(Added -s to see the informational print statements!)
"""

import pytest
import numpy as np
import scipy.io as sio
from pathlib import Path
from src.utils.config import Config

PREPROCESSED_DIR = Config.PROCESSED_ECG_ROOT_DIR / "preprocessed"


@pytest.fixture(scope="module")
def mat_files():
    """Collect all .mat files from the preprocessed ECG directory."""
    if not PREPROCESSED_DIR.exists():
        pytest.skip(f"Preprocessed ECG dir not found: {PREPROCESSED_DIR}")

    files = [f for f in PREPROCESSED_DIR.iterdir() if f.suffix == ".mat"]
    if not files:
        pytest.skip("No .mat files found in preprocessed ECG directory")

    return files


def test_preprocessed_dir_exists():
    """Verify the preprocessed ECG directory exists."""
    assert PREPROCESSED_DIR.exists(), (
        f"Preprocessed ECG directory not found: {PREPROCESSED_DIR}"
    )


def test_preprocessed_files_count(mat_files):
    """Verify the exact number of preprocessed ECG files matches the cohort (7146)."""
    count = len(mat_files)
    assert count == 7146, f"Expected 7146 preprocessed .mat files, found {count}"


def test_preprocessed_files_have_feats_key(mat_files):
    """Every .mat file should contain a 'feats' key. Checks ALL files."""
    missing_key = []
    for fpath in mat_files:
        try:
            mat = sio.loadmat(fpath)
            if "feats" not in mat:
                missing_key.append(fpath.name)
        except Exception as e:
            pytest.fail(f"Could not read {fpath.name}: {e}")

    assert len(missing_key) == 0, (
        f"{len(missing_key)} files missing 'feats' key: {missing_key[:10]}"
    )


def test_preprocessed_files_shape(mat_files):
    """Verify feats have expected shape: (12, num_samples). Checks ALL files."""
    bad_shape = []
    for fpath in mat_files:
        try:
            mat = sio.loadmat(fpath)
            if "feats" not in mat:
                continue
            data = mat["feats"]
            if data.shape[0] != 12:
                bad_shape.append((fpath.name, data.shape))
        except Exception as e:
            pytest.fail(f"Could not read {fpath.name}: {e}")

    assert len(bad_shape) == 0, (
        f"{len(bad_shape)} files with unexpected shape (expected 12 leads): {bad_shape[:10]}"
    )


def test_ecg_preprocessed_files_validity(mat_files):
    """
    Verify preprocessed ECG .mat files are not completely corrupted.
    It is normal for clinical ECGs to have missing/flat leads (NaNs),
    but the entire 12-lead recording should NOT be 100% NaN.
    """
    totally_blank_files = []
    files_with_nans_count = 0

    for fpath in mat_files:
        try:
            mat = sio.loadmat(fpath)
            if "feats" not in mat:
                continue
            data = mat["feats"]

            # Check if there are ANY NaNs in the file
            if np.isnan(data).any():
                files_with_nans_count += 1

                # Check if ALL values across ALL leads are NaN (fatal error)
                if np.isnan(data).all():
                    totally_blank_files.append(fpath.name)

        except Exception as e:
            pytest.fail(f"Could not read {fpath.name}: {e}")

    print(
        f"\n[INFO] Found {files_with_nans_count}/{len(mat_files)} files with partial NaNs (Expected clinical behavior)."
    )

    assert len(totally_blank_files) == 0, (
        f"Found {len(totally_blank_files)} completely broken files (100% NaN): "
        f"{totally_blank_files[:10]}"
    )

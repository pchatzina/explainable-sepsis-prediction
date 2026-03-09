"""
Custom MEDS Reader Verification Script

This script verifies that a MEDS Reader database matches the source MEDS Parquet dataset,
with added normalization for handling empty strings ('') vs. None in string properties
(e.g., 'text_value'). It replicates the logic of `meds_reader_verify` but applies
normalization during comparison to account for minor representation differences that
do not affect data integrity.

Usage:
    python custom_meds_reader_verify.py <meds_dataset> <meds_reader_database>

Arguments:
    meds_dataset (str): Path to the MEDS Parquet dataset directory.
    meds_reader_database (str): Path to the MEDS Reader database directory.

The script randomly selects one Parquet shard from the dataset, loads its events,
groups them by subject_id, and compares them against the corresponding subjects in
the reader database. String properties with empty strings are treated as equivalent
to None during assertion.

If the verification passes, it prints "Custom test passed!". Otherwise, it raises
an AssertionError with details on the mismatch.

Dependencies:
    - meds_reader (for SubjectDatabase)
    - pyarrow (for Parquet reading)
    - Other standard libraries: argparse, collections, glob, os, random, sys

Note:
    This is a modified version of the original meds_reader_verify to handle
    specific edge cases in data representation. For production use, consider
    upstream fixes or data preprocessing.
"""

from __future__ import annotations

import argparse
import collections
import glob
import os
import random
import sys
import warnings

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from meds_reader import SubjectDatabase  # Import from installed package


def custom_assert_same(pyarrow_subject, reader_subject, properties):
    assert len(pyarrow_subject) == len(reader_subject.events), (
        f"{len(pyarrow_subject)} {len(reader_subject.events)}"
    )
    for pyarrow_event, reader_event in zip(pyarrow_subject, reader_subject.events):
        for prop in properties:
            actual = getattr(reader_event, prop)
            expected = pyarrow_event[prop]

            # Normalize empty strings vs None for string properties
            if isinstance(expected, str) and expected == "":
                expected = None
            if isinstance(actual, str) and actual == "":
                actual = None

            assert actual == expected, (
                f"Got {actual} expected {expected} for {reader_subject} {prop}"
                f" {pyarrow_event['time']} {reader_event.time}"
            )


def main():
    parser = argparse.ArgumentParser(
        description="Custom verify that a meds_reader dataset matches a source dataset with normalization"
    )
    parser.add_argument(
        "meds_dataset", type=str, help="A MEDS dataset to compare against"
    )
    parser.add_argument(
        "meds_reader_database", type=str, help="A meds_reader database to verify"
    )

    args = parser.parse_args()

    database = SubjectDatabase(args.meds_reader_database)

    random.seed(3452342)  # Same seed as original for reproducibility

    files = sorted(
        glob.glob(
            os.path.join(args.meds_dataset, "data", "**", "*.parquet"), recursive=True
        )
    )

    if not files:
        print("No Parquet files found in the dataset.")
        sys.exit(1)

    file = random.choice(files)
    reference = pq.ParquetFile(file)

    row_group = reference.read()

    custom_fields = sorted(set(row_group.schema.names) - {"subject_id"})
    all_properties = {k: row_group.schema.field(k).type for k in custom_fields}

    missing = set(all_properties) - set(database.properties)
    extra = set(database.properties) - set(all_properties)

    assert len(missing) == 0, f"Had missing properties {missing}"
    assert len(extra) == 0, f"Had extra properties {extra}"

    assert all_properties == database.properties

    python_objects = row_group.to_pylist()

    subject_objects = collections.defaultdict(list)

    for obj in python_objects:
        subject_id = obj["subject_id"]
        del obj["subject_id"]
        subject_objects[subject_id].append(obj)

    for subject_id, pyarrow_subject in subject_objects.items():
        reader_subject = database[subject_id]

        custom_assert_same(pyarrow_subject, reader_subject, database.properties)

    print("Custom test passed!")


if __name__ == "__main__":
    main()

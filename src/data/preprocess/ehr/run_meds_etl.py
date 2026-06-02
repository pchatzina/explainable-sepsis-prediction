"""
Module: run_meds_etl.py
========================
Purpose: CLI wrapper for MIMIC-IV EHR export and MEDS (Medical Event Data Standard) ETL pipeline.

Architecture Overview:
  This module orchestrates 4 sequential stages of EHR processing:
  1. export_cohort()      → Query PostgreSQL → export raw EHR CSV
  2. export_pretraining() → Query PostgreSQL → export raw EHR CSV
  3. run_pipeline("cohort")      → CSV → MEDS Format
  4. run_pipeline("pretraining") → CSV → MEDS Format

Key Design Decisions:
  - Two separate datasets: "cohort" (15,513 train+val+test) and "pretraining" (all MIMIC-IV patients except cohort test split)
    Rationale: Cohort is for downstream prediction task; pretraining is for MOTOR foundation model pretraining.
  - MEDS standardization: Raw MIMIC codes → standardized MEDS codes
    Rationale: Reproduces MOTOR pretraining dataset format + enables cross-institutional transfer

CLI Commands:
  python -m src.data.preprocess.ehr.run_meds_etl export-cohort
  python -m src.data.preprocess.ehr.run_meds_etl export-pretraining
  python -m src.data.preprocess.ehr.run_meds_etl meds-pipeline cohort
  python -m src.data.preprocess.ehr.run_meds_etl meds-pipeline pretraining
"""

import argparse
import logging
import subprocess
import os
import sys
from pathlib import Path

from src.utils.config import Config

logger = logging.getLogger(__name__)


def run_script(
    script_path: str,
    script_args: list[str] | None = None,
    env_vars: dict[str, str] | None = None,
) -> None:
    """
    Run a bash script with specific environment variables.

    Args:
        script_path: Absolute path to bash script to execute
        script_args: Optional list of command-line arguments (appended to bash command)
        env_vars: Optional dict of environment variables to inject/override
    """
    script = Path(script_path)
    if not script.is_file():
        logger.error("Script not found: %s", script)
        sys.exit(1)

    # Merge current environment with custom config
    full_env = os.environ.copy()
    if env_vars:
        full_env.update(env_vars)

    # Build command
    cmd = ["bash", str(script)]
    if script_args:
        cmd.extend(script_args)

    logger.info("Running: %s with args %s", script, script_args)

    try:
        subprocess.run(cmd, check=True, env=full_env)
        logger.info("Success: %s", script)
    except subprocess.CalledProcessError as e:
        logger.error("Error running %s. Exit code: %d", script, e.returncode)
        sys.exit(e.returncode)


def export_cohort() -> None:
    """Invoke export_cohort_data.sh to extract raw EHR CSVs for the 15,513 cohort patients."""
    env = {
        "BASE_OUTPUT_DIR": str(Config.RAW_EHR_COHORT_DIR),
        "DB": Config.DB_NAME,
    }
    run_script(str(Config.SRC_EHR_EXPORTS_DIR / "export_cohort_data.sh"), env_vars=env)


def export_pretraining() -> None:
    """Invoke export_pretraining_data.sh to extract raw EHR CSVs for all MIMIC-IV patients except the test split."""
    env = {
        "BASE_OUTPUT_DIR": str(Config.RAW_EHR_PRETRAINING_DIR),
        "DB": Config.DB_NAME,
    }
    run_script(
        str(Config.SRC_EHR_EXPORTS_DIR / "export_pretraining_data.sh"), env_vars=env
    )


def run_pipeline(dataset: str) -> None:
    """
    Run the full MEDS ETL pipeline for the specified dataset.

    Purpose:
        Transform raw CSV → standardized MEDS format.

    Args:
        dataset: "cohort" or "pretraining"

    Data Flow:
        Raw CSV (from export_cohort/export_pretraining)
        → mimic_to_meds.sh (bash script invokes MEDS library)
    """
    dataset_config = {
        "pretraining": (
            Config.RAW_EHR_PRETRAINING_DIR,
            Config.PROCESSED_EHR_PRETRAINING_DIR,
        ),
        "cohort": (Config.RAW_EHR_COHORT_DIR, Config.PROCESSED_EHR_COHORT_DIR),
    }

    if dataset not in dataset_config:
        logger.error(
            "Unknown dataset: %s. Must be one of %s.", dataset, list(dataset_config)
        )
        sys.exit(1)

    raw_base, processed_base = dataset_config[dataset]

    env = {
        "RAW_BASE": str(raw_base.parent),
        "PROCESSED_BASE": str(processed_base),
    }
    run_script(
        str(Config.EHR_TRANSFORMATIONS_SRC_DIR / "mimic_to_meds.sh"),
        script_args=[dataset],
        env_vars=env,
    )


def main() -> None:
    """
    CLI entry point for Data Processing.

    Commands:
        export-cohort: Stage 1 (extract raw EHR for cohort)
        export-pretraining: Stage 2 (extract raw EHR for pretraining)
        meds-pipeline: Stage 3-4 (transform CSV→MEDS format)
    """
    Config.setup_logging()

    parser = argparse.ArgumentParser(
        description="CLI Wrapper for Project Pipelines",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Command: export-cohort
    cohort_parser = subparsers.add_parser("export-cohort", help="Export cohort data")
    cohort_parser.set_defaults(func=lambda args: export_cohort())

    # Command: export-pretraining
    pretraining_parser = subparsers.add_parser(
        "export-pretraining",
        help="Export pretraining data (excludes test split)",
    )
    pretraining_parser.set_defaults(func=lambda args: export_pretraining())

    # Command: meds-pipeline
    meds_parser = subparsers.add_parser("meds-pipeline", help="Run MEDS ETL pipeline")
    meds_parser.add_argument(
        "dataset",
        choices=["pretraining", "cohort"],
        help="Name of the dataset to process",
    )
    meds_parser.set_defaults(func=lambda args: run_pipeline(args.dataset))

    args = parser.parse_args()

    # Execute the function associated with the parsed subparser
    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()

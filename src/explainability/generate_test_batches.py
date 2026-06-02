"""
Generates tokenized FEMR batches for the test split, strictly truncated
at the Sepsis anchor times.

The resulting batches are hierarchical token sequences ready for the frozen MOTOR
transformer and downstream Captum-based attribution analysis.

Usage:
    python -m src.explainability.generate_test_batches
"""

import logging
import pickle

import pandas as pd
import meds_reader
import femr.models.processor
import femr.models.tasks
import femr.models.tokenizer

from src.utils.config import Config

logger = logging.getLogger(__name__)


def main() -> None:
    """
    Main pipeline orchestrator for test batch generation.
    """
    Config.setup_logging()

    output_dir = Config.RESULTS_DIR / "explainability"
    output_dir.mkdir(parents=True, exist_ok=True)
    test_batches_path = output_dir / "test_batches"

    # 1. Get Test Subject IDs from the static Parquet Cohort
    logger.info("Loading test subject IDs from master cohort parquet...")
    cohort_df = pd.read_parquet(Config.PROCESSED_COHORT_PARQUET_FILE)
    test_ids_df = cohort_df[cohort_df["dataset_split"] == "test"]
    test_subject_ids = set(test_ids_df["subject_id"].astype(int).tolist())

    # 2. Load Labels to enforce Anchor Time truncation
    logger.info("Loading Sepsis Labels for Anchor Time truncation...")
    labels_df = pd.read_parquet(Config.PROCESSED_EHR_LABELS_DIR / "labels.parquet")
    labels_df = labels_df[labels_df["subject_id"].isin(test_subject_ids)]

    labels_df["prediction_time"] = pd.to_datetime(labels_df["prediction_time"]).apply(
        lambda x: x.to_pydatetime()
    )
    labels = labels_df.to_dict("records")

    # Create the LabeledSubjectTask to enforce the temporal truncation
    task = femr.models.tasks.LabeledSubjectTask(labels)

    # 3. Load Tokenizer (with ontology for hierarchical structure)
    prep_dir = Config.MODEL_EHR_MOTOR_PRETRAINING_FILES_DIR
    with open(prep_dir / "ontology.pkl", "rb") as f:
        ontology = pickle.load(f)

    tokenizer = femr.models.tokenizer.HierarchicalTokenizer.from_pretrained(
        Config.MODEL_EHR_MOTOR_WEIGHTS_DIR, ontology=ontology
    )

    # Initialize processor with the LabeledSubjectTask
    # This ensures batches are truncated at prediction_time
    processor = femr.models.processor.FEMRBatchProcessor(tokenizer, task=task)

    # 4. Load Database and Convert
    logger.info(f"Loading Cohort MEDS Database...")
    with meds_reader.SubjectDatabase(
        str(Config.PROCESSED_EHR_MEDS_COHORT_DIR)
    ) as database:
        db_subjects = set(database)
        valid_test_ids = [sid for sid in test_subject_ids if sid in db_subjects]

        logger.info(
            f"Converting {len(valid_test_ids)} matching test subjects to batches..."
        )
        test_database = database.filter(valid_test_ids)

        # convert_dataset handles hierarchical tokenization and batching
        test_batches = processor.convert_dataset(
            test_database, tokens_per_batch=8192, num_proc=4, min_subjects_per_batch=1
        )

        # Format as PyTorch tensors for efficient GPU operations
        test_batches.set_format("pt")
        test_batches.save_to_disk(test_batches_path)
        logger.info(
            f"Success! Saved strictly truncated test batches to {test_batches_path}"
        )


if __name__ == "__main__":
    main()

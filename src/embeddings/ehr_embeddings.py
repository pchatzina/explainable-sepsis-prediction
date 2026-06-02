"""Extract EHR embeddings using the frozen MOTOR foundation model via EHRExtractor.

Usage:
    python -m src.embeddings.ehr_embeddings
"""

import logging
import pickle

import meds_reader
import numpy as np
import pandas as pd
import torch

import femr.models.transformer

from src.embeddings.base_extractor import BaseEmbeddingExtractor
from src.utils.config import Config

logger = logging.getLogger(__name__)


class EHRExtractor(BaseEmbeddingExtractor):
    """Extracts EHR embeddings using the frozen MOTOR transformer via the femr library.

    MOTOR is a foundation model pretrained on MIMIC-IV EHR data.
    It converts sequences of medical events (medications, procedures, diagnoses) into fixed-size
    embeddings that capture longitudinal EHR patterns relevant to sepsis prediction.
    """

    def __init__(self):
        # Initialize the base class with the EHR-specific output directory
        super().__init__(output_dir=Config.PROCESSED_EHR_RAW_EMBEDDINGS_DIR)

    def extract_and_save(self) -> None:
        """Orchestrates end-to-end EHR embedding extraction via MOTOR.

        Workflow:
            1. Loads preprocessed EHR as a MEDS dataset (via meds_reader).
            2. Uses femr.models.transformer.compute_features() to run MOTOR on the cohort.
            3. Maps extracted features to the master cohort splits (train/valid/test).
            4. Saves per-modality embeddings following the canonical .pt format."""

        prep_dir = Config.MODEL_EHR_MOTOR_PRETRAINING_FILES_DIR

        # === Load Labels ===
        # Labels file is output from Phase 2.
        # It contains prediction times (at t=-6h anchor) and binary sepsis labels.
        labels_path = Config.PROCESSED_EHR_LABELS_DIR / "labels.parquet"
        if not labels_path.exists():
            logger.error("Labels file not found: %s", labels_path)
            return

        labels_df = pd.read_parquet(labels_path)
        logger.info("Loaded %d labels from %s", len(labels_df), labels_path)

        # === Convert prediction_time to Native Python datetime ===
        # femr requires native datetime objects.
        labels_df["prediction_time"] = pd.to_datetime(
            labels_df["prediction_time"]
        ).apply(lambda x: x.to_pydatetime())

        # === Load Ontology ===
        # Ontology maps medical event codes to embeddings; built during MOTOR pretraining.
        ontology_path = prep_dir / "ontology.pkl"
        if not ontology_path.exists():
            logger.error("Ontology file not found: %s", ontology_path)
            return

        with open(ontology_path, "rb") as f:
            ontology = pickle.load(f)

        # === Compute MOTOR Features ===
        # Opens MEDS database, runs transformer encoder, returns [N, embedding_dim] array.
        # This is the most computationally intensive step; tokens_per_batch controls memory usage.
        logger.info("Computing MOTOR features (this may take a while)...")
        with meds_reader.SubjectDatabase(
            str(Config.PROCESSED_EHR_MEDS_COHORT_DIR)
        ) as database:
            features = femr.models.transformer.compute_features(
                db=database,
                model_path=str(Config.MODEL_EHR_MOTOR_WEIGHTS_DIR),
                labels=labels_df.to_dict("records"),
                ontology=ontology,
                device=self.device,
                tokens_per_batch=8192,  # Batch size (tokens, not samples)
                num_proc=8,  # CPU processes for data loading
            )

        feature_array = features["features"]
        subject_ids = features["subject_ids"]
        logger.info("Extracted features for %d subjects, dim=%d", *feature_array.shape)
        # === Load Master Cohort File ===
        # Used to determine train/valid/test splits and verify subject presence.
        if not Config.PROCESSED_COHORT_PARQUET_FILE.exists():
            logger.error(
                f"Master cohort not found at {Config.PROCESSED_COHORT_PARQUET_FILE}. Run extract_cohort_splits.py first."
            )
            return

        splits_df = pd.read_parquet(Config.PROCESSED_COHORT_PARQUET_FILE)
        sid_to_idx = {int(sid): idx for idx, sid in enumerate(subject_ids)}

        # === Extract and Delegate Saving per Split ===
        # For each train/valid/test split, filter the cohort to only subjects
        # that have MOTOR embeddings and save in canonical format.
        standard_splits = ["train", "valid", "test"]

        for split_name in standard_splits:
            split_df = splits_df[splits_df["dataset_split"] == split_name]
            all_sids = split_df["subject_id"].tolist()
            split_sids = [s for s in all_sids if s in sid_to_idx]

            n_skipped = len(all_sids) - len(split_sids)
            if n_skipped > 0:
                logger.warning(
                    "Split '%s': %d / %d subjects missing from extracted features",
                    split_name,
                    n_skipped,
                    len(all_sids),
                )

            # === Map Bulk Array to Split Samples ===
            # Extract only the embeddings for subjects in this split.
            indices = [sid_to_idx[s] for s in split_sids]
            if not indices:
                logger.warning(
                    f"No matching subjects found for {split_name} split. Skipping."
                )
                continue

            embeddings_tensor = torch.from_numpy(
                feature_array[indices].astype(np.float32)
            )

            # === Map Labels from Master Cohort ===
            # Ensure label consistency by looking up in the original cohort file.
            label_lookup = split_df.set_index("subject_id")["sepsis_label"]
            labels_list = [int(label_lookup[s]) for s in split_sids]
            subject_ids_list = [int(s) for s in split_sids]

            # === Delegate Saving to Base Class ===
            self.save_split(
                split_name=split_name,
                embeddings=embeddings_tensor,
                labels=labels_list,
                subject_ids=subject_ids_list,
            )


if __name__ == "__main__":
    extractor = EHRExtractor()
    extractor.extract_and_save()

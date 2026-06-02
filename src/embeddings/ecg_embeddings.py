"""
Extract ECG embeddings using the offline pretrained wanglab/ecg-fm model.

Usage:
    python -m src.embeddings.ecg_embeddings
"""

import logging
import scipy.io as sio
import pandas as pd
import torch
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

from fairseq_signals.models import build_model_from_checkpoint

from src.embeddings.base_extractor import BaseEmbeddingExtractor
from src.utils.config import Config

logger = logging.getLogger(__name__)


class ECGDataset(Dataset):
    """Loads ECG signals from .mat files and dynamically chunks them into 5-second windows.

    ECG signals are preprocessed offline as .mat files with shape [leads, samples].
    This dataset chunks variable-length signals into fixed-size windows (5 seconds @ 500Hz = 2500 samples)
    with padding applied to shorter chunks. This allows max batching on GPU without OOM issues.

    Attributes:
        df (pd.DataFrame): Cohort dataframe with columns: subject_id, ecg_study_id, sepsis_label.
        mat_dir (Path): Directory containing preprocessed ECG .mat files.
    """

    def __init__(self, df: pd.DataFrame, mat_dir: Path):
        """Initializes the ECG dataset."""
        self.df = df.reset_index(drop=True)
        self.mat_dir = mat_dir

    def __len__(self) -> int:
        """Returns the number of unique ECG studies in the cohort."""
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple:
        """Loads and chunks an ECG signal into 5-second windows.

        Args:
            idx (int): Index into the dataframe.

        Returns:
            tuple: (chunks_tensor, sepsis_label, subject_id)
        """
        row = self.df.iloc[idx]
        subject_id = int(row["subject_id"])
        study_id = int(row["ecg_study_id"])
        label = int(row["sepsis_label"])

        mat_path = self.mat_dir / f"{study_id}.mat"

        # === Load .mat File ===
        mat_data = sio.loadmat(str(mat_path))
        signal = torch.from_numpy(mat_data["feats"]).float()

        # === Handle NaN/Inf Values ===
        # Replace any anomalous values with bounded alternatives.
        if torch.isnan(signal).any() or torch.isinf(signal).any():
            signal = torch.nan_to_num(signal, nan=0.0, posinf=1e5, neginf=-1e5)

        # === Ensure Correct Orientation ===
        # ECG signals should be [leads, samples]. If wider than tall, transpose.
        if signal.shape[0] > signal.shape[1]:
            signal = signal.T

        # === Dynamic Chunking into 5-Second Windows ===
        # Standard: 12-lead ECG sampled at 500Hz = 2500 samples per 5-second chunk.
        # Supports variable-length signals (e.g., 10s, 15s) by creating multiple chunks.
        chunk_size = 2500
        n_samples = signal.shape[1]
        chunks = []

        for i in range(0, n_samples, chunk_size):
            chunk = signal[:, i : i + chunk_size]
            # === Padding: Zero-pad short chunks ===
            # If last chunk is shorter than 2500, pad to match chunk_size.
            if chunk.shape[1] < chunk_size:
                chunk = torch.nn.functional.pad(chunk, (0, chunk_size - chunk.shape[1]))
            chunks.append(chunk)

        # === Stack All Chunks ===
        # Shape: [num_chunks, n_leads, 2500]
        chunks_tensor = torch.stack(chunks)

        return chunks_tensor, label, subject_id


def ecg_collate_fn(batch: list) -> tuple:
    """Custom collate function for ECG batches with variable numbers of chunks.

    Problem: Each patient can have a different number of 5-second chunks depending
    on signal length. Standard PyTorch collate fails with mismatched batch dimensions.

    Solution: Flatten all chunks into a single batch tensor, tracking chunk counts
    per patient. After model inference, chunks are re-aggregated by patient.

    Args:
        batch (list): List of (chunks_tensor, label, subject_id) tuples from ECGDataset.

    Returns:
        tuple: (batch_chunks, labels, subject_ids, chunks_per_patient)
            - batch_chunks: Shape [total_chunks_in_batch, n_leads, 2500]
            - labels: List[int] (length = batch_size)
            - subject_ids: List[int] (length = batch_size)
            - chunks_per_patient: List[int] indicating chunk count per patient
    """
    chunks_list, labels, subject_ids, chunks_per_patient = [], [], [], []

    # === Flatten All Chunks ===
    # Concatenate chunks from all patients in the batch into a single tensor.
    for chunks_tensor, label, subj_id in batch:
        chunks_list.append(chunks_tensor)
        labels.append(label)
        subject_ids.append(subj_id)
        chunks_per_patient.append(chunks_tensor.shape[0])  # Remember chunk count

    # === Concatenate Along Batch Dimension ===
    # Resulting shape: [total_chunks_in_batch, n_leads, 2500]
    # chunks_per_patient is used downstream to split the embeddings back per-patient.
    batch_chunks = torch.cat(chunks_list, dim=0)

    return batch_chunks, labels, subject_ids, chunks_per_patient


class ECGExtractor(BaseEmbeddingExtractor):
    """Extracts ECG embeddings using the frozen pretrained ECG-FM model (from fairseq_signals).

    The ECG-FM model is a transformer trained on MIMIC-IV ECG and PhysioNet 2021 data.
    It captures physiological patterns relevant to sepsis without task-specific fine-tuning.
    Extracted features are averaged across time and chunks to produce a subject-level embedding.
    """

    def __init__(self):
        """Initializes the ECG extractor with cached ECG-FM checkpoint."""
        super().__init__(output_dir=Config.PROCESSED_ECG_RAW_EMBEDDINGS_DIR)

        self.checkpoint_path = (
            Config.MODEL_ECG_PRETRAINED_DIR / "mimic_iv_ecg_physionet_pretrained.pt"
        )
        self.mat_dir = Config.PROCESSED_ECG_ROOT_DIR / "preprocessed"

        self.model = self._load_model()

    def _load_model(self):
        """Loads the ECG-FM pretrained checkpoint using fairseq_signals utilities.

        Raises:
            FileNotFoundError: If checkpoint does not exist at the configured path.
        """
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Missing model weights at {self.checkpoint_path}.")

        logger.info("Loading ecg-fm checkpoint from %s", self.checkpoint_path)
        # === Load Checkpoint ===
        model = build_model_from_checkpoint(str(self.checkpoint_path))
        model = model.to(self.device)
        model.eval()
        return model

    def extract_and_save(self) -> None:
        """Orchestrates end-to-end ECG embedding extraction with chunk aggregation.

        Workflow:
            1. Load master cohort, filter to records with valid ECG .mat files.
            2. For each split, create DataLoader using custom collate_fn.
            3. Batch-forward flattened chunks through ECG-FM (seq-to-seq encoder).
            4. Average chunk embeddings across time; re-aggregate per patient.
            5. Average across chunks to obtain subject-level embedding.
            6. Save per split using parent class.
        """
        # === Validation: Ensure master cohort exists ===
        if not Config.PROCESSED_COHORT_PARQUET_FILE.exists():
            logger.error(
                f"Master cohort not found at {Config.PROCESSED_COHORT_PARQUET_FILE}. Run extract_cohort_splits.py first."
            )
            return

        logger.info("Loading cohort splits from parquet...")
        df = pd.read_parquet(Config.PROCESSED_COHORT_PARQUET_FILE)

        logger.info("Pre-filtering valid .mat files...")
        # === Filter to subjects with ECG modality ===
        df = df[df["ecg_study_id"].notna()]

        # === Early Validation: Check file existence ===
        # Prevents DataLoader crashes later by detecting missing files early.
        valid_mask = df["ecg_study_id"].apply(
            lambda sid: (self.mat_dir / f"{int(sid)}.mat").exists()
        )

        df = df[valid_mask]

        logger.info("Loaded %d valid ECG records for processing.", len(df))

        splits = ["train", "valid", "test"]

        # === Gradient Disabled ===
        with torch.no_grad():
            for split_name in splits:
                split_df = df[df["dataset_split"] == split_name]

                if split_df.empty:
                    logger.warning(f"No data found for {split_name} split. Skipping.")
                    continue

                logger.info(
                    "Processing '%s' split (%d ECGs)...", split_name, len(split_df)
                )

                dataset = ECGDataset(split_df, self.mat_dir)

                # === DataLoader with Custom Collate Function ===
                # Flattens variable-chunk ECG signals for efficient batching.
                dataloader = DataLoader(
                    dataset,
                    batch_size=32,
                    shuffle=False,
                    num_workers=4,
                    collate_fn=ecg_collate_fn,
                    pin_memory=True,
                )

                embeddings_list, labels_list, subject_ids_list = [], [], []

                # === Batch-wise Inference with Chunk Re-aggregation ===
                for batch_chunks, batch_labels, batch_ids, chunks_per_patient in tqdm(
                    dataloader, desc=f"{split_name} split"
                ):
                    # Transfer all flattened chunks to GPU
                    batch_chunks = batch_chunks.to(self.device)

                    # === Forward Pass Through ECG-FM Encoder ===
                    output = self.model.extract_features(
                        batch_chunks, padding_mask=None
                    )
                    encoder_out = output["x"]

                    # === Average Over Sequence Length ===
                    # Output is [n_chunks, seqlen, hidden_dim], average -> [n_chunks, hidden_dim]
                    chunk_embeddings = encoder_out.mean(dim=1)

                    # === Re-aggregate Chunks per Patient ===
                    # Split flattened chunk embeddings back to individual patients.
                    # For each patient, average their chunks to obtain patient-level embedding.
                    start_idx = 0
                    for num_chunks in chunks_per_patient:
                        end_idx = start_idx + num_chunks

                        # Extract embeddings for this patient's chunks
                        patient_embedding = chunk_embeddings[start_idx:end_idx].mean(
                            dim=0  # Average across chunks
                        )

                        embeddings_list.append(patient_embedding.cpu())

                        start_idx = end_idx

                    # Record labels and subject IDs (one per patient, not per chunk)
                    labels_list.extend(batch_labels)
                    subject_ids_list.extend(batch_ids)

                # === Concatenate All Patient Embeddings ===
                # Stack into a single [n_patients, hidden_dim] tensor.
                final_embeddings = torch.stack(embeddings_list)

                # === Delegate Saving to Base Class ===
                self.save_split(
                    split_name=split_name,
                    embeddings=final_embeddings,
                    labels=labels_list,
                    subject_ids=subject_ids_list,
                )


if __name__ == "__main__":
    extractor = ECGExtractor()
    extractor.extract_and_save()

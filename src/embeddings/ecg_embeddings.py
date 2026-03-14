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
    """Custom Dataset to load .mat files and dynamically chunk the signals."""

    def __init__(self, df: pd.DataFrame, mat_dir: Path):
        self.df = df.reset_index(drop=True)
        self.mat_dir = mat_dir

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        subject_id = int(row["subject_id"])
        study_id = int(row["ecg_study_id"])
        label = int(row["sepsis_label"])

        mat_path = self.mat_dir / f"{study_id}.mat"

        mat_data = sio.loadmat(str(mat_path))
        signal = torch.from_numpy(mat_data["feats"]).float()

        if torch.isnan(signal).any() or torch.isinf(signal).any():
            signal = torch.nan_to_num(signal, nan=0.0, posinf=1e5, neginf=-1e5)

        if signal.shape[0] > signal.shape[1]:
            signal = signal.T

        # Chunking logic (5-second windows at 500Hz = 2500 samples)
        chunk_size = 2500
        n_samples = signal.shape[1]
        chunks = []

        for i in range(0, n_samples, chunk_size):
            chunk = signal[:, i : i + chunk_size]
            if chunk.shape[1] < chunk_size:
                chunk = torch.nn.functional.pad(chunk, (0, chunk_size - chunk.shape[1]))
            chunks.append(chunk)

        # Shape: [num_chunks, n_leads, 2500]
        chunks_tensor = torch.stack(chunks)

        return chunks_tensor, label, subject_id


def ecg_collate_fn(batch):
    """
    Takes a batch of patients with varying chunk amounts and flattens them
    for maximum GPU utilization, while tracking how to split them back up.
    """
    chunks_list, labels, subject_ids, chunks_per_patient = [], [], [], []

    for chunks_tensor, label, subj_id in batch:
        chunks_list.append(chunks_tensor)
        labels.append(label)
        subject_ids.append(subj_id)
        chunks_per_patient.append(chunks_tensor.shape[0])

    # Concatenate all chunks along the batch dimension
    # Final Shape: [total_chunks_in_batch, n_leads, 2500]
    batch_chunks = torch.cat(chunks_list, dim=0)

    return batch_chunks, labels, subject_ids, chunks_per_patient


class ECGExtractor(BaseEmbeddingExtractor):
    def __init__(self):
        super().__init__(output_dir=Config.PROCESSED_ECG_RAW_EMBEDDINGS_DIR)

        self.checkpoint_path = (
            Config.MODEL_ECG_PRETRAINED_DIR / "mimic_iv_ecg_physionet_pretrained.pt"
        )
        self.mat_dir = Config.PROCESSED_ECG_ROOT_DIR / "preprocessed"

        self.model = self._load_model()

    def _load_model(self):
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Missing model weights at {self.checkpoint_path}.")

        logger.info("Loading ecg-fm checkpoint from %s", self.checkpoint_path)
        result = build_model_from_checkpoint(str(self.checkpoint_path))

        model = (
            result[0][0]
            if isinstance(result, tuple) and isinstance(result[0], list)
            else (result[0] if isinstance(result, tuple) else result)
        )
        model = model.to(self.device)
        model.eval()
        return model

    def extract_and_save(self):
        if not Config.PROCESSED_COHORT_PARQUET_FILE.exists():
            logger.error(
                f"Master cohort not found at {Config.PROCESSED_COHORT_PARQUET_FILE}. Run extract_cohort_splits.py first."
            )
            return

        logger.info("Loading cohort splits from parquet...")
        df = pd.read_parquet(Config.PROCESSED_COHORT_PARQUET_FILE)

        logger.info("Pre-filtering valid .mat files to save CPU overhead...")

        df = df[df["ecg_study_id"].notna()]

        valid_mask = df["ecg_study_id"].apply(
            lambda sid: (self.mat_dir / f"{int(sid)}.mat").exists()
        )

        df = df[valid_mask]

        logger.info("Loaded %d valid ECG records for processing.", len(df))

        splits = ["train", "valid", "test"]

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

                dataloader = DataLoader(
                    dataset,
                    batch_size=32,
                    shuffle=False,
                    num_workers=4,
                    collate_fn=ecg_collate_fn,
                    pin_memory=True,
                )

                embeddings_list, labels_list, subject_ids_list = [], [], []

                for batch_chunks, batch_labels, batch_ids, chunks_per_patient in tqdm(
                    dataloader, desc=f"{split_name} split"
                ):
                    batch_chunks = batch_chunks.to(self.device)

                    output = self.model.extract_features(
                        batch_chunks, padding_mask=None
                    )

                    if isinstance(output, dict) and "x" in output:
                        encoder_out = output["x"]
                    elif isinstance(output, tuple):
                        encoder_out = output[0]
                    else:
                        encoder_out = output

                    # Average over the sequence length for each chunk
                    # Shape becomes: [total_chunks_in_batch, hidden_dim]
                    chunk_embeddings = encoder_out.mean(dim=1)

                    # Now, split the chunks back out to their respective patients
                    start_idx = 0
                    for num_chunks in chunks_per_patient:
                        end_idx = start_idx + num_chunks

                        # Take the chunks for this specific patient and average them
                        patient_embedding = chunk_embeddings[start_idx:end_idx].mean(
                            dim=0
                        )
                        embeddings_list.append(patient_embedding.cpu())

                        start_idx = end_idx

                    labels_list.extend(batch_labels)
                    subject_ids_list.extend(batch_ids)

                final_embeddings = torch.stack(embeddings_list)

                self.save_split(
                    split_name=split_name,
                    embeddings=final_embeddings,
                    labels=labels_list,
                    subject_ids=subject_ids_list,
                )


if __name__ == "__main__":
    extractor = ECGExtractor()
    extractor.extract_and_save()

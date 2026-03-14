"""Extract CXR Text embeddings using the offline Bio_ClinicalBERT Foundation Model.

Usage:
    python -m src.embeddings.cxr_txt_embeddings
"""

import logging
import re
import zipfile
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

from src.embeddings.base_extractor import BaseEmbeddingExtractor
from src.utils.config import Config

logger = logging.getLogger(__name__)


def clean_report_text(text: str) -> str:
    """Cleans MIMIC-CXR report text by removing underscore artifacts."""
    text = re.sub(r"_+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()


class CXRTextDataset(Dataset):
    """Custom Dataset yielding plain text strings from RAM."""

    def __init__(self, df: pd.DataFrame):
        self.df = df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        return row["report_text"], int(row["sepsis_label"]), int(row["subject_id"])


class CXRTextExtractor(BaseEmbeddingExtractor):
    def __init__(self):
        super().__init__(output_dir=Config.PROCESSED_CXR_TXT_RAW_EMBEDDINGS_DIR)

        model_name = "emilyalsentzer/Bio_ClinicalBERT"
        model_dir = Config.MODEL_CXR_TXT_PRETRAINED_DIR

        logger.info(f"Loading cached foundation model: {model_name}...")

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, cache_dir=model_dir, local_files_only=True
        )
        self.model = AutoModel.from_pretrained(
            model_name, cache_dir=model_dir, local_files_only=True
        ).to(self.device)
        self.model.eval()

    def extract_and_save(self):
        if not Config.PROCESSED_COHORT_PARQUET_FILE.exists():
            logger.error(
                f"Master cohort not found at {Config.PROCESSED_COHORT_PARQUET_FILE}. Run extract_cohort_splits.py first."
            )
            return

        logger.info("Loading cohort splits from parquet...")
        df = pd.read_parquet(Config.PROCESSED_COHORT_PARQUET_FILE)

        # Filter for subjects that actually have a CXR study
        df = df[df["cxr_study_id"].notnull()].copy()
        df["cxr_study_id"] = df["cxr_study_id"].astype(int)

        zip_path = Config.RAW_CXR_TXT_DIR / Config.CXR_REPORTS_FILE
        if not zip_path.exists():
            logger.error(f"❌ ZIP file not found at: {zip_path}")
            return

        logger.info("Pre-loading and cleaning text reports into RAM...")
        report_texts = []
        valid_indices = []

        with zipfile.ZipFile(zip_path, "r") as zf:
            # Index internal paths structured as p{subject_id}/s{study_id}.txt
            zip_path_map = {}
            for name in zf.namelist():
                if name.endswith(".txt"):
                    match = re.search(r"p(\d{8})/s(\d{8})\.txt$", name)
                    if match:
                        subj_id = int(match.group(1))
                        stud_id = int(match.group(2))
                        zip_path_map[(subj_id, stud_id)] = name

            # Extract only the texts we need
            for idx, row in tqdm(df.iterrows(), total=len(df), desc="Reading ZIP"):
                subj_id = int(row["subject_id"])
                stud_id = int(row["cxr_study_id"])
                internal_path = zip_path_map.get((subj_id, stud_id))

                if internal_path:
                    raw_text = zf.read(internal_path).decode("utf-8")
                    report_texts.append(clean_report_text(raw_text))
                    valid_indices.append(idx)
                else:
                    logger.debug(
                        f"Missing in ZIP for subject_id {subj_id}, study_id {stud_id}"
                    )

        # Keep only rows where we successfully found the text
        df = df.loc[valid_indices].copy()
        df["report_text"] = report_texts
        logger.info(f"Successfully loaded {len(df)} text reports into memory.")

        splits = ["train", "valid", "test"]

        with torch.no_grad():
            for split_name in splits:
                split_df = df[df["dataset_split"] == split_name]

                if split_df.empty:
                    logger.warning(f"No data found for {split_name} split. Skipping.")
                    continue

                logger.info(
                    f"Processing '{split_name}' split ({len(split_df)} reports)..."
                )

                dataset = CXRTextDataset(split_df)

                # We can use num_workers=0 here because the data is already in RAM as strings.
                dataloader = DataLoader(
                    dataset,
                    batch_size=64,
                    shuffle=False,
                    num_workers=0,
                    pin_memory=False,
                )

                embeddings_list, labels_list, subject_ids_list = [], [], []

                for batch_texts, batch_labels, batch_ids in tqdm(
                    dataloader, desc=f"{split_name} split"
                ):
                    # Tokenize the entire batch of 64 strings at once and push to GPU
                    inputs = self.tokenizer(
                        list(batch_texts),
                        return_tensors="pt",
                        truncation=True,
                        padding=True,
                        max_length=512,
                    ).to(self.device)

                    outputs = self.model(**inputs)

                    # Extract CLS token embedding: [Batch, 768]
                    cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu()

                    embeddings_list.append(cls_embeddings)
                    labels_list.extend(batch_labels.tolist())
                    subject_ids_list.extend(batch_ids.tolist())

                # Concatenate the lists of batch tensors into one massive tensor
                final_embeddings = torch.cat(embeddings_list, dim=0)

                self.save_split(
                    split_name=split_name,
                    embeddings=final_embeddings,
                    labels=labels_list,
                    subject_ids=subject_ids_list,
                )


if __name__ == "__main__":
    extractor = CXRTextExtractor()
    extractor.extract_and_save()

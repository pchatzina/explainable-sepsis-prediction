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
    """Cleans MIMIC-CXR report text by removing structural artifacts.

    Removes underscore sequences and normalizes whitespace
    before tokenization. Improves embedding quality by
    reducing noise in the input to Bio_ClinicalBERT.

    Args:
        text (str): Raw CXR report text from ZIP.

    Returns:
        str: Cleaned, lowercase report text ready for tokenization.
    """
    text = re.sub(r"_+", " ", text)  # Replace underscore sequences with space
    text = re.sub(r"\s+", " ", text)  # Normalize whitespace
    return text.strip().lower()


class CXRTextDataset(Dataset):
    """Provides plain-text CXR reports from a pre-loaded DataFrame.

    This dataset holds report text in memory (loaded upfront in extract_and_save())
    to avoid repeated ZIP file I/O during DataLoader iteration. Enables efficient
    batch-wise tokenization on GPU.

    Attributes:
        df (pd.DataFrame): Cohort dataframe with columns: subject_id, sepsis_label, report_text.
    """

    def __init__(self, df: pd.DataFrame):
        """Initializes the CXR text dataset.

        Args:
            df (pd.DataFrame): Dataframe with report_text already pre-loaded.
        """
        self.df = df.reset_index(drop=True)

    def __len__(self) -> int:
        """Returns total number of CXR reports."""
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple:
        """Retrieves report text and metadata for a subject.

        Returns:
            tuple: (report_text, sepsis_label, subject_id)
        """
        row = self.df.iloc[idx]
        return row["report_text"], int(row["sepsis_label"]), int(row["subject_id"])


class CXRTextExtractor(BaseEmbeddingExtractor):
    """Extracts CXR report text embeddings using the frozen Bio_ClinicalBERT model.

    Bio_ClinicalBERT is a BERT variant fine-tuned on clinical and biomedical text,
    making it well-suited for MIMIC-CXR radiology reports. The model output is
    the CLS token embedding (768-dim), which represents the whole-document semantic.
    """

    def __init__(self):
        """Initializes the CXR text extractor with cached Bio_ClinicalBERT weights."""
        super().__init__(output_dir=Config.PROCESSED_CXR_TXT_RAW_EMBEDDINGS_DIR)

        model_name = "emilyalsentzer/Bio_ClinicalBERT"
        model_dir = Config.MODEL_CXR_TXT_PRETRAINED_DIR

        logger.info(f"Loading cached foundation model: {model_name}...")

        # === Load Tokenizer and Model (offline, from cache) ===
        # Both are kept in local_files_only mode to avoid re-downloading.
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, cache_dir=model_dir, local_files_only=True
        )
        self.model = AutoModel.from_pretrained(
            model_name, cache_dir=model_dir, local_files_only=True
        ).to(self.device)
        self.model.eval()

    def extract_and_save(self) -> None:
        """Orchestrates end-to-end CXR text embedding extraction.

        Workflow:
            1. Load master cohort, filter to records with CXR studies.
            2. Read all relevant CXR reports from the ZIP archive into RAM.
            3. Clean and store report text in the dataframe.
            4. For each split, instantiate DataLoader.
            5. Batch-tokenize and forward through Bio_ClinicalBERT.
            6. Extract CLS token embeddings (768-dim) and save per split.
        """
        # === Validation: Ensure master cohort exists ===
        if not Config.PROCESSED_COHORT_PARQUET_FILE.exists():
            logger.error(
                f"Master cohort not found at {Config.PROCESSED_COHORT_PARQUET_FILE}. Run extract_cohort_splits.py first."
            )
            return

        logger.info("Loading cohort splits from parquet...")
        df = pd.read_parquet(Config.PROCESSED_COHORT_PARQUET_FILE)

        # === Filter to subjects with CXR modality ===
        # Only process records where cxr_study_id is not null
        df = df[df["cxr_study_id"].notnull()].copy()
        df["cxr_study_id"] = df["cxr_study_id"].astype(int)

        zip_path = Config.RAW_CXR_TXT_DIR / Config.CXR_REPORTS_FILE
        if not zip_path.exists():
            logger.error(f"ZIP file not found at: {zip_path}")
            return

        logger.info("Pre-loading and cleaning text reports into RAM...")
        report_texts = []
        valid_indices = []

        # === Efficient ZIP Indexing ===
        # Read ZIP directory once and build in-memory map: (subject_id, study_id) -> internal_path
        # This avoids repeated ZIP file traversals during extraction.
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

            # === Extract Only Needed Reports ===
            # Loop through cohort dataframe and retrieve text from ZIP for each unique study.
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

        # === Keep Only Rows with Successfully Retrieved Text ===
        # Filter the dataframe to only subjects whose reports were found in the ZIP file.
        df = df.loc[valid_indices].copy()
        df["report_text"] = report_texts
        logger.info(f"Successfully loaded {len(df)} text reports into memory.")

        splits = ["train", "valid", "test"]

        # === Gradient Disabled ===
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

                # === DataLoader with Zero Workers ===
                # Data is already in RAM as strings, so num_workers=0 avoids pickling overhead.
                dataloader = DataLoader(
                    dataset,
                    batch_size=64,
                    shuffle=False,
                    num_workers=0,
                    pin_memory=False,
                )

                embeddings_list, labels_list, subject_ids_list = [], [], []

                # === Batch-wise Tokenization and Encoding ===
                for batch_texts, batch_labels, batch_ids in tqdm(
                    dataloader, desc=f"{split_name} split"
                ):
                    # === Tokenize Entire Batch at Once ===
                    # Handles variable-length texts with truncation and padding to 512 tokens.
                    inputs = self.tokenizer(
                        list(batch_texts),
                        return_tensors="pt",
                        truncation=True,
                        padding=True,
                        max_length=512,
                    ).to(self.device)

                    # === Forward Pass Through Bio_ClinicalBERT ===
                    # outputs.last_hidden_state shape: [Batch, SeqLen, 768]
                    outputs = self.model(**inputs)

                    # === Extract CLS Token Embedding ===
                    # CLS token is always at position 0; it aggregates document-level semantics.
                    # Shape after indexing: [Batch, 768]
                    cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu()

                    # Accumulate batch embeddings and metadata
                    embeddings_list.append(cls_embeddings)
                    labels_list.extend(batch_labels.tolist())
                    subject_ids_list.extend(batch_ids.tolist())

                # === Concatenate All Batch Tensors ===
                # Stack across batch dim to obtain a single [N, 768] tensor.
                final_embeddings = torch.cat(embeddings_list, dim=0)

                # === Delegate Saving to Base Class ===
                self.save_split(
                    split_name=split_name,
                    embeddings=final_embeddings,
                    labels=labels_list,
                    subject_ids=subject_ids_list,
                )


if __name__ == "__main__":
    extractor = CXRTextExtractor()
    extractor.extract_and_save()

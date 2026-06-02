"""Extract CXR Image embeddings using the frozen torchxrayvision Foundation Model.

Usage:
    python -m src.embeddings.cxr_img_embeddings
"""

import logging
from pathlib import Path
import pandas as pd
import skimage.io
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchxrayvision as xrv
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from src.embeddings.base_extractor import BaseEmbeddingExtractor
from src.utils.config import Config

import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

logger = logging.getLogger(__name__)


class CXRDataset(Dataset):
    """Loads and preprocesses CXR images on CPU before GPU transfer.

    Handles image loading, grayscale normalization, and augmentation via transforms.

    Attributes:
        df (pd.DataFrame): Cohort dataframe with columns: subject_id, sepsis_label, cxr_study_path.
        img_dir (Path): Root directory where CXR .jpg files are located.
        transform (callable): Optional torchvision transforms to apply (e.g., center crop, resize).
    """

    def __init__(self, df: pd.DataFrame, img_dir: Path, transform=None):
        """Initializes the CXR dataset."""
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self) -> int:
        """Returns total number of CXR images."""
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple:
        """Loads and preprocesses a single CXR image.

        Returns:
            tuple: (image_tensor, sepsis_label, subject_id)
        """
        row = self.df.iloc[idx]
        subject_id = int(row["subject_id"])
        label = int(row["sepsis_label"])
        img_path = self.img_dir / row["cxr_study_path"]

        # === Image Loading & Preprocessing ===
        # Load image in grayscale. MIMIC-CXR already provides single-channel imagery.
        img = skimage.io.imread(img_path, as_gray=True)

        # Normalize pixel intensities from [0, 255] to [-1, 1] using xrv's standard.
        img = xrv.datasets.normalize(img, 255)

        # Add channel dimension: [H, W] -> [1, H, W] for PyTorch convention
        img = img[None, ...]

        # Apply augmentation transforms if provided (e.g., center crop, resize)
        if self.transform:
            img = self.transform(img)

        # Convert to torch tensor with float32 precision
        img_tensor = torch.from_numpy(img).float()

        return img_tensor, label, subject_id


class CXRImageExtractor(BaseEmbeddingExtractor):
    """Extracts CXR image embeddings using the frozen torchxrayvision DenseNet121 model.

    The DenseNet121 serves as the foundation model. Features are extracted from the
    adaptive average pooling layer (1024-dim). This extractor handles batch processing
    with PyTorch DataLoader for GPU efficiency.
    """

    def __init__(self):
        """Initializes the CXR image extractor with cached foundation model weights."""
        super().__init__(output_dir=Config.PROCESSED_CXR_IMG_RAW_EMBEDDINGS_DIR)

        self.raw_img_dir = Config.RAW_CXR_IMG_DIR

        # === Load Frozen Foundation Model ===
        # Kept in eval() mode; no gradients are computed.
        logger.info("Loading densenet121-res224-all foundation model...")
        self.model = xrv.models.DenseNet(weights="densenet121-res224-all").to(
            self.device
        )
        self.model.eval()

        # === Preprocessing Pipeline for Images ===
        # Center-crop to isolate lung fields, then resize to 224x224 (DenseNet standard input).
        self.transform = transforms.Compose(
            [xrv.datasets.XRayCenterCrop(), xrv.datasets.XRayResizer(224)]
        )

    def extract_and_save(self) -> None:
        """Extract CXR image embeddings and persist per split.

        Uses DenseNet's feature layer (before the classification head), applies ReLU,
        then global average pools spatial features from [B, 1024, 7, 7] to [B, 1024].
        Delegates persistence to the base class save_split().
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
        # Only process records where cxr_study_path is not null
        df = df[df["cxr_study_path"].notnull()].copy()

        logger.info("Pre-filtering valid image paths...")
        # === Early Validation: Check file existence ===
        # Prevents DataLoader crashes later by detecting missing files early.
        valid_mask = df["cxr_study_path"].apply(
            lambda p: (self.raw_img_dir / p).exists()
        )
        df = df[valid_mask]

        logger.info("Loaded %d valid CXR records for processing.", len(df))

        splits = ["train", "valid", "test"]

        with torch.no_grad():
            for split_name in splits:
                split_df = df[df["dataset_split"] == split_name]

                if split_df.empty:
                    logger.warning(f"No data found for {split_name} split. Skipping.")
                    continue

                logger.info(
                    "Processing '%s' split (%d images)...", split_name, len(split_df)
                )

                # === Create PyTorch DataLoader with Multiprocessing ===
                # num_workers=4 parallelizes image I/O; pin_memory=True speeds GPU transfer.
                dataset = CXRDataset(split_df, self.raw_img_dir, self.transform)
                dataloader = DataLoader(
                    dataset,
                    batch_size=64,
                    shuffle=False,
                    num_workers=4,
                    pin_memory=True,
                )

                embeddings_list, labels_list, subject_ids_list = [], [], []

                # === Batch-wise Forward Pass ===
                for batch_imgs, batch_labels, batch_ids in tqdm(
                    dataloader, desc=f"{split_name} split"
                ):
                    # Transfer images to GPU
                    batch_imgs = batch_imgs.to(self.device)

                    # === Feature Extraction ===
                    # Extract features from DenseNet feature layers (before classification head).
                    # Output shape: [B, 1024, 7, 7] since input is pooled/resized to 224x224.
                    features = self.model.features(batch_imgs)
                    features = F.relu(features, inplace=True)  # Apply ReLU activation

                    # === Global Average Pooling to Embedding Vector ===
                    # Reduce spatial dimensions [B, 1024, 7, 7] -> [B, 1024]
                    # This creates a subject-level embedding, aggregating spatial information.
                    pooled_2d = F.adaptive_avg_pool2d(features, (1, 1)).view(
                        features.size(0), -1
                    )

                    # Accumulate batch embeddings and metadata
                    embeddings_list.append(pooled_2d.cpu())
                    labels_list.extend(batch_labels.tolist())
                    subject_ids_list.extend(batch_ids.tolist())

                # === Concatenate All Batch Tensors ===
                # Stack across the batch dimension to obtain a single [N, 1024] tensor.
                final_embeddings = torch.cat(embeddings_list, dim=0)

                # === Delegate Saving to Base Class ===
                self.save_split(
                    split_name=split_name,
                    embeddings=final_embeddings,
                    labels=labels_list,
                    subject_ids=subject_ids_list,
                )


if __name__ == "__main__":
    extractor = CXRImageExtractor()
    extractor.extract_and_save()

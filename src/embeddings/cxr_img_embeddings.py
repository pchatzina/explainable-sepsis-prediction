"""Extract CXR Image embeddings using the frozen torchxrayvision Foundation Model.

Usage:
    python -m src.embeddings.cxr_img_embeddings
"""

import logging
from pathlib import Path
import numpy as np
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
    """Custom Dataset to handle loading and transforming CXR images on the CPU."""

    def __init__(self, df: pd.DataFrame, img_dir: Path, transform=None):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        subject_id = int(row["subject_id"])
        label = int(row["sepsis_label"])
        img_path = self.img_dir / row["cxr_study_path"]

        # Load and process
        img = skimage.io.imread(img_path, as_gray=True)
        img = xrv.datasets.normalize(img, 255)
        img = img[None, ...]  # Add channel dimension

        if self.transform:
            img = self.transform(img)

        img_tensor = torch.from_numpy(img).float()

        return img_tensor, label, subject_id


class CXRImageExtractor(BaseEmbeddingExtractor):
    def __init__(self):
        super().__init__(output_dir=Config.PROCESSED_CXR_IMG_RAW_EMBEDDINGS_DIR)

        self.raw_img_dir = Config.RAW_CXR_IMG_DIR

        # Load model
        logger.info("Loading densenet121-res224-all foundation model...")
        self.model = xrv.models.DenseNet(weights="densenet121-res224-all").to(
            self.device
        )
        self.model.eval()

        self.transform = transforms.Compose(
            [xrv.datasets.XRayCenterCrop(), xrv.datasets.XRayResizer(224)]
        )

    def extract_and_save(self):
        if not Config.PROCESSED_COHORT_PARQUET_FILE.exists():
            logger.error(
                f"Master cohort not found at {Config.PROCESSED_COHORT_PARQUET_FILE}. Run extract_cohort_splits.py first."
            )
            return

        logger.info("Loading cohort splits from parquet...")
        df = pd.read_parquet(Config.PROCESSED_COHORT_PARQUET_FILE)

        # Filter only subjects that have CXR paths
        df = df[df["cxr_study_path"].notnull()].copy()

        logger.info("Pre-filtering valid image paths to save CPU overhead...")
        # Check if files exist to prevent DataLoader crashes
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

                # Create Dataset and DataLoader
                dataset = CXRDataset(split_df, self.raw_img_dir, self.transform)
                dataloader = DataLoader(
                    dataset,
                    batch_size=64,
                    shuffle=False,
                    num_workers=4,
                    pin_memory=True,
                )

                embeddings_list, labels_list, subject_ids_list = [], [], []

                for batch_imgs, batch_labels, batch_ids in tqdm(
                    dataloader, desc=f"{split_name} split"
                ):
                    batch_imgs = batch_imgs.to(self.device)

                    features = self.model.features(batch_imgs)
                    features = F.relu(features, inplace=True)

                    # Global Average Pooling to [B, 1024, 1, 1] then flatten to [B, 1024]
                    pooled_2d = F.adaptive_avg_pool2d(features, (1, 1)).view(
                        features.size(0), -1
                    )

                    # Append batches to lists
                    embeddings_list.append(pooled_2d.cpu())
                    labels_list.extend(batch_labels.tolist())
                    subject_ids_list.extend(batch_ids.tolist())

                # Concatenate all batched tensors into one large tensor
                final_embeddings = torch.cat(embeddings_list, dim=0)

                # Delegate saving to the Base Class
                self.save_split(
                    split_name=split_name,
                    embeddings=final_embeddings,
                    labels=labels_list,
                    subject_ids=subject_ids_list,
                )


if __name__ == "__main__":
    extractor = CXRImageExtractor()
    extractor.extract_and_save()

import pytorch_lightning as pl
from typing import List, Optional
from torch.utils.data import DataLoader
from src.data.loaders.multimodal_dataset import MultimodalSepsisDataset


class FusionDataModule(pl.LightningDataModule):
    """PyTorch Lightning DataModule for multimodal fusion training with EHR dropout.

    Lazily instantiates train/val/test datasets on first call to setup(). Caches datasets
    in RAM for efficient batching across epochs. Supports dynamic EHR dropout injection:
    the tuner or main pipeline can update self.ehr_dropout_rate before calling setup()
    again, and the train dataset will be re-instantiated with the new dropout rate while
    val/test remain unchanged (always ehr_dropout_rate=0.0).

    This design allows Optuna trials to experiment with different EHR dropout rates
    without reloading embeddings from disk.
    """

    def __init__(
        self,
        active_modalities: List[str],
        batch_size: int = 64,
        num_workers: int = 4,
        ehr_dropout_rate: float = 0.0,
    ):
        """Initialize the fusion data module.

        Args:
            active_modalities: List of modality keys to load.
            batch_size: Batch size for DataLoaders.
            num_workers: Number of worker processes for data loading.
            ehr_dropout_rate: Initial EHR dropout rate for training. Can be updated
                by calling datamodule.ehr_dropout_rate = new_value and then
                datamodule.setup(stage='fit') again.
        """
        super().__init__()
        self.active_modalities = active_modalities
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.ehr_dropout_rate = ehr_dropout_rate

    def setup(self, stage: Optional[str] = None) -> None:
        """Create or refresh train/val/test datasets as needed.

        This method is called by Lightning at the start of each fit/val/test stage.
        On first call, datasets are instantiated and cached in RAM. On subsequent
        calls (e.g., Optuna injection of a new EHR dropout rate), the train dataset
        is recreated with the updated dropout rate, while val/test are created only once.

        Args:
            stage: One of 'fit', 'validate', 'test', or None (meaning all stages).
        """
        if stage == "fit" or stage is None:
            # Only instantiate the train dataset if it doesn't exist yet
            if not hasattr(self, "train_dataset"):
                self.train_dataset = MultimodalSepsisDataset(
                    split="train",
                    active_modalities=self.active_modalities,
                    ehr_dropout_rate=self.ehr_dropout_rate,
                )
            else:
                # Tensors are already in RAM; just update the dropout rate for this trial.
                # This allows Optuna to inject different dropout rates without reloading.
                # NOTE: not thread-safe for parallel Optuna trials (n_jobs > 1).
                self.train_dataset.ehr_dropout_rate = self.ehr_dropout_rate

            # Validate dataset only needs to be loaded once (dropout is always 0.0)
            if not hasattr(self, "val_dataset"):
                self.val_dataset = MultimodalSepsisDataset(
                    split="valid",
                    active_modalities=self.active_modalities,
                    ehr_dropout_rate=0.0,
                )

        if stage == "test" or stage is None:
            if not hasattr(self, "test_dataset"):
                self.test_dataset = MultimodalSepsisDataset(
                    split="test",
                    active_modalities=self.active_modalities,
                    ehr_dropout_rate=0.0,
                )

    def train_dataloader(self) -> DataLoader:
        """Return the training DataLoader with shuffling and EHR dropout.

        Returns DataLoader with shuffle=True and drop_last=True to ensure
        consistent batch sizes during training.
        """
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        """Validation DataLoader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
        """Test DataLoader."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

import logging
import torch
import random
from torch.utils.data import Dataset
from src.data.loaders.helpers import load_embeddings
from src.utils.config import Config

logger = logging.getLogger(__name__)


class MultimodalSepsisDataset(Dataset):
    """PyTorch Dataset for multimodal sepsis prediction with dynamic missing modalities.

    Supports EHR dropout: randomly masking EHR for patients who have auxiliary modalities.
    Validation and test sets always use ehr_dropout_rate=0.0 for consistent evaluation.

    Attributes:
        split: Dataset split ('train', 'valid', or 'test').
        active_modalities: List of modality keys.
        ehr_dropout_rate: Probability of masking EHR during training (only for patients
            with auxiliary modalities). Set to 0.0 for val/test.
        subject_ids: List of MIMIC-IV subject IDs.
        labels: Binary sepsis labels (0 or 1) aligned with subject_ids.
        data_store: Dict mapping modality -> torch tensor of normalized embeddings.
        idx_maps: Dict mapping modality -> {subject_id: row_index_in_embedding_matrix}.
        dims: Dict mapping modality -> embedding dimension.
    """

    def __init__(
        self,
        split: str,
        active_modalities: list,
        ehr_dropout_rate: float = 0.0,
    ):
        """Initialize the multimodal dataset.

        Loads all embeddings for the specified split into RAM. EHR is treated as the
        anchor modality and **must** be present for all patients. Auxiliary modalities
        are loaded if available; missing modalities trigger zero-padding and mask=0.

        Args:
            split: Dataset split name ('train', 'valid', 'test').
            active_modalities: List of modality keys to load.
            ehr_dropout_rate: Probability of masking EHR for training. Only applied to
                patients with at least one auxiliary modality. Ignored for val/test (set to 0.0).

        Raises:
            FileNotFoundError: If EHR embeddings are missing (EHR is the anchor).
        """
        super().__init__()
        self.split = split
        self.active_modalities = active_modalities
        self.ehr_dropout_rate = ehr_dropout_rate

        # Map modality names to their normalized embedding directories
        dir_map = {
            "ehr": Config.PROCESSED_EHR_EMBEDDINGS_DIR,
            "ecg": Config.PROCESSED_ECG_EMBEDDINGS_DIR,
            "cxr_img": Config.PROCESSED_CXR_IMG_EMBEDDINGS_DIR,
            "cxr_txt": Config.PROCESSED_CXR_TXT_EMBEDDINGS_DIR,
        }

        self.data_store = {}
        self.idx_maps = {}
        self.dims = {}

        # 1. Load Anchor Modality (EHR)
        ehr_path = dir_map["ehr"] / f"{split}_embeddings.pt"
        X_ehr, y_ehr, sids_ehr = load_embeddings(ehr_path)

        self.subject_ids = sids_ehr
        self.labels = y_ehr
        self.data_store["ehr"] = torch.from_numpy(X_ehr)
        self.dims["ehr"] = X_ehr.shape[1]
        self.idx_maps["ehr"] = {sid: idx for idx, sid in enumerate(self.subject_ids)}

        # 2. Load Auxiliary Modalities
        for mod in active_modalities:
            if mod == "ehr":
                continue

            mod_path = dir_map[mod] / f"{split}_embeddings.pt"
            if mod_path.exists():
                X_mod, _, sids_mod = load_embeddings(mod_path)
                self.data_store[mod] = torch.from_numpy(X_mod)
                self.dims[mod] = X_mod.shape[1]
                self.idx_maps[mod] = {sid: idx for idx, sid in enumerate(sids_mod)}
            else:
                # If a modality has no data for a split, store empty mappings
                self.data_store[mod] = None
                self.dims[mod] = Config.EMBEDDING_DIMS[mod]
                self.idx_maps[mod] = {}

    def __len__(self) -> int:
        """Return the number of patients in this split."""
        return len(self.subject_ids)

    def __getitem__(self, idx: int) -> dict:
        """Retrieve embeddings, masks, and label for a single patient.

        Logic for handling missing modalities and EHR dropout:
          1. Check if patient has any auxiliary modality (ECG, CXR).
          2. If ehr_dropout_rate > 0 and patient has aux modalities, randomly mask EHR.
          3. For each modality:
             - If present and not dropped: embed with mask=1.0
             - If missing or dropped: zero-pad vector with mask=0.0

        Args:
            idx: Index into this dataset.

        Returns:
            Dictionary with keys:
              - 'subject_id': MIMIC-IV subject ID
              - 'embeddings': Dict[modality -> embedded tensor]
              - 'masks': Dict[modality -> binary mask tensor]
              - 'label': Binary sepsis label (0 or 1)
        """
        subject_id = self.subject_ids[idx]
        label = self.labels[idx]

        embeddings = {}
        masks = {}

        # Check if patient has auxiliary modalities
        has_aux = any(
            subject_id in self.idx_maps[mod]
            for mod in self.active_modalities
            if mod != "ehr"
        )

        # EHR Dropout Logic
        drop_ehr = False
        if self.ehr_dropout_rate > 0.0 and has_aux:
            if random.random() < self.ehr_dropout_rate:
                drop_ehr = True

        # Process Modalities
        for mod in self.active_modalities:
            if mod == "ehr" and drop_ehr:
                embeddings[mod] = torch.zeros(self.dims[mod], dtype=torch.float32)
                masks[mod] = torch.tensor([0.0], dtype=torch.float32)
                continue
            elif subject_id in self.idx_maps[mod]:
                row_idx = self.idx_maps[mod][subject_id]
                embeddings[mod] = self.data_store[mod][row_idx]
                masks[mod] = torch.tensor([1.0], dtype=torch.float32)
            else:
                embeddings[mod] = torch.zeros(self.dims[mod], dtype=torch.float32)
                masks[mod] = torch.tensor([0.0], dtype=torch.float32)

        return {
            "subject_id": subject_id,
            "embeddings": embeddings,
            "masks": masks,
            "label": torch.tensor(label, dtype=torch.float32),
        }

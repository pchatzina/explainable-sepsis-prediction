"""Tests for embedding .pt files across all modalities.

Validates that the embedding files produced by the extraction scripts
exist, have correct structure, and are internally consistent.

These are **integration tests** against real pipeline outputs — individual
tests skip when the relevant artifacts have not yet been generated.

Run:
    pytest tests/test_embeddings.py -v
"""

import pytest
import torch

from src.utils.config import Config

# ==========================================
# PATHS
# ==========================================

SPLITS = ["train", "valid", "test"]


# ==========================================
# SHARED EMBEDDING VALIDATION
# ==========================================


class _BaseEmbeddingTests:
    """Reusable embedding tests — subclassed per modality."""

    EMBEDDINGS_DIR = None  # Override in subclass

    @pytest.fixture(scope="class", params=SPLITS)
    def embedding_data(self, request):
        if self.EMBEDDINGS_DIR is None:
            pytest.skip("Base class")

        path = self.EMBEDDINGS_DIR / f"{request.param}_embeddings.pt"
        if not path.exists():
            pytest.skip(f"{path.name} not found")

        return torch.load(path, map_location="cpu", weights_only=False), request.param

    def test_required_keys(self, embedding_data):
        data, _ = embedding_data
        for key in ("embeddings", "labels", "subject_ids"):
            assert key in data, f"Missing key: {key}"

    def test_embeddings_are_2d_tensor(self, embedding_data):
        data, _ = embedding_data
        emb = data["embeddings"]
        assert isinstance(emb, torch.Tensor)
        assert emb.ndim == 2
        assert emb.shape[0] > 0, "No samples"
        assert emb.shape[1] > 0, "Embedding dim is 0"

    def test_embeddings_dtype_float32(self, embedding_data):
        data, _ = embedding_data
        assert data["embeddings"].dtype == torch.float32

    def test_no_nans_or_infs(self, embedding_data):
        data, _ = embedding_data
        emb = data["embeddings"]
        assert not torch.isnan(emb).any(), "NaN in embeddings"
        assert not torch.isinf(emb).any(), "Inf in embeddings"

    def test_embeddings_not_collapsed(self, embedding_data):
        """Ensures embeddings have some variance (ported from manual verify script)."""
        data, _ = embedding_data
        emb = data["embeddings"]
        assert emb.std().item() > 0.0, (
            "Embeddings standard deviation is 0 (collapsed representation)"
        )

    def test_labels_are_binary(self, embedding_data):
        """Allows both integer and string representations of binary labels."""
        data, _ = embedding_data
        assert all(str(l) in ("0", "1") for l in data["labels"])

    def test_subject_ids_unique(self, embedding_data):
        data, _ = embedding_data
        ids = data["subject_ids"]
        assert len(ids) == len(set(ids)), "Duplicate sample IDs"

    def test_lengths_consistent(self, embedding_data):
        data, _ = embedding_data
        n = data["embeddings"].shape[0]
        assert len(data["labels"]) == n
        assert len(data["subject_ids"]) == n

    def test_embedding_dim_consistent_across_splits(self):
        dims = []
        for split in SPLITS:
            path = self.EMBEDDINGS_DIR / f"{split}_embeddings.pt"
            if not path.exists():
                pytest.skip("Not all splits present yet")
            data = torch.load(path, map_location="cpu", weights_only=False)
            dims.append(data["embeddings"].shape[1])
        assert len(set(dims)) == 1, f"Inconsistent dims: {dims}"

    def test_no_sample_id_leakage(self):
        all_ids = []
        for split in SPLITS:
            path = self.EMBEDDINGS_DIR / f"{split}_embeddings.pt"
            if not path.exists():
                pytest.skip("Not all splits present yet")
            data = torch.load(path, map_location="cpu", weights_only=False)
            all_ids.extend(data["subject_ids"])
        assert len(all_ids) == len(set(all_ids)), "Sample ID leakage across splits"

    def test_has_both_classes(self, embedding_data):
        data, _ = embedding_data
        # Cast to strings for safe set comparison regardless of underlying type
        labels = set(str(l) for l in data["labels"])
        assert labels == {"0", "1"}, f"Expected both classes, got {labels}"

    def test_data_types_are_integers(self, embedding_data):
        """Ensures that both labels and subject_ids are saved strictly as integers."""
        data, _ = embedding_data

        # Check labels
        assert all(isinstance(label, int) for label in data["labels"]), (
            "All labels must be of type int (not str)"
        )

        # Check subject_ids
        assert all(isinstance(sid, int) for sid in data["subject_ids"]), (
            "All subject_ids must be of type int (not str) for Late-Fusion alignment"
        )


# ==========================================
# EHR EMBEDDINGS
# ==========================================


class TestEHREmbeddings(_BaseEmbeddingTests):
    """EHR embedding validation."""

    EMBEDDINGS_DIR = Config.PROCESSED_EHR_EMBEDDINGS_DIR


# ==========================================
# ECG EMBEDDINGS
# ==========================================


class TestECGEmbeddings(_BaseEmbeddingTests):
    """ECG embedding validation."""

    EMBEDDINGS_DIR = Config.PROCESSED_ECG_EMBEDDINGS_DIR


# ==========================================
# CXR IMAGE EMBEDDINGS
# ==========================================


class TestCXRImgEmbeddings(_BaseEmbeddingTests):
    """CXR Image embedding validation."""

    EMBEDDINGS_DIR = Config.PROCESSED_CXR_IMG_EMBEDDINGS_DIR


# ==========================================
# CXR REPORTS (TEXT) EMBEDDINGS
# ==========================================


class TestCXRTxtEmbeddings(_BaseEmbeddingTests):
    """CXR Text (Reports) embedding validation."""

    EMBEDDINGS_DIR = Config.PROCESSED_CXR_TXT_EMBEDDINGS_DIR

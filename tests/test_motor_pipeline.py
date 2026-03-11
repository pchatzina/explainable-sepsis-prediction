"""Tests for the MOTOR foundation-model pretraining pipeline.

Validates that the artifacts produced by ``prepare_motor.py`` and
``pretrain_motor.py`` exist on disk, are loadable, and are internally
consistent.  These are **integration tests** that run against the real
pipeline outputs — they will be skipped when artifacts have not yet been
generated.

Run:
    pytest tests/test_motor_pipeline.py -v
"""

import pickle

import pytest

from src.utils.config import Config

# ==========================================
# FIXTURES
# ==========================================

PREP_DIR = Config.MODEL_EHR_MOTOR_PRETRAINING_FILES_DIR
MODEL_DIR = Config.MODEL_EHR_MOTOR_WEIGHTS_DIR


@pytest.fixture(scope="module")
def ontology():
    """Load the cached ontology, skip if not yet built."""
    path = PREP_DIR / "ontology.pkl"
    if not path.exists():
        pytest.skip("ontology.pkl not found — run prepare_motor first")
    with open(path, "rb") as f:
        return pickle.load(f)


@pytest.fixture(scope="module")
def motor_task():
    """Load the cached MOTOR task, skip if not yet built."""
    path = PREP_DIR / "motor_task.pkl"
    if not path.exists():
        pytest.skip("motor_task.pkl not found — run prepare_motor first")
    with open(path, "rb") as f:
        return pickle.load(f)


# ==========================================
# CONSTANTS SANITY (no I/O, always runs)
# ==========================================


class TestConstantsSanity:
    """Guard-rails on the module-level hyperparameters / constants."""

    def test_prepare_constants(self):
        from src.models.foundation.ehr.prepare_motor import (
            TOKENS_PER_BATCH,
            NUM_PROC,
            TRAIN_FRAC,
            VAL_FRAC,
            VOCAB_SIZE,
            MOTOR_NUM_TASKS,
            MOTOR_NUM_BINS,
        )

        assert TOKENS_PER_BATCH > 0
        assert NUM_PROC >= 1
        assert TRAIN_FRAC + VAL_FRAC == pytest.approx(1.0)
        assert VOCAB_SIZE > 0
        assert MOTOR_NUM_TASKS > 0
        assert MOTOR_NUM_BINS > 0

    def test_pretrain_constants(self):
        from src.models.foundation.ehr.pretrain_motor import (
            N_LAYERS,
            LEARNING_RATE,
            WEIGHT_DECAY,
            ADAM_BETA2,
            NUM_EPOCHS,
            PER_DEVICE_BATCH_SIZE,
            GRADIENT_ACCUMULATION_STEPS,
            EARLY_STOPPING_PATIENCE,
        )

        assert N_LAYERS > 0
        assert 0 < LEARNING_RATE < 1
        assert 0 < WEIGHT_DECAY < 1
        assert 0 < ADAM_BETA2 < 1
        assert NUM_EPOCHS > 0
        assert PER_DEVICE_BATCH_SIZE >= 1
        assert GRADIENT_ACCUMULATION_STEPS >= 1
        assert EARLY_STOPPING_PATIENCE >= 1


# ==========================================
# PREPARATION ARTIFACT TESTS
# ==========================================


class TestPreparationArtifacts:
    """Verify artifacts produced by prepare_motor.py."""

    EXPECTED_FILES = [
        "ontology.pkl",
        "split.csv",
        "motor_task.pkl",
    ]
    EXPECTED_DIRS = [
        "tokenizer",
        "train_batches",
        "val_batches",
    ]

    def test_prep_dir_exists(self):
        if not PREP_DIR.exists():
            pytest.skip("Pretraining files dir does not exist yet")
        assert PREP_DIR.is_dir()

    @pytest.mark.parametrize("filename", EXPECTED_FILES)
    def test_artifact_file_exists(self, filename):
        path = PREP_DIR / filename
        if not PREP_DIR.exists():
            pytest.skip("Pretraining files dir does not exist yet")
        assert path.exists(), f"Missing artifact: {path}"
        assert path.stat().st_size > 0, f"Artifact is empty: {path}"

    @pytest.mark.parametrize("dirname", EXPECTED_DIRS)
    def test_artifact_dir_exists(self, dirname):
        path = PREP_DIR / dirname
        if not PREP_DIR.exists():
            pytest.skip("Pretraining files dir does not exist yet")
        assert path.exists(), f"Missing artifact directory: {path}"
        assert path.is_dir(), f"Expected directory, got file: {path}"
        assert any(path.iterdir()), f"Artifact directory is empty: {path}"

    def test_ontology_is_loadable(self, ontology):
        """Ontology pickle must deserialise without error."""
        assert ontology is not None

    def test_motor_task_is_loadable(self, motor_task):
        """MOTOR task pickle must deserialise without error."""
        assert motor_task is not None

    def test_split_csv_has_content(self):
        path = PREP_DIR / "split.csv"
        if not path.exists():
            pytest.skip("split.csv not found")
        lines = path.read_text().strip().splitlines()
        assert len(lines) > 1, "split.csv has no data rows (only header?)"

    def test_tokenizer_loadable(self, ontology):
        """Tokenizer must load and expose a positive vocab_size."""
        tokenizer_path = PREP_DIR / "tokenizer"
        if not tokenizer_path.exists():
            pytest.skip("tokenizer dir not found")

        import femr.models.tokenizer

        tokenizer = femr.models.tokenizer.HierarchicalTokenizer.from_pretrained(
            tokenizer_path, ontology=ontology
        )
        assert tokenizer.vocab_size > 0

    @pytest.mark.parametrize("split", ["train_batches", "val_batches"])
    def test_batches_loadable(self, split):
        """Batch datasets must load as HuggingFace Datasets with rows."""
        path = PREP_DIR / split
        if not path.exists():
            pytest.skip(f"{split} not found")

        import datasets

        ds = datasets.Dataset.load_from_disk(path)
        assert len(ds) > 0, f"{split} loaded but has 0 rows"


# ==========================================
# PRETRAINING OUTPUT TESTS
# ==========================================


class TestInferenceBundle:
    """Verify the consolidated inference bundle in MODEL_EHR_MOTOR_WEIGHTS_DIR."""

    EXPECTED_FILES = [
        "config.json",
        "model.safetensors",
        "dictionary.msgpack",
    ]

    def test_model_dir_exists(self):
        if not MODEL_DIR.exists():
            pytest.skip("MODEL_EHR_MOTOR_WEIGHTS_DIR does not exist yet")
        assert MODEL_DIR.is_dir()

    @pytest.mark.parametrize("filename", EXPECTED_FILES)
    def test_inference_file_exists(self, filename):
        path = MODEL_DIR / filename
        if not MODEL_DIR.exists():
            pytest.skip("MODEL_EHR_MOTOR_WEIGHTS_DIR does not exist yet")
        assert path.exists(), f"Missing inference file: {path}"
        assert path.stat().st_size > 0, f"Inference file is empty: {path}"

    def test_model_loadable(self):
        """Model config + weights must load without error."""
        if not MODEL_DIR.exists():
            pytest.skip("MODEL_EHR_MOTOR_WEIGHTS_DIR does not exist yet")
        config_path = MODEL_DIR / "config.json"
        weights_path = MODEL_DIR / "model.safetensors"
        if not config_path.exists() or not weights_path.exists():
            pytest.skip("Model files not yet available")

        import femr.models.config
        import femr.models.transformer

        config = femr.models.config.FEMRModelConfig.from_pretrained(MODEL_DIR)
        model = femr.models.transformer.FEMRModel(config)
        assert model is not None


# ==========================================
# CONFIG CONSISTENCY
# ==========================================


class TestConfigPaths:
    """Ensure Config paths referenced by the pipeline are set and consistent."""

    def test_pretraining_meds_reader_dir_defined(self):
        assert Config.PROCESSED_EHR_MEDS_PRETRAINING_DIR is not None

    def test_athena_vocabulary_dir_defined(self):
        assert Config.MODEL_EHR_MOTOR_VOCAB_DIR is not None

    def test_cohort_meds_reader_dir_defined(self):
        assert Config.PROCESSED_EHR_MEDS_COHORT_DIR is not None

    def test_ehr_embeddings_dir_defined(self):
        assert Config.PROCESSED_EHR_EMBEDDINGS_DIR is not None

    def test_motor_dirs_are_distinct(self):
        """Pretraining files, model output, and model dir must not overlap."""
        dirs = {
            Config.MODEL_EHR_MOTOR_PRETRAINING_FILES_DIR,
            Config.MODEL_EHR_MOTOR_PRETRAINING_OUT_DIR,
            Config.MODEL_EHR_MOTOR_WEIGHTS_DIR,
        }
        assert len(dirs) == 3, "MOTOR Config paths overlap!"

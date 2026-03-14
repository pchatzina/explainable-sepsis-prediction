import logging
import os
import random
import sys
from pathlib import Path
from dotenv import load_dotenv

import numpy as np
import torch

# Load environment variables from .env file
load_dotenv()

logger = logging.getLogger(__name__)


class Config:
    """Centralized Configuration for the MIMIC-IV Multimodal Pipeline."""

    # =========================================================================
    # 1. PROJECT & BASE DIRECTORIES
    # =========================================================================
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

    _raw_env = os.getenv("RAW_DATA_DIR")
    _processed_env = os.getenv("PROCESSED_DATA_DIR")
    _models_env = os.getenv("MODELS_DATA_DIR")
    _logs_env = os.getenv("LOGS_DATA_DIR")

    if not all([_raw_env, _processed_env, _models_env, _logs_env]):
        logger.error("Missing required data directories in .env file.")
        sys.exit(1)

    DIR_RAW = Path(_raw_env)
    DIR_PROCESSED = Path(_processed_env)
    DIR_MODELS = Path(_models_env)
    DIR_LOGS = Path(_logs_env)

    # =========================================================================
    # 2. DATABASE & CREDENTIALS
    # =========================================================================
    DB_HOST = os.getenv("DB_HOST", "localhost")
    DB_PORT = os.getenv("DB_PORT", "5432")
    DB_NAME = os.getenv("DB_NAME", "mimiciv")
    DB_USER = os.getenv("DB_USER", "postgres")
    DB_PASSWORD = os.getenv("DB_PASSWORD", "postgres")

    PHYSIONET_USER = os.getenv("PHYSIONET_USER")
    PHYSIONET_PASS = os.getenv("PHYSIONET_PASS")

    # =========================================================================
    # 3. REMOTE URLS & METADATA FILES
    # =========================================================================
    URL_CXR_BASE = "https://physionet.org/files/mimic-cxr/2.1.0/"
    URL_CXR_JPG_BASE = "https://physionet.org/files/mimic-cxr-jpg/2.1.0/"
    URL_ECG_BASE = "https://physionet.org/files/mimic-iv-ecg/1.0/"

    CXR_METADATA_FILES = [
        ("cxr-record-list.csv.gz", URL_CXR_BASE),
        ("cxr-study-list.csv.gz", URL_CXR_BASE),
        ("mimic-cxr-2.0.0-metadata.csv.gz", URL_CXR_JPG_BASE),
    ]
    CXR_REPORTS_FILE = "mimic-cxr-reports.zip"
    ECG_METADATA_FILES = ["record_list.csv", "machine_measurements.csv"]

    # =========================================================================
    # 4. RAW DATA PATHS (Convention: RAW_{MODALITY}_{DESCRIPTOR}_DIR)
    # =========================================================================
    # EHR
    RAW_EHR_COHORT_DIR = DIR_RAW / "ehr" / "cohort" / "2.2"
    RAW_EHR_PRETRAINING_DIR = DIR_RAW / "ehr" / "pretraining" / "2.2"

    # ECG & CXR
    RAW_ECG_DIR = DIR_RAW / "ecg"
    RAW_CXR_IMG_DIR = DIR_RAW / "cxr_img"
    RAW_CXR_TXT_DIR = DIR_RAW / "cxr_txt"

    # Source code specific mappings
    SRC_EHR_EXPORTS_DIR = (
        PROJECT_ROOT / "src" / "data" / "preprocess" / "ehr" / "exports"
    )
    EHR_TRANSFORMATIONS_SRC_DIR = (
        PROJECT_ROOT / "src" / "data" / "preprocess" / "ehr" / "transformations"
    )
    SRC_DB_SETUP_DIR = PROJECT_ROOT / "src" / "data" / "acquisition" / "db" / "setup"
    HYDRA_CONFIG_DIR = PROJECT_ROOT / "conf"

    # =========================================================================
    # 5. PROCESSED DATA PATHS
    # =========================================================================
    # EHR
    PROCESSED_EHR_COHORT_DIR = DIR_PROCESSED / "ehr" / "cohort"
    PROCESSED_EHR_PRETRAINING_DIR = DIR_PROCESSED / "ehr" / "pretraining"
    PROCESSED_EHR_LABELS_DIR = DIR_PROCESSED / "ehr" / "labels"
    PROCESSED_EHR_EMBEDDINGS_DIR = DIR_PROCESSED / "ehr" / "embeddings"

    # EHR MEDS Reader dependencies
    PROCESSED_EHR_MEDS_COHORT_DIR = (
        DIR_PROCESSED / "ehr" / "cohort" / "mimic-iv-meds-reader"
    )
    PROCESSED_EHR_MEDS_PRETRAINING_DIR = (
        DIR_PROCESSED / "ehr" / "pretraining" / "mimic-iv-meds-reader"
    )
    PROCESSED_EHR_MEDS_METADATA_FILE = (
        DIR_PROCESSED
        / "ehr"
        / "pretraining"
        / "mimic-iv-meds"
        / "metadata"
        / "codes.parquet"
    )

    PROCESSED_COHORT_PARQUET_FILE = DIR_PROCESSED / "master_cohort.parquet"

    # ECG
    PROCESSED_ECG_ROOT_DIR = DIR_PROCESSED / "ecg"
    PROCESSED_ECG_EMBEDDINGS_DIR = DIR_PROCESSED / "ecg" / "embeddings"

    # CXR
    PROCESSED_CXR_IMG_EMBEDDINGS_DIR = DIR_PROCESSED / "cxr_img" / "embeddings"
    PROCESSED_CXR_TXT_EMBEDDINGS_DIR = DIR_PROCESSED / "cxr_txt" / "embeddings"

    # =========================================================================
    # 6. MODEL ARTIFACT PATHS
    # =========================================================================
    # EHR - MOTOR Foundation Model
    MODEL_EHR_MOTOR_PRETRAINING_FILES_DIR = (
        DIR_MODELS / "ehr" / "motor" / "pretraining_files"
    )
    MODEL_EHR_MOTOR_PRETRAINING_OUT_DIR = (
        DIR_MODELS / "ehr" / "motor" / "pretraining_output"
    )
    MODEL_EHR_MOTOR_WEIGHTS_DIR = DIR_MODELS / "ehr" / "motor" / "model"
    MODEL_EHR_MOTOR_VOCAB_DIR = DIR_MODELS / "ehr" / "motor" / "athena_vocabulary"

    # EHR - Downstream Models
    MODEL_EHR_LR_DIR = DIR_MODELS / "ehr" / "lr"
    MODEL_EHR_XGBOOST_DIR = DIR_MODELS / "ehr" / "xgboost"
    MODEL_EHR_MLP_DIR = DIR_MODELS / "ehr" / "mlp"

    # ECG
    MODEL_ECG_PRETRAINED_DIR = DIR_MODELS / "ecg" / "pretrained"
    MODEL_ECG_LR_DIR = DIR_MODELS / "ecg" / "lr"
    MODEL_ECG_XGBOOST_DIR = DIR_MODELS / "ecg" / "xgboost"
    MODEL_ECG_MLP_DIR = DIR_MODELS / "ecg" / "mlp"

    # CXR Images
    MODEL_CXR_IMG_PRETRAINED_DIR = DIR_MODELS / "cxr_img" / "pretrained"
    MODEL_CXR_IMG_LR_DIR = DIR_MODELS / "cxr_img" / "lr"
    MODEL_CXR_IMG_XGBOOST_DIR = DIR_MODELS / "cxr_img" / "xgboost"
    MODEL_CXR_IMG_MLP_DIR = DIR_MODELS / "cxr_img" / "mlp"

    # CXR Reports (Text)
    MODEL_CXR_TXT_PRETRAINED_DIR = DIR_MODELS / "cxr_txt" / "pretrained"
    MODEL_CXR_TXT_LR_DIR = DIR_MODELS / "cxr_txt" / "lr"
    MODEL_CXR_TXT_XGBOOST_DIR = DIR_MODELS / "cxr_txt" / "xgboost"
    MODEL_CXR_TXT_MLP_DIR = DIR_MODELS / "cxr_txt" / "mlp"

    # Fusion
    MODEL_FUSION_DIR = DIR_MODELS / "fusion"

    # =========================================================================
    # 7. RESULTS & LOGGING PATHS
    # =========================================================================
    LOGS_ROOT_DIR = DIR_LOGS
    TENSORBOARD_DIR = LOGS_ROOT_DIR / "tensorboard"
    RESULTS_DIR = PROJECT_ROOT / "results"
    LOGS_ROOT_DIR = DIR_LOGS

    # =========================================================================
    # 8. UTILITY METHODS
    # =========================================================================
    @classmethod
    def check_dirs(cls):
        """
        Dynamically scans the Config class for any attribute ending in '_DIR'
        that is a Path object, and creates the directory if it doesn't exist.
        """
        for attr_name in dir(cls):
            if attr_name.endswith("_DIR"):
                path_obj = getattr(cls, attr_name)
                if isinstance(path_obj, Path):
                    if not path_obj.exists():
                        path_obj.mkdir(parents=True, exist_ok=True)
                        logger.info("Created directory: %s", path_obj)

    @classmethod
    def get_db_url(cls):
        """Returns the connection URL for SQLAlchemy/Pandas."""
        return f"postgresql://{cls.DB_USER}:{cls.DB_PASSWORD}@{cls.DB_HOST}:{cls.DB_PORT}/{cls.DB_NAME}"

    @staticmethod
    def setup_logging(level: int = logging.INFO) -> None:
        """Configure root logger with the project-wide format."""
        logging.basicConfig(
            level=level,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        )

    @staticmethod
    def set_seed(seed: int = 42) -> None:
        """Locks all random seeds for strict reproducibility across the project."""
        logger.info(f"Setting global random seed to {seed}...")

        os.environ["PYTHONHASHSEED"] = str(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

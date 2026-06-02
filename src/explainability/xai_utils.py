"""
Shared utilities for the explainability analysis scripts (clinical_inference_xai.py,
extract_clinical_archetypes.py, find_dataset_noise.py).

EHREndToEndWrapper bridges the MOTOR transformer and the trained EHR MLP into a single
nn.Module compatible with Captum attribution methods. The wrapper runs the MOTOR forward
pass under bfloat16 autocast — required by the FlashAttention kernel MOTOR uses — and
returns logits cast to float32 so attribution gradients propagate cleanly. This
precision-bridge pattern is load-bearing for the Chapter 9 explainability analysis.

Patient isolation: the wrapper's forward pass accepts a patient_pos argument that
extracts a single subject's representation from the packed FEMR batch before computing
logits. This prevents cross-patient gradient leakage when Captum integrates gradients
over a single sample.

Other helpers provided:
  load_pipeline        — loads MOTOR + EHR MLP, wraps them, and returns the wrapper
                         along with calibration scalars (temperature, optimal threshold).
  prepare_batch        — recursively adds a batch dimension and moves tensors to the
                         target device; required by Captum's expected input shape.
  load_raw_msgpack_dictionary
                       — reads the MOTOR vocabulary from its MessagePack dictionary file.
  get_token_string_resilient
                       — converts a FEMR token ID to a human-readable label, with
                         fallback through Athena and MIMIC-IV lookup tables.
  load_athena_mapping  — loads Athena CONCEPT.csv into a fast-lookup dict.
  load_mimic_mapping   — queries the MIMIC-IV item-definition tables for ICU/lab labels.
"""
import logging
from typing import Dict, Any, List, Optional, Tuple, Union

import json
import msgpack
import torch
import torch.nn as nn
import pandas as pd
import femr.models.transformer

from src.utils.database import query_to_df
from src.utils.config import Config
from src.models.unimodal.mlp import UnimodalModule

logger = logging.getLogger(__name__)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class EHREndToEndWrapper(nn.Module):
    """
    Wraps the MOTOR transformer and unimodal EHR MLP into single module.
    """

    def __init__(self, motor_model: nn.Module, mlp_model: nn.Module):
        super().__init__()
        self.motor = motor_model
        self.motor.config.task_config = None
        self.mlp = mlp_model

        # Freeze both models (inference-only)
        self.motor.eval()
        for param in self.motor.parameters():
            param.requires_grad = False
        self.mlp.eval()

    def forward(
        self,
        dummy_input: torch.Tensor,
        batch_dict: Dict[str, Any],
        patient_pos: int = None,
    ) -> torch.Tensor:
        """
        Forward pass through MOTOR + MLP.

        Args:
            dummy_input: Placeholder tensor (required by Captum)
            batch_dict: FEMR batch dict
            patient_pos: Index of the target patient in the batch to isolate gradients
        """
        with torch.autocast(device_type=DEVICE.type, dtype=torch.bfloat16):
            _, motor_outputs = self.motor(
                batch_dict, return_loss=True, return_reprs=True
            )
            patient_rep = motor_outputs["representations"]
            logits = self.mlp(patient_rep)

        # Isolate target patient to prevent cross-patient gradient leakage
        # Unsqueeze maintains the [Batch, Class] shape required by Captum.
        if patient_pos is not None:
            logits = logits[patient_pos].unsqueeze(0)

        return logits.to(torch.float32)


def load_pipeline() -> Tuple[EHREndToEndWrapper, float, float]:
    """
    Loads and prepares MOTOR foundation model and unimodal EHR MLP for attribution analysis.

    Steps:
      1. Load the MOTOR foundation model
      2. Load the trained EHR MLP checkpoint
      3. Load calibration temperature
      4. Wrap both models in EHREndToEndWrapper for Captum

    Returns:
        Tuple[EHREndToEndWrapper, float, float]:
            - wrapper: EHREndToEndWrapper instance ready for Captum
            - t_val: Temperature scaling factor for calibration
            - optimal_threshold: Optimal classification threshold
    """
    logger.info("Loading MOTOR Foundation Model...")
    motor_model = femr.models.transformer.FEMRModel.from_pretrained(
        str(Config.MODEL_EHR_MOTOR_WEIGHTS_DIR)
    ).to(DEVICE)

    logger.info("Loading Unimodal EHR MLP...")
    ehr_checkpoint_dir = Config.DIR_MODELS / "ehr" / "mlp"
    model_path = ehr_checkpoint_dir / "best_ehr_mlp_calibrated.ckpt"
    ehr_model = UnimodalModule.load_from_checkpoint(model_path, weights_only=False)
    t_val = ehr_model.temperature

    optimal_threshold = 0.5
    metrics_file = (
        Config.RESULTS_DIR / "unimodal" / "metrics" / "ehr" / "mlp_metrics.json"
    )
    if metrics_file.exists():
        with open(metrics_file, "r") as f:
            optimal_threshold = json.load(f).get("threshold", 0.5)

    ehr_model.to(DEVICE)

    wrapper = EHREndToEndWrapper(motor_model, ehr_model.model)
    return wrapper, t_val, optimal_threshold


def prepare_batch(data: Any) -> Any:
    """
    Recursively adds batch dimension and moves tensors to GPU.

    This is necessary because Captum's LayerGradientXActivation expects
    batch-dimension consistency across all inputs.
    """
    if isinstance(data, dict):
        return {k: prepare_batch(v) for k, v in data.items()}
    elif isinstance(data, torch.Tensor):
        return data.unsqueeze(0).to(DEVICE)
    return data


def load_raw_msgpack_dictionary() -> List[Dict[str, Any]]:
    """
    Loads the raw MOTOR vocabulary dictionary with MessagePack format.
    """
    dict_path = Config.MODEL_EHR_MOTOR_WEIGHTS_DIR / "dictionary.msgpack"
    with open(dict_path, "rb") as f:
        return msgpack.unpackb(f.read())["vocab"]


def get_token_string_resilient(
    vocab_list: List[Dict[str, Any]],
    token_id: Union[int, float, str],
    athena_mapping: Optional[Dict] = None,
    mimic_mapping: Optional[Dict] = None,
) -> str:
    """
    Converts a token ID to a human-readable string with error handling.

    Attempts to resolve token IDs through multiple lookup hierarchies:
    1. Direct vocabulary lookup → extract token metadata by type
    2. For "code" tokens: Try Athena mapping first, then MIMIC lookup
    3. For "text" tokens: Format as property: text_string
    4. For numeric ranges: Format as property: [val_start, val_end)
    5. On any error: Return descriptive fallback string

    Args:
        vocab_list: FEMR vocabulary loaded via load_raw_msgpack_dictionary()
        token_id: Token index to resolve
        athena_mapping: Mapping from "VOCAB/CODE" to concept names
        mimic_mapping: Mapping from itemid to MIMIC-IV labels

    Returns:
        Human-readable token description
    """
    token_id = int(token_id)
    if token_id >= len(vocab_list) or token_id < 0:
        return f"Out_of_Bounds_{token_id}"

    item = vocab_list[token_id]
    try:
        if item.get("type") == "code" and "code_string" in item:
            raw_code = item["code_string"].strip()

            if athena_mapping and raw_code in athena_mapping:
                return f"{raw_code} ({athena_mapping[raw_code]})"

            if mimic_mapping and ("MIMIC_IV_" in raw_code):
                itemid = raw_code.split("/")[-1]
                if itemid in mimic_mapping:
                    return f"{raw_code} ({mimic_mapping[itemid]})"

            return raw_code

        elif item.get("type") == "text" and "text_string" in item:
            return (
                f"{item.get('property', 'text').strip()}: {item['text_string'].strip()}"
            )

        elif "val_start" in item and "val_end" in item:
            prop = item.get(
                "property", item.get("code_string", "Numeric_Feature")
            ).strip()

            # Same lookup logic for numeric features if they use standard codes
            if athena_mapping and prop in athena_mapping:
                prop = f"{prop} ({athena_mapping[prop]})"

            start_str = (
                f"{item['val_start']:.2f}"
                if isinstance(item["val_start"], float)
                else str(item["val_start"])
            )
            end_str = (
                f"{item['val_end']:.2f}"
                if isinstance(item["val_end"], float)
                else str(item["val_end"])
            )
            return f"{prop}: [{start_str}, {end_str})"

        else:
            return str({k: v for k, v in item.items() if k != "weight"})

    except Exception:
        return f"Parse_Error_{token_id}"


def load_mimic_mapping() -> dict:
    """
    Fetches human-readable clinical labels for MIMIC-IV item IDs from the database.

    Queries two separate MIMIC-IV tables to build a unified mapping of itemid to label:
    - mimiciv_hosp.d_labitems: Laboratory test definitions (blood work, microbiology, etc.)
    - mimiciv_icu.d_items: ICU charted events (vitals, medications, complications, etc.)
    """
    logger.info("Fetching MIMIC-IV item definitions from database...")
    query = """
        SELECT CAST(itemid AS VARCHAR) AS itemid, label 
        FROM mimiciv_hosp.d_labitems
        UNION ALL
        SELECT CAST(itemid AS VARCHAR) AS itemid, label 
        FROM mimiciv_icu.d_items
    """
    try:
        df = query_to_df(query)
        # Drop duplicates in case an itemid exists in multiple places
        return df.drop_duplicates("itemid").set_index("itemid")["label"].to_dict()
    except Exception as e:
        logger.warning(f"Failed to load MIMIC mappings: {e}")
        return {}


def load_athena_mapping(csv_path: str) -> Dict[str, str]:
    """
    Loads Athena CONCEPT vocabulary CSV into a fast lookup dictionary for token resolution.

    The Athena CONCEPT.csv file contains standardized clinical concept definitions from
    multiple vocabularies. This function creates a mapping from token keys to
    human-readable concept names.

    Args:
        csv_path (str): Path to Athena CONCEPT.csv file (tab-separated)

    Returns:
        Dict[str, str]: Dictionary mapping token keys to concept names.
            If file not found or parsing fails, returns empty dict with warning.
    """
    logger.info(f"Loading Athena vocabulary mapping...")
    try:
        # Load only the columns we need
        df = pd.read_csv(
            csv_path,
            sep="\t",
            dtype=str,
            usecols=["vocabulary_id", "concept_code", "concept_name"],
        )
        # Create a combined key matching FEMR format
        df["token_key"] = df["vocabulary_id"] + "/" + df["concept_code"]

        # Drop duplicates just in case, then convert to a dictionary
        mapping_dict = (
            df.drop_duplicates("token_key")
            .set_index("token_key")["concept_name"]
            .to_dict()
        )
        return mapping_dict
    except Exception as e:
        logger.warning(
            f"Failed to load Athena mapping: {e}. Falling back to raw tokens."
        )
        return {}

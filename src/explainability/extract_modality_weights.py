"""
Extracts the modality gating weights and synergy coefficient
from the fusion model across the test set.

This module performs inference on the test split using the trained fusion model
and extracts the dynamic gating weights that control the relative contribution
of each modality (EHR, ECG, CXR Image, CXR Text) to the final sepsis risk prediction.

Key concepts:
  - w_ehr, w_ecg, w_cxr_img, w_cxr_txt: Individual modality attention weights
  - beta (synergy coefficient): Defined as 1 - max(w_i). Measures multimodal synergy.
  - Beta = 0: One modality dominates (unimodal behavior)
  - Beta > 0: Genuine multimodal synergy (auxiliary modalities contribute)

The extracted weights are saved to CSV for downstream analysis.

Usage:
    python -m src.explainability.extract_modality_weights
"""

import logging
import pandas as pd
import torch

from src.utils.config import Config
from src.data.loaders.fusion_datamodule import FusionDataModule
from src.models.fusion.late_fusion_module import LateFusionModule

logger = logging.getLogger(__name__)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main() -> None:
    """
    Main pipeline for extracting modality gating weights from the fusion model.

    Steps:
      1. Load the best 4-modality late-fusion model
      2. Setup the FusionDataModule (test split only)
      3. Perform inference on the test set with no EHR dropout
      4. Extract modality weights (w_ehr, w_ecg, w_cxr_img, w_cxr_txt)
      5. Compute synergy coefficient (beta = 1 - max(w_i))
      6. Save results to CSV for downstream XAI analysis
    """
    Config.setup_logging()
    base_exp_dir = (
        Config.RESULTS_DIR / "explainability" / "modality_weights" / "4mod_architecture"
    )
    base_exp_dir.mkdir(parents=True, exist_ok=True)
    csv_path = base_exp_dir / "test_set_modality_weights.csv"

    # 1. Load the model
    run_name = Config.FUSION_RUN_PRETRAINED
    checkpoint_dir = Config.DIR_MODELS / "fusion" / run_name
    model_path = checkpoint_dir / f"best_{run_name}_calibrated.ckpt"
    logger.info(f"Loading model from {model_path.name}")
    model = LateFusionModule.load_from_checkpoint(model_path, weights_only=False)
    model.eval()
    model.to(DEVICE)

    # 2. Setup DataModule
    # Note: ehr_dropout_rate=0.0 ensures strict evaluation
    datamodule = FusionDataModule(
        active_modalities=model.active_modalities,
        batch_size=256,
        num_workers=4,
        ehr_dropout_rate=0.0,
    )
    datamodule.setup()
    test_loader = datamodule.test_dataloader()

    # 3. Extract Weights
    logger.info("Extracting modality weights across the Test Set...")
    results = {"subject_id": [], "true_label": [], "p_final": [], "beta": []}
    for mod in model.active_modalities:
        results[f"w_{mod}"] = []

    with torch.no_grad():
        for batch in test_loader:
            # Move embeddings and masks to GPU
            embeddings = {k: v.to(DEVICE) for k, v in batch["embeddings"].items()}
            masks = {k: v.to(DEVICE) for k, v in batch["masks"].items()}
            targets = batch["label"].numpy()

            # Handle subject IDs safely
            sids = batch["subject_id"]
            if isinstance(sids, torch.Tensor):
                sids = sids.cpu().numpy().tolist()
            elif isinstance(sids, list) and isinstance(sids[0], torch.Tensor):
                sids = [s.item() for s in sids]

            # Forward pass: get model outputs including gating weights
            outputs = model(embeddings, masks)

            results["subject_id"].extend(sids)
            results["true_label"].extend(targets)
            results["p_final"].extend(outputs["p_final"].cpu().numpy().flatten())
            results["beta"].extend(outputs["beta"].cpu().numpy().flatten())

            # outputs["weights"] has shape [Batch, num_modalities]
            w_batch = outputs["weights"].cpu().numpy()
            for i, mod in enumerate(model.active_modalities):
                results[f"w_{mod}"].extend(w_batch[:, i])

    # Convert to DataFrame and save
    df = pd.DataFrame(results)
    df.to_csv(csv_path, index=False)
    logger.info(f"Saved tabular weight data to {csv_path}")

    logger.info("Modality Weights Extraction Complete.")


if __name__ == "__main__":
    main()

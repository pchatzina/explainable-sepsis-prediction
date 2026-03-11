"""
Pretrain a MOTOR transformer on MEDS-formatted EHR data.

Loads the artifacts produced by ``prepare_motor.py`` (ontology, tokenizer,
MOTOR task, tokenised batches) and trains a FEMRModel from scratch using
HuggingFace's Trainer with the MOTOR pretraining objective.

The trained model is saved to ``Config.MODEL_EHR_MOTOR_PRETRAINING_OUT_DIR`` with
checkpoints managed by the Trainer.  After training, the script consolidates
the inference-ready bundle (model weights, config, tokenizer dictionary) into
``Config.MODEL_EHR_MOTOR_WEIGHTS_DIR``.

Usage:
    python -m src.models.foundation.ehr.pretrain_motor
"""

import logging
import pickle
import shutil

import datasets
import torch
import transformers

import femr.models.config
import femr.models.processor
import femr.models.tokenizer
import femr.models.transformer

from src.utils.config import Config

logger = logging.getLogger(__name__)

# ==========================================
# ARCHITECTURE
# ==========================================

# Transformer depth.
N_LAYERS = 12

# ==========================================
# TRAINING HYPERPARAMETERS
# ==========================================

LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.1
ADAM_BETA2 = 0.95
WARMUP_STEPS = 500
NUM_EPOCHS = 30

# Stop training when eval loss does not improve for N consecutive evaluations.
# With EVAL_EVERY_STEPS=500 this means 2500 optimiser steps of patience.
EARLY_STOPPING_PATIENCE = 5

# Effective batch size = per_device × gradient_accumulation = 1 × 32 = 32.
PER_DEVICE_BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 32

# Logging / evaluation cadence (in optimiser steps).
LOG_EVERY_STEPS = 100
EVAL_EVERY_STEPS = 500

# DataLoader workers.
DATALOADER_NUM_WORKERS = 8
DATALOADER_PREFETCH_FACTOR = 2

# Keep the N most recent checkpoints.
SAVE_TOTAL_LIMIT = 2


# ==========================================
# MAIN
# ==========================================


def main():
    """Load pretraining artifacts and run MOTOR pretraining."""
    Config.setup_logging()

    prep_dir = Config.MODEL_EHR_MOTOR_PRETRAINING_FILES_DIR
    out_dir = Config.MODEL_EHR_MOTOR_PRETRAINING_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load pretraining artifacts produced by prepare_motor.py
    # ------------------------------------------------------------------
    logger.info("Loading ontology and tokenizer from %s", prep_dir)

    required_artifacts = [
        prep_dir / "ontology.pkl",
        prep_dir / "tokenizer",
        prep_dir / "motor_task.pkl",
        prep_dir / "train_batches",
        prep_dir / "val_batches",
    ]
    for artifact in required_artifacts:
        if not artifact.exists():
            logger.error("Missing artifact: %s — run prepare_motor.py first", artifact)
            return

    with open(prep_dir / "ontology.pkl", "rb") as f:
        ontology = pickle.load(f)

    tokenizer = femr.models.tokenizer.HierarchicalTokenizer.from_pretrained(
        prep_dir / "tokenizer", ontology=ontology
    )

    with open(prep_dir / "motor_task.pkl", "rb") as f:
        motor_task = pickle.load(f)

    processor = femr.models.processor.FEMRBatchProcessor(tokenizer, motor_task)

    # ------------------------------------------------------------------
    # 2. Load tokenised datasets (memory-mapped, lazy)
    # ------------------------------------------------------------------
    logger.info("Loading train/val batches")
    train_batches = datasets.Dataset.load_from_disk(prep_dir / "train_batches")
    val_batches = datasets.Dataset.load_from_disk(prep_dir / "val_batches")

    # ------------------------------------------------------------------
    # 3. Build model
    # ------------------------------------------------------------------
    transformer_config = femr.models.config.FEMRTransformerConfig(
        vocab_size=tokenizer.vocab_size,
        is_hierarchical=True,
        n_layers=N_LAYERS,
        use_normed_ages=True,
        use_bias=False,
        hidden_act="swiglu",
    )

    model_config = femr.models.config.FEMRModelConfig.from_transformer_task_configs(
        transformer_config, motor_task.get_task_config()
    )

    model = femr.models.transformer.FEMRModel(model_config)
    model = model.to(torch.device("cuda"))
    logger.info(
        "Model initialised (%d layers, vocab %d)", N_LAYERS, tokenizer.vocab_size
    )

    # ------------------------------------------------------------------
    # 4. Configure Trainer
    # ------------------------------------------------------------------
    trainer_config = transformers.TrainingArguments(
        output_dir=str(out_dir),
        per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
        per_device_eval_batch_size=PER_DEVICE_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        adam_beta2=ADAM_BETA2,
        warmup_steps=WARMUP_STEPS,
        num_train_epochs=NUM_EPOCHS,
        remove_unused_columns=False,
        bf16=True,
        report_to="tensorboard",
        logging_strategy="steps",
        logging_steps=LOG_EVERY_STEPS,
        disable_tqdm=False,
        eval_strategy="steps",
        eval_steps=EVAL_EVERY_STEPS,
        save_strategy="steps",
        save_steps=EVAL_EVERY_STEPS,
        prediction_loss_only=True,
        dataloader_num_workers=DATALOADER_NUM_WORKERS,
        dataloader_prefetch_factor=DATALOADER_PREFETCH_FACTOR,
        dataloader_pin_memory=True,
        save_total_limit=SAVE_TOTAL_LIMIT,
        load_best_model_at_end=True,
        metric_for_best_model="loss",
    )

    trainer = transformers.Trainer(
        model=model,
        data_collator=processor.collate,
        train_dataset=train_batches,
        eval_dataset=val_batches,
        args=trainer_config,
        callbacks=[
            transformers.EarlyStoppingCallback(
                early_stopping_patience=EARLY_STOPPING_PATIENCE
            ),
        ],
    )

    # ------------------------------------------------------------------
    # 5. Train
    # ------------------------------------------------------------------
    logger.info("Starting training — output dir: %s", out_dir)
    trainer.train()

    # Automatically save and copy the best model bundle to match your manual process
    final_dir = out_dir / "final"
    trainer.save_model(str(final_dir))
    logger.info("Best model saved to %s", final_dir)

    model_dir = Config.MODEL_EHR_MOTOR_WEIGHTS_DIR
    model_dir.mkdir(parents=True, exist_ok=True)

    for fname in ("config.json", "model.safetensors"):
        src = final_dir / fname
        if src.exists():
            shutil.copy2(src, model_dir / fname)
            logger.info("Copied %s → %s", src, model_dir / fname)
        else:
            logger.warning("Expected %s not found in final/ — skipping", fname)

    dict_src = prep_dir / "tokenizer" / "dictionary.msgpack"
    if dict_src.exists():
        shutil.copy2(dict_src, model_dir / "dictionary.msgpack")
        logger.info("Copied %s → %s", dict_src, model_dir / "dictionary.msgpack")
    else:
        logger.warning("dictionary.msgpack not found at %s — skipping", dict_src)

    logger.info("Inference bundle ready at %s", model_dir)


if __name__ == "__main__":
    main()

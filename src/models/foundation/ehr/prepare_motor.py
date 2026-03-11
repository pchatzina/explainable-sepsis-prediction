"""
Prepare MOTOR pretraining artifacts from MEDS-formatted EHR data.

Builds the intermediate artifacts required before MOTOR pretraining can start:
    1. Ontology   — pruned OHDSI concept graph (via Athena vocabularies)
    2. Splits     — deterministic 90/10 train/val hash-based split
    3. Tokenizer  — hierarchical vocabulary over the medical ontology
    4. MOTOR task — time-to-event distribution head parameters
    5. Batches    — tokenised HuggingFace Datasets (train + val)

Every step is idempotent: if the output artifact already exists on disk it is
loaded instead of recomputed, so the script can safely be re-run after a crash.

Usage:
    python -m src.models.foundation.ehr.prepare_motor
"""

import logging
import pickle

import femr.models.processor
import femr.models.tasks
import femr.models.tokenizer
import femr.ontology
import femr.splits
import meds_reader

from src.utils.config import Config

logger = logging.getLogger(__name__)

# ==========================================
# CONSTANTS
# ==========================================

# Tokens fed to the model per batch during dataset conversion.
# 8192 is conservative for 16 GB VRAM.
TOKENS_PER_BATCH = 8192

# Parallel workers for meds_reader and batch conversion.
# 4 is conservative for 16 GB RAM.
NUM_PROC = 4

# Pretraining split fractions (no test set — downstream evaluation uses the
# cohort splits defined in PostgreSQL).
TRAIN_FRAC = 0.90
VAL_FRAC = 0.10

# Seed used by femr.splits.generate_hash_split for deterministic splitting.
SPLIT_SEED = 97

# Ontologies pruned from the OHDSI graph to save memory.  LOINC is retained
# because it maps well to MIMIC lab codes.
PRUNED_ONTOLOGIES = {"SPL", "HemOnc"}

# Tokenizer vocabulary size — 16 384 (16 × 1024) is the MOTOR default.
VOCAB_SIZE = 16 * 1024

# Minimum code frequency to be included in the tokenizer vocabulary.
TOKENIZER_MIN_FRACTION = 1e-4

# MOTOR task head parameters.
MOTOR_NUM_TASKS = 8 * 1024
MOTOR_NUM_BINS = 8
MOTOR_FINAL_LAYER_SIZE = 512


# ==========================================
# PIPELINE STEPS
# ==========================================


def build_or_load_ontology(database, ontology_path):
    """Build a pruned OHDSI ontology or load it from cache."""
    if ontology_path.exists():
        logger.info("Loading existing ontology from %s", ontology_path)
        with open(ontology_path, "rb") as f:
            return pickle.load(f)

    logger.info("Building ontology (Athena: %s)", Config.MODEL_EHR_MOTOR_VOCAB_DIR)
    ontology = femr.ontology.Ontology(
        Config.MODEL_EHR_MOTOR_VOCAB_DIR, Config.PROCESSED_EHR_MEDS_METADATA_FILE
    )

    logger.info("Pruning ontology (removing %s)", PRUNED_ONTOLOGIES)
    ontology.prune_to_dataset(
        database,
        prune_all_descriptions=True,
        remove_ontologies=PRUNED_ONTOLOGIES,
    )

    with open(ontology_path, "wb") as f:
        pickle.dump(ontology, f)
    logger.info("Ontology saved to %s", ontology_path)
    return ontology


def create_or_load_split(database, split_path):
    """Create a deterministic hash split or reload from CSV."""
    if split_path.exists():
        logger.info("Loading existing split from %s", split_path)
        return femr.splits.PatientSplit.load_from_csv(split_path)

    logger.info(
        "Generating %d/%d split (seed=%d)",
        int(TRAIN_FRAC * 100),
        int(VAL_FRAC * 100),
        SPLIT_SEED,
    )
    split = femr.splits.generate_hash_split(
        list(database), SPLIT_SEED, frac_test=VAL_FRAC
    )
    split.save_to_csv(split_path)
    logger.info("Split saved to %s", split_path)
    return split


def train_or_load_tokenizer(database, ontology, tokenizer_path):
    """Train a hierarchical tokenizer or load from disk."""
    if tokenizer_path.exists():
        logger.info("Loading existing tokenizer from %s", tokenizer_path)
        return femr.models.tokenizer.HierarchicalTokenizer.from_pretrained(
            tokenizer_path, ontology=ontology
        )

    logger.info("Training tokenizer (vocab_size=%d)", VOCAB_SIZE)
    tokenizer = femr.models.tokenizer.HierarchicalTokenizer.train(
        database,
        vocab_size=VOCAB_SIZE,
        ontology=ontology,
        min_fraction=TOKENIZER_MIN_FRACTION,
    )
    tokenizer.save_pretrained(tokenizer_path)
    logger.info("Tokenizer saved to %s", tokenizer_path)
    return tokenizer


def fit_or_load_motor_task(train_database, tokenizer, task_path):
    """Fit MOTOR pretraining task distributions or load from cache."""
    if task_path.exists():
        logger.info("Loading existing MOTOR task from %s", task_path)
        with open(task_path, "rb") as f:
            return pickle.load(f)

    logger.info(
        "Fitting MOTOR task (num_tasks=%d, num_bins=%d)",
        MOTOR_NUM_TASKS,
        MOTOR_NUM_BINS,
    )
    motor_task = femr.models.tasks.MOTORTask.fit_pretraining_task_info(
        train_database,
        tokenizer,
        num_tasks=MOTOR_NUM_TASKS,
        num_bins=MOTOR_NUM_BINS,
        final_layer_size=MOTOR_FINAL_LAYER_SIZE,
    )
    with open(task_path, "wb") as f:
        pickle.dump(motor_task, f)
    logger.info("MOTOR task saved to %s", task_path)
    return motor_task


def convert_or_load_batches(processor, database, batches_path, split_name):
    """Tokenise a database view into HuggingFace batches or skip if cached."""
    if batches_path.exists():
        logger.info(
            "%s batches already exist at %s — skipping", split_name, batches_path
        )
        return

    logger.info(
        "Converting %s batches (tokens_per_batch=%d, num_proc=%d)",
        split_name,
        TOKENS_PER_BATCH,
        NUM_PROC,
    )
    batches = processor.convert_dataset(
        database,
        tokens_per_batch=TOKENS_PER_BATCH,
        num_proc=NUM_PROC,
    )
    batches.set_format("pt")
    batches.save_to_disk(batches_path)
    logger.info("%s batches saved to %s", split_name, batches_path)


# ==========================================
# MAIN
# ==========================================


def main():
    """Run the full MOTOR preparation pipeline."""
    Config.setup_logging()
    Config.set_seed()

    output_dir = Config.MODEL_EHR_MOTOR_PRETRAINING_FILES_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Opening MEDS database: %s", Config.PROCESSED_EHR_MEDS_PRETRAINING_DIR)

    with meds_reader.SubjectDatabase(
        str(Config.PROCESSED_EHR_MEDS_PRETRAINING_DIR), num_threads=NUM_PROC
    ) as database:
        # 1. Ontology
        ontology = build_or_load_ontology(database, output_dir / "ontology.pkl")

        # 2. Split
        main_split = create_or_load_split(database, output_dir / "split.csv")

        train_database = database.filter(main_split.train_subject_ids)
        val_database = database.filter(main_split.test_subject_ids)
        logger.info(
            "Train subjects: %d | Val subjects: %d",
            len(train_database),
            len(val_database),
        )

        # 3. Tokenizer
        tokenizer = train_or_load_tokenizer(
            database, ontology, output_dir / "tokenizer"
        )

        # 4. MOTOR task
        motor_task = fit_or_load_motor_task(
            train_database, tokenizer, output_dir / "motor_task.pkl"
        )

        # 5. Batch conversion
        processor = femr.models.processor.FEMRBatchProcessor(tokenizer, motor_task)

        convert_or_load_batches(
            processor, train_database, output_dir / "train_batches", "Train"
        )
        convert_or_load_batches(
            processor, val_database, output_dir / "val_batches", "Val"
        )

    logger.info("MOTOR preparation complete — artifacts in %s", output_dir)


if __name__ == "__main__":
    main()

"""
Inspect the statistical distributions of extracted embeddings for any modality.
Outputs a consolidated Markdown report.

Usage:
    python -m src.embeddings.inspect_embeddings --modality ehr
    python -m src.embeddings.inspect_embeddings --all
"""

import argparse
import logging
import torch
import sys
from pathlib import Path

from src.utils.config import Config

logger = logging.getLogger(__name__)

MODALITY_DIRS = {
    "ehr": Config.PROCESSED_EHR_EMBEDDINGS_DIR,
    "ecg": Config.PROCESSED_ECG_EMBEDDINGS_DIR,
    "cxr_img": Config.PROCESSED_CXR_IMG_EMBEDDINGS_DIR,
    "cxr_txt": Config.PROCESSED_CXR_TXT_EMBEDDINGS_DIR,
}


def inspect_split(file_path: Path, file_out):
    """Loads a .pt embedding file and writes its statistical summary in Markdown."""
    file_out.write(f"### Split: `{file_path.name}`\n\n")

    if not file_path.exists():
        file_out.write(f"**[!] Warning:** `{file_path.name}` not found.\n\n")
        return

    data = torch.load(file_path, map_location="cpu", weights_only=False)

    embeddings = data["embeddings"]
    labels = data["labels"]
    subject_ids = data["subject_ids"]

    file_out.write(f"- **Embeddings shape**: `{list(embeddings.shape)}`\n")
    file_out.write(f"- **Number of samples**: `{len(subject_ids)}`\n")
    file_out.write(f"- **Embedding dimension**: `{embeddings.shape[1]}`\n\n")

    file_out.write(f"#### Embedding Statistics\n")
    file_out.write(f"- **Mean**: `{embeddings.mean().item():.6f}`\n")
    file_out.write(f"- **Std**: `{embeddings.std().item():.6f}`\n")
    file_out.write(f"- **Min**: `{embeddings.min().item():.6f}`\n")
    file_out.write(f"- **Max**: `{embeddings.max().item():.6f}`\n\n")

    has_nan = torch.isnan(embeddings).any().item()
    has_inf = torch.isinf(embeddings).any().item()

    # Flag critical issues aggressively using Markdown bolding
    nan_str = "**YES (CRITICAL ERROR)** ❌" if has_nan else "No ✅"
    inf_str = "**YES (CRITICAL ERROR)** ❌" if has_inf else "No ✅"
    file_out.write(f"- **Contains NaN**: {nan_str}\n")
    file_out.write(f"- **Contains Inf**: {inf_str}\n\n")

    if labels is not None and len(labels) > 0:
        int_labels = [int(l) for l in labels]
        unique_labels = set(int_labels)
        file_out.write(f"#### Label Distribution\n")
        for label in sorted(unique_labels):
            count = int_labels.count(label)
            percentage = (count / len(int_labels)) * 100
            file_out.write(f"- **Class {label}**: {count} ({percentage:.1f}%)\n")

    file_out.write("\n---\n\n")


def main():
    Config.setup_logging(level=logging.WARNING)
    parser = argparse.ArgumentParser(description="Inspect embedding statistics.")
    parser.add_argument("--modality", type=str, choices=list(MODALITY_DIRS.keys()))
    parser.add_argument(
        "--all", action="store_true", help="Inspect all modalities and save a report."
    )
    args = parser.parse_args()

    if not args.modality and not args.all:
        parser.error("Must specify either --modality or --all")

    modalities_to_run = MODALITY_DIRS.keys() if args.all else [args.modality]
    splits = ["train", "valid", "test"]

    # If running all, save to the new REPORT_EMBEDDINGS_DIR as a Markdown file
    Config.REPORT_EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = Config.REPORT_EMBEDDINGS_DIR / "embeddings_health_report.md"

    # Use context manager to handle either file output or stdout
    with open(report_path, "w") if args.all else sys.stdout as out_file:
        out_file.write("# Embedding Inspection Report\n\n")

        for mod in modalities_to_run:
            out_file.write(f"## Modality: {mod.upper()}\n\n")

            for split in splits:
                file_path = MODALITY_DIRS[mod] / f"{split}_embeddings.pt"
                inspect_split(file_path, out_file)

    if args.all:
        print(f"✅ Full Markdown report successfully generated at: {report_path}")


if __name__ == "__main__":
    main()

"""
Negative-result diagnostic for ontology leaf-concept filtering feasibility.

See src/explainability/README.md for methodology context.
"""

import logging
import pickle
from collections import Counter

import pandas as pd

from src.utils.config import Config
from src.explainability.xai_utils import load_raw_msgpack_dictionary

logger = logging.getLogger(__name__)

# Same clinical-domain prefixes the attribution scripts use.
VALID_CLINICAL_PREFIXES = (
    "LOINC/",
    "SNOMED/",
    "NDC/",
    "RxNorm/",
    "ATC/",
    "ICD9",
    "ICD10",
    "HCPCS/",
    "CVX/",
    "MIMIC_IV_LABITEM/",
    "MIMIC_IV_OMR/",
    "MIMIC_IV_ITEM/",
    "MIMIC_IV_Drug/",
    "MIMIC_IV_Microbiology",
)

# Curated spot-check: codes observed as TOP DRIVERS in the archetype output.
# Verifies the leaf criterion mechanically does what path 2 needs.
EXPECTED_ROLLUP = [  # should classify as internal_code
    "SNOMED/71388002",
    "SNOMED/386053000",
    "SNOMED/430925007",
    "SNOMED/363787002",
    "LOINC/LP7834-7",
    "LOINC/LP7803-2",
    "LOINC/LP30605-7",
    "LOINC/LP432695-7",
    "LOINC/LP40317-7",
    "LOINC/LP386984-1",
    "ATC/A",
    "ATC/B05",
    "ATC/C03",
]
EXPECTED_SPECIFIC = [  # should classify as leaf_code
    "LOINC/86243-3",
    "LOINC/49136-5",
    "RxNorm/1807634",
    "NDC/63323026201",
    "MIMIC_IV_ITEM/225640",
    "MIMIC_IV_MicrobiologyTest/HCV VIRAL LOAD",
    "ATC/N02BA77",
]


def classify_vocabulary(vocab, ontology) -> pd.DataFrame:
    """Classify every token id in the MOTOR vocabulary."""
    rows = []
    for token_id, item in enumerate(vocab):
        ttype = item.get("type")

        if ttype == "code":
            code = (item.get("code_string") or "").strip()
            is_clinical = code.startswith(VALID_CLINICAL_PREFIXES)
            in_ontology = (
                code in ontology.description_map
                or code in ontology.children_map
                or code in ontology.parents_map
            )
            n_children = len(ontology.get_children(code))

            if not is_clinical:
                cls = "non_clinical_code"
            elif n_children == 0:
                cls = "leaf_code"
            else:
                cls = "internal_code"

            rows.append(
                dict(
                    token_id=token_id,
                    cls=cls,
                    ttype=ttype,
                    code_string=code,
                    vocab_prefix=(code.split("/")[0] if "/" in code else code),
                    is_loinc_part=code.startswith("LOINC/LP"),
                    property=None,
                    in_ontology=in_ontology,
                    n_children=n_children,
                )
            )

        elif ttype in ("numeric", "text"):
            rows.append(
                dict(
                    token_id=token_id,
                    cls=ttype,
                    ttype=ttype,
                    code_string=None,
                    vocab_prefix=None,
                    is_loinc_part=False,
                    property=item.get("property"),
                    in_ontology=False,
                    n_children=0,
                )
            )

        else:
            rows.append(
                dict(
                    token_id=token_id,
                    cls="other",
                    ttype=str(ttype),
                    code_string=None,
                    vocab_prefix=None,
                    is_loinc_part=False,
                    property=item.get("property"),
                    in_ontology=False,
                    n_children=0,
                )
            )

    return pd.DataFrame(rows)


def _spot_check(df: pd.DataFrame, codes, expected_cls: str) -> None:
    lookup = df.set_index("code_string")
    for code in codes:
        if code in lookup.index:
            row = lookup.loc[code]
            # A code_string is unique in the vocab, but guard against duplicates.
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            flag = "OK " if row["cls"] == expected_cls else ">> MISMATCH"
            logger.info(
                f"  {flag} | {code:<55} -> {row['cls']:<17} "
                f"(children={row['n_children']}, in_ontology={row['in_ontology']})"
            )
        else:
            logger.info(f"  ABSENT | {code:<55} -> not present in vocabulary")


def main() -> None:
    Config.setup_logging()

    # 1. Load the FEMR ontology
    ontology_path = Config.MODEL_EHR_MOTOR_PRETRAINING_FILES_DIR / "ontology.pkl"
    logger.info(f"Loading ontology from {ontology_path} ...")
    with open(ontology_path, "rb") as f:
        ontology = pickle.load(f)

    # 2. Load the MOTOR vocabulary
    vocab = load_raw_msgpack_dictionary()
    logger.info(f"Loaded vocabulary with {len(vocab)} tokens.")

    # 3. Classify
    df = classify_vocabulary(vocab, ontology)

    # ------------------------------------------------------------------
    # DIAGNOSTICS
    # ------------------------------------------------------------------
    logger.info("\n=== CLASS COUNTS (all tokens) ===")
    for cls, n in df["cls"].value_counts().items():
        logger.info(f"  {cls:<18} {n}")

    logger.info("\n=== NUMERIC TOKENS: distinct `property` values ===")
    for prop, n in Counter(df.loc[df["cls"] == "numeric", "property"]).most_common():
        logger.info(f"  {str(prop):<25} {n}")

    logger.info("\n=== TEXT TOKENS: distinct `property` values ===")
    for prop, n in Counter(df.loc[df["cls"] == "text", "property"]).most_common():
        logger.info(f"  {str(prop):<25} {n}")

    logger.info("\n=== CODE TOKENS: vocabulary prefix x class ===")
    code_df = df[df["ttype"] == "code"]
    crosstab = code_df.groupby(["vocab_prefix", "cls"]).size().unstack(fill_value=0)
    for line in crosstab.to_string().splitlines():
        logger.info("  " + line)

    logger.info("\n=== leaf_code: true ontology leaf vs absent-from-ontology ===")
    leaf_df = df[df["cls"] == "leaf_code"]
    logger.info(
        f"  in_ontology=True  (true leaf)        : {int((leaf_df['in_ontology']).sum())}"
    )
    logger.info(
        f"  in_ontology=False (absent, leaf-by-default): {int((~leaf_df['in_ontology']).sum())}"
    )

    logger.info("\n=== LOINC 'LP' (LOINC Part) codes: class breakdown ===")
    lp_df = df[df["is_loinc_part"]]
    logger.info(f"  total LOINC/LP tokens: {len(lp_df)}")
    for cls, n in lp_df["cls"].value_counts().items():
        logger.info(f"    {cls:<18} {n}")

    logger.info(
        "\n=== SPOT-CHECK: codes expected to be ROLLUPS (want internal_code) ==="
    )
    _spot_check(df, EXPECTED_ROLLUP, "internal_code")

    logger.info("\n=== SPOT-CHECK: codes expected to be SPECIFIC (want leaf_code) ===")
    _spot_check(df, EXPECTED_SPECIFIC, "leaf_code")

    # ------------------------------------------------------------------
    # SAVE
    # ------------------------------------------------------------------
    out_path = Config.RESULTS_DIR / "explainability" / "token_classification.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    logger.info(f"\nSaved token classification table to {out_path}")
    logger.info(
        "Review the spot-check and the LOINC/LP breakdown above before re-running "
        "the attribution scripts."
    )


if __name__ == "__main__":
    main()

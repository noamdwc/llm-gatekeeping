"""
CLI runner for synthetic benign generation and validation.

Generates diverse benign prompts, applies heuristic + optional LLM-judge
validation, and saves to data/processed/synthetic_benign.parquet.

Usage:
    # Generate 10 category-C samples, skip LLM validation
    python -m src.cli.generate_synthetic_benign --category C --limit 10 --skip-judge-validation

    # Generate all categories with LLM validation
    python -m src.cli.generate_synthetic_benign --category all --limit 100

    # Use custom config
    python -m src.cli.generate_synthetic_benign --category E --limit 50 --config configs/default.yaml
"""

import argparse
from pathlib import Path

import dotenv
import pandas as pd

from src.utils import load_config, DATA_DIR
from src.synthetic_benign import SyntheticBenignGenerator, _CATEGORY_META
from src.validators import HeuristicBenignValidator, JudgeBenignValidator

dotenv.load_dotenv()

DEFAULT_OUTPUT_PATH = DATA_DIR / "synthetic_benign.parquet"


def run_generation(
    category: str,
    limit: int | None,
    skip_judge_validation: bool,
    cfg: dict,
    output_path: Path,
) -> pd.DataFrame:
    """Run generation + validation pipeline for one or all categories.

    Args:
        category: Category letter ("A"–"F") or "all".
        limit: Max samples per category (None = use config quotas).
        skip_judge_validation: If True, skip LLM judge validation step.
        cfg: Configuration dict.
        output_path: Where to save the output parquet.

    Returns:
        DataFrame of validated synthetic benign records.
    """
    synth_cfg = cfg.get("benign", {}).get("synthetic", {})
    gen = SyntheticBenignGenerator(cfg)
    heuristic = HeuristicBenignValidator()

    # Determine categories and quotas
    if category == "all":
        categories = list(_CATEGORY_META.keys())
    else:
        if category not in _CATEGORY_META:
            raise ValueError(f"Unknown category {category!r}. Valid: {list(_CATEGORY_META)}")
        categories = [category]

    default_quotas = synth_cfg.get("quotas", {cat: 100 for cat in _CATEGORY_META})

    all_records: list[dict] = []

    for cat in categories:
        target_n = limit if limit is not None else default_quotas.get(cat, 100)
        print(f"\n[Category {cat}] Generating {target_n} samples ({_CATEGORY_META[cat]['name']})")

        # Generation
        texts = gen.generate_category(cat, target_n)
        print(f"  Generated: {len(texts)} raw samples")

        # Layer 1: Heuristic filter
        texts_filtered = heuristic.filter_batch(texts)
        n_rejected = len(texts) - len(texts_filtered)
        print(f"  After heuristic filter: {len(texts_filtered)} (+{n_rejected} rejected)")

        if not texts_filtered:
            print(f"  Warning: no samples passed heuristic filter for category {cat}")
            continue

        # Layer 2: LLM judge validation (optional)
        val_scores: list[float | None] = [None] * len(texts_filtered)
        validated_flags: list[bool] = [False] * len(texts_filtered)

        if not skip_judge_validation:
            print(f"  Running LLM judge validation on {len(texts_filtered)} samples...")
            from src.llm_classifier.llm_classifier import (
                HierarchicalLLMClassifier,
                build_few_shot_examples,
            )
            from src.utils import SPLITS_DIR
            df_train = pd.read_parquet(SPLITS_DIR / "train.parquet")
            few_shot, _ = build_few_shot_examples(df_train, cfg)
            classifier = HierarchicalLLMClassifier(cfg, few_shot)
            judge_validator = JudgeBenignValidator(classifier)
            judge_results = judge_validator.validate(texts_filtered)

            accepted_texts = []
            accepted_scores = []
            n_judge_rejected = 0
            for res in judge_results:
                if res["accepted"]:
                    accepted_texts.append(res["text"])
                    accepted_scores.append(res["val_score"])
                else:
                    n_judge_rejected += 1

            texts_filtered = accepted_texts
            val_scores = accepted_scores
            validated_flags = [True] * len(texts_filtered)
            print(f"  After judge validation: {len(texts_filtered)} (+{n_judge_rejected} rejected)")
        else:
            val_method = "heuristic"
            print(f"  Skipping LLM judge validation (--skip-judge-validation).")
            val_method = "heuristic"

        val_method = "judge" if not skip_judge_validation else "heuristic"
        model_name = gen.generation_model

        records = gen.to_records(
            texts_filtered,
            category=cat,
            source="llm_generated",
            model=model_name,
            validated=not skip_judge_validation,
            val_scores=val_scores if not skip_judge_validation else None,
            val_method=val_method,
        )
        all_records.extend(records)
        print(f"  Final: {len(records)} records for category {cat}")

    if not all_records:
        print("\nNo records generated.")
        return pd.DataFrame()

    df_new = pd.DataFrame(all_records)

    # Merge with existing synthetic benign parquet (if any)
    if output_path.exists():
        df_existing = pd.read_parquet(output_path)
        # Deduplicate by prompt_hash
        existing_hashes = set(df_existing["prompt_hash"].tolist())
        df_new = df_new[~df_new["prompt_hash"].isin(existing_hashes)]
        df_out = pd.concat([df_existing, df_new], ignore_index=True)
        print(f"\nMerged with existing: {len(df_existing)} + {len(df_new)} new = {len(df_out)} total")
    else:
        df_out = df_new

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_parquet(output_path, index=False)
    print(f"\nSaved {len(df_out)} records → {output_path}")
    print(f"Schema:\n{df_out.dtypes}")
    print(f"\nSample:\n{df_out.head(3).to_string()}")

    return df_out


def main():
    parser = argparse.ArgumentParser(
        description="Generate and validate synthetic benign prompts"
    )
    parser.add_argument(
        "--category",
        default="all",
        help="Category to generate: A, B, C, D, E, F, or 'all' (default: all)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max samples per category (default: use config quotas)",
    )
    parser.add_argument(
        "--skip-judge-validation",
        action="store_true",
        default=False,
        help="Skip LLM judge validation (faster but lower quality)",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to config YAML (default: configs/default.yaml)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help=f"Output parquet path (default: {DEFAULT_OUTPUT_PATH})",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    output_path = Path(args.output) if args.output else DEFAULT_OUTPUT_PATH

    run_generation(
        category=args.category,
        limit=args.limit,
        skip_judge_validation=args.skip_judge_validation,
        cfg=cfg,
        output_path=output_path,
    )


if __name__ == "__main__":
    main()

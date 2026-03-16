"""
CLI runner for synthetic benign generation and validation.

Generates diverse benign prompts, applies heuristic + optional LLM-judge
validation, and saves per-category parquet files.

Usage:
    # Generate single category
    python -m src.cli.generate_synthetic_benign --category C --limit 10 --skip-judge-validation

    # Generate all categories (one file each)
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

SYNTHETIC_DIR = DATA_DIR / "synthetic_benign"


def _output_path_for_category(cat: str, output_dir: Path) -> Path:
    return output_dir / f"synthetic_benign_{cat}.parquet"


def run_generation_single(
    category: str,
    limit: int | None,
    skip_judge_validation: bool,
    cfg: dict,
    output_dir: Path,
) -> pd.DataFrame:
    """Run generation + validation pipeline for a single category.

    Args:
        category: Category letter ("A"–"F").
        limit: Max samples (None = use config quotas).
        skip_judge_validation: If True, skip LLM judge validation step.
        cfg: Configuration dict.
        output_dir: Directory for per-category parquet files.

    Returns:
        DataFrame of validated synthetic benign records.
    """
    if category not in _CATEGORY_META:
        raise ValueError(f"Unknown category {category!r}. Valid: {list(_CATEGORY_META)}")

    synth_cfg = cfg.get("benign", {}).get("synthetic", {})
    gen = SyntheticBenignGenerator(cfg)
    heuristic = HeuristicBenignValidator()

    default_quotas = synth_cfg.get("quotas", {cat: 100 for cat in _CATEGORY_META})
    target_n = limit if limit is not None else default_quotas.get(category, 100)

    print(f"\n[Category {category}] Generating {target_n} samples ({_CATEGORY_META[category]['name']})")

    # Generation
    texts = gen.generate_category(category, target_n)
    print(f"  Generated: {len(texts)} raw samples")

    # Layer 1: Heuristic filter
    texts_filtered = heuristic.filter_batch(texts)
    n_rejected = len(texts) - len(texts_filtered)
    print(f"  After heuristic filter: {len(texts_filtered)} (+{n_rejected} rejected)")

    if not texts_filtered:
        print(f"  Warning: no samples passed heuristic filter for category {category}")
        return pd.DataFrame()

    # Layer 2: LLM judge validation (optional)
    val_scores: list[float | None] = [None] * len(texts_filtered)

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
        print(f"  After judge validation: {len(texts_filtered)} (+{n_judge_rejected} rejected)")

    val_method = "judge" if not skip_judge_validation else "heuristic"
    model_name = gen.generation_model

    records = gen.to_records(
        texts_filtered,
        category=category,
        source="llm_generated",
        model=model_name,
        validated=not skip_judge_validation,
        val_scores=val_scores if not skip_judge_validation else None,
        val_method=val_method,
    )
    print(f"  Final: {len(records)} records for category {category}")

    if not records:
        print("\nNo records generated.")
        return pd.DataFrame()

    df_new = pd.DataFrame(records)

    # Merge with existing category parquet (if any) for crash resilience
    output_path = _output_path_for_category(category, output_dir)
    if output_path.exists():
        df_existing = pd.read_parquet(output_path)
        existing_hashes = set(df_existing["prompt_hash"].tolist())
        df_new = df_new[~df_new["prompt_hash"].isin(existing_hashes)]
        df_out = pd.concat([df_existing, df_new], ignore_index=True)
        print(f"\nMerged with existing: {len(df_existing)} + {len(df_new)} new = {len(df_out)} total")
    else:
        df_out = df_new

    output_dir.mkdir(parents=True, exist_ok=True)
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
        "--output-dir",
        default=None,
        help=f"Output directory for per-category parquets (default: {SYNTHETIC_DIR})",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    output_dir = Path(args.output_dir) if args.output_dir else SYNTHETIC_DIR

    if args.category == "all":
        categories = list(_CATEGORY_META.keys())
    else:
        categories = [args.category]

    for cat in categories:
        run_generation_single(
            category=cat,
            limit=args.limit,
            skip_judge_validation=args.skip_judge_validation,
            cfg=cfg,
            output_dir=output_dir,
        )


if __name__ == "__main__":
    main()

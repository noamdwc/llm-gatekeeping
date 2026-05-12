"""Score a split with the trained escalating model.

This is the lightweight inference entry point that produces
``escalating_model_eval_{split}.parquet`` — the same artifact the DVC
``train_escalating_model`` stage writes — from already-existing
classifier and DeBERTa prediction parquets. It does not retrain the
escalating model; it loads ``escalating_model.pkl`` and applies it.

Usage:
    python -m src.cli.score_escalation --split test
    python -m src.cli.score_escalation --split test \
        --classifier-predictions data/processed/predictions/llm_predictions_test_colab_local_classifier.parquet \
        --deberta-predictions data/processed/predictions/deberta_predictions_test.parquet \
        --output data/processed/research/escalating_model_eval_test.parquet
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.escalating_model import (
    EVAL_SUMMARY_COLS,
    EscalatingDataset,
    EscalatingModel,
    evaluate_escalating_split,
)
from src.utils import MODELS_DIR, PREDICTIONS_DIR, RESEARCH_DIR, load_config


def default_classifier_path(split: str) -> Path:
    return PREDICTIONS_DIR / f"llm_predictions_{split}_colab_local_classifier.parquet"


def default_deberta_path(split: str) -> Path:
    return PREDICTIONS_DIR / f"deberta_predictions_{split}.parquet"


def default_output_path(split: str) -> Path:
    return RESEARCH_DIR / f"escalating_model_eval_{split}.parquet"


def score_split(
    split: str,
    *,
    classifier_df: pd.DataFrame,
    deberta_df: pd.DataFrame,
    model: EscalatingModel,
) -> tuple[pd.DataFrame, dict]:
    ds = EscalatingDataset(classifier_df, deberta_df)
    scores = model.predict_escalation_batch(ds.df)
    return evaluate_escalating_split(split, ds, scores)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Score a split with the escalating model.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--split", required=True)
    parser.add_argument(
        "--classifier-predictions",
        type=Path,
        default=None,
        help="Cheap classifier predictions parquet (default: predictions/llm_predictions_{split}_colab_local_classifier.parquet)",
    )
    parser.add_argument(
        "--deberta-predictions",
        type=Path,
        default=None,
        help="DeBERTa predictions parquet (default: predictions/deberta_predictions_{split}.parquet)",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=None,
        help="Trained escalating model pickle (default: hybrid.escalating_model.model_path)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output parquet (default: research/escalating_model_eval_{split}.parquet)",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config)
    escalating_cfg = cfg.get("hybrid", {}).get("escalating_model", {})

    classifier_path = args.classifier_predictions or default_classifier_path(args.split)
    deberta_path = args.deberta_predictions or default_deberta_path(args.split)
    model_path = args.model or Path(
        escalating_cfg.get("model_path", MODELS_DIR / "escalating_model.pkl")
    )
    output_path = args.output or default_output_path(args.split)

    model = EscalatingModel.load(model_path)
    classifier_df = pd.read_parquet(classifier_path)
    deberta_df = pd.read_parquet(deberta_path)
    scored, summary = score_split(
        args.split,
        classifier_df=classifier_df,
        deberta_df=deberta_df,
        model=model,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    scored.to_parquet(output_path, index=False)

    summary_row = {col: summary.get(col) for col in EVAL_SUMMARY_COLS}
    print(f"Scored {summary_row['rows_joined']} joined rows -> {output_path}")
    print(
        f"cheap_error_rate={summary_row['cheap_error_rate']}, "
        f"roc_auc={summary_row['roc_auc']}, pr_auc={summary_row['pr_auc']}"
    )


if __name__ == "__main__":
    main()

"""Train the offline escalating model POC.

Usage:
    python -m src.cli.train_escalating_model
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.escalating_model import (
    ESCALATING_FEATURE_COLS,
    EVAL_SUMMARY_COLS,
    EscalatingDataset,
    EscalatingModel,
    evaluate_escalating_split,
    evaluate_threshold_sweep,
    write_escalating_report,
)
from src.utils import MODELS_DIR, PREDICTIONS_DIR, RESEARCH_DIR, ROOT, load_config


DEFAULT_EVAL_SPLITS = ["test", "unseen_val", "unseen_test", "safeguard_test"]


def _default_colab_path(split: str) -> Path:
    return PREDICTIONS_DIR / f"llm_predictions_{split}_colab_local_classifier.parquet"


def _default_deberta_path(split: str) -> Path:
    return PREDICTIONS_DIR / f"deberta_predictions_{split}.parquet"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the offline escalating model POC.")
    parser.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    parser.add_argument(
        "--train-colab-predictions",
        default=str(_default_colab_path("val")),
    )
    parser.add_argument(
        "--train-deberta-predictions",
        default=str(_default_deberta_path("val")),
    )
    parser.add_argument(
        "--eval-split",
        nargs=3,
        action="append",
        metavar=("SPLIT", "COLAB_PARQUET", "DEBERTA_PARQUET"),
        help=(
            "Evaluation split triple. Repeat for multiple splits. "
            "Defaults to test, unseen_val, unseen_test, and safeguard_test."
        ),
    )
    parser.add_argument(
        "--model-output",
        default=str(MODELS_DIR / "escalating_model.pkl"),
    )
    parser.add_argument(
        "--research-output-dir",
        default=str(RESEARCH_DIR),
    )
    parser.add_argument(
        "--summary-output",
        default=str(RESEARCH_DIR / "escalating_model_summary.csv"),
    )
    parser.add_argument(
        "--threshold-sweep-output",
        default=str(RESEARCH_DIR / "escalating_model_threshold_sweep_unseen_val.csv"),
    )
    parser.add_argument(
        "--report-output",
        default=str(ROOT / "reports" / "escalating_model_poc.md"),
    )
    return parser


def _resolve_eval_splits(args: argparse.Namespace) -> list[tuple[str, Path, Path]]:
    if args.eval_split:
        return [
            (split, Path(colab_path), Path(deberta_path))
            for split, colab_path, deberta_path in args.eval_split
        ]
    return [
        (split, _default_colab_path(split), _default_deberta_path(split))
        for split in DEFAULT_EVAL_SPLITS
    ]


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    escalating_cfg = cfg.get("hybrid", {}).get("escalating_model", {})
    model_output = Path(args.model_output or escalating_cfg.get(
        "model_path",
        MODELS_DIR / "escalating_model.pkl",
    ))
    research_output_dir = Path(args.research_output_dir)
    summary_output = Path(args.summary_output)
    threshold_sweep_output = Path(args.threshold_sweep_output)
    report_output = Path(args.report_output)

    train_colab = pd.read_parquet(args.train_colab_predictions)
    train_deberta = pd.read_parquet(args.train_deberta_predictions)
    train_ds = EscalatingDataset(train_colab, train_deberta)

    print(
        "Training escalating model on "
        f"{train_ds.rows_joined} joined val rows "
        f"({train_ds.rows_dropped_colab_only} Colab-only dropped, "
        f"{train_ds.rows_dropped_deberta_only} DeBERTa-only dropped)"
    )
    model = EscalatingModel.train(
        train_ds.X,
        train_ds.y,
        feature_cols=list(ESCALATING_FEATURE_COLS),
    )
    model.save(model_output)
    print(f"Saved escalating model to {model_output}")

    research_output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    unseen_val_scored = None
    for split, colab_path, deberta_path in _resolve_eval_splits(args):
        colab_df = pd.read_parquet(colab_path)
        deberta_df = pd.read_parquet(deberta_path)
        ds = EscalatingDataset(colab_df, deberta_df)
        scores = model.predict_escalation_batch(ds.df)
        scored, summary = evaluate_escalating_split(split, ds, scores)

        eval_output = research_output_dir / f"escalating_model_eval_{split}.parquet"
        scored.to_parquet(eval_output, index=False)
        if split == "unseen_val":
            unseen_val_scored = scored
        summaries.append(summary)
        print(
            f"Wrote {split} evaluation to {eval_output} "
            f"({summary['rows_joined']} joined rows)"
        )

    summary_df = pd.DataFrame(summaries, columns=EVAL_SUMMARY_COLS)
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(summary_output, index=False)
    threshold_sweep_df = None
    if unseen_val_scored is not None:
        threshold_sweep_df = evaluate_threshold_sweep(unseen_val_scored)
        threshold_sweep_output.parent.mkdir(parents=True, exist_ok=True)
        threshold_sweep_df.to_csv(threshold_sweep_output, index=False)
        print(f"Wrote unseen_val threshold sweep to {threshold_sweep_output}")
    write_escalating_report(summary_df, report_output, threshold_sweep_df)
    print(f"Wrote summary to {summary_output}")
    print(f"Wrote report to {report_output}")


if __name__ == "__main__":
    main()

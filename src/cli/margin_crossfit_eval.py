"""Cross-fitted exploratory threshold evaluation for the saved margin trace."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from src.margin_trace import apply_threshold_override, compute_binary_metrics_from_predictions
from src.utils import RESEARCH_DIR, REPORTS_ARTIFACTS_DIR


def _select_threshold(train_df: pd.DataFrame, thresholds: list[float], objective: str) -> float:
    best_threshold = thresholds[0]
    best_value = -np.inf
    for threshold in thresholds:
        preds = apply_threshold_override(train_df, threshold=threshold)
        metrics = compute_binary_metrics_from_predictions(train_df["true_label"], preds)
        value = metrics[objective]
        if value > best_value:
            best_value = value
            best_threshold = threshold
    return best_threshold


def main() -> None:
    parser = argparse.ArgumentParser(description="Cross-fitted exploratory threshold evaluation.")
    parser.add_argument("--trace", default=str(RESEARCH_DIR / "hybrid_margin_trace_test.parquet"))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--objective", choices=["youden_j", "balanced_accuracy"], default="youden_j")
    args = parser.parse_args()

    df = pd.read_parquet(args.trace)
    thresholds = [round(x * 0.5, 2) for x in range(13)]
    splitter = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=42)

    rows = []
    for fold_idx, (train_idx, test_idx) in enumerate(splitter.split(df, df["true_label"]), start=1):
        train_df = df.iloc[train_idx].reset_index(drop=True)
        test_df = df.iloc[test_idx].reset_index(drop=True)
        threshold = _select_threshold(train_df, thresholds, args.objective)
        preds = apply_threshold_override(test_df, threshold=threshold)
        metrics = compute_binary_metrics_from_predictions(test_df["true_label"], preds)
        rows.append({"fold": fold_idx, "selected_threshold": threshold, **metrics})

    result_df = pd.DataFrame(rows)
    summary_df = result_df.drop(columns=["fold"]).agg(["mean", "std"]).reset_index().rename(columns={"index": "stat"})

    REPORTS_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    fold_path = REPORTS_ARTIFACTS_DIR / "margin_crossfit_folds.csv"
    summary_path = REPORTS_ARTIFACTS_DIR / "margin_crossfit_summary.csv"
    md_path = REPORTS_ARTIFACTS_DIR / "margin_crossfit_summary.md"
    result_df.to_csv(fold_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    md_path.write_text(
        "# Cross-fitted Margin Evaluation\n\n"
        "Warning: this remains exploratory because the base LLM predictions were generated once on the same dataset.\n\n"
        "## Fold Metrics\n\n"
        f"{result_df.to_markdown(index=False)}\n\n"
        "## Aggregate Summary\n\n"
        f"{summary_df.to_markdown(index=False)}\n"
    )
    print(f"Cross-fitted summary saved -> {md_path}")


if __name__ == "__main__":
    main()

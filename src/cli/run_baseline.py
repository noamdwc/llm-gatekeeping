"""Run external HuggingFace baseline detectors on internal or external datasets."""

from __future__ import annotations

import argparse

import pandas as pd

from src.baselines.hf_detector import HFDetector
from src.eval_external import load_external_dataset
from src.utils import BASELINES_DIR, SPLITS_DIR, ensure_dirs, load_config

INTERNAL_SPLITS = ["val", "test", "test_unseen"]
EXTERNAL_KEYS = ["deepset", "jackhhao", "safeguard"]


def _resolve_requested(value: str | None, allowed: list[str]) -> list[str]:
    if value is None:
        return []
    if value == "all":
        return allowed
    return [value]


def _dataset_specs(cfg: dict, split_arg: str | None, external_arg: str | None) -> list[tuple[str, pd.DataFrame, str]]:
    specs = []
    for split in _resolve_requested(split_arg, INTERNAL_SPLITS):
        path = SPLITS_DIR / f"{split}.parquet"
        specs.append((split, pd.read_parquet(path), "modified_sample"))
    for key in _resolve_requested(external_arg, EXTERNAL_KEYS):
        specs.append((f"external_{key}", load_external_dataset(cfg["external_datasets"][key]), "modified_sample"))
    return specs


def run_baseline_for_dataset(
    baseline_key: str,
    cfg: dict,
    dataset_key: str,
    df: pd.DataFrame,
    text_col: str,
    batch_size: int | None = None,
    device: str | None = None,
    threshold: float | None = None,
    max_length: int | None = None,
) -> pd.DataFrame:
    detector = HFDetector.from_config(
        baseline_key,
        cfg,
        batch_size=batch_size,
        device=device,
        threshold=threshold,
        max_length=max_length,
    )
    preds = detector.predict_dataframe(df, text_col)
    preds["baseline_key"] = baseline_key
    preds["dataset_key"] = dataset_key
    preds["threshold_used"] = float(detector.threshold)
    preds["resolved_positive_label"] = detector.positive_label_resolved
    preds["label_mapping_method"] = detector.label_mapping_method
    return preds


def main():
    parser = argparse.ArgumentParser(description="Run HuggingFace baseline detectors")
    parser.add_argument("--baseline", choices=["sentinel_v2", "protectai_v2", "all"], required=True)
    parser.add_argument("--split", choices=INTERNAL_SPLITS + ["all"], default=None)
    parser.add_argument("--external", choices=EXTERNAL_KEYS + ["all"], default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    if args.split is None and args.external is None:
        parser.error("At least one of --split or --external is required.")

    cfg = load_config(args.config)
    ensure_dirs()

    baseline_keys = _resolve_requested(args.baseline, ["sentinel_v2", "protectai_v2"])
    dataset_specs = _dataset_specs(cfg, args.split, args.external)

    for baseline_key in baseline_keys:
        for dataset_key, df, text_col in dataset_specs:
            print(f"Running baseline={baseline_key} dataset={dataset_key} rows={len(df)}")
            preds = run_baseline_for_dataset(
                baseline_key=baseline_key,
                cfg=cfg,
                dataset_key=dataset_key,
                df=df,
                text_col=text_col,
                batch_size=args.batch_size,
                device=args.device,
                threshold=args.threshold,
                max_length=args.max_length,
            )
            path = BASELINES_DIR / f"{baseline_key}_{dataset_key}.parquet"
            preds.to_parquet(path, index=False)
            print(f"  Saved -> {path}")


if __name__ == "__main__":
    main()

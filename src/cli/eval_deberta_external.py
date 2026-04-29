"""Evaluate a saved DeBERTa classifier on configured external datasets.

Usage:
    python -m src.cli.eval_deberta_external --dataset all --device cpu
    python -m src.cli.eval_deberta_external --dataset deepset
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd

from src.evaluate import binary_metrics, calibration_metrics
from src.eval_external import load_external_dataset
from src.models.deberta_classifier import DeBERTaClassifier
from src.utils import (
    DEBERTA_ARTIFACTS_DIR,
    PREDICTIONS_EXTERNAL_DIR,
    REPORTS_EXTERNAL_DIR,
    build_sample_id,
    load_config,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate saved DeBERTa artifacts on external datasets"
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="External dataset key from config, or 'all'",
    )
    parser.add_argument("--config", default=None, help="Path to config YAML")
    parser.add_argument(
        "--artifacts-dir",
        default=str(DEBERTA_ARTIFACTS_DIR),
        help="Directory containing saved DeBERTa model/tokenizer artifacts",
    )
    parser.add_argument(
        "--predictions-dir",
        default=str(PREDICTIONS_EXTERNAL_DIR),
        help="Directory for external DeBERTa prediction parquet outputs",
    )
    parser.add_argument(
        "--reports-dir",
        default=str(REPORTS_EXTERNAL_DIR),
        help="Directory for external DeBERTa reports",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda", "mps"],
        default="auto",
        help="Inference device. Defaults to auto.",
    )
    parser.add_argument("--cpu", action="store_true", help="Force inference on CPU")
    parser.add_argument("--limit", type=int, default=None, help="Max samples per dataset")
    return parser


def _format_metric_value(value) -> str:
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def sanitize_for_json(value):
    if isinstance(value, dict):
        return {key: sanitize_for_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [sanitize_for_json(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def generate_report(
    ds_key: str,
    ds_cfg: dict,
    n_samples: int,
    binary: dict,
    calibration: dict,
) -> str:
    lines = [
        f"# DeBERTa External Evaluation — {ds_key}",
        "",
        f"- **Dataset**: `{ds_cfg['name']}`",
        f"- **Split**: `{ds_cfg['split']}`",
        f"- **Samples**: {n_samples}",
        "",
        "## Binary Detection",
        "",
        "| Metric | Value |",
        "|--------|-------|",
    ]
    for key, value in binary.items():
        lines.append(f"| {key} | {_format_metric_value(value)} |")

    lines.extend([
        "",
        "## Calibration",
        "",
        "| Bin | Count | Avg Confidence | Accuracy |",
        "|-----|-------|----------------|----------|",
    ])
    for bucket in calibration.get("calibration_buckets", []):
        lines.append(
            f"| {bucket['bin']} | {bucket['count']} | "
            f"{bucket['avg_confidence']:.3f} | {bucket['accuracy']:.3f} |"
        )
    lines.append("")
    return "\n".join(lines)


def evaluate_deberta(df: pd.DataFrame, model, text_col: str = "modified_sample") -> tuple[dict, dict, pd.DataFrame]:
    preds = model.predict(df, text_col)
    binary = binary_metrics(df["label_binary"], preds["deberta_pred_binary"])
    calibration = calibration_metrics(
        df["label_binary"],
        preds["deberta_pred_binary"],
        preds["deberta_conf_binary"],
    )
    return binary, calibration, preds


def build_predictions_df(df: pd.DataFrame, preds: pd.DataFrame) -> pd.DataFrame:
    gt_cols = [
        col
        for col in ["modified_sample", "label_binary", "label_category", "label_type"]
        if col in df.columns
    ]
    out = pd.concat([df[gt_cols].reset_index(drop=True), preds.reset_index(drop=True)], axis=1)
    out.insert(0, "sample_id", out["modified_sample"].apply(build_sample_id))
    return out


def run_single_dataset(
    ds_key: str,
    ds_cfg: dict,
    cfg: dict,
    artifacts_dir: str | Path = DEBERTA_ARTIFACTS_DIR,
    predictions_dir: str | Path = PREDICTIONS_EXTERNAL_DIR,
    reports_dir: str | Path = REPORTS_EXTERNAL_DIR,
    model=None,
    device: str = "auto",
    force_cpu: bool = False,
    limit: int | None = None,
) -> dict:
    print(f"\n{'=' * 60}")
    print(f"DeBERTa external eval: {ds_key} ({ds_cfg['name']})")
    print(f"{'=' * 60}")

    df = load_external_dataset(ds_cfg)
    if limit and limit < len(df):
        df = df.sample(n=limit, random_state=42).reset_index(drop=True)
    print(
        f"  Loaded {len(df)} samples "
        f"({(df['label_binary'] == 'adversarial').sum()} adversarial, "
        f"{(df['label_binary'] == 'benign').sum()} benign)"
    )

    if model is None:
        model = DeBERTaClassifier.load(
            artifacts_dir,
            cfg,
            force_cpu=force_cpu,
            device=None if device == "auto" else device,
        )

    binary, calibration, preds = evaluate_deberta(df, model)
    out = build_predictions_df(df, preds)

    predictions_dir = Path(predictions_dir)
    reports_dir = Path(reports_dir)
    predictions_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    pred_path = predictions_dir / f"deberta_predictions_external_{ds_key}.parquet"
    report_path = reports_dir / f"eval_deberta_external_{ds_key}.md"
    out.to_parquet(pred_path, index=False)
    report_path.write_text(generate_report(ds_key, ds_cfg, len(df), binary, calibration))

    print(f"\n--- {ds_key} DeBERTa Results ---")
    print(f"  Accuracy:            {binary['accuracy']:.4f}")
    print(f"  Adversarial F1:      {binary['adversarial_f1']:.4f}")
    print(f"  Benign F1:           {binary['benign_f1']:.4f}")
    print(f"  False-positive rate: {binary['false_positive_rate']:.4f}")
    print(f"  False-negative rate: {binary['false_negative_rate']:.4f}")
    print(f"  Predictions saved -> {pred_path}")
    print(f"  Report saved -> {report_path}")

    return {
        "dataset": ds_key,
        "binary": binary,
        "calibration": calibration,
        "predictions_path": str(pred_path),
        "report_path": str(report_path),
    }


def main(argv: list[str] | None = None):
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config)
    ext_datasets = cfg.get("external_datasets", {})
    if not ext_datasets:
        raise SystemExit("No external_datasets defined in config.")

    if args.dataset == "all":
        selected = ext_datasets
    else:
        if args.dataset not in ext_datasets:
            raise SystemExit(
                f"Unknown dataset key: {args.dataset!r}. "
                f"Available: {list(ext_datasets.keys()) + ['all']}"
            )
        selected = {args.dataset: ext_datasets[args.dataset]}

    model = DeBERTaClassifier.load(
        args.artifacts_dir,
        cfg,
        force_cpu=args.cpu,
        device=None if args.device == "auto" else args.device,
    )

    results = [
        run_single_dataset(
            ds_key,
            ds_cfg,
            cfg,
            artifacts_dir=args.artifacts_dir,
            predictions_dir=args.predictions_dir,
            reports_dir=args.reports_dir,
            model=model,
            device=args.device,
            force_cpu=args.cpu,
            limit=args.limit,
        )
        for ds_key, ds_cfg in selected.items()
    ]

    reports_dir = Path(args.reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    summary_path = reports_dir / "eval_deberta_external_summary.json"
    summary_path.write_text(json.dumps(sanitize_for_json(results), indent=2, allow_nan=False))
    print(f"\nSummary saved -> {summary_path}")


if __name__ == "__main__":
    main()

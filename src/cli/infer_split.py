"""
Lightweight split inference + report generation.

This is the code-backed replacement for the long `python -c ...` block in
`run_inference.sh` (ML mode).

Usage:
  python -m src.cli.infer_split --mode ml --split test
  python -m src.cli.infer_split --mode ml --split test --output reports/research/inference_ml_test.md
"""

import argparse
from pathlib import Path

import pandas as pd

from src.utils import load_config, SPLITS_DIR, MODELS_DIR, REPORTS_RESEARCH_DIR
from src.ml_classifier.ml_baseline import MLBaseline
from src.evaluate import (
    binary_metrics,
    category_metrics,
    type_metrics,
    calibration_metrics,
    generate_report,
)


def infer_ml_split(split: str, config_path: str | None = None) -> tuple[pd.DataFrame, dict, dict, dict, dict]:
    cfg = load_config(config_path)
    text_col = cfg["dataset"]["text_col"]

    df = pd.read_parquet(SPLITS_DIR / f"{split}.parquet")

    ml = MLBaseline(cfg)
    ml.load(str(MODELS_DIR / "ml_baseline.pkl"))
    preds = ml.predict(df, text_col)

    binary = binary_metrics(df["label_binary"], preds["pred_label_binary"])
    cat = category_metrics(df["label_category"], preds["pred_label_category"])
    types = type_metrics(df["label_type"], preds["pred_label_type"])
    cal = calibration_metrics(df["label_binary"], preds["pred_label_binary"], preds["confidence_label_binary"])

    return df, binary, cat, types, cal


def main():
    parser = argparse.ArgumentParser(description="Infer on a split parquet and generate report")
    parser.add_argument("--mode", choices=["ml"], default="ml")
    parser.add_argument("--split", default="test", choices=["test", "val", "test_unseen"])
    parser.add_argument("--config", default=None, help="Path to config YAML")
    parser.add_argument("--output", default=None, help="Output report path (markdown)")
    args = parser.parse_args()

    if args.mode != "ml":
        raise ValueError("Only mode=ml is supported by src.cli.infer_split for now")

    df, binary, cat, types, cal = infer_ml_split(args.split, args.config)

    report = generate_report(df, binary, cat, types, cal)
    report = report.replace("# LLM Classifier Evaluation Report", f"# ML Inference Report — {args.split}")

    REPORTS_RESEARCH_DIR.mkdir(parents=True, exist_ok=True)
    out = Path(args.output) if args.output else (REPORTS_RESEARCH_DIR / f"inference_ml_{args.split}.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report)

    print(f"Loaded {len(df)} samples from {args.split} split")
    print(f"Report saved -> {out}")
    print(f"Binary accuracy:    {binary['accuracy']:.4f}")
    print(f"False-negative rate: {binary['false_negative_rate']:.4f}")


if __name__ == "__main__":
    main()


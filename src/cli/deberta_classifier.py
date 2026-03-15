"""CLI entry point for DeBERTa classifier training and evaluation.

Usage:
    python -m src.cli.deberta_classifier --research --no-wandb
    python -m src.cli.deberta_classifier --train-only
    python -m src.cli.deberta_classifier --predict-only
"""

import os

# MPS (Apple Silicon) fragments memory aggressively; disable the watermark
# so PyTorch can spill to system RAM instead of OOMing.
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")

import argparse
import json
import logging

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.models.deberta_classifier import DeBERTaClassifier
from src.utils import (
    DEBERTA_ARTIFACTS_DIR,
    DEBERTA_REPORTS_DIR,
    PREDICTIONS_DIR,
    SPLITS_DIR,
    build_sample_id,
    ensure_dirs,
    load_config,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

GROUND_TRUTH_COLS = [
    "modified_sample",
    "original_sample",
    "attack_name",
    "label_binary",
    "label_category",
    "label_type",
    "prompt_hash",
]

EVAL_SPLITS = ["val", "test", "test_unseen"]


def compute_split_metrics(df: pd.DataFrame, label_col: str = "label_binary") -> dict:
    """Compute binary classification metrics for a split."""
    y_true = df[label_col]
    y_pred = df["deberta_pred_binary"]
    y_prob = df["deberta_proba_binary_adversarial"]

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, pos_label="adversarial", zero_division=0),
        "recall": recall_score(y_true, y_pred, pos_label="adversarial", zero_division=0),
        "f1": f1_score(y_true, y_pred, pos_label="adversarial", zero_division=0),
        "roc_auc": roc_auc_score((y_true == "adversarial").astype(int), y_prob),
        "average_precision": average_precision_score((y_true == "adversarial").astype(int), y_prob),
        "benign_recall": recall_score(y_true, y_pred, pos_label="benign", zero_division=0),
        "n_samples": len(df),
    }


def save_predictions(df_split: pd.DataFrame, preds: pd.DataFrame,
                     split_name: str, text_col: str):
    """Merge ground truth with predictions and save parquet."""
    gt_cols = [c for c in GROUND_TRUTH_COLS if c in df_split.columns]
    gt = df_split[gt_cols].reset_index(drop=True)
    out = pd.concat([gt, preds.reset_index(drop=True)], axis=1)
    out.insert(0, "sample_id", out[text_col].apply(build_sample_id))

    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    path = PREDICTIONS_DIR / f"deberta_predictions_{split_name}.parquet"
    out.to_parquet(path, index=False)
    logger.info(f"Predictions saved → {path} ({out.shape})")


def generate_summary(all_metrics: dict) -> str:
    """Generate a markdown summary table."""
    lines = ["# DeBERTa Classifier Evaluation\n"]
    lines.append("| Split | Accuracy | Precision | Recall | F1 | ROC-AUC | Avg-Prec | Benign Recall | N |")
    lines.append("|-------|----------|-----------|--------|-----|---------|----------|---------------|---|")
    for split, m in all_metrics.items():
        lines.append(
            f"| {split} | {m['accuracy']:.4f} | {m['precision']:.4f} | "
            f"{m['recall']:.4f} | {m['f1']:.4f} | {m['roc_auc']:.4f} | "
            f"{m['average_precision']:.4f} | {m['benign_recall']:.4f} | {m['n_samples']} |"
        )
    lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="DeBERTa classifier training and evaluation")
    parser.add_argument("--config", default=None)
    parser.add_argument("--no-wandb", action="store_true", help="Disable wandb (unused, for CLI compat)")
    parser.add_argument("--research", action="store_true",
                        help="Train + predict on all splits + save reports")
    parser.add_argument("--train-only", action="store_true", help="Only train, skip prediction")
    parser.add_argument("--predict-only", action="store_true", help="Only predict from saved model")
    args = parser.parse_args()

    cfg = load_config(args.config)
    text_col = cfg["dataset"]["text_col"]
    ensure_dirs()

    should_train = not args.predict_only
    should_predict = not args.train_only

    # ── Train ─────────────────────────────────────────────────────────────
    if should_train:
        logger.info("Loading train/val splits...")
        df_train = pd.read_parquet(SPLITS_DIR / "train.parquet")
        df_val = pd.read_parquet(SPLITS_DIR / "val.parquet")
        logger.info(f"Train: {len(df_train)}, Val: {len(df_val)}")

        clf = DeBERTaClassifier(cfg)
        clf.train(df_train, df_val, text_col=text_col, label_col="label_binary")
        clf.save(DEBERTA_ARTIFACTS_DIR)
        logger.info(f"Model saved to {DEBERTA_ARTIFACTS_DIR}")

    # ── Predict + Evaluate ────────────────────────────────────────────────
    if should_predict:
        clf = DeBERTaClassifier.load(DEBERTA_ARTIFACTS_DIR, cfg)

        all_metrics = {}
        all_reports = {}

        for split in EVAL_SPLITS:
            split_path = SPLITS_DIR / f"{split}.parquet"
            if not split_path.exists():
                logger.warning(f"Split {split} not found, skipping.")
                continue

            df_split = pd.read_parquet(split_path)
            logger.info(f"Predicting on {split} ({len(df_split)} samples)...")

            preds = clf.predict(df_split, text_col)
            save_predictions(df_split, preds, split, text_col)

            # Merge for metrics
            merged = pd.concat([df_split.reset_index(drop=True), preds.reset_index(drop=True)], axis=1)
            metrics = compute_split_metrics(merged)
            all_metrics[split] = metrics

            report = classification_report(
                merged["label_binary"], merged["deberta_pred_binary"], zero_division=0,
            )
            all_reports[split] = report
            print(f"\n{'='*60}")
            print(f"  {split.upper()} Split Results")
            print(f"{'='*60}")
            print(report)

        # Save reports
        DEBERTA_REPORTS_DIR.mkdir(parents=True, exist_ok=True)

        (DEBERTA_REPORTS_DIR / "metrics.json").write_text(
            json.dumps(all_metrics, indent=2)
        )
        (DEBERTA_REPORTS_DIR / "classification_report.json").write_text(
            json.dumps(all_reports, indent=2)
        )

        summary = generate_summary(all_metrics)
        (DEBERTA_REPORTS_DIR / "summary.md").write_text(summary)

        logger.info(f"Reports saved to {DEBERTA_REPORTS_DIR}")


if __name__ == "__main__":
    main()

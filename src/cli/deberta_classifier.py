"""CLI entry point for DeBERTa classifier training and evaluation.

Usage:
    python -m src.cli.deberta_classifier --research --no-wandb
    python -m src.cli.deberta_classifier --train-only
    python -m src.cli.deberta_classifier --predict-only
    python -m src.cli.deberta_classifier --research --cpu --debug-numerics --debug-first-n-batches 5
    python -m src.cli.deberta_classifier --research --cpu --sanity-forward-only --sanity-batches 3
"""

import os

# MPS (Apple Silicon) fragments memory aggressively; disable the watermark
# so PyTorch can spill to system RAM instead of OOMing.
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")

import argparse
import json
import logging
import sys

import pandas as pd
import wandb
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
from src.models.debug_numerics import DebugConfig
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
    "benign_source",
    "is_synthetic_benign",
]

EVAL_SPLITS = ["val", "test", "unseen_val", "unseen_test"]
MONITOR_SPLITS = ["unseen_val", "unseen_test"]


def compute_split_metrics(df: pd.DataFrame, label_col: str = "label_binary") -> dict:
    """Compute binary classification metrics for a split."""
    y_true = df[label_col]
    y_pred = df["deberta_pred_binary"]
    y_prob = df["deberta_proba_binary_adversarial"]

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, pos_label="adversarial", zero_division=0),
        "recall": recall_score(y_true, y_pred, pos_label="adversarial", zero_division=0),
        "f1": f1_score(y_true, y_pred, pos_label="adversarial", zero_division=0),
        "roc_auc": roc_auc_score((y_true == "adversarial").astype(int), y_prob),
        "average_precision": average_precision_score((y_true == "adversarial").astype(int), y_prob),
        "benign_recall": recall_score(y_true, y_pred, pos_label="benign", zero_division=0),
        "n_samples": len(df),
    }
    for label in sorted(set(y_true) | set(y_pred)):
        metrics[f"f1_{label}"] = f1_score(y_true, y_pred, pos_label=label, zero_division=0)
    return metrics


def compute_non_synthetic_benign_metrics(df: pd.DataFrame, label_col: str = "label_binary") -> dict | None:
    if "is_synthetic_benign" not in df.columns:
        return None
    benign_mask = (df[label_col] == "benign") & (~df["is_synthetic_benign"].fillna(False).astype(bool))
    n_benign = int(benign_mask.sum())
    if n_benign == 0:
        return None
    subset = df[(df[label_col] == "adversarial") | benign_mask].copy()
    metrics = compute_split_metrics(subset, label_col=label_col)
    metrics["n_non_synthetic_benign"] = n_benign
    return metrics


def load_monitor_splits() -> dict[str, pd.DataFrame]:
    """Load unseen splits for training-time monitoring when available."""
    monitor_dfs = {}
    for split in MONITOR_SPLITS:
        split_path = SPLITS_DIR / f"{split}.parquet"
        if not split_path.exists():
            logger.warning(f"Monitor split {split} not found, skipping.")
            continue
        monitor_dfs[split] = pd.read_parquet(split_path)
        logger.info(f"Monitor {split}: {len(monitor_dfs[split])}")
    return monitor_dfs


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
    ns_splits = {split: m for split, m in all_metrics.items() if "non_synthetic_benign" in m}
    if ns_splits:
        lines.extend([
            "",
            "## Non-Synthetic Benign Slice",
            "",
            "| Split | Accuracy | Precision | Recall | F1 | ROC-AUC | Avg-Prec | Benign Recall | Non-Synth Benign N | Total N |",
            "|-------|----------|-----------|--------|-----|---------|----------|---------------|--------------------|---------|",
        ])
        for split, m in ns_splits.items():
            ns = m["non_synthetic_benign"]
            lines.append(
                f"| {split} | {ns['accuracy']:.4f} | {ns['precision']:.4f} | "
                f"{ns['recall']:.4f} | {ns['f1']:.4f} | {ns['roc_auc']:.4f} | "
                f"{ns['average_precision']:.4f} | {ns['benign_recall']:.4f} | "
                f"{ns['n_non_synthetic_benign']} | {ns['n_samples']} |"
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
    parser.add_argument("--cpu", action="store_true", help="Force training/inference on CPU")

    # Debug flags
    debug_group = parser.add_argument_group("debug", "Numeric debug instrumentation")
    debug_group.add_argument("--debug-numerics", action="store_true",
                             help="Enable numeric debug instrumentation")
    debug_group.add_argument("--debug-first-n-batches", type=int, default=3,
                             help="Number of batches for verbose debug logging (default: 3)")
    debug_group.add_argument("--debug-save-bad-batch", action="store_true",
                             help="Save batch artifacts when NaN is detected")
    debug_group.add_argument("--debug-stop-on-nan", action="store_true", default=True,
                             help="Stop training on first NaN (default: true)")
    debug_group.add_argument("--debug-log-param-stats", action="store_true",
                             help="Log parameter statistics during debug batches")
    debug_group.add_argument("--debug-log-batch-text", action="store_true",
                             help="Log decoded batch text during debug batches")

    sanity_group = parser.add_argument_group("sanity", "Sanity forward-only mode")
    sanity_group.add_argument("--sanity-forward-only", action="store_true",
                              help="Run forward passes only (no backward/optimizer)")
    sanity_group.add_argument("--sanity-batches", type=int, default=3,
                              help="Number of batches for sanity forward (default: 3)")

    args = parser.parse_args()

    cfg = load_config(args.config)
    text_col = cfg["dataset"]["text_col"]
    ensure_dirs()

    dcfg = cfg["deberta"]

    if not args.no_wandb:
        wandb.init(
            project="llm-gatekeeping",
            name="deberta-classifier",
            config={
                "model_name": dcfg["model_name"],
                "max_length": dcfg["max_length"],
                "num_epochs": dcfg["num_epochs"],
                "batch_size": dcfg["batch_size"],
                "eval_batch_size": dcfg.get("eval_batch_size", 8),
                "learning_rate": dcfg["learning_rate"],
                "warmup_ratio": dcfg["warmup_ratio"],
                "weight_decay": dcfg["weight_decay"],
                "early_stopping_patience": dcfg["early_stopping_patience"],
                "max_grad_norm": dcfg.get("max_grad_norm", 0.5),
                "label_order": dcfg.get("label_order", ["benign", "adversarial"]),
                "research": args.research,
                "train_only": args.train_only,
                "predict_only": args.predict_only,
                "force_cpu": args.cpu,
                "debug_numerics": args.debug_numerics,
                "sanity_forward_only": args.sanity_forward_only,
            },
        )

    # Build DebugConfig
    debug = DebugConfig(
        enabled=args.debug_numerics,
        first_n_batches=args.debug_first_n_batches,
        save_bad_batch=args.debug_save_bad_batch,
        stop_on_nan=args.debug_stop_on_nan,
        log_param_stats=args.debug_log_param_stats,
        log_batch_text=args.debug_log_batch_text,
        sanity_forward_only=args.sanity_forward_only,
        sanity_batches=args.sanity_batches,
    )

    should_train = not args.predict_only
    should_predict = not args.train_only

    # ── Train ─────────────────────────────────────────────────────────────
    if should_train:
        logger.info("Loading train/val splits...")
        df_train = pd.read_parquet(SPLITS_DIR / "train.parquet")
        df_val = pd.read_parquet(SPLITS_DIR / "val.parquet")
        logger.info(f"Train: {len(df_train)}, Val: {len(df_val)}")

        if wandb.run is not None:
            wandb.log({
                "train_samples": len(df_train),
                "val_samples": len(df_val),
            })

        monitor_dfs = load_monitor_splits()
        clf = DeBERTaClassifier(cfg)
        result = clf.train(df_train, df_val, text_col=text_col, label_col="label_binary",
                           force_cpu=args.cpu, debug=debug, monitor_dfs=monitor_dfs)

        if wandb.run is not None and result.train_history:
            for epoch_metrics in result.train_history:
                log_payload = {
                    "epoch": epoch_metrics["epoch"],
                    "train_loss": epoch_metrics["train_loss"],
                    "eval/accuracy": epoch_metrics["eval_accuracy"],
                    "eval/f1": epoch_metrics["eval_f1"],
                    "eval/macro_f1": epoch_metrics["eval_macro_f1"],
                    "eval/precision": epoch_metrics["eval_precision"],
                    "eval/recall": epoch_metrics["eval_recall"],
                }
                for metric_name, metric_value in epoch_metrics.items():
                    if metric_name.startswith("eval_f1_"):
                        log_payload[f"eval/{metric_name.removeprefix('eval_')}"] = metric_value
                    for split in MONITOR_SPLITS:
                        prefix = f"{split}_"
                        if metric_name.startswith(prefix):
                            monitor_metric = metric_name.removeprefix(prefix)
                            log_payload[f"monitor/{split}/{monitor_metric}"] = metric_value
                wandb.log(log_payload, step=epoch_metrics["epoch"])
            wandb.log({
                "best_epoch": result.best_epoch,
                "best_metric_name": result.best_metric_name,
                "best_metric_value": result.best_metric_value,
                "stopped_early": int(result.stopped_early),
            })

        if not result.success:
            if wandb.run is not None:
                wandb.log({
                    "training_success": 0,
                    "failed_epoch": result.first_bad_epoch,
                    "failed_step": result.first_bad_step,
                    "failed_stage": result.first_bad_stage,
                    "failed_param": result.first_bad_param,
                })
                wandb.finish(exit_code=1)
            logger.error(f"Training failed: {result.failed_reason}")
            logger.error(f"  epoch={result.first_bad_epoch}, step={result.first_bad_step}, "
                         f"stage={result.first_bad_stage}, param={result.first_bad_param}")
            if result.debug_artifact_paths:
                logger.error(f"  debug artifacts: {result.debug_artifact_paths}")
            sys.exit(1)

        clf.save(DEBERTA_ARTIFACTS_DIR)
        logger.info(f"Model saved to {DEBERTA_ARTIFACTS_DIR}")

        if wandb.run is not None:
            wandb.log({"training_success": 1})

    # ── Predict + Evaluate ────────────────────────────────────────────────
    if should_predict:
        clf = DeBERTaClassifier.load(DEBERTA_ARTIFACTS_DIR, cfg, force_cpu=args.cpu)

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
            non_synth_metrics = compute_non_synthetic_benign_metrics(merged)
            if non_synth_metrics is not None:
                metrics["non_synthetic_benign"] = non_synth_metrics
            all_metrics[split] = metrics

            if wandb.run is not None:
                wandb.log({f"{split}/{metric}": value for metric, value in metrics.items()})

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

    if wandb.run is not None:
        wandb.finish()


if __name__ == "__main__":
    main()

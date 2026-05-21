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
from copy import deepcopy
from dataclasses import dataclass
import json
import logging
import sys
from pathlib import Path

import pandas as pd
import torch
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

EVAL_SPLITS = ["val", "test", "unseen_val", "unseen_test", "safeguard_test"]
MONITOR_SPLITS = ["unseen_val", "unseen_test"]
REQUIRED_SPLITS = ["train", "val"]
REQUIRED_LABELS = ["benign", "adversarial"]


@dataclass(frozen=True)
class RuntimePaths:
    splits_dir: Path
    artifacts_dir: Path
    predictions_dir: Path
    reports_dir: Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DeBERTa classifier training and evaluation")
    parser.add_argument("--config", default=None)
    parser.add_argument("--no-wandb", action="store_true", help="Disable wandb (unused, for CLI compat)")
    parser.add_argument("--research", action="store_true",
                        help="Train + predict on all splits + save reports")
    parser.add_argument("--train-only", action="store_true", help="Only train, skip prediction")
    parser.add_argument("--predict-only", action="store_true", help="Only predict from saved model")
    parser.add_argument("--cpu", action="store_true", help="Force training/inference on CPU")
    parser.add_argument("--splits-dir", default=None,
                        help="Directory containing train/val/eval split parquet files")
    parser.add_argument("--artifacts-dir", default=None,
                        help="Directory for DeBERTa model artifacts")
    parser.add_argument("--output-dir", default=None,
                        help="Alias for --artifacts-dir")
    parser.add_argument("--predictions-dir", default=None,
                        help="Directory for prediction parquet outputs")
    parser.add_argument("--reports-dir", default=None,
                        help="Directory for DeBERTa report outputs")
    parser.add_argument("--wandb-project", default=None,
                        help="Override W&B project name")
    parser.add_argument("--wandb-run-name", default=None,
                        help="Override W&B run name")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto",
                        help="Training/inference device. Defaults to auto.")
    parser.add_argument("--num-epochs", type=int, default=None,
                        help="Override configs/default.yaml deberta.num_epochs")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Override configs/default.yaml deberta.batch_size")
    parser.add_argument("--learning-rate", type=float, default=None,
                        help="Override configs/default.yaml deberta.learning_rate")

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
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.artifacts_dir and args.output_dir and Path(args.artifacts_dir) != Path(args.output_dir):
        parser.error("--artifacts-dir and --output-dir refer to the same destination; provide only one")
    return args


def resolve_runtime_paths(args: argparse.Namespace) -> RuntimePaths:
    artifacts_dir = args.artifacts_dir or args.output_dir
    return RuntimePaths(
        splits_dir=Path(args.splits_dir) if args.splits_dir else SPLITS_DIR,
        artifacts_dir=Path(artifacts_dir) if artifacts_dir else DEBERTA_ARTIFACTS_DIR,
        predictions_dir=Path(args.predictions_dir) if args.predictions_dir else PREDICTIONS_DIR,
        reports_dir=Path(args.reports_dir) if args.reports_dir else DEBERTA_REPORTS_DIR,
    )


def resolve_wandb_settings(args: argparse.Namespace) -> tuple[str, str]:
    return (
        args.wandb_project or "llm-gatekeeping",
        args.wandb_run_name or "deberta-classifier",
    )


def apply_training_overrides(cfg: dict, args: argparse.Namespace) -> dict:
    cfg = deepcopy(cfg)
    dcfg = cfg["deberta"]
    if args.num_epochs is not None:
        dcfg["num_epochs"] = args.num_epochs
    if args.batch_size is not None:
        dcfg["batch_size"] = args.batch_size
    if args.learning_rate is not None:
        dcfg["learning_rate"] = args.learning_rate
    return cfg


def resolve_device(device: str, force_cpu: bool = False) -> str:
    if force_cpu:
        if device not in (None, "auto", "cpu"):
            raise SystemExit("--cpu cannot be combined with --device other than cpu/auto")
        return "cpu"
    if device in (None, "auto"):
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("Requested device 'cuda' is not available")
    if device == "mps" and not torch.backends.mps.is_available():
        raise SystemExit("Requested device 'mps' is not available")
    return device


def _validate_split_frame(path: Path, df: pd.DataFrame, text_col: str, label_order: list[str]):
    if not isinstance(df, pd.DataFrame):
        return
    missing_cols = [c for c in [text_col, "label_binary"] if c not in df.columns]
    if missing_cols:
        raise SystemExit(f"Split {path} missing required columns: {missing_cols}")
    invalid_labels = sorted(set(df["label_binary"].dropna()) - set(label_order))
    if invalid_labels:
        raise SystemExit(f"Split {path} has invalid label_binary values: {invalid_labels}")
    if df["label_binary"].isna().any():
        raise SystemExit(f"Split {path} has missing label_binary values")


def validate_split_inputs(
    splits_dir: Path,
    text_col: str,
    label_order: list[str],
    optional_splits: list[str] | None = None,
) -> dict[str, pd.DataFrame]:
    split_dfs = {}
    for split in REQUIRED_SPLITS:
        path = splits_dir / f"{split}.parquet"
        if not path.exists():
            raise SystemExit(f"Missing required split: {path}")
        df = pd.read_parquet(path)
        _validate_split_frame(path, df, text_col, label_order)
        split_dfs[split] = df

    for split in optional_splits if optional_splits is not None else EVAL_SPLITS:
        if split in split_dfs:
            continue
        path = splits_dir / f"{split}.parquet"
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        _validate_split_frame(path, df, text_col, label_order)
        split_dfs[split] = df
    return split_dfs


def ensure_writable_dirs(paths: list[Path]):
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / ".write_test"
        try:
            probe.write_text("ok")
        except OSError as exc:
            raise SystemExit(f"Output directory is not writable: {path}") from exc
        finally:
            if probe.exists():
                probe.unlink()


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


def load_monitor_splits(splits_dir: Path) -> dict[str, pd.DataFrame]:
    """Load unseen splits for training-time monitoring when available."""
    monitor_dfs = {}
    for split in MONITOR_SPLITS:
        split_path = splits_dir / f"{split}.parquet"
        if not split_path.exists():
            logger.warning(f"Monitor split {split} not found, skipping.")
            continue
        monitor_dfs[split] = pd.read_parquet(split_path)
        logger.info(f"Monitor {split}: {len(monitor_dfs[split])}")
    return monitor_dfs


def build_training_log_payload(epoch_metrics: dict) -> dict:
    """Convert training history keys to WandB metric names."""
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
    return log_payload


def build_training_batch_log_payload(batch_metrics: dict) -> dict:
    """Convert training batch metrics to WandB metric names."""
    return {
        "epoch": batch_metrics["epoch"],
        "train/batch": batch_metrics["batch"],
        "train/loss_step": batch_metrics["train_loss_step"],
        "train/learning_rate": batch_metrics["learning_rate"],
    }


def save_predictions(df_split: pd.DataFrame, preds: pd.DataFrame,
                     split_name: str, text_col: str, predictions_dir: Path):
    """Merge ground truth with predictions and save parquet."""
    gt_cols = [c for c in GROUND_TRUTH_COLS if c in df_split.columns]
    gt = df_split[gt_cols].reset_index(drop=True)
    out = pd.concat([gt, preds.reset_index(drop=True)], axis=1)
    out.insert(0, "sample_id", out[text_col].apply(build_sample_id))

    predictions_dir.mkdir(parents=True, exist_ok=True)
    path = predictions_dir / f"deberta_predictions_{split_name}.parquet"
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


def run_deberta(args: argparse.Namespace):
    cfg = apply_training_overrides(load_config(args.config), args)
    text_col = cfg["dataset"]["text_col"]
    dcfg = cfg["deberta"]
    label_order = dcfg.get("label_order", REQUIRED_LABELS)
    runtime_paths = resolve_runtime_paths(args)
    selected_device = resolve_device(args.device, force_cpu=args.cpu)

    should_train = not args.predict_only
    should_predict = not args.train_only
    optional_splits = []
    if should_train:
        optional_splits.extend(MONITOR_SPLITS)
    if should_predict:
        optional_splits.extend(EVAL_SPLITS)
    split_dfs = validate_split_inputs(
        runtime_paths.splits_dir,
        text_col,
        label_order,
        optional_splits=optional_splits,
    )
    output_dirs = [runtime_paths.artifacts_dir]
    if should_predict:
        output_dirs.extend([runtime_paths.predictions_dir, runtime_paths.reports_dir])
    ensure_writable_dirs(output_dirs)

    if not args.no_wandb:
        wandb_project, wandb_run_name = resolve_wandb_settings(args)
        wandb.init(
            project=wandb_project,
            name=wandb_run_name,
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
                "device": selected_device,
                "splits_dir": str(runtime_paths.splits_dir),
                "artifacts_dir": str(runtime_paths.artifacts_dir),
                "predictions_dir": str(runtime_paths.predictions_dir),
                "reports_dir": str(runtime_paths.reports_dir),
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

    # ── Train ─────────────────────────────────────────────────────────────
    if should_train:
        logger.info("Loading train/val splits...")
        df_train = split_dfs["train"]
        df_val = split_dfs["val"]
        logger.info(f"Train: {len(df_train)}, Val: {len(df_val)}")

        if wandb.run is not None:
            wandb.log({
                "train_samples": len(df_train),
                "val_samples": len(df_val),
            })

        monitor_dfs = {
            split: split_dfs[split]
            for split in MONITOR_SPLITS
            if split in split_dfs
        }
        clf = DeBERTaClassifier(cfg)
        on_epoch_end = None
        on_train_batch_end = None
        if wandb.run is not None:
            def on_epoch_end(epoch_metrics):
                wandb.log(
                    build_training_log_payload(epoch_metrics),
                    step=epoch_metrics["epoch"],
                )
            def on_train_batch_end(batch_metrics):
                wandb.log(
                    build_training_batch_log_payload(batch_metrics),
                    step=batch_metrics["global_step"],
                )

        result = clf.train(df_train, df_val, text_col=text_col, label_col="label_binary",
                           force_cpu=args.cpu, device=selected_device, debug=debug, monitor_dfs=monitor_dfs,
                           on_epoch_end=on_epoch_end, on_train_batch_end=on_train_batch_end)

        if wandb.run is not None and result.train_history and on_epoch_end is None:
            for epoch_metrics in result.train_history:
                wandb.log(
                    build_training_log_payload(epoch_metrics),
                    step=epoch_metrics["epoch"],
                )
        if wandb.run is not None:
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

        clf.save(runtime_paths.artifacts_dir)
        logger.info(f"Model saved to {runtime_paths.artifacts_dir}")

        if wandb.run is not None:
            wandb.log({"training_success": 1})

    # ── Predict + Evaluate ────────────────────────────────────────────────
    if should_predict:
        clf = DeBERTaClassifier.load(runtime_paths.artifacts_dir, cfg,
                                     force_cpu=args.cpu, device=selected_device)

        all_metrics = {}
        all_reports = {}

        for split in EVAL_SPLITS:
            if split not in split_dfs:
                logger.warning(f"Split {split} not found, skipping.")
                continue

            df_split = split_dfs[split]
            logger.info(f"Predicting on {split} ({len(df_split)} samples)...")

            preds = clf.predict(df_split, text_col)
            save_predictions(df_split, preds, split, text_col, runtime_paths.predictions_dir)

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
        runtime_paths.reports_dir.mkdir(parents=True, exist_ok=True)

        (runtime_paths.reports_dir / "metrics.json").write_text(
            json.dumps(all_metrics, indent=2)
        )
        (runtime_paths.reports_dir / "classification_report.json").write_text(
            json.dumps(all_reports, indent=2)
        )

        summary = generate_summary(all_metrics)
        (runtime_paths.reports_dir / "summary.md").write_text(summary)

        logger.info(f"Reports saved to {runtime_paths.reports_dir}")

    if wandb.run is not None:
        wandb.finish()


def main():
    run_deberta(parse_args())


if __name__ == "__main__":
    main()

"""Evaluate external baseline detector outputs and compare against repo models."""

from __future__ import annotations

import argparse
import math

import pandas as pd

from src.baselines.threshold import (
    evaluate_at_threshold,
    tune_threshold_bounded_fpr,
    tune_threshold_low_fnr,
)
from src.eval_external import load_external_dataset
from src.evaluate import binary_metrics
from src.utils import (
    BASELINES_DIR,
    REPORTS_BASELINES_DIR,
    RESEARCH_DIR,
    RESEARCH_EXTERNAL_DIR,
    SPLITS_DIR,
    build_sample_id,
    ensure_dirs,
    load_config,
)

BASELINE_LABELS = {
    "sentinel_v2": "Sentinel v2",
    "protectai_v2": "ProtectAI v2",
}


def _fmt(value) -> str:
    if value is None:
        return "-"
    if isinstance(value, str):
        return value
    if isinstance(value, (int, bool)):
        return str(value)
    if math.isnan(value):
        return "-"
    return f"{value:.4f}"


def _fmt_threshold(value: float) -> str:
    if value < 0.0:
        return "<0.0"
    if value > 1.0:
        return ">1.0"
    return f"{value:.4f}"


def _load_ground_truth(cfg: dict, dataset_key: str) -> pd.DataFrame:
    if dataset_key.startswith("external_"):
        external_key = dataset_key.removeprefix("external_")
        return load_external_dataset(cfg["external_datasets"][external_key])
    return pd.read_parquet(SPLITS_DIR / f"{dataset_key}.parquet")


def _load_research_df(dataset_key: str) -> pd.DataFrame:
    if dataset_key.startswith("external_"):
        external_key = dataset_key.removeprefix("external_")
        return pd.read_parquet(RESEARCH_EXTERNAL_DIR / f"research_external_{external_key}.parquet")
    return pd.read_parquet(RESEARCH_DIR / f"research_{dataset_key}.parquet")


def _research_path_exists(dataset_key: str) -> bool:
    if dataset_key.startswith("external_"):
        external_key = dataset_key.removeprefix("external_")
        return (RESEARCH_EXTERNAL_DIR / f"research_external_{external_key}.parquet").exists()
    return (RESEARCH_DIR / f"research_{dataset_key}.parquet").exists()


def _available_dataset_keys() -> list[str]:
    keys = []
    for path in sorted(BASELINES_DIR.glob("*.parquet")):
        stem = path.stem
        for baseline_key in BASELINE_LABELS:
            prefix = f"{baseline_key}_"
            if stem.startswith(prefix):
                keys.append(stem[len(prefix):])
                break
    keys = sorted(set(keys))
    return [key for key in keys if key != "val"]


def _load_baseline_eval_frame(cfg: dict, baseline_key: str, dataset_key: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    pred_path = BASELINES_DIR / f"{baseline_key}_{dataset_key}.parquet"
    preds = pd.read_parquet(pred_path)
    if dataset_key != "val" and _research_path_exists(dataset_key):
        gt = _load_research_df(dataset_key)[["sample_id", "label_binary"]].copy()
    else:
        gt = _load_ground_truth(cfg, dataset_key)
        gt = gt[["modified_sample", "label_binary"]].copy()
        gt.insert(0, "sample_id", gt["modified_sample"].map(build_sample_id))
    merged = gt.merge(preds, on="sample_id", how="inner", validate="one_to_one")
    return preds, merged


def _tuned_thresholds(cfg: dict, baseline_key: str) -> dict[str, dict]:
    _, merged = _load_baseline_eval_frame(cfg, baseline_key, "val")
    y_true = merged["label_binary"]
    scores = merged["adversarial_score"]
    return {
        "default": {
            "threshold": float(cfg["baselines"][baseline_key]["default_threshold"]),
            "constraint_met": True,
            "target_metric": "default",
            "target_value": None,
        },
        "low_fnr": tune_threshold_low_fnr(y_true, scores, max_fnr=0.02),
        "bounded_fpr": tune_threshold_bounded_fpr(y_true, scores, max_fpr=0.05),
    }


def _row_for_existing_model(name: str, y_true: pd.Series, y_pred: pd.Series, auroc=None, auprc=None) -> dict:
    metrics = binary_metrics(y_true, y_pred)
    adv_mask = y_true == "adversarial"
    ben_mask = y_true == "benign"
    tp = int((adv_mask & (y_pred == "adversarial")).sum())
    fn = int((adv_mask & (y_pred == "benign")).sum())
    fp = int((ben_mask & (y_pred == "adversarial")).sum())
    tn = int((ben_mask & (y_pred == "benign")).sum())
    return {
        "Model": name,
        "Threshold": "-",
        "Accuracy": _fmt(metrics["accuracy"]),
        "AUROC": _fmt(auroc),
        "AUPRC": _fmt(auprc),
        "Adv Recall": _fmt(metrics["adversarial_recall"]),
        "Benign Recall": _fmt(metrics["benign_recall"]),
        "FPR": _fmt(metrics["false_positive_rate"]),
        "FNR": _fmt(metrics["false_negative_rate"]),
        "TP": str(tp),
        "FP": str(fp),
        "TN": str(tn),
        "FN": str(fn),
    }


def _existing_model_rows(research_df: pd.DataFrame) -> list[dict]:
    y_true = research_df["label_binary"]
    ml_row = _row_for_existing_model(
        "Our ML",
        y_true,
        research_df["ml_pred_binary"],
        auroc=evaluate_at_threshold(y_true, research_df["ml_proba_binary_adversarial"], 0.5)["auroc"],
        auprc=evaluate_at_threshold(y_true, research_df["ml_proba_binary_adversarial"], 0.5)["auprc"],
    )
    hybrid_row = _row_for_existing_model("Our Hybrid", y_true, research_df["hybrid_pred_binary"])
    return [ml_row, hybrid_row]


def _baseline_rows(merged: pd.DataFrame, baseline_name: str, thresholds: dict[str, dict]) -> list[dict]:
    rows = []
    for threshold_name in ["default", "low_fnr", "bounded_fpr"]:
        tuned = thresholds[threshold_name]
        metrics = evaluate_at_threshold(
            merged["label_binary"],
            merged["adversarial_score"],
            tuned["threshold"],
        )
        rows.append({
            "Model": baseline_name,
            "Threshold": f"{threshold_name} ({_fmt_threshold(tuned['threshold'])})",
            "Accuracy": _fmt(metrics["accuracy"]),
            "AUROC": _fmt(metrics["auroc"]),
            "AUPRC": _fmt(metrics["auprc"]),
            "Adv Recall": _fmt(metrics["adversarial_recall"]),
            "Benign Recall": _fmt(metrics["benign_recall"]),
            "FPR": _fmt(metrics["false_positive_rate"]),
            "FNR": _fmt(metrics["false_negative_rate"]),
            "TP": str(metrics["tp"]),
            "FP": str(metrics["fp"]),
            "TN": str(metrics["tn"]),
            "FN": str(metrics["fn"]),
        })
    return rows


def _threshold_summary_line(label: str, tuned: dict) -> str:
    target = tuned["target_value"]
    target_text = "-" if target is None else f"{target:.4f}"
    return (
        f"- {label}: threshold={_fmt_threshold(tuned['threshold'])}, "
        f"target_metric={tuned['target_metric']}, target={target_text}, "
        f"constraint_met={tuned['constraint_met']}"
    )


def _render_dataset_section(dataset_key: str, research_df: pd.DataFrame, baseline_sections: list[tuple[str, dict[str, dict], pd.DataFrame]]) -> str:
    lines = [f"## {dataset_key}", ""]
    for baseline_name, thresholds, _ in baseline_sections:
        lines.append(f"### {baseline_name} Threshold Tuning")
        lines.append(_threshold_summary_line("default", thresholds["default"]))
        lines.append(_threshold_summary_line("low_fnr", thresholds["low_fnr"]))
        lines.append(_threshold_summary_line("bounded_fpr", thresholds["bounded_fpr"]))
        lines.append("")

    rows = _existing_model_rows(research_df)
    for baseline_name, thresholds, merged in baseline_sections:
        rows.extend(_baseline_rows(merged, baseline_name, thresholds))

    columns = ["Model", "Threshold", "Accuracy", "AUROC", "AUPRC", "Adv Recall", "Benign Recall", "FPR", "FNR", "TP", "FP", "TN", "FN"]
    lines.append("| " + " | ".join(columns) + " |")
    lines.append("|" + "|".join(["---"] * len(columns)) + "|")
    for row in rows:
        lines.append("| " + " | ".join(row[col] for col in columns) + " |")
    lines.append("")
    return "\n".join(lines)


def generate_report(cfg: dict) -> str:
    sections = ["# External Baseline Comparison", ""]
    skipped = []
    for dataset_key in _available_dataset_keys():
        if not _research_path_exists(dataset_key):
            skipped.append(dataset_key)
            continue
        research_df = _load_research_df(dataset_key)
        baseline_sections = []
        for baseline_key, baseline_name in BASELINE_LABELS.items():
            pred_path = BASELINES_DIR / f"{baseline_key}_{dataset_key}.parquet"
            if not pred_path.exists():
                continue
            _, merged = _load_baseline_eval_frame(cfg, baseline_key, dataset_key)
            thresholds = _tuned_thresholds(cfg, baseline_key)
            baseline_sections.append((baseline_name, thresholds, merged))
        if baseline_sections:
            sections.append(_render_dataset_section(dataset_key, research_df, baseline_sections))
    if skipped:
        sections.extend([
            "## Skipped Datasets",
            "",
            "No matching research parquet was available for these baseline datasets: "
            + ", ".join(sorted(skipped)),
            "",
        ])
    return "\n".join(sections).rstrip() + "\n"


def main():
    parser = argparse.ArgumentParser(description="Evaluate HuggingFace baseline outputs")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    ensure_dirs()
    report = generate_report(cfg)
    out_path = REPORTS_BASELINES_DIR / "comparison_report.md"
    out_path.write_text(report)
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()

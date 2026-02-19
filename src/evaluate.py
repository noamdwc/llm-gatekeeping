"""
Evaluation framework for the hierarchical classifier.

Computes metrics at each hierarchy level:
  - Binary: precision, recall, F1 (focus on false-negative rate)
  - Category: accuracy, confusion matrix
  - Per-type: macro F1, per-class breakdown (unicode types only)
  - Calibration: confidence vs accuracy buckets
  - Cost: tokens, latency

Usage:
    python -m src.evaluate --predictions data/processed/predictions_test.csv \
                           [--output reports/eval_report_llm.md]
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support
)

from src.utils import REPORTS_RESEARCH_DIR


def binary_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict:
    """Compute binary detection metrics (adversarial vs benign)."""
    labels = ["adversarial", "benign"]
    p, r, f, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )
    acc = accuracy_score(y_true, y_pred)

    # False-negative rate: adversarial samples predicted as benign
    adv_mask = y_true == "adversarial"
    fn_rate = 0.0
    if adv_mask.sum() > 0:
        fn_rate = (y_pred[adv_mask] == "benign").mean()

    return {
        "accuracy": acc,
        "adversarial_precision": p[0],
        "adversarial_recall": r[0],
        "adversarial_f1": f[0],
        "benign_precision": p[1],
        "benign_recall": r[1],
        "benign_f1": f[1],
        "false_negative_rate": fn_rate,
        "support_adversarial": int(adv_mask.sum()),
        "support_benign": int((~adv_mask).sum()),
    }


def category_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict:
    """Compute category-level metrics (unicode_attack vs nlp_attack)."""
    # Filter to adversarial ground truth only; keep FNs (adv predicted as benign) as errors
    mask = y_true != "benign"
    yt, yp = y_true[mask], y_pred[mask]
    if len(yt) == 0:
        return {"category_accuracy": 0.0, "category_f1_macro": 0.0}

    labels = ["unicode_attack", "nlp_attack"]
    acc = accuracy_score(yt, yp)
    f1 = f1_score(yt, yp, labels=labels, average="macro", zero_division=0)
    cm = confusion_matrix(yt, yp, labels=labels)

    return {
        "category_accuracy": acc,
        "category_f1_macro": f1,
        "confusion_matrix": cm.tolist(),
        "confusion_labels": labels,
    }


def type_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict:
    """Compute per-type metrics for unicode attack sub-types."""
    # Filter to unicode predictions
    mask = (y_true != "benign") & (y_true != "nlp_attack")
    yt, yp = y_true[mask], y_pred[mask]
    if len(yt) == 0:
        return {"type_accuracy": 0.0, "type_f1_macro": 0.0}

    labels = sorted(yt.unique())
    acc = accuracy_score(yt, yp)
    f1 = f1_score(yt, yp, labels=labels, average="macro", zero_division=0)
    report = classification_report(yt, yp, labels=labels, zero_division=0, output_dict=True)

    return {
        "type_accuracy": acc,
        "type_f1_macro": f1,
        "type_report": report,
    }


def calibration_metrics(
    y_true: pd.Series, y_pred: pd.Series, confidences: pd.Series, n_bins: int = 10
) -> dict:
    """Compute calibration: confidence vs accuracy in buckets."""
    correct = (y_true == y_pred).astype(float)
    conf = confidences.fillna(0.5).astype(float)

    bins = np.linspace(0, 1, n_bins + 1)
    buckets = []
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (conf >= lo) & (conf < hi) if i < n_bins - 1 else (conf >= lo) & (conf <= hi)
        n = mask.sum()
        if n == 0:
            continue
        buckets.append({
            "bin": f"{lo:.1f}-{hi:.1f}",
            "count": int(n),
            "avg_confidence": float(conf[mask].mean()),
            "accuracy": float(correct[mask].mean()),
        })

    return {"calibration_buckets": buckets}


def generate_report(
    df: pd.DataFrame,
    binary: dict,
    category: dict,
    types: dict,
    calibration: dict,
    usage: dict | None = None,
    title: str = "LLM Classifier Evaluation Report",
) -> str:
    """Generate a Markdown evaluation report."""
    lines = [f"# {title}\n"]

    # Binary
    lines.append("## Binary Detection (Adversarial vs Benign)\n")
    lines.append(f"| Metric | Value |")
    lines.append(f"|--------|-------|")
    for k, v in binary.items():
        if k.startswith("support"):
            lines.append(f"| {k} | {v} |")
        else:
            lines.append(f"| {k} | {v:.4f} |")
    lines.append("")

    # Category
    lines.append("## Category Classification (Unicode vs NLP)\n")
    lines.append(f"- Accuracy: {category.get('category_accuracy', 0):.4f}")
    lines.append(f"- Macro F1: {category.get('category_f1_macro', 0):.4f}")
    if "confusion_matrix" in category:
        lines.append(f"\nConfusion matrix (rows=true, cols=pred):")
        lines.append(f"Labels: {category['confusion_labels']}")
        for row in category["confusion_matrix"]:
            lines.append(f"  {row}")
    lines.append("")

    # Type
    lines.append("## Per-Type Classification (Unicode Sub-Types)\n")
    lines.append(f"- Accuracy: {types.get('type_accuracy', 0):.4f}")
    lines.append(f"- Macro F1: {types.get('type_f1_macro', 0):.4f}")
    if "type_report" in types:
        lines.append("\n| Type | Precision | Recall | F1 | Support |")
        lines.append("|------|-----------|--------|-----|---------|")
        for label, vals in types["type_report"].items():
            if isinstance(vals, dict) and "precision" in vals:
                lines.append(
                    f"| {label} | {vals['precision']:.2f} | {vals['recall']:.2f} "
                    f"| {vals['f1-score']:.2f} | {int(vals['support'])} |"
                )
    lines.append("")

    # Calibration
    lines.append("## Calibration\n")
    lines.append("| Bin | Count | Avg Confidence | Accuracy |")
    lines.append("|-----|-------|----------------|----------|")
    for b in calibration.get("calibration_buckets", []):
        lines.append(
            f"| {b['bin']} | {b['count']} | {b['avg_confidence']:.3f} | {b['accuracy']:.3f} |"
        )
    lines.append("")

    # Usage
    if usage:
        lines.append("## Cost / Usage\n")
        for k, v in usage.items():
            lines.append(f"- {k}: {v}")
        lines.append("")

    return "\n".join(lines)


def evaluate(predictions_path: str, output_path: str = None):
    """Run full evaluation on a predictions CSV."""
    df = pd.read_csv(predictions_path)

    # Determine which columns hold ground truth vs predictions
    # Ground truth columns: label_binary, label_category, label_type (from dataset)
    # Predicted columns from llm_classifier: also named label_binary, etc.
    # We handle the case where predictions CSV has both (suffixed)
    gt_binary = df.get("label_binary", pd.Series(dtype=str))
    gt_category = df.get("label_category", pd.Series(dtype=str))
    gt_type = df.get("label_type", pd.Series(dtype=str))

    # Check for predicted columns (from concat in llm_classifier.py they'll be duplicated)
    # If there are duplicate columns, pandas appends .1
    pred_binary = df.get("label_binary.1", gt_binary)
    pred_category = df.get("label_category.1", gt_category)
    pred_type = df.get("label_type.1", gt_type)
    if "confidence_binary" in df.columns:
        conf_binary = df["confidence_binary"]
    elif "confidence" in df.columns:
        conf_binary = df["confidence"]
    else:
        conf_binary = pd.Series([0.5] * len(df))

    # Handle case where there's a separate 'pred_' prefix
    if "pred_label_binary" in df.columns:
        pred_binary = df["pred_label_binary"]
        pred_category = df["pred_label_category"]
        pred_type = df["pred_label_type"]

    binary = binary_metrics(gt_binary, pred_binary)
    cat = category_metrics(gt_category, pred_category)
    types = type_metrics(gt_type, pred_type)
    cal = calibration_metrics(gt_binary, pred_binary, conf_binary)

    report = generate_report(df, binary, cat, types, cal)

    REPORTS_RESEARCH_DIR.mkdir(parents=True, exist_ok=True)
    out = output_path or str(REPORTS_RESEARCH_DIR / "eval_report_llm.md")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        f.write(report)
    print(f"Report saved → {out}")

    # Also print summary
    print(f"\n--- Summary ---")
    print(f"Binary accuracy:   {binary['accuracy']:.4f}")
    print(f"False-negative rate: {binary['false_negative_rate']:.4f}")
    print(f"Category accuracy: {cat.get('category_accuracy', 'N/A')}")
    print(f"Type accuracy:     {types.get('type_accuracy', 'N/A')}")

    return binary, cat, types, cal


def evaluate_dataframe(
    df_eval: pd.DataFrame,
    predictions: list[dict],
    output_path: str = None,
    usage: dict = None,
) -> tuple[dict, dict, dict, dict]:
    """
    Evaluate directly from a DataFrame and prediction dicts.
    Used programmatically by other modules.
    """
    preds = pd.DataFrame(predictions)
    gt_binary = df_eval["label_binary"].reset_index(drop=True)
    gt_category = df_eval["label_category"].reset_index(drop=True)
    gt_type = df_eval["label_type"].reset_index(drop=True)
    pred_binary = preds["label_binary"]
    pred_category = preds["label_category"]
    pred_type = preds["label_type"]
    conf_binary = preds.get("confidence_binary", pd.Series([0.5] * len(preds)))

    binary = binary_metrics(gt_binary, pred_binary)
    cat = category_metrics(gt_category, pred_category)
    types = type_metrics(gt_type, pred_type)
    cal = calibration_metrics(gt_binary, pred_binary, conf_binary)

    report = generate_report(
        pd.concat([df_eval.reset_index(drop=True), preds], axis=1),
        binary, cat, types, cal, usage,
    )

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            f.write(report)
        print(f"Report saved → {output_path}")

    return binary, cat, types, cal


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate classifier predictions")
    parser.add_argument("--predictions", required=True, help="Path to predictions CSV")
    parser.add_argument("--output", default=None, help="Output report path")
    args = parser.parse_args()
    evaluate(args.predictions, args.output)

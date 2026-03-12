"""Generate calibration-focused reports from the row-level hybrid margin trace."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.margin_trace import (
    apply_threshold_override,
    bucketize_margin,
    compute_binary_metrics_from_predictions,
    expected_accuracy_from_rates,
)
from src.utils import load_config, RESEARCH_DIR, REPORTS_ARTIFACTS_DIR


def _ece_and_buckets(correct: pd.Series, confidence: pd.Series, bins: int = 10) -> tuple[float, pd.DataFrame]:
    frame = pd.DataFrame({"correct": correct.astype(float), "confidence": confidence.astype(float)}).dropna()
    if frame.empty:
        return float("nan"), pd.DataFrame(columns=["bucket", "count", "avg_confidence", "accuracy"])
    edges = np.linspace(0.0, 1.0, bins + 1)
    rows = []
    ece = 0.0
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (frame["confidence"] >= lo) & (frame["confidence"] < hi)
        if i == bins - 1:
            mask = (frame["confidence"] >= lo) & (frame["confidence"] <= hi)
        bucket = frame.loc[mask]
        if bucket.empty:
            continue
        avg_conf = float(bucket["confidence"].mean())
        acc = float(bucket["correct"].mean())
        ece += (len(bucket) / len(frame)) * abs(avg_conf - acc)
        rows.append({
            "bucket": f"[{lo:.2f}, {hi:.2f}{']' if i == bins - 1 else ')'}",
            "count": int(len(bucket)),
            "avg_confidence": avg_conf,
            "accuracy": acc,
        })
    return ece, pd.DataFrame(rows)


def _margin_confidence_proxy(series: pd.Series) -> pd.Series:
    clipped = series.clip(lower=0.0, upper=6.0)
    return clipped / 6.0


def _margin_bucket_table(df: pd.DataFrame, bins: list[float]) -> pd.DataFrame:
    work = df.copy()
    work["margin_bucket"] = bucketize_margin(work["margin"], bins=bins)
    rows = []
    for bucket, group in work.groupby("margin_bucket", observed=False):
        if len(group) == 0:
            continue
        rows.append({
            "margin_bucket": str(bucket),
            "count": int(len(group)),
            "avg_margin": float(group["margin"].mean()) if group["margin"].notna().any() else np.nan,
            "accuracy": float(group["is_correct"].mean()),
            "fn_risk_predicted_benign": float(
                ((group["predicted_label"] == "benign") & (group["true_label"] == "adversarial")).sum()
                / max((group["predicted_label"] == "benign").sum(), 1)
            ),
        })
    return pd.DataFrame(rows)


def _safe_auc(y_true: pd.Series, scores: pd.Series) -> float:
    frame = pd.DataFrame({"y": y_true.astype(int), "s": scores.astype(float)}).dropna()
    if frame.empty or frame["y"].nunique() < 2:
        return float("nan")
    return float(roc_auc_score(frame["y"], frame["s"]))


def _save_histogram(df: pd.DataFrame, column: str, by: str, output_path: Path, title: str) -> None:
    plt.figure(figsize=(8, 5))
    for key, group in df.groupby(by):
        vals = group[column].dropna()
        if vals.empty:
            continue
        plt.hist(vals, bins=30, alpha=0.45, label=str(key), density=True)
    plt.xlabel(column)
    plt.ylabel("Density")
    plt.title(title)
    plt.legend()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def _save_risk_plot(risk_df: pd.DataFrame, output_path: Path) -> None:
    plt.figure(figsize=(8, 4.5))
    plt.plot(risk_df["margin_mid"], risk_df["risk"], marker="o")
    plt.xlabel("Margin bucket midpoint")
    plt.ylabel("P(true_adversarial | predicted_benign, bucket)")
    plt.title("False-negative risk among predicted benign by margin")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def _save_fpr_fnr_plot(sweep_df: pd.DataFrame, output_path: Path) -> None:
    plt.figure(figsize=(8, 4.5))
    plt.plot(sweep_df["threshold"], sweep_df["fpr"], marker="o", label="FPR")
    plt.plot(sweep_df["threshold"], sweep_df["fnr"], marker="s", label="FNR")
    plt.xlabel("Margin threshold")
    plt.ylabel("Rate")
    plt.title("FPR and FNR vs Margin Threshold")
    plt.ylim(0.0, 1.0)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def _build_risk_table(df: pd.DataFrame, bins: list[float]) -> pd.DataFrame:
    sub = df[df["predicted_label"] == "benign"].copy()
    sub["margin_bucket"] = bucketize_margin(sub["margin"], bins=bins)
    rows = []
    for bucket, group in sub.groupby("margin_bucket", observed=False):
        if len(group) == 0:
            continue
        risk = float((group["true_label"] == "adversarial").mean())
        interval = bucket
        mid = (interval.left + interval.right) / 2 if hasattr(interval, "left") else np.nan
        rows.append({
            "margin_bucket": str(bucket),
            "margin_mid": mid,
            "count": int(len(group)),
            "risk": risk,
        })
    return pd.DataFrame(rows)


def _policy_sweep(
    trace_df: pd.DataFrame,
    thresholds: list[float],
    *,
    subset_name: str,
    subset_mask: pd.Series | None = None,
    production_prior: float = 0.1,
) -> pd.DataFrame:
    rows = []
    for threshold in thresholds:
        preds = apply_threshold_override(trace_df, threshold=threshold, subset=subset_mask)
        metrics = compute_binary_metrics_from_predictions(trace_df["true_label"], preds)
        rows.append({
            "subset": subset_name,
            "threshold": threshold,
            **metrics,
            "expected_accuracy_observed_prior": metrics["accuracy"],
            "expected_accuracy_balanced_prior": expected_accuracy_from_rates(
                tpr=metrics["tpr"], fpr=metrics["fpr"], adversarial_prior=0.5
            ),
            "expected_accuracy_prod_prior": expected_accuracy_from_rates(
                tpr=metrics["tpr"], fpr=metrics["fpr"], adversarial_prior=production_prior
            ),
        })
    return pd.DataFrame(rows)


def build_report(trace_df: pd.DataFrame, cfg: dict, split: str, output_dir: Path) -> tuple[str, dict[str, pd.DataFrame]]:
    bins = [-0.01, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.01]
    production_prior = float(cfg["hybrid"].get("production_adversarial_prior", 0.1))
    trace_df = trace_df.copy()
    trace_df["confidence_proxy"] = _margin_confidence_proxy(trace_df["margin"])

    reliability_all_ece, reliability_all = _ece_and_buckets(trace_df["is_correct"], trace_df["confidence_proxy"])
    pred_benign = trace_df["predicted_label"] == "benign"
    classifier_only = trace_df["route_bucket"] == "classifier_only"
    reliability_pred_benign_ece, reliability_pred_benign = _ece_and_buckets(
        trace_df.loc[pred_benign, "is_correct"],
        trace_df.loc[pred_benign, "confidence_proxy"],
    )
    reliability_classifier_only_ece, reliability_classifier_only = _ece_and_buckets(
        trace_df.loc[classifier_only, "is_correct"],
        trace_df.loc[classifier_only, "confidence_proxy"],
    )

    all_auc = _safe_auc(trace_df["is_correct"], trace_df["margin"])
    pred_benign_auc = _safe_auc(trace_df.loc[pred_benign, "is_correct"], trace_df.loc[pred_benign, "margin"])
    classifier_only_auc = _safe_auc(trace_df.loc[classifier_only, "is_correct"], trace_df.loc[classifier_only, "margin"])

    margin_bucket_all = _margin_bucket_table(trace_df, bins)
    risk_table = _build_risk_table(trace_df, bins)

    thresholds = [round(x * 0.5, 2) for x in range(13)]
    sweep_all = _policy_sweep(
        trace_df,
        thresholds,
        subset_name="all",
        production_prior=production_prior,
    )
    sweep_classifier_only = _policy_sweep(
        trace_df,
        thresholds,
        subset_name="classifier_only",
        subset_mask=classifier_only,
        production_prior=production_prior,
    )

    self_reported_auc = _safe_auc(trace_df["is_correct"], trace_df["self_reported_confidence"])
    recommendations = [
        f"1. Raw margin is {'useful' if np.isfinite(all_auc) and all_auc > 0.55 else 'weak'} as a correctness/risk signal on this trace (AUC={all_auc:.3f} when defined).",
        f"2. Self-reported confidence is {'weak' if not np.isfinite(self_reported_auc) or self_reported_auc < all_auc else 'non-trivially useful'} relative to margin (AUC={self_reported_auc:.3f} when defined).",
        "3. The current hard-flip policy should not be treated as production-grade calibration when tuned on the same trace rows; use it only as a baseline comparator.",
        (
            "4. Threshold ranges that look most defensible on this trace are the thresholds near the top balanced-accuracy / Youden-J rows in the sweep tables; "
            "prefer separate interpretation for observed skew and balanced prior."
        ),
        "5. Escalation-style routing is likely safer than direct adversarial override when the deployment cost of false positives is material.",
    ]

    lines = [
        "# Margin Calibration Report",
        "",
        f"- split: `{split}`",
        f"- rows: {len(trace_df)}",
        f"- production adversarial prior: {production_prior:.2f}",
        "",
        "## Margin Distribution",
        f"- overall usable margins: {int(trace_df['margin'].notna().sum())}/{len(trace_df)}",
        "",
        "## Margin As Correctness Signal",
        f"- ECE all: {reliability_all_ece:.4f}",
        f"- ECE predicted benign: {reliability_pred_benign_ece:.4f}",
        f"- ECE classifier-only: {reliability_classifier_only_ece:.4f}",
        f"- AUC correctness from margin (all): {all_auc:.4f}",
        f"- AUC correctness from margin (predicted benign): {pred_benign_auc:.4f}",
        f"- AUC correctness from margin (classifier-only): {classifier_only_auc:.4f}",
        "",
        "## False-Negative Risk Among Predicted Benign",
        "",
        risk_table.to_markdown(index=False) if not risk_table.empty else "No predicted-benign rows available.",
        "",
        "## Threshold Sweep",
        "",
        "### All Rows",
        sweep_all.to_markdown(index=False),
        "",
        "### Classifier-Only Subset",
        sweep_classifier_only.to_markdown(index=False),
        "",
        "## Reliability Tables",
        "",
        "### All Rows",
        reliability_all.to_markdown(index=False),
        "",
        "### Predicted Benign",
        reliability_pred_benign.to_markdown(index=False),
        "",
        "### Classifier-Only",
        reliability_classifier_only.to_markdown(index=False),
        "",
        "## Recommendations",
        "",
        *recommendations,
        "",
        "Warning: this report is exploratory when built from a single saved trace without a fresh validation split.",
        "",
    ]
    return "\n".join(lines), {
        "reliability_all": reliability_all,
        "reliability_pred_benign": reliability_pred_benign,
        "reliability_classifier_only": reliability_classifier_only,
        "margin_bucket_all": margin_bucket_all,
        "risk_table": risk_table,
        "sweep_all": sweep_all,
        "sweep_classifier_only": sweep_classifier_only,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a calibration-focused margin report.")
    parser.add_argument("--split", default="test")
    parser.add_argument("--trace", default=None)
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    trace_path = Path(args.trace) if args.trace else (RESEARCH_DIR / f"hybrid_margin_trace_{args.split}.parquet")
    trace_df = pd.read_parquet(trace_path)

    output_dir = REPORTS_ARTIFACTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    report_text, tables = build_report(trace_df, cfg, args.split, output_dir)

    report_path = output_dir / f"margin_calibration_{args.split}.md"
    report_path.write_text(report_text)

    for name, table in tables.items():
        table.to_csv(output_dir / f"{name}_{args.split}.csv", index=False)

    _save_histogram(trace_df, "margin", "predicted_label", output_dir / f"margin_by_predicted_label_{args.split}.png", "Margin by predicted label")
    _save_histogram(trace_df, "margin", "true_label", output_dir / f"margin_by_true_label_{args.split}.png", "Margin by true label")
    _save_histogram(trace_df, "margin", "route_bucket", output_dir / f"margin_by_route_{args.split}.png", "Margin by route bucket")
    _save_fpr_fnr_plot(tables["sweep_all"], output_dir / f"margin_threshold_fpr_fnr_{args.split}.png")
    if not tables["risk_table"].empty:
        _save_risk_plot(tables["risk_table"], output_dir / f"predicted_benign_risk_{args.split}.png")

    print(f"Margin calibration report saved -> {report_path}")


if __name__ == "__main__":
    main()

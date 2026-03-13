"""Threshold-based metrics and tuning helpers for external baselines."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


def _series(values) -> pd.Series:
    return pd.Series(values).reset_index(drop=True)


def evaluate_at_threshold(y_true, scores, threshold: float) -> dict:
    """Compute binary metrics for a fixed adversarial score threshold."""
    y_true = _series(y_true).astype(str)
    scores = _series(scores).astype(float)
    y_pred = pd.Series(
        np.where(scores >= float(threshold), "adversarial", "benign"),
        index=y_true.index,
    )

    adv_mask = y_true == "adversarial"
    ben_mask = y_true == "benign"

    tp = int((adv_mask & (y_pred == "adversarial")).sum())
    fn = int((adv_mask & (y_pred == "benign")).sum())
    fp = int((ben_mask & (y_pred == "adversarial")).sum())
    tn = int((ben_mask & (y_pred == "benign")).sum())

    support_adversarial = int(adv_mask.sum())
    support_benign = int(ben_mask.sum())
    total = len(y_true)

    def _safe_div(num: float, den: float) -> float:
        return float(num / den) if den else 0.0

    adv_precision = _safe_div(tp, tp + fp)
    adv_recall = _safe_div(tp, tp + fn)
    benign_precision = _safe_div(tn, tn + fn)
    benign_recall = _safe_div(tn, tn + fp)
    adv_f1 = _safe_div(2 * adv_precision * adv_recall, adv_precision + adv_recall)
    benign_f1 = _safe_div(2 * benign_precision * benign_recall, benign_precision + benign_recall)
    accuracy = _safe_div(tp + tn, total)
    false_positive_rate = _safe_div(fp, fp + tn)
    false_negative_rate = _safe_div(fn, fn + tp)

    binary_true = adv_mask.astype(int)
    try:
        auroc = float(roc_auc_score(binary_true, scores))
    except ValueError:
        auroc = float("nan")
    try:
        auprc = float(average_precision_score(binary_true, scores))
    except ValueError:
        auprc = float("nan")

    return {
        "threshold": float(threshold),
        "accuracy": accuracy,
        "adversarial_precision": adv_precision,
        "adversarial_recall": adv_recall,
        "adversarial_f1": adv_f1,
        "benign_precision": benign_precision,
        "benign_recall": benign_recall,
        "benign_f1": benign_f1,
        "false_positive_rate": false_positive_rate,
        "false_negative_rate": false_negative_rate,
        "auroc": auroc,
        "auprc": auprc,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "support_adversarial": support_adversarial,
        "support_benign": support_benign,
    }


def _candidate_thresholds(scores) -> list[float]:
    unique_scores = sorted(set(float(v) for v in _series(scores).astype(float).tolist()))
    if not unique_scores:
        return [0.5]
    low = float(np.nextafter(0.0, -1.0))
    high = float(np.nextafter(1.0, 2.0))
    return [low] + unique_scores + [high]


def tune_threshold_low_fnr(y_true, scores, max_fnr: float = 0.02) -> dict:
    """Pick the highest threshold with FNR <= max_fnr, else nearest violation."""
    evaluations = [evaluate_at_threshold(y_true, scores, threshold) for threshold in _candidate_thresholds(scores)]
    feasible = [row for row in evaluations if row["false_negative_rate"] <= max_fnr]
    if feasible:
        best = max(feasible, key=lambda row: row["threshold"])
        best = {**best, "constraint_met": True}
    else:
        best = min(
            evaluations,
            key=lambda row: (
                row["false_negative_rate"] - max_fnr,
                -row["threshold"],
            ),
        )
        best = {**best, "constraint_met": False}
    return {
        **best,
        "target_metric": "false_negative_rate",
        "target_value": float(max_fnr),
    }


def tune_threshold_bounded_fpr(y_true, scores, max_fpr: float = 0.05) -> dict:
    """Pick the lowest threshold with FPR <= max_fpr, else nearest violation."""
    evaluations = [evaluate_at_threshold(y_true, scores, threshold) for threshold in _candidate_thresholds(scores)]
    feasible = [row for row in evaluations if row["false_positive_rate"] <= max_fpr]
    if feasible:
        best = min(feasible, key=lambda row: row["threshold"])
        best = {**best, "constraint_met": True}
    else:
        best = min(
            evaluations,
            key=lambda row: (
                row["false_positive_rate"] - max_fpr,
                row["threshold"],
            ),
        )
        best = {**best, "constraint_met": False}
    return {
        **best,
        "target_metric": "false_positive_rate",
        "target_value": float(max_fpr),
    }

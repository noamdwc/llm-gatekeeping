"""Row-level margin trace utilities and threshold evaluation helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.logprob_margin import extract_preferred_margin_features_from_row, infer_route_bucket


TRACE_SCHEMA_COMMENT = """
Row-level hybrid margin trace schema:
- Identity/context: sample_id, dataset, split, provider, model_name
- Labels: true_label, predicted_label, final_label
- Routing/policy: route, route_bucket, policy_name, policy_outcome, override_applied, override_reason
- LLM parse/raw: raw_response_text, parse_success, self_reported_confidence
- Margin extraction: label_start_position, top1_logprob, top2_logprob, margin, top_k_tokens_raw
- Diagnostics: token_names_missing, token_strings_available, is_correct, is_fp, is_fn
"""


def build_margin_trace(
    research_df: pd.DataFrame,
    *,
    dataset: str,
    split: str,
) -> pd.DataFrame:
    """Build a reusable row-level margin trace from the research parquet."""
    rows: list[dict] = []
    for record in research_df.to_dict(orient="records"):
        margin = extract_preferred_margin_features_from_row(record)
        route_bucket = infer_route_bucket(record)
        predicted_label = record.get("llm_pred_binary")
        if predicted_label is None:
            predicted_label = record.get("hybrid_pred_binary") if route_bucket == "ml_fastpath" else record.get("ml_pred_binary")
        provider_name = (
            record.get(f"{margin.source_stage}_provider_name")
            if margin.source_stage in {"clf", "judge"}
            else None
        ) or record.get("llm_provider_name")
        model_name = (
            record.get(f"{margin.source_stage}_model_name")
            if margin.source_stage in {"clf", "judge"}
            else None
        ) or record.get("llm_model_name")
        raw_response_text = (
            record.get(f"{margin.source_stage}_raw_response_text")
            if margin.source_stage in {"clf", "judge"}
            else None
        ) or record.get("llm_raw_response_text")
        parse_success = (
            record.get(f"{margin.source_stage}_parse_success")
            if margin.source_stage in {"clf", "judge"}
            else None
        )
        if parse_success is None:
            parse_success = record.get("llm_parse_success")

        true_label = record.get("label_binary")
        final_label = record.get("hybrid_pred_binary")
        is_correct = final_label == true_label if true_label is not None and final_label is not None else False
        is_fp = true_label == "benign" and final_label == "adversarial"
        is_fn = true_label == "adversarial" and final_label == "benign"

        rows.append({
            "sample_id": record.get("sample_id"),
            "dataset": dataset,
            "split": split,
            "true_label": true_label,
            "predicted_label": predicted_label,
            "final_label": final_label,
            "route": record.get("hybrid_routed_to"),
            "route_bucket": route_bucket,
            "provider": provider_name,
            "model_name": model_name,
            "raw_response_text": raw_response_text,
            "parse_success": parse_success,
            "label_start_position": margin.label_start_position,
            "top1_logprob": margin.top1_logprob,
            "top2_logprob": margin.top2_logprob,
            "margin": margin.margin,
            "self_reported_confidence": record.get("llm_conf_binary"),
            "override_applied": bool(record.get("hybrid_override_applied", False)),
            "override_reason": record.get("hybrid_override_reason"),
            "policy_name": record.get("hybrid_margin_policy"),
            "policy_outcome": record.get("hybrid_policy_outcome"),
            "margin_source_stage": margin.source_stage,
            "top_k_tokens_raw": margin.top_k_tokens,
            "token_strings_available": margin.token_strings_available,
            "token_names_missing": margin.token_names_missing,
            "llm_stages_run": record.get("llm_stages_run"),
            "is_correct": is_correct,
            "is_fp": is_fp,
            "is_fn": is_fn,
        })

    return pd.DataFrame(rows)


def write_margin_trace(trace_df: pd.DataFrame, parquet_path: Path, csv_path: Path | None = None) -> None:
    """Persist the margin trace to parquet and optionally CSV."""
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    trace_df.to_parquet(parquet_path, index=False)
    if csv_path is not None:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        trace_df.to_csv(csv_path, index=False)


def compute_confusion_counts(y_true: pd.Series, y_pred: pd.Series) -> dict[str, int]:
    """Return confusion counts for the adversarial-positive binary task."""
    tp = int(((y_true == "adversarial") & (y_pred == "adversarial")).sum())
    tn = int(((y_true == "benign") & (y_pred == "benign")).sum())
    fp = int(((y_true == "benign") & (y_pred == "adversarial")).sum())
    fn = int(((y_true == "adversarial") & (y_pred == "benign")).sum())
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn}


def compute_binary_metrics_from_predictions(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    """Compute binary metrics from a concrete prediction vector."""
    counts = compute_confusion_counts(y_true, y_pred)
    tp = counts["tp"]
    tn = counts["tn"]
    fp = counts["fp"]
    fn = counts["fn"]
    total = max(len(y_true), 1)
    tpr = tp / max(tp + fn, 1)
    fpr = fp / max(fp + tn, 1)
    fnr = fn / max(tp + fn, 1)
    precision = tp / max(tp + fp, 1)
    accuracy = (tp + tn) / total
    balanced_accuracy = 0.5 * (tpr + (tn / max(tn + fp, 1)))
    youden_j = tpr - fpr
    return {
        "tpr": tpr,
        "fpr": fpr,
        "fnr": fnr,
        "precision": precision,
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "youden_j": youden_j,
        "fp_count": fp,
        "fn_count": fn,
    }


def apply_threshold_override(
    trace_df: pd.DataFrame,
    threshold: float,
    *,
    subset: pd.Series | None = None,
) -> pd.Series:
    """Apply the baseline hard-flip override to a final-label vector."""
    pred = trace_df["final_label"].astype(object).copy()
    mask = (
        (trace_df["predicted_label"] == "benign")
        & trace_df["margin"].notna()
        & (trace_df["margin"] < threshold)
    )
    if subset is not None:
        mask &= subset
    pred.loc[mask] = "adversarial"
    return pred


def expected_accuracy_from_rates(
    *,
    tpr: float,
    fpr: float,
    adversarial_prior: float,
) -> float:
    """Reweight accuracy under a different class prior using fixed TPR/FPR."""
    benign_prior = 1.0 - adversarial_prior
    return (adversarial_prior * tpr) + (benign_prior * (1.0 - fpr))


def bucketize_margin(values: pd.Series, bins: list[float] | np.ndarray) -> pd.Categorical:
    """Bucket margins with stable left-inclusive intervals."""
    return pd.cut(values, bins=bins, include_lowest=True, right=False)

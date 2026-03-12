"""
Research mode pipeline — reads pre-computed ML and (optionally) LLM prediction
parquets, computes hybrid routing, and produces a wide research parquet plus
evaluation reports.

In the DVC research pipeline:
  - ml_model stage produces: predictions/ml_predictions_{split}.parquet
  - llm_classifier stage produces: predictions/llm_predictions_{split}.parquet
  - This stage merges them + computes hybrid routing + generates reports.

Usage:
    python -m src.research --split test
"""

import argparse

import numpy as np
import pandas as pd

from src.logprob_margin import (
    apply_margin_policy,
    extract_preferred_margin_features_from_row,
    infer_route_bucket,
    resolve_margin_policy_config,
)
from src.margin_trace import build_margin_trace, write_margin_trace
from src.utils import (
    load_config,
    build_sample_id,
    SPLITS_DIR, PREDICTIONS_DIR, RESEARCH_DIR, REPORTS_DIR,
    REPORTS_RESEARCH_DIR, REPORTS_ARTIFACTS_DIR,
)
from src.evaluate import (
    binary_metrics, category_metrics, type_metrics,
    calibration_metrics, generate_report, compute_fpr_views,
)


ADVERSARIAL_LABEL_ALIASES = {
    "adversarial",
    "adversary",
    "adv",
    "attack",
    "attacker",
    "malicious",
    "jailbreak",
    "prompt_injection",
    "prompt injection",
    "injection",
}


def _normalize_binary_label(label) -> str:
    return str(label).strip().lower().replace("-", "_")


def _normalize_attack_token(value) -> str:
    return str(value).strip().lower().replace("-", "_").replace(" ", "_")


def _is_adversarial_label(label) -> bool:
    norm = _normalize_binary_label(label)
    return norm in ADVERSARIAL_LABEL_ALIASES or norm.startswith("adv")


def _format_llm_required_error(
    message: str,
    llm_required_path: str | None = None,
    llm_generation_hint: str | None = None,
) -> str:
    lines = [message]
    if llm_required_path:
        lines.append(f"Expected LLM predictions artifact: {llm_required_path}")
    if llm_generation_hint:
        lines.append(f"Generate it via: {llm_generation_hint}")
    return "\n".join(lines)


def _compute_unicode_lane_mask(
    df: pd.DataFrame,
    category_col: str = "ml_pred_category",
    type_col: str = "ml_pred_type",
    unicode_types: list[str] | set[str] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (unicode_lane_mask, lane_reliable_mask)."""
    n = len(df)
    if n == 0:
        return np.zeros(0, dtype=bool), np.zeros(0, dtype=bool)

    if category_col in df.columns:
        category = df[category_col]
        cat_has_value = category.notna().values
        cat_norm = category.astype(str).map(_normalize_attack_token)
        unicode_by_category = (cat_norm == "unicode_attack").values
    else:
        cat_has_value = np.zeros(n, dtype=bool)
        unicode_by_category = np.zeros(n, dtype=bool)
        cat_norm = pd.Series([""] * n)

    if type_col in df.columns:
        attack_type = df[type_col]
        type_has_value = attack_type.notna().values
        type_norm = attack_type.astype(str).map(_normalize_attack_token)
    else:
        type_has_value = np.zeros(n, dtype=bool)
        type_norm = pd.Series([""] * n)

    unicode_types_norm = {
        _normalize_attack_token(v)
        for v in (unicode_types or [])
        if v is not None and pd.notna(v)
    }
    if not unicode_types_norm and type_col in df.columns and category_col in df.columns:
        unicode_types_norm = {
            t for t, is_unicode_cat in zip(type_norm.values, unicode_by_category) if is_unicode_cat and t
        }

    unicode_by_type = type_norm.isin(unicode_types_norm).values if unicode_types_norm else np.zeros(n, dtype=bool)
    unicode_lane = unicode_by_category | unicode_by_type
    lane_reliable = cat_has_value | type_has_value
    return unicode_lane, lane_reliable


def compute_routing_diagnostics(
    df: pd.DataFrame,
    ml_pred_col: str = "ml_pred_binary",
    route_col: str = "hybrid_routed_to",
    ml_category_col: str = "ml_pred_category",
    ml_type_col: str = "ml_pred_type",
    unicode_types: list[str] | set[str] | None = None,
) -> dict:
    """Compute additive routing diagnostics for hybrid reports."""
    total = int(len(df))
    routed_ml = int((df[route_col] == "ml").sum()) if total else 0
    routed_llm = int((df[route_col] == "llm").sum()) if total else 0
    routed_abstain = int((df[route_col] == "abstain").sum()) if total else 0

    ml_is_adv = df[ml_pred_col].map(_is_adversarial_label) if total else pd.Series(dtype=bool)
    ben_mask = ~ml_is_adv if total else pd.Series(dtype=bool)
    adv_mask = ml_is_adv if total else pd.Series(dtype=bool)
    esc_mask = (df[route_col] == "llm") | (df[route_col] == "abstain") if total else pd.Series(dtype=bool)

    ben_total = int(ben_mask.sum()) if total else 0
    ben_to_ml = int(((df[route_col] == "ml") & ben_mask).sum()) if total else 0
    ben_to_llm = int(((df[route_col] == "llm") & ben_mask).sum()) if total else 0
    ben_to_abstain = int(((df[route_col] == "abstain") & ben_mask).sum()) if total else 0

    adv_total = int(adv_mask.sum()) if total else 0
    adv_to_ml = int(((df[route_col] == "ml") & adv_mask).sum()) if total else 0
    adv_to_llm = int(((df[route_col] == "llm") & adv_mask).sum()) if total else 0
    adv_to_abstain = int(((df[route_col] == "abstain") & adv_mask).sum()) if total else 0

    unicode_lane, lane_reliable = _compute_unicode_lane_mask(
        df,
        category_col=ml_category_col,
        type_col=ml_type_col,
        unicode_types=unicode_types,
    )
    unicode_true_total = int(unicode_lane.sum()) if total else 0
    unicode_false_total = int((~unicode_lane).sum()) if total else 0
    unknown_lane_total = int((~lane_reliable).sum()) if total else 0
    unicode_true_fastpath_ml = int(((df[route_col] == "ml") & unicode_lane).sum()) if total else 0
    unicode_true_escalated = int((esc_mask & unicode_lane).sum()) if total else 0
    unicode_false_fastpath_ml = int(((df[route_col] == "ml") & ~unicode_lane).sum()) if total else 0
    unicode_false_escalated = int((esc_mask & ~unicode_lane).sum()) if total else 0

    return {
        "total_samples": total,
        "routed_ml": routed_ml,
        "routed_llm": routed_llm,
        "routed_abstain": routed_abstain,
        "routed_ml_rate": (routed_ml / total) if total else 0.0,
        "routed_llm_rate": (routed_llm / total) if total else 0.0,
        "routed_abstain_rate": (routed_abstain / total) if total else 0.0,
        "ml_pred_benign_total": ben_total,
        "ml_pred_benign_routed_ml": ben_to_ml,
        "ml_pred_benign_routed_llm": ben_to_llm,
        "ml_pred_benign_routed_abstain": ben_to_abstain,
        "ml_pred_benign_escalation_rate": ((ben_to_llm + ben_to_abstain) / ben_total) if ben_total else 0.0,
        "ml_pred_adversarial_total": adv_total,
        "ml_pred_adversarial_routed_ml": adv_to_ml,
        "ml_pred_adversarial_routed_llm": adv_to_llm,
        "ml_pred_adversarial_routed_abstain": adv_to_abstain,
        "ml_pred_adversarial_escalation_rate": ((adv_to_llm + adv_to_abstain) / adv_total) if adv_total else 0.0,
        "unicode_lane_true_total": unicode_true_total,
        "unicode_lane_false_total": unicode_false_total,
        "unicode_lane_unknown_total": unknown_lane_total,
        "unicode_lane_true_fastpath_ml": unicode_true_fastpath_ml,
        "unicode_lane_true_escalated": unicode_true_escalated,
        "unicode_lane_false_fastpath_ml": unicode_false_fastpath_ml,
        "unicode_lane_false_escalated": unicode_false_escalated,
    }


def render_routing_diagnostics_markdown(diag: dict) -> str:
    """Render routing diagnostics as an additive markdown section."""
    lines = [
        "## Routing Diagnostics",
        "",
        f"- total_samples: {diag['total_samples']}",
        f"- routed_ml: {diag['routed_ml']} ({diag['routed_ml_rate']:.4f})",
        f"- routed_llm: {diag['routed_llm']} ({diag['routed_llm_rate']:.4f})",
        f"- routed_abstain: {diag['routed_abstain']} ({diag['routed_abstain_rate']:.4f})",
        f"- unicode_lane_unknown_total: {diag['unicode_lane_unknown_total']}",
        "",
        "| ml_pred_label | routed_ml | routed_llm | routed_abstain | escalation_rate |",
        "|---------------|-----------|------------|----------------|-----------------|",
        (
            f"| benign | {diag['ml_pred_benign_routed_ml']} | "
            f"{diag['ml_pred_benign_routed_llm']} | "
            f"{diag['ml_pred_benign_routed_abstain']} | "
            f"{diag['ml_pred_benign_escalation_rate']:.4f} |"
        ),
        (
            f"| adversarial | {diag['ml_pred_adversarial_routed_ml']} | "
            f"{diag['ml_pred_adversarial_routed_llm']} | "
            f"{diag['ml_pred_adversarial_routed_abstain']} | "
            f"{diag['ml_pred_adversarial_escalation_rate']:.4f} |"
        ),
        "",
        "| unicode_lane | total | fastpath_ml | escalated_llm_or_abstain |",
        "|--------------|-------|-------------|---------------------------|",
        (
            f"| True | {diag['unicode_lane_true_total']} | "
            f"{diag['unicode_lane_true_fastpath_ml']} | "
            f"{diag['unicode_lane_true_escalated']} |"
        ),
        (
            f"| False | {diag['unicode_lane_false_total']} | "
            f"{diag['unicode_lane_false_fastpath_ml']} | "
            f"{diag['unicode_lane_false_escalated']} |"
        ),
        "",
    ]
    return "\n".join(lines)

def compute_hybrid_routing(
    ml_df: pd.DataFrame,
    llm_df: pd.DataFrame | None,
    threshold: float,
    llm_conf_threshold: float = 0.7,
    logprob_margin_threshold: float | None = None,
    margin_policy_cfg: dict | None = None,
    unicode_types: list[str] | set[str] | None = None,
    require_llm_for_escalations: bool = False,
    llm_required_path: str | None = None,
    llm_generation_hint: str | None = None,
) -> pd.DataFrame:
    """Compute hybrid routing decisions from ML prediction + confidence threshold.

    If LLM results are available, escalated samples use LLM predictions.
    Otherwise, escalated samples fall back to ML predictions.
    Rows are matched between ml_df and llm_df via the ``sample_id`` column.

    Specialist policy:
      - ML benign (or non-adversarial) predictions always escalate to LLM.
      - ML adversarial predictions route to ML only when confidence >= threshold
        and the prediction is in unicode lane.
      - If escalated LLM confidence < llm_conf_threshold, route to abstain and
        force binary output to adversarial.

    Returns DataFrame with: sample_id, hybrid_routed_to, hybrid_pred_{binary,category,type}
    """
    conf_col = "ml_conf_binary_cal" if "ml_conf_binary_cal" in ml_df.columns else "ml_conf_binary"
    print(f"  [research routing] confidence source={conf_col}")
    ml_conf = ml_df[conf_col].values
    ml_pred_binary = ml_df["ml_pred_binary"].values

    ml_adv_mask = np.array([_is_adversarial_label(v) for v in ml_pred_binary], dtype=bool)
    unicode_lane, lane_reliable = _compute_unicode_lane_mask(ml_df, unicode_types=unicode_types)
    confident = ml_adv_mask & unicode_lane & (ml_conf >= threshold)
    n_ml_fastpath = int(confident.sum())
    n_llm_candidates = int((~confident).sum())
    print(
        "  [research routing] "
        f"threshold={threshold} | llm_conf_threshold={llm_conf_threshold} | "
        f"ml_confident_unicode_adv_fastpath={n_ml_fastpath} | "
        f"llm_escalation_candidates={n_llm_candidates}"
    )
    print(
        "  [research routing] lane coverage | "
        f"unicode_lane_true={int(unicode_lane.sum())} | "
        f"unicode_lane_false={int((~unicode_lane).sum())} | "
        f"unicode_lane_unknown={int((~lane_reliable).sum())}"
    )

    if require_llm_for_escalations and llm_df is None:
        raise RuntimeError(
            _format_llm_required_error(
                "Hybrid routing requires LLM predictions but llm_df is missing.",
                llm_required_path=llm_required_path,
                llm_generation_hint=llm_generation_hint,
            )
        )
    if require_llm_for_escalations and llm_df is not None and llm_df.empty:
        raise RuntimeError(
            _format_llm_required_error(
                "Hybrid routing requires non-empty LLM predictions but llm_df is empty.",
                llm_required_path=llm_required_path,
                llm_generation_hint=llm_generation_hint,
            )
        )

    # Start with ML predictions for every row (default / fallback)
    result = pd.DataFrame({
        "sample_id": ml_df["sample_id"].values,
        "hybrid_routed_to": np.where(confident, "ml", "llm"),
        "hybrid_pred_binary": ml_df["ml_pred_binary"].values,
        "hybrid_pred_category": ml_df["ml_pred_category"].values,
        "hybrid_pred_type": ml_df["ml_pred_type"].values,
        "hybrid_llm_pred_binary_pre_policy": pd.Series([None] * len(ml_df), dtype=object),
        "hybrid_margin": pd.Series([None] * len(ml_df), dtype=object),
        "hybrid_margin_source_stage": pd.Series([None] * len(ml_df), dtype=object),
        "hybrid_label_start_position": pd.Series([None] * len(ml_df), dtype=object),
        "hybrid_top1_logprob": pd.Series([None] * len(ml_df), dtype=object),
        "hybrid_top2_logprob": pd.Series([None] * len(ml_df), dtype=object),
        "hybrid_top_logprobs_raw": pd.Series([None] * len(ml_df), dtype=object),
        "hybrid_token_names_missing": pd.Series([None] * len(ml_df), dtype=object),
        "hybrid_token_strings_available": pd.Series([None] * len(ml_df), dtype=object),
        "hybrid_margin_policy": pd.Series([None] * len(ml_df), dtype=object),
        "hybrid_policy_outcome": pd.Series([None] * len(ml_df), dtype=object),
        "hybrid_override_applied": False,
        "hybrid_override_reason": pd.Series([None] * len(ml_df), dtype=object),
        "hybrid_route_bucket": pd.Series([None] * len(ml_df), dtype=object),
    })
    effective_policy_cfg = margin_policy_cfg or {
        "policy": "baseline",
        "threshold": logprob_margin_threshold,
        "low_threshold": logprob_margin_threshold,
        "high_threshold": logprob_margin_threshold,
        "classifier_only_threshold": logprob_margin_threshold,
        "judge_threshold": logprob_margin_threshold,
    }

    if llm_df is not None:
        escalated = ~confident
        llm_indexed = llm_df.set_index("sample_id")

        # Among escalated rows, find which have matching LLM predictions
        esc_ids = result.loc[escalated, "sample_id"]
        has_llm = esc_ids.isin(llm_indexed.index)

        # Override escalated rows that have LLM predictions
        override_idx = has_llm[has_llm].index
        if len(override_idx) > 0:
            matched_ids = result.loc[override_idx, "sample_id"]
            llm_rows = llm_indexed.loc[matched_ids]
            result.loc[override_idx, "hybrid_llm_pred_binary_pre_policy"] = llm_rows["llm_pred_binary"].values
            result.loc[override_idx, "hybrid_pred_binary"] = llm_rows["llm_pred_binary"].values
            # Override category if LLM provides it (llm_pred_category exists when LLM derives category)
            if "llm_pred_category" in llm_indexed.columns:
                result.loc[override_idx, "hybrid_pred_category"] = llm_rows["llm_pred_category"].values
            if "llm_conf_binary" in llm_indexed.columns:
                llm_conf = llm_rows["llm_conf_binary"].fillna(0.5).astype(float).values
                abstain_mask = llm_conf < llm_conf_threshold
                abstain_idx = override_idx[abstain_mask]
                if len(abstain_idx) > 0:
                    result.loc[abstain_idx, "hybrid_routed_to"] = "abstain"
                    result.loc[abstain_idx, "hybrid_pred_binary"] = "adversarial"
            # LLM does not provide type-level predictions; hybrid_pred_type stays as ML's prediction

            for idx in override_idx:
                sample_id = result.at[idx, "sample_id"]
                llm_row = llm_indexed.loc[sample_id]
                margin = extract_preferred_margin_features_from_row(llm_row)
                route_bucket = infer_route_bucket({
                    "hybrid_routed_to": result.at[idx, "hybrid_routed_to"],
                    "llm_stages_run": llm_row.get("llm_stages_run"),
                })
                policy_result = apply_margin_policy(
                    current_route=result.at[idx, "hybrid_routed_to"],
                    predicted_binary=result.at[idx, "hybrid_pred_binary"],
                    predicted_label=(
                        llm_row.get("llm_pred_raw")
                        if llm_row.get("llm_pred_raw") in ("benign", "adversarial", "uncertain")
                        else result.at[idx, "hybrid_pred_binary"]
                    ),
                    margin=margin.margin,
                    policy_cfg=effective_policy_cfg,
                    route_bucket=route_bucket,
                )
                result.at[idx, "hybrid_margin"] = margin.margin
                result.at[idx, "hybrid_margin_source_stage"] = margin.source_stage
                result.at[idx, "hybrid_label_start_position"] = margin.label_start_position
                result.at[idx, "hybrid_top1_logprob"] = margin.top1_logprob
                result.at[idx, "hybrid_top2_logprob"] = margin.top2_logprob
                result.at[idx, "hybrid_top_logprobs_raw"] = margin.top_k_tokens
                result.at[idx, "hybrid_token_names_missing"] = margin.token_names_missing
                result.at[idx, "hybrid_token_strings_available"] = margin.token_strings_available
                result.at[idx, "hybrid_margin_policy"] = policy_result["policy_name"]
                result.at[idx, "hybrid_policy_outcome"] = policy_result["policy_outcome"]
                result.at[idx, "hybrid_override_applied"] = bool(policy_result["override_applied"])
                result.at[idx, "hybrid_override_reason"] = policy_result["override_reason"]
                result.at[idx, "hybrid_route_bucket"] = route_bucket
                result.at[idx, "hybrid_routed_to"] = policy_result["route"]
                result.at[idx, "hybrid_pred_binary"] = policy_result["final_binary"]

            n_overrides = int(result.loc[override_idx, "hybrid_override_applied"].sum())
            n_candidates = int(len(override_idx))
            if n_candidates > 0 and effective_policy_cfg.get("threshold") is not None:
                print(
                    "  [research routing] margin policy | "
                    f"policy={effective_policy_cfg['policy']} | "
                    f"threshold={effective_policy_cfg.get('threshold')} | "
                    f"candidates={n_candidates} | "
                    f"overrides={n_overrides}"
                )

        # Escalated rows without LLM fall back to ML (predictions already set);
        # correct routing label from "llm" → "ml"
        no_llm_idx = has_llm[~has_llm].index
        if require_llm_for_escalations and len(no_llm_idx) > 0:
            raise RuntimeError(
                _format_llm_required_error(
                    (
                        "Hybrid routing requires LLM coverage for all escalated samples, "
                        f"but {len(no_llm_idx)} escalated sample(s) are missing from llm_df."
                    ),
                    llm_required_path=llm_required_path,
                    llm_generation_hint=llm_generation_hint,
                )
            )
        result.loc[no_llm_idx, "hybrid_routed_to"] = "ml"
    else:
        if require_llm_for_escalations:
            raise RuntimeError(
                _format_llm_required_error(
                    "Hybrid routing requires LLM predictions but llm_df is missing.",
                    llm_required_path=llm_required_path,
                    llm_generation_hint=llm_generation_hint,
                )
            )
        # No LLM at all — everything routes to ML
        result["hybrid_routed_to"] = "ml"
        print("  [research routing] llm_df missing -> ML fallback for all rows")

    result.loc[result["hybrid_routed_to"] == "ml", "hybrid_route_bucket"] = "ml_fastpath"

    route_diag = compute_routing_diagnostics(
        result.assign(
            ml_pred_binary=ml_df["ml_pred_binary"].values,
            ml_pred_category=(
                ml_df["ml_pred_category"].values if "ml_pred_category" in ml_df.columns else None
            ),
            ml_pred_type=(
                ml_df["ml_pred_type"].values if "ml_pred_type" in ml_df.columns else None
            ),
        ),
        unicode_types=unicode_types,
    )
    print(f"  [research routing] final routing counts={result['hybrid_routed_to'].value_counts().to_dict()}")
    print(
        "  [research routing] diagnostics | "
        f"ml_pred_benign_to_llm_rate={route_diag['ml_pred_benign_escalation_rate']:.4f} | "
        f"ml_pred_adv_to_llm_rate={route_diag['ml_pred_adversarial_escalation_rate']:.4f} | "
        f"routed_abstain={route_diag['routed_abstain']}"
    )

    return result


def build_research_dataframe(
    ml_df: pd.DataFrame,
    hybrid_df: pd.DataFrame,
    llm_df: pd.DataFrame | None = None,
    split_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Merge ML predictions, LLM predictions, and hybrid results into one wide DataFrame.

    All DataFrames are joined on ``sample_id`` so row order doesn't matter.
    The ML predictions parquet already contains ground-truth columns, so we
    don't need the original split parquet.

    If ``split_df`` is provided, ``synth_*`` metadata columns are joined from it.
    """
    result = ml_df.merge(hybrid_df, on="sample_id", validate="one_to_one")
    if llm_df is not None:
        gt_cols = {
            "sample_id",
            "modified_sample",
            "original_sample",
            "attack_name",
            "label_binary",
            "label_category",
            "label_type",
            "prompt_hash",
        }
        llm_cols = ["sample_id"] + [c for c in llm_df.columns if c not in gt_cols and c != "sample_id"]
        result = result.merge(llm_df[llm_cols], on="sample_id", how="left", validate="one_to_one")
    # Preserve synth_* metadata from split parquet
    if split_df is not None:
        synth_cols = [c for c in split_df.columns if c.startswith("synth_")]
        if synth_cols:
            split_synth = split_df[["modified_sample"] + synth_cols].copy()
            split_synth["sample_id"] = split_synth["modified_sample"].apply(build_sample_id)
            result = result.merge(
                split_synth[["sample_id"] + synth_cols],
                on="sample_id", how="left", validate="one_to_one",
            )
    return result


def generate_ml_report(research_df: pd.DataFrame, output_path: str):
    """Generate ML-only evaluation report — evaluated on ML domain only (benign + unicode)."""
    # ML is a unicode specialist; NLP attacks are intentionally deferred to LLM
    df = research_df[research_df["label_category"] != "nlp_attack"].copy()
    n_excluded = len(research_df) - len(df)

    binary = binary_metrics(df["label_binary"], df["ml_pred_binary"])
    cat = category_metrics(df["label_category"], df["ml_pred_category"])
    types = type_metrics(df["label_type"], df["ml_pred_type"])
    cal = calibration_metrics(
        df["label_binary"], df["ml_pred_binary"],
        df["ml_conf_binary"],
    )
    usage = {
        "eval_scope": "benign_plus_unicode_only",
        "nlp_rows_excluded": n_excluded,
    }
    report = generate_report(df, binary, cat, types, cal, usage=usage,
                             title="ML Classifier Evaluation Report")
    with open(output_path, "w") as f:
        f.write(report)
    print(f"  ML report saved → {output_path}")
    return binary


def generate_hybrid_report(
    research_df: pd.DataFrame,
    output_path: str,
    unicode_types: list[str] | set[str] | None = None,
):
    """Generate hybrid evaluation report from the research DataFrame."""
    binary = binary_metrics(research_df["label_binary"], research_df["hybrid_pred_binary"])
    cat = category_metrics(research_df["label_category"], research_df["hybrid_pred_category"])
    types = type_metrics(research_df["label_type"], research_df["hybrid_pred_type"])

    # Use ML confidence as proxy for hybrid confidence
    cal = calibration_metrics(
        research_df["label_binary"], research_df["hybrid_pred_binary"],
        research_df["ml_conf_binary"],
    )

    # Add routing stats as usage info
    routing_diag = compute_routing_diagnostics(research_df, unicode_types=unicode_types)
    usage = {
        "routed_ml": routing_diag["routed_ml"],
        "routed_llm": routing_diag["routed_llm"],
        "routed_abstain": routing_diag["routed_abstain"],
        "ml_pred_benign_routed_ml": routing_diag["ml_pred_benign_routed_ml"],
        "ml_pred_benign_routed_llm": routing_diag["ml_pred_benign_routed_llm"],
        "ml_pred_benign_routed_abstain": routing_diag["ml_pred_benign_routed_abstain"],
        "ml_pred_adversarial_routed_ml": routing_diag["ml_pred_adversarial_routed_ml"],
        "ml_pred_adversarial_routed_llm": routing_diag["ml_pred_adversarial_routed_llm"],
        "ml_pred_adversarial_routed_abstain": routing_diag["ml_pred_adversarial_routed_abstain"],
    }

    report = generate_report(
        research_df,
        binary,
        cat,
        types,
        cal,
        usage,
        title="Hybrid Router Evaluation Report (Strict LLM Coverage)",
    )
    report = f"{report}\n{render_routing_diagnostics_markdown(routing_diag)}"

    is_clean = research_df.get("synth_validated")
    is_clean_benign = (
        is_clean.fillna(False).astype(bool)
        if is_clean is not None else None
    )
    fpr_views = compute_fpr_views(
        research_df["label_binary"],
        research_df["hybrid_pred_binary"],
        routed_to=research_df.get("hybrid_routed_to"),
        is_clean_benign=is_clean_benign,
    )
    fpr_rows = [
        "## FPR Diagnostic Views",
        "",
        "| View | FPR | Notes |",
        "|------|-----|-------|",
        f"| Standard | {fpr_views['fpr_standard']:.4f} | All samples, abstain=adversarial |",
        f"| Abstain-excluded | {fpr_views['fpr_abstain_excluded']:.4f} | {fpr_views['n_abstain']} abstain samples removed |",
        f"| Abstain rate | {fpr_views['abstain_rate']:.4f} | {fpr_views['n_abstain']}/{fpr_views['n_total']} samples |",
    ]
    if fpr_views["fpr_clean_benign"] is not None:
        n_cb = fpr_views["n_clean_benign"]
        n_cb_abs = fpr_views["n_clean_benign_abstain"]
        fpr_rows.extend([
            f"| Clean-benign | {fpr_views['fpr_clean_benign']:.4f} | {n_cb} validated synthetic benigns only |",
            f"| Clean-benign + abstain-excluded | {fpr_views['fpr_clean_benign_abstain_excluded']:.4f} | Clean benigns, {n_cb_abs} abstain removed |",
            f"| Clean-benign abstain rate | {fpr_views['clean_benign_abstain_rate']:.4f} | {n_cb_abs}/{n_cb} clean benign samples abstained |",
        ])
    fpr_rows.append("")
    fpr_section = "\n".join(fpr_rows)
    report = f"{report}\n{fpr_section}"

    with open(output_path, "w") as f:
        f.write(report)
    print(f"  Hybrid report saved → {output_path}")
    return binary


def generate_llm_report(research_df: pd.DataFrame, output_path: str):
    """Generate LLM-only evaluation report from the research DataFrame."""
    # Only evaluate rows that have LLM predictions (left merge may leave NaN)
    df = research_df.dropna(subset=["llm_pred_binary"])
    if df.empty:
        print(f"  Skipping LLM report — no LLM predictions available (0/{len(research_df)} samples)")
        return None
    judge_decisions = df.get("judge_computed_decision") if "judge_computed_decision" in df.columns else None
    binary = binary_metrics(df["label_binary"], df["llm_pred_binary"], judge_decisions=judge_decisions)
    # Category metrics only if LLM provides category predictions
    if "llm_pred_category" in df.columns:
        cat = category_metrics(df["label_category"], df["llm_pred_category"])
        # LLM does not predict type-level labels; skip type metrics
        types = {"type_accuracy": 0.0, "type_f1_macro": 0.0}
    else:
        cat = {"category_accuracy": 0.0, "category_f1_macro": 0.0}
        types = {"type_accuracy": 0.0, "type_f1_macro": 0.0}
    cal = calibration_metrics(
        df["label_binary"], df["llm_pred_binary"],
        df["llm_conf_binary"],
    )
    report = generate_report(df, binary, cat, types, cal)
    with open(output_path, "w") as f:
        f.write(report)
    print(f"  LLM report saved → {output_path} ({len(df)}/{len(research_df)} samples with LLM predictions)")
    return binary


def write_margin_code_path_note(path, *, split: str) -> None:
    """Document the verified executed margin/routing code path for experiments."""
    lines = [
        "# Margin Calibration Code Path",
        "",
        f"- Split: `{split}`",
        "- Canonical experiment entrypoint: `python -m src.llm_classifier.llm_classifier --split test --research`",
        "- Canonical hybrid routing entrypoint: `python -m src.research --split test`",
        "- Canonical report entrypoint: `python -m src.cli.eval_new --split test`",
        "",
        "## Verified Executed Path",
        "",
        "1. `src.llm_classifier.llm_classifier.HierarchicalLLMClassifier._call_llm()` performs the API call and captures raw response text, parse status, and token logprobs.",
        "2. `src.llm_classifier.llm_classifier.HierarchicalLLMClassifier.predict()` normalizes classifier/judge output and emits persisted LLM prediction rows.",
        "3. `src.research.compute_hybrid_routing()` is the code path used by the current DVC research experiments. It applies ML fast-path routing, LLM abstain handling, and the configured margin policy.",
        "4. `src.logprob_margin.extract_preferred_margin_features_from_row()` selects the label-start token position and computes the preferred margin (judge first, classifier fallback).",
        "5. `src.margin_trace.build_margin_trace()` writes row-level margin traces for downstream calibration analysis.",
        "6. `src.cli.eval_new` consumes `data/processed/research/research_{split}.parquet` for markdown reports; notebook-only sweeps are no longer the primary analysis path.",
        "",
        "## Duplicate / Secondary Paths",
        "",
        "- `src.hybrid_router.HybridRouter` is still used for live CLI prediction and some external evaluation flows, but it is not the canonical code path for the current DVC threshold experiments.",
        "- Margin extraction and policy logic are shared through `src.logprob_margin` to avoid stale duplicate implementations.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Research stage: merge predictions, compute hybrid routing, generate reports"
    )
    parser.add_argument("--config", default=None)
    parser.add_argument("--split", default="test", help="Which split to run on")
    args = parser.parse_args()

    cfg = load_config(args.config)
    threshold = cfg["hybrid"]["ml_confidence_threshold"]
    llm_threshold = cfg["hybrid"]["llm_confidence_threshold"]
    logprob_margin_threshold = cfg["hybrid"].get("logprob_margin_threshold")
    margin_policy_cfg = resolve_margin_policy_config(cfg)
    unicode_types = cfg.get("labels", {}).get("unicode_attacks", [])

    # ── Read pre-computed ML predictions ─────────────────────────────────────
    ml_path = PREDICTIONS_DIR / f"ml_predictions_{args.split}.parquet"
    if not ml_path.exists():
        raise FileNotFoundError(
            f"ML predictions not found: {ml_path}\n"
            "Run the ml_model stage first (dvc repro ml_model)."
        )
    ml_df = pd.read_parquet(ml_path)
    print(f"Loaded ML predictions: {ml_path} ({len(ml_df)} samples)")

    # ── Read pre-computed LLM predictions (required for strict hybrid) ───────
    llm_path = PREDICTIONS_DIR / f"llm_predictions_{args.split}.parquet"
    llm_df = None
    if llm_path.exists():
        llm_df = pd.read_parquet(llm_path)
        print(f"Loaded LLM predictions: {llm_path} ({len(llm_df)} samples)")
    else:
        print(f"No LLM predictions found at {llm_path} — strict hybrid report cannot be generated")

    # ── Compute hybrid routing ───────────────────────────────────────────────
    print(f"Computing hybrid routing (threshold={threshold})...")
    hybrid_df = compute_hybrid_routing(
        ml_df,
        llm_df,
        threshold,
        llm_conf_threshold=llm_threshold,
        logprob_margin_threshold=logprob_margin_threshold,
        margin_policy_cfg=margin_policy_cfg,
        unicode_types=unicode_types,
        require_llm_for_escalations=True,
        llm_required_path=str(llm_path),
        llm_generation_hint=f"dvc repro llm_classifier research eval_new --force",
    )

    # ── Load split parquet for synth_* metadata ──────────────────────────────
    split_path = SPLITS_DIR / f"{args.split}.parquet"
    split_df = pd.read_parquet(split_path) if split_path.exists() else None

    # ── Build wide research DataFrame ────────────────────────────────────────
    research_df = build_research_dataframe(ml_df, hybrid_df, llm_df, split_df=split_df)

    # ── Save research parquet ────────────────────────────────────────────────
    RESEARCH_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESEARCH_DIR / f"research_{args.split}.parquet"
    research_df.to_parquet(out_path, index=False)
    print(f"\nResearch parquet saved → {out_path}")
    print(f"Shape: {research_df.shape}")
    print(f"Columns: {research_df.columns.tolist()}")

    trace_df = build_margin_trace(
        research_df,
        dataset=cfg["dataset"]["name"],
        split=args.split,
    )
    trace_parquet_path = RESEARCH_DIR / f"hybrid_margin_trace_{args.split}.parquet"
    trace_csv_path = RESEARCH_DIR / f"hybrid_margin_trace_{args.split}.csv"
    write_margin_trace(trace_df, trace_parquet_path, trace_csv_path)
    print(f"Margin trace saved → {trace_parquet_path}")

    code_path_note = REPORTS_DIR / "margin_code_path.md"
    write_margin_code_path_note(code_path_note, split=args.split)
    print(f"Margin code path note saved → {code_path_note}")

    # ── Generate evaluation reports ──────────────────────────────────────────
    REPORTS_RESEARCH_DIR.mkdir(parents=True, exist_ok=True)

    print("\nGenerating reports...")
    generate_ml_report(research_df, str(REPORTS_RESEARCH_DIR / "eval_report_ml.md"))
    generate_hybrid_report(
        research_df,
        str(REPORTS_RESEARCH_DIR / "eval_report_hybrid.md"),
        unicode_types=unicode_types,
    )
    if llm_df is not None:
        generate_llm_report(research_df, str(REPORTS_RESEARCH_DIR / "eval_report_llm.md"))

    # ── Quick sanity check ───────────────────────────────────────────────────
    binary_cols = [c for c in research_df.columns if c.startswith("ml_proba_binary_")]
    if binary_cols:
        sums = research_df[binary_cols].sum(axis=1)
        print(f"\nBinary proba sum: min={sums.min():.6f}, max={sums.max():.6f}")

    routing_counts = hybrid_df["hybrid_routed_to"].value_counts()
    print(f"\nRouting: {routing_counts.to_dict()}")


if __name__ == "__main__":
    main()

"""Shared logprob margin extraction and policy helpers."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from typing import Any


DEFAULT_THRESHOLD_GRID = [round(x * 0.5, 2) for x in range(13)]


@dataclass
class MarginFeatures:
    """Canonical row-level margin features for one chosen LLM stage."""

    source_stage: str | None = None
    label_start_position: int | None = None
    top1_logprob: float | None = None
    top2_logprob: float | None = None
    margin: float | None = None
    top_k_tokens: list[dict[str, Any]] | None = None
    token_names_missing: bool | None = None
    token_strings_available: bool | None = None
    semantic_margin_supported: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def safe_json_loads(value: Any) -> Any:
    """Parse JSON from various storage formats."""
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, (list, dict)):
        return value
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    if not stripped or stripped in {"null", "None", "nan", "NaN"}:
        return None
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        return None


def find_label_start_position(token_logprobs: list[dict] | None, mode: str = "clf") -> int | None:
    """Locate the label-start token position for classifier or judge JSON output."""
    if not token_logprobs or not isinstance(token_logprobs, list):
        return None

    if mode == "clf":
        return 4 if len(token_logprobs) > 4 else None

    for i, tok in enumerate(token_logprobs):
        if tok.get("token") == "_label":
            candidate = i + 3
            if candidate < len(token_logprobs):
                return candidate
            return None
    return None


def extract_margin_features(
    token_logprobs: list[dict] | None,
    mode: str,
    source_stage: str | None = None,
) -> MarginFeatures:
    """Extract top-k logprob margin features from one token-logprob payload.

    Future semantic margin support should hang off this layer once providers
    return stable token strings for the class tokens.
    """
    label_idx = find_label_start_position(token_logprobs, mode=mode)
    if label_idx is None or token_logprobs is None or label_idx >= len(token_logprobs):
        return MarginFeatures(source_stage=source_stage)

    token_payload = token_logprobs[label_idx] or {}
    top_items = token_payload.get("top_logprobs") or []
    top_values = [
        float(item["logprob"])
        for item in top_items
        if isinstance(item, dict) and isinstance(item.get("logprob"), (int, float))
    ]
    top1 = top_values[0] if len(top_values) >= 1 else None
    top2 = top_values[1] if len(top_values) >= 2 else None
    margin = (top1 - top2) if top1 is not None and top2 is not None else None

    token_names = [item.get("token") for item in top_items if isinstance(item, dict)]
    token_strings_available = any(isinstance(tok, str) and tok != "" for tok in token_names)
    token_names_missing = len(token_names) > 0 and not token_strings_available

    return MarginFeatures(
        source_stage=source_stage,
        label_start_position=label_idx,
        top1_logprob=top1,
        top2_logprob=top2,
        margin=margin,
        top_k_tokens=top_items or None,
        token_names_missing=token_names_missing,
        token_strings_available=token_strings_available,
    )


def extract_preferred_margin_features_from_row(row: dict | Any) -> MarginFeatures:
    """Prefer judge-stage margin when available, otherwise classifier-stage."""
    judge = extract_margin_features(
        safe_json_loads(row.get("judge_token_logprobs")),
        mode="judge",
        source_stage="judge",
    )
    if judge.margin is not None or judge.label_start_position is not None:
        return judge

    return extract_margin_features(
        safe_json_loads(row.get("clf_token_logprobs")),
        mode="clf",
        source_stage="clf",
    )


def extract_preferred_margin_features_from_result(result: dict[str, Any]) -> MarginFeatures:
    """Prefer judge-stage margin when available, otherwise classifier-stage."""
    judge = extract_margin_features(
        result.get("judge_token_logprobs"),
        mode="judge",
        source_stage="judge",
    )
    if judge.margin is not None or judge.label_start_position is not None:
        return judge

    return extract_margin_features(
        result.get("clf_token_logprobs"),
        mode="clf",
        source_stage="clf",
    )


def resolve_margin_policy_config(cfg: dict) -> dict[str, Any]:
    """Normalize hybrid margin policy config into a single dict."""
    hybrid_cfg = cfg.get("hybrid", {})
    threshold = hybrid_cfg.get("logprob_margin_threshold")
    return {
        "policy": hybrid_cfg.get("margin_policy", "baseline"),
        "threshold": threshold,
        "low_threshold": hybrid_cfg.get("margin_low_threshold", threshold),
        "high_threshold": hybrid_cfg.get("margin_high_threshold", threshold),
        "classifier_only_threshold": hybrid_cfg.get(
            "margin_threshold_classifier_only",
            threshold,
        ),
        "judge_threshold": hybrid_cfg.get("margin_threshold_judge", threshold),
    }


def apply_margin_policy(
    *,
    current_route: str,
    predicted_binary: str,
    predicted_label: str,
    margin: float | None,
    policy_cfg: dict[str, Any],
    route_bucket: str | None,
) -> dict[str, Any]:
    """Apply the configured benign-margin policy without changing defaults silently."""
    result = {
        "route": current_route,
        "final_binary": predicted_binary,
        "final_label": predicted_label,
        "override_applied": False,
        "override_reason": None,
        "policy_name": policy_cfg["policy"],
        "policy_outcome": "accepted",
        "policy_threshold_used": None,
    }

    if current_route != "llm" or predicted_binary != "benign":
        return result

    policy = policy_cfg["policy"]
    threshold = policy_cfg.get("threshold")
    low_threshold = policy_cfg.get("low_threshold")
    high_threshold = policy_cfg.get("high_threshold")
    route_threshold = threshold
    if policy == "route_specific":
        if route_bucket == "judge_involved":
            route_threshold = policy_cfg.get("judge_threshold")
        else:
            route_threshold = policy_cfg.get("classifier_only_threshold")

    if policy == "baseline":
        result["policy_threshold_used"] = threshold
        if threshold is not None and margin is not None and margin < threshold:
            result["final_binary"] = "adversarial"
            result["final_label"] = "adversarial"
            result["override_applied"] = True
            result["override_reason"] = "low_margin_force_adversarial"
            result["policy_outcome"] = "forced_adversarial"
        return result

    if policy == "escalate_band":
        result["policy_threshold_used"] = threshold
        if threshold is not None and margin is not None and margin < threshold:
            result["route"] = "abstain"
            result["final_binary"] = "adversarial"
            result["final_label"] = "uncertain"
            result["override_applied"] = True
            result["override_reason"] = "low_margin_escalate"
            result["policy_outcome"] = "escalated"
        return result

    if policy == "three_zone":
        result["policy_threshold_used"] = {
            "low": low_threshold,
            "high": high_threshold,
        }
        if margin is None or low_threshold is None or high_threshold is None:
            return result
        if margin < low_threshold:
            result["final_binary"] = "adversarial"
            result["final_label"] = "adversarial"
            result["override_applied"] = True
            result["override_reason"] = "three_zone_low_force_adversarial"
            result["policy_outcome"] = "forced_adversarial"
        elif margin < high_threshold:
            result["route"] = "abstain"
            result["final_binary"] = "adversarial"
            result["final_label"] = "uncertain"
            result["override_applied"] = True
            result["override_reason"] = "three_zone_mid_escalate"
            result["policy_outcome"] = "escalated"
        return result

    if policy == "route_specific":
        result["policy_threshold_used"] = route_threshold
        if route_threshold is not None and margin is not None and margin < route_threshold:
            result["final_binary"] = "adversarial"
            result["final_label"] = "adversarial"
            result["override_applied"] = True
            result["override_reason"] = f"route_specific_low_margin_{route_bucket or 'unknown'}"
            result["policy_outcome"] = "forced_adversarial"
        return result

    raise ValueError(f"Unknown margin policy: {policy}")


def infer_route_bucket(row: dict | Any) -> str:
    """Collapse execution path into the buckets used by trace/reporting."""
    routed_to = row.get("hybrid_routed_to") or row.get("routed_to")
    stages_run = row.get("llm_stages_run")
    try:
        stages_run = int(stages_run) if stages_run is not None else None
    except (TypeError, ValueError):
        stages_run = None

    if routed_to == "ml":
        return "ml_fastpath"
    if stages_run == 2:
        return "judge_involved"
    if stages_run == 1:
        return "classifier_only"
    if routed_to == "abstain":
        return "abstain"
    return "llm_unknown"

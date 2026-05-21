"""Routing diagnostic helpers shared by legacy research and external eval."""

from __future__ import annotations

import numpy as np
import pandas as pd


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


def normalize_binary_label(label) -> str:
    return str(label).strip().lower().replace("-", "_")


def normalize_attack_token(value) -> str:
    return str(value).strip().lower().replace("-", "_").replace(" ", "_")


def is_adversarial_label(label) -> bool:
    norm = normalize_binary_label(label)
    return norm in ADVERSARIAL_LABEL_ALIASES or norm.startswith("adv")


def compute_unicode_lane_mask(
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
        cat_norm = category.astype(str).map(normalize_attack_token)
        unicode_by_category = (cat_norm == "unicode_attack").values
    else:
        cat_has_value = np.zeros(n, dtype=bool)
        unicode_by_category = np.zeros(n, dtype=bool)

    if type_col in df.columns:
        attack_type = df[type_col]
        type_has_value = attack_type.notna().values
        type_norm = attack_type.astype(str).map(normalize_attack_token)
    else:
        type_has_value = np.zeros(n, dtype=bool)
        type_norm = pd.Series([""] * n)

    unicode_types_norm = {
        normalize_attack_token(v)
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

    ml_is_adv = df[ml_pred_col].map(is_adversarial_label) if total else pd.Series(dtype=bool)
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

    unicode_lane, lane_reliable = compute_unicode_lane_mask(
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

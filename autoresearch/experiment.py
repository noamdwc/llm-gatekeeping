"""
autoresearch/experiment.py — Routing logic the agent edits.

prepare.py merges all pre-computed predictions (ML, DeBERTa, LLM) into a
single DataFrame with one row per sample. This file defines how to route
each row to a final binary prediction.

Available columns per row:
  ML:      ml_pred_binary, ml_conf_binary, ml_pred_category, ml_pred_type,
           ml_conf_category, ml_conf_type, ml_proba_binary_adversarial
  DeBERTa: deberta_pred_binary, deberta_conf_binary, deberta_proba_binary_adversarial
  LLM:     llm_pred_binary, llm_conf_binary, llm_pred_category, llm_stages_run,
           llm_evidence, clf_confidence, judge_independent_confidence
  Margin:  margin (logprob nats), top1_logprob, top2_logprob,
           margin_source_stage, is_judge_stage
  Risk:    risk_score (P(adversarial) from trained risk model, 0-1)
"""

# === Thresholds ===
ML_CONFIDENCE_THRESHOLD = 0.85
DEBERTA_ADV_THRESHOLD = 0.75
DEBERTA_BEN_THRESHOLD = 0.80
DEBERTA_CONFIDENCE_THRESHOLD = DEBERTA_BEN_THRESHOLD  # compat for prepare.py logging
RISK_GATE_FOR_BENIGN = 0.30        # DeBERTa benign only trusted if risk < this
LLM_CONFIDENCE_THRESHOLD = 0.9
MARGIN_THRESHOLD = 2.0
RISK_THRESHOLD = 0.30

# === Unicode attack types (populated by prepare.py from config) ===
UNICODE_TYPES: list[str] = []


def route(row: dict) -> str:
    """Route a single sample to a final binary prediction.

    Args:
        row: dict with all pre-computed predictions (see columns above).

    Returns:
        "adversarial" or "benign"
    """
    ml_pred = row["ml_pred_binary"]
    ml_conf = row["ml_conf_binary"]
    ml_type = row.get("ml_pred_type", "")
    ml_category = row.get("ml_pred_category", "")

    deberta_pred = row.get("deberta_pred_binary")
    deberta_conf = row.get("deberta_conf_binary", 0.0)

    llm_pred = row.get("llm_pred_binary")
    llm_conf = row.get("llm_conf_binary", 0.0)
    margin = row.get("margin")
    risk_score = row.get("risk_score")

    is_unicode = ml_type in UNICODE_TYPES or ml_category == "unicode_attack"

    # Step 1: ML fast path — high-confidence unicode adversarial
    if ml_pred == "adversarial" and ml_conf >= ML_CONFIDENCE_THRESHOLD and is_unicode:
        return "adversarial"

    # Step 2: DeBERTa adversarial fast path (using proba for better precision)
    deberta_proba = row.get("deberta_proba_binary_adversarial", 0.5)
    if deberta_proba >= 0.98:
        return "adversarial"

    # Step 3: DeBERTa benign — gated on risk model
    if deberta_pred == "benign" and deberta_conf >= DEBERTA_BEN_THRESHOLD:
        if risk_score is None or risk_score < 0.15:
            return "benign"
        # Otherwise fall through to LLM/risk

    # Step 5: LLM decision (with DeBERTa adversarial override)
    if llm_pred and llm_conf >= LLM_CONFIDENCE_THRESHOLD:
        # If LLM says benign but DeBERTa strongly says adversarial, trust DeBERTa
        if llm_pred == "benign" and deberta_proba >= 0.95:
            return "adversarial"
        return llm_pred

    # Step 6: Abstain resolution via risk model (adaptive threshold)
    if risk_score is not None:
        clf_conf = row.get("clf_confidence") or 0
        # If LLM ran (val only) but low-conf, and DeBERTa says benign,
        # require very high risk to override
        if clf_conf > 0 and deberta_pred == "benign":
            effective_threshold = 0.60
        elif deberta_proba < 0.20:
            effective_threshold = 0.50
        else:
            effective_threshold = RISK_THRESHOLD
        return "adversarial" if risk_score > effective_threshold else "benign"

    # Default: conservative
    return "adversarial"

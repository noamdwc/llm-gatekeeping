def decide_accept_or_override(judge_out: dict, cand_out: dict) -> str:
    independent_label = judge_out.get("independent_label", "")
    cand_label = cand_out.get("label", "")

    # Garbage judge output → override
    if not independent_label:
        return "override_candidate"

    # Uncertain judge → always override (flag for review, don't silently accept)
    if independent_label == "uncertain":
        return "override_candidate"

    # Normalize to binary for comparison
    ind_binary = "benign" if independent_label == "benign" else "adversarial"
    cand_binary = "benign" if cand_label == "benign" else ("adversarial" if cand_label else "")

    if not cand_binary or ind_binary != cand_binary:
        return "override_candidate"

    if ind_binary != "adversarial":
        return "accept_candidate"

    # Both adversarial: require evidence match
    je = (judge_out.get("independent_evidence") or "").strip()
    ce = (cand_out.get("evidence") or "").strip()

    if not je and not ce:
        # Both evidence empty — valid for NLP attacks which have no extractable span.
        # Accept if either side identifies an NLP attack type; otherwise override.
        j_nlp = judge_out.get("nlp_attack_type", "none") not in ("none", "", None)
        c_nlp = cand_out.get("nlp_attack_type", "none") not in ("none", "", None)
        return "accept_candidate" if (j_nlp or c_nlp) else "override_candidate"

    if not je or not ce:
        return "override_candidate"
    if je in ce or ce in je:
        return "accept_candidate"

    return "override_candidate"

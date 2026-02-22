def decide_accept_or_override(judge_out: dict, cand_out: dict) -> str:
    independent_label = judge_out.get("independent_label", "")
    cand_label = cand_out.get("label", "")

    # Normalize both to binary for comparison
    ind_binary = "benign" if independent_label == "benign" else ("adversarial" if independent_label else "")
    cand_binary = "benign" if cand_label == "benign" else ("adversarial" if cand_label else "")

    if not ind_binary or ind_binary != cand_binary:
        return "override_candidate"

    if ind_binary != "adversarial":
        return "accept_candidate"

    # Both adversarial: require evidence match
    je = (judge_out.get("independent_evidence") or "").strip()
    ce = (cand_out.get("evidence") or "").strip()

    if not je or not ce:
        return "override_candidate"
    if je in ce or ce in je:
        return "accept_candidate"

    return "override_candidate"

def decide_accept_or_override(judge_out: dict, cand_out: dict) -> str:
    if judge_out["final_label"] != cand_out.get("label"):
        return "override_candidate"

    if judge_out["final_label"] != "adversarial":
        return "accept_candidate"

    # adversarial: require evidence and type match
    if judge_out.get("nlp_attack_type") != cand_out.get("nlp_attack_type"):
        return "override_candidate"

    je = (judge_out.get("final_evidence") or "").strip()
    ce = (cand_out.get("evidence") or "").strip()

    # allow minor differences: one contains the other
    if not je or not ce:
        return "override_candidate"
    if je in ce or ce in je:
        return "accept_candidate"

    return "override_candidate"

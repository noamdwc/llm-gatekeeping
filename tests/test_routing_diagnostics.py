import pandas as pd

from src.routing_diagnostics import (
    compute_routing_diagnostics,
    compute_unicode_lane_mask,
    is_adversarial_label,
    render_routing_diagnostics_markdown,
)


def test_is_adversarial_label_normalizes_common_aliases():
    assert is_adversarial_label("prompt-injection")
    assert is_adversarial_label("Adv")
    assert not is_adversarial_label("benign")


def test_compute_unicode_lane_mask_uses_category_and_configured_types():
    df = pd.DataFrame(
        {
            "ml_pred_category": ["unicode_attack", "nlp_attack", None],
            "ml_pred_type": ["Diacritcs", "Homoglyphs", None],
        }
    )

    unicode_lane, lane_reliable = compute_unicode_lane_mask(
        df,
        unicode_types={"Homoglyphs"},
    )

    assert unicode_lane.tolist() == [True, True, False]
    assert lane_reliable.tolist() == [True, True, False]


def test_compute_and_render_routing_diagnostics():
    df = pd.DataFrame(
        {
            "ml_pred_binary": ["adversarial", "benign", "adversarial", "benign"],
            "ml_pred_category": ["unicode_attack", "benign", "nlp_attack", None],
            "ml_pred_type": ["Diacritcs", "benign", "nlp_attack", None],
            "hybrid_routed_to": ["ml", "llm", "abstain", "llm"],
        }
    )

    diag = compute_routing_diagnostics(df)

    assert diag["total_samples"] == 4
    assert diag["routed_ml"] == 1
    assert diag["routed_llm"] == 2
    assert diag["routed_abstain"] == 1
    assert diag["ml_pred_benign_escalation_rate"] == 1.0
    assert diag["ml_pred_adversarial_escalation_rate"] == 0.5
    assert diag["unicode_lane_true_fastpath_ml"] == 1
    assert diag["unicode_lane_false_escalated"] == 3

    markdown = render_routing_diagnostics_markdown(diag)

    assert "## Routing Diagnostics" in markdown
    assert (
        "| ml_pred_label | routed_ml | routed_llm | routed_abstain | escalation_rate |" in markdown
    )
    assert "| unicode_lane | total | fastpath_ml | escalated_llm_or_abstain |" in markdown

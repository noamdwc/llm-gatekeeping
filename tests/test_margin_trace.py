"""Tests for row-level margin trace generation."""

import pandas as pd

from src.margin_trace import build_margin_trace, compute_binary_metrics_from_predictions


def test_build_margin_trace_populates_correctness_fields():
    research_df = pd.DataFrame({
        "sample_id": ["a", "b"],
        "label_binary": ["benign", "adversarial"],
        "llm_pred_binary": ["benign", "benign"],
        "hybrid_pred_binary": ["adversarial", "benign"],
        "hybrid_routed_to": ["llm", "abstain"],
        "hybrid_margin_policy": ["baseline", "baseline"],
        "hybrid_policy_outcome": ["forced_adversarial", "accepted"],
        "hybrid_override_applied": [True, False],
        "hybrid_override_reason": ["low_margin_force_adversarial", None],
        "hybrid_margin": [0.4, 1.2],
        "llm_conf_binary": [0.9, 0.6],
        "llm_stages_run": [1, 2],
        "clf_token_logprobs": [[{}, {}, {}, {}, {"top_logprobs": [{"token": "ben", "logprob": -0.1}, {"token": "adv", "logprob": -0.5}]}], None],
        "judge_token_logprobs": [None, [{"token": "{"}, {"token": "_label"}, {"token": "\":"}, {"token": " \""}, {"top_logprobs": [{"token": "adv", "logprob": -0.1}, {"token": "ben", "logprob": -1.0}]}]],
        "clf_provider_name": ["nim", "nim"],
        "judge_provider_name": [None, "nim"],
        "clf_model_name": ["m1", "m1"],
        "judge_model_name": [None, "m2"],
        "clf_raw_response_text": ["{}", "{}"],
        "judge_raw_response_text": [None, "{}"],
        "clf_parse_success": [True, True],
        "judge_parse_success": [None, True],
    })
    trace = build_margin_trace(research_df, dataset="demo", split="test")
    assert list(trace["is_fp"]) == [True, False]
    assert list(trace["is_fn"]) == [False, True]
    assert list(trace["is_correct"]) == [False, False]
    assert list(trace["route_bucket"]) == ["classifier_only", "judge_involved"]


def test_compute_binary_metrics_from_predictions():
    y_true = pd.Series(["adversarial", "benign", "adversarial", "benign"])
    y_pred = pd.Series(["adversarial", "adversarial", "benign", "benign"])
    metrics = compute_binary_metrics_from_predictions(y_true, y_pred)
    assert metrics["fp_count"] == 1
    assert metrics["fn_count"] == 1
    assert metrics["accuracy"] == 0.5

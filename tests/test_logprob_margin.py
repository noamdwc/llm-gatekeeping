"""Tests for shared logprob margin extraction and policy helpers."""

from src.logprob_margin import (
    apply_margin_policy,
    extract_margin_features,
    extract_preferred_margin_features_from_row,
    find_label_start_position,
)


def test_find_classifier_label_start_position():
    tokens = [{"token": "{"}, {"token": "\""}, {"token": "label"}, {"token": "\":"}, {"token": "ben"}]
    assert find_label_start_position(tokens, mode="clf") == 4


def test_find_judge_label_start_position():
    tokens = [
        {"token": "{"},
        {"token": "_label"},
        {"token": "\":"},
        {"token": " \""},
        {"token": "adv"},
    ]
    assert find_label_start_position(tokens, mode="judge") == 4


def test_extract_margin_features_handles_missing_token_names():
    tokens = [
        {"token": "{"},
        {"token": "\""},
        {"token": "label"},
        {"token": "\":"},
        {
            "token": "ben",
            "top_logprobs": [
                {"token": "", "logprob": -0.1},
                {"token": "", "logprob": -1.3},
            ],
        },
    ]
    result = extract_margin_features(tokens, mode="clf", source_stage="clf")
    assert result.margin == 1.2
    assert result.token_names_missing is True
    assert result.token_strings_available is False


def test_preferred_margin_uses_judge_before_classifier():
    row = {
        "clf_token_logprobs": [
            {}, {}, {}, {},
            {"top_logprobs": [{"token": "ben", "logprob": -0.2}, {"token": "adv", "logprob": -0.8}]},
        ],
        "judge_token_logprobs": [
            {"token": "{"},
            {"token": "_label"},
            {"token": "\":"},
            {"token": " \""},
            {"top_logprobs": [{"token": "adv", "logprob": -0.1}, {"token": "ben", "logprob": -1.6}]},
        ],
    }
    result = extract_preferred_margin_features_from_row(row)
    assert result.source_stage == "judge"
    assert result.margin == 1.5


def test_apply_margin_policy_baseline_forces_adversarial():
    result = apply_margin_policy(
        current_route="llm",
        predicted_binary="benign",
        predicted_label="benign",
        margin=0.5,
        policy_cfg={"policy": "baseline", "threshold": 1.0},
        route_bucket="classifier_only",
    )
    assert result["final_binary"] == "adversarial"
    assert result["override_applied"] is True
    assert result["policy_outcome"] == "forced_adversarial"


def test_apply_margin_policy_escalate_band_routes_to_abstain():
    result = apply_margin_policy(
        current_route="llm",
        predicted_binary="benign",
        predicted_label="benign",
        margin=0.5,
        policy_cfg={"policy": "escalate_band", "threshold": 1.0},
        route_bucket="classifier_only",
    )
    assert result["route"] == "abstain"
    assert result["final_label"] == "uncertain"
    assert result["policy_outcome"] == "escalated"


def test_apply_margin_policy_route_specific_uses_judge_threshold():
    result = apply_margin_policy(
        current_route="llm",
        predicted_binary="benign",
        predicted_label="benign",
        margin=1.2,
        policy_cfg={
            "policy": "route_specific",
            "threshold": 2.0,
            "classifier_only_threshold": 2.5,
            "judge_threshold": 1.5,
        },
        route_bucket="judge_involved",
    )
    assert result["override_applied"] is True
    assert result["policy_threshold_used"] == 1.5

from __future__ import annotations

from unittest.mock import MagicMock

import pandas as pd

from src.cli import judge_colab_local_predictions as judge_cli


def _classifier_only_row(**overrides):
    row = {
        "sample_id": "s1",
        "modified_sample": "Summarize this document.",
        "llm_pred_binary": "benign",
        "llm_pred_raw": "benign",
        "llm_pred_category": "benign",
        "llm_conf_binary": 0.91,
        "llm_evidence": "",
        "llm_stages_run": 1,
        "llm_provider_name": "transformers-local",
        "llm_model_name": "meta/llama-3.1-8b-instruct",
        "llm_raw_response_text": "{}",
        "llm_parse_success": True,
        "clf_label": "benign",
        "clf_category": "benign",
        "clf_confidence": 0.91,
        "clf_evidence": "",
        "clf_nlp_attack_type": "none",
        "clf_provider_name": "transformers-local",
        "clf_model_name": "meta/llama-3.1-8b-instruct",
        "clf_raw_response_text": "{}",
        "clf_parse_success": True,
        "clf_token_logprobs": "null",
    }
    row.update(overrides)
    return row


def test_apply_judge_runs_all_rows_by_default_and_preserves_classifier_columns(sample_config):
    predictions = pd.DataFrame(
        [
            _classifier_only_row(sample_id="high", clf_confidence=0.91, llm_conf_binary=0.91),
            _classifier_only_row(sample_id="low", clf_confidence=0.42, llm_conf_binary=0.42),
        ]
    )
    classifier = MagicMock()
    classifier.cfg = sample_config
    classifier.judge.return_value = {
        "independent_label": "adversarial",
        "independent_confidence": 97,
        "independent_evidence": "override instruction",
        "final_label": "adversarial",
        "final_confidence": 97,
        "nlp_attack_type": "none",
        "computed_decision": "override_candidate",
        "judge_benign_task_override": False,
        "judge_override_reason": None,
        "_provider_name": "openai",
        "_model_name": "gpt-4o",
        "_raw_response_text": '{"independent_label":"adversarial"}',
        "_parse_success": True,
        "_token_logprobs": [{"token": "adversarial"}],
    }

    out = judge_cli.apply_judge_to_predictions(predictions, classifier)

    assert classifier.judge.call_count == 2
    assert [call.args[0] for call in classifier.judge.call_args_list] == [
        "Summarize this document.",
        "Summarize this document.",
    ]
    high = out.loc[out["sample_id"] == "high"].iloc[0]
    low = out.loc[out["sample_id"] == "low"].iloc[0]
    assert high["llm_stages_run"] == 1
    assert high["llm_pred_binary"] == "benign"
    assert high["llm_pred_raw"] == "benign"
    assert high["llm_pred_category"] == "benign"
    assert high["llm_conf_binary"] == 0.91
    assert high["llm_model_name"] == "meta/llama-3.1-8b-instruct"
    assert high["llm_stages_run"] == 1
    assert low["llm_stages_run"] == 1
    assert low["llm_pred_binary"] == "benign"
    assert low["llm_pred_raw"] == "benign"
    assert low["llm_pred_category"] == "benign"
    assert low["llm_conf_binary"] == 0.42
    assert low["llm_model_name"] == "meta/llama-3.1-8b-instruct"
    assert high["judge_final_label"] == "adversarial"
    assert high["judge_final_pred_binary"] == "adversarial"
    assert high["judge_final_category"] == "unicode_attack"
    assert high["judge_final_confidence"] == 0.97
    assert low["judge_independent_label"] == "adversarial"
    assert low["judge_token_logprobs"] == '[{"token": "adversarial"}]'


def test_default_paths_use_colab_local_classifier_input_and_judged_output():
    args = judge_cli.parse_args(["--split", "unseen_val"])

    assert args.input.name == "llm_predictions_unseen_val_colab_local_classifier.parquet"
    assert args.output.name == "llm_predictions_unseen_val_colab_local_judged.parquet"

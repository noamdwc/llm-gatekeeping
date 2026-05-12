"""Shared schema for classifier-only Colab handoff artifacts."""

from __future__ import annotations


REQUIRED_COLUMNS = [
    "sample_id",
    "modified_sample",
    "llm_pred_binary",
    "llm_pred_raw",
    "llm_pred_category",
    "llm_conf_binary",
    "llm_stages_run",
    "llm_provider_name",
    "llm_model_name",
    "llm_raw_response_text",
    "llm_parse_success",
    "clf_label",
    "clf_category",
    "clf_confidence",
    "clf_evidence",
    "clf_nlp_attack_type",
    "clf_provider_name",
    "clf_model_name",
    "clf_raw_response_text",
    "clf_parse_success",
    "clf_token_logprobs",
]

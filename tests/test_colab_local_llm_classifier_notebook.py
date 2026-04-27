"""Structural tests for the Colab local LLM classifier notebook."""

from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "colab_local_llm_classifier.ipynb"


def _notebook() -> dict:
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def _all_source() -> str:
    nb = _notebook()
    return "\n".join(
        "".join(cell.get("source", []))
        for cell in nb["cells"]
    )


def test_notebook_exists_and_targets_colab_gpu():
    nb = _notebook()

    assert nb["nbformat"] == 4
    assert nb["metadata"]["accelerator"] == "GPU"
    assert nb["metadata"]["kernelspec"]["name"] == "python3"


def test_notebook_uses_current_classifier_model_and_vllm_backend():
    source = _all_source()

    assert "MODEL_ID = 'meta/llama-3.1-8b-instruct'" in source
    assert "python -m vllm.entrypoints.openai.api_server" in source
    assert "VLLM_BASE_URL = 'http://127.0.0.1:8000/v1'" in source
    assert "api_key='EMPTY'" in source


def test_notebook_reuses_project_classifier_helpers():
    source = _all_source()

    assert "from src.llm_classifier.llm_classifier import build_few_shot_examples" in source
    assert "from src.llm_classifier.prompts import build_classifier_messages" in source
    assert "from src.utils import build_sample_id, load_config" in source


def test_output_contract_is_classifier_only():
    source = _all_source()

    expected_columns = [
        "llm_pred_binary",
        "llm_pred_raw",
        "llm_pred_category",
        "llm_conf_binary",
        "llm_evidence",
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
    for column in expected_columns:
        assert column in source

    match = re.search(r"PREDICTION_COLUMNS = \[(.*?)\]", source, flags=re.S)
    assert match is not None
    assert "judge_" not in match.group(1)
    assert "assert not any(col.startswith('judge_')" in source


def test_notebook_has_checkpoint_and_resume_logic():
    source = _all_source()

    assert "CHECKPOINT_EVERY = 50" in source
    assert "CHECKPOINT_PATH" in source
    assert "completed_ids" in source
    assert "sample_id" in source
    assert "to_parquet" in source

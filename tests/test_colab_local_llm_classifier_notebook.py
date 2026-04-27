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
    assert "HF_MODEL_ID = 'meta-llama/Llama-3.1-8B-Instruct'" in source
    assert "python -m vllm.entrypoints.openai.api_server" in source
    assert "'--model', HF_MODEL_ID" in source
    assert "'--served-model-name', MODEL_ID" in source
    assert "VLLM_BASE_URL = 'http://127.0.0.1:8000/v1'" in source
    assert "api_key='EMPTY'" in source


def test_notebook_captures_vllm_startup_logs_for_debugging():
    source = _all_source()

    assert "VLLM_LOG_PATH" in source
    assert "vllm_server_{SPLIT}_{OUTPUT_SUFFIX}.log" in source
    assert "stdout=vllm_log_file" in source
    assert "stderr=subprocess.STDOUT" in source
    assert "def tail_log(path: Path, line_count: int = 80) -> str" in source
    assert "def raise_vllm_startup_error(reason: str) -> None" in source
    assert "Last vLLM log lines" in source
    assert "print(f'vLLM log path: {VLLM_LOG_PATH}')" in source


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


def test_notebook_filters_invalid_checkpoint_and_final_rows():
    source = _all_source()

    assert "def valid_prediction_mask(df: pd.DataFrame) -> pd.Series" in source
    assert "required_non_null_columns = ['sample_id', *PREDICTION_COLUMNS]" in source
    assert "df[required_non_null_columns].notna().all(axis=1)" in source
    assert "valid_checkpoint_df = checkpoint_df[valid_prediction_mask(checkpoint_df)].copy()" in source
    assert "invalid_checkpoint_rows = len(checkpoint_df) - len(valid_checkpoint_df)" in source
    assert "final_df = final_df[valid_prediction_mask(final_df)].copy()" in source
    assert "assert valid_prediction_mask(out_df).all()" in source


def test_notebook_encodes_parse_failures_explicitly():
    source = _all_source()

    assert "if not isinstance(payload, dict) or not payload:" in source
    assert "raw_response_text = raw_response_text or ''" in source
    assert "'confidence': 0.0" in source
    assert "'evidence': 'parse_failure'" in source
    assert "'reason': 'classifier response could not be parsed'" in source

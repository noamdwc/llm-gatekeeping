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


def test_notebook_code_cells_are_valid_python_or_colab_magics():
    nb = _notebook()

    for cell in nb["cells"]:
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        normalized = "\n".join(
            "pass" if line.startswith("%") else line
            for line in source.splitlines()
        )
        compile(normalized, f"{NOTEBOOK}:{cell.get('id')}", "exec")


def test_notebook_uses_current_classifier_model_and_transformers_backend():
    source = _all_source()

    assert "MODEL_ID = 'meta/llama-3.1-8b-instruct'" in source
    assert "HF_MODEL_ID = 'meta-llama/Llama-3.1-8B-Instruct'" in source
    assert "MODEL_PROVIDER_NAME = 'transformers-local'" in source
    assert "AutoTokenizer.from_pretrained" in source
    assert "AutoModelForCausalLM.from_pretrained" in source
    assert "model.generate(**model_inputs, **generation_kwargs)" in source
    assert "python -m vllm.entrypoints.openai.api_server" not in source
    assert "VLLM_BASE_URL" not in source
    assert "api_key='EMPTY'" not in source


def test_notebook_defines_batch_output_targets_and_paths():
    source = _all_source()

    assert "MAIN_SPLITS = ['train', 'val', 'test', 'unseen_val', 'unseen_test', 'safeguard_test']" in source
    assert "EXTERNAL_DATASETS = ['deepset', 'jackhhao']" in source
    assert "PREDICTIONS_EXTERNAL_DIR = f'{DRIVE_ROOT}/data/processed/predictions_external'" in source
    assert "def make_main_target(split: str) -> dict:" in source
    assert "def make_external_target(dataset_key: str) -> dict:" in source
    assert "llm_checkpoint_{split}_{OUTPUT_SUFFIX}.parquet" in source
    assert "llm_predictions_{split}_{OUTPUT_SUFFIX}.parquet" in source
    assert "llm_checkpoint_external_{dataset_key}_{OUTPUT_SUFFIX}.parquet" in source
    assert "llm_predictions_external_{dataset_key}.parquet" in source


def test_notebook_reuses_project_classifier_helpers():
    source = _all_source()

    assert "from src.llm_classifier.llm_classifier import build_few_shot_examples" in source
    assert "from src.llm_classifier.prompts import build_classifier_messages" in source
    assert "from src.utils import build_sample_id, load_config" in source
    assert "from src.eval_external import load_external_dataset" in source


def test_notebook_loads_main_and_external_targets_separately():
    source = _all_source()

    assert "train_df = pd.read_parquet(train_path)" in source
    assert "def load_main_target_df(target: dict) -> pd.DataFrame:" in source
    assert "def load_external_target_df(target: dict) -> pd.DataFrame:" in source
    assert "cfg['external_datasets'][dataset_key]" in source
    assert "load_external_dataset(ds_cfg)" in source
    assert "def prepare_target_df(target: dict) -> pd.DataFrame:" in source
    assert "if target['kind'] == 'main':" in source
    assert "elif target['kind'] == 'external':" in source


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
    assert "target['checkpoint_path']" in source
    assert "completed_ids" in source
    assert "sample_id" in source
    assert "to_parquet" in source
    assert "def run_target(target: dict) -> dict:" in source


def test_notebook_filters_invalid_checkpoint_and_final_rows():
    source = _all_source()

    assert "def valid_prediction_mask(df: pd.DataFrame) -> pd.Series" in source
    assert "required_non_null_columns = ['sample_id', *PREDICTION_COLUMNS]" in source
    assert "df[required_non_null_columns].notna().all(axis=1)" in source
    assert "valid_checkpoint_df = checkpoint_df[valid_prediction_mask(checkpoint_df)].copy()" in source
    assert "invalid_checkpoint_rows = len(checkpoint_df) - len(valid_checkpoint_df)" in source
    assert "final_df = final_df[valid_prediction_mask(final_df)].copy()" in source
    assert "assert_valid_output(out_df, target)" in source


def test_notebook_encodes_parse_failures_explicitly():
    source = _all_source()

    assert "if not isinstance(payload, dict) or not payload:" in source
    assert "raw_response_text = raw_response_text or ''" in source
    assert "'confidence': 0.0" in source
    assert "'evidence': 'parse_failure'" in source
    assert "'reason': 'classifier response could not be parsed'" in source


def test_notebook_runs_all_targets_and_reports_summary():
    source = _all_source()

    assert "TARGETS = [make_main_target(split) for split in MAIN_SPLITS]" in source
    assert "TARGETS.extend(make_external_target(dataset_key) for dataset_key in EXTERNAL_DATASETS)" in source
    assert "run_results = []" in source
    assert "for target in TARGETS:" in source
    assert "run_results.append(run_target(target))" in source
    assert "Batch output summary" in source
    assert "failure_status" in source

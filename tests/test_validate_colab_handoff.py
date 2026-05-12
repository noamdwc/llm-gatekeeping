from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from src.cli import validate_colab_handoff


def _classifier_frame(sample_ids=("s1", "s2")) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "sample_id": sample_id,
                "modified_sample": f"text {sample_id}",
                "label_binary": "benign" if sample_id.endswith("1") else "adversarial",
                "llm_pred_binary": "benign",
                "llm_pred_raw": "benign",
                "llm_pred_category": "benign",
                "llm_conf_binary": 0.9,
                "llm_stages_run": 1,
                "llm_provider_name": "transformers-local",
                "llm_model_name": "meta/llama-3.1-8b-instruct",
                "llm_raw_response_text": "{}",
                "llm_parse_success": True,
                "clf_label": "benign",
                "clf_category": "benign",
                "clf_confidence": 0.9,
                "clf_evidence": "",
                "clf_nlp_attack_type": "none",
                "clf_provider_name": "transformers-local",
                "clf_model_name": "meta/llama-3.1-8b-instruct",
                "clf_raw_response_text": "{}",
                "clf_parse_success": True,
                "clf_token_logprobs": json.dumps([{"token": " benign", "logprob": -0.1}]),
            }
            for sample_id in sample_ids
        ]
    )


def _deberta_frame(sample_ids=("s1", "s2")) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "sample_id": sample_id,
                "label_binary": "benign" if sample_id.endswith("1") else "adversarial",
                "deberta_proba_binary_adversarial": 0.2,
            }
            for sample_id in sample_ids
        ]
    )


def test_importing_validation_cli_does_not_import_judge_cli():
    code = "\n".join(
        [
            "import sys",
            "import src.cli.validate_colab_handoff",
            "assert 'src.cli.judge_colab_local_predictions' not in sys.modules",
        ]
    )

    subprocess.run([sys.executable, "-c", code], check=True)


def test_validate_artifact_passes_for_valid_classifier_and_deberta_join(tmp_path: Path):
    classifier_path = tmp_path / "llm_predictions_test_colab_local_classifier.parquet"
    deberta_path = tmp_path / "deberta_predictions_test.parquet"
    _classifier_frame().to_parquet(classifier_path, index=False)
    _deberta_frame().to_parquet(deberta_path, index=False)

    result = validate_colab_handoff.validate_artifact_pair(
        name="test",
        classifier_path=classifier_path,
        deberta_path=deberta_path,
    )

    assert result["name"] == "test"
    assert result["rows_classifier"] == 2
    assert result["rows_deberta"] == 2
    assert result["rows_joined"] == 2
    assert result["rows_dropped_classifier_only"] == 0
    assert result["rows_dropped_deberta_only"] == 0


def test_validate_artifact_fails_with_exact_missing_path(tmp_path: Path):
    classifier_path = tmp_path / "missing.parquet"
    deberta_path = tmp_path / "deberta_predictions_test.parquet"
    _deberta_frame().to_parquet(deberta_path, index=False)

    with pytest.raises(FileNotFoundError, match=str(classifier_path)):
        validate_colab_handoff.validate_artifact_pair(
            name="test",
            classifier_path=classifier_path,
            deberta_path=deberta_path,
        )


def test_validate_artifact_rejects_judge_columns(tmp_path: Path):
    classifier_path = tmp_path / "llm_predictions_test_colab_local_classifier.parquet"
    deberta_path = tmp_path / "deberta_predictions_test.parquet"
    _classifier_frame().assign(judge_ran=True).to_parquet(classifier_path, index=False)
    _deberta_frame().to_parquet(deberta_path, index=False)

    with pytest.raises(ValueError, match="judge columns"):
        validate_colab_handoff.validate_artifact_pair(
            name="test",
            classifier_path=classifier_path,
            deberta_path=deberta_path,
        )


def test_validate_artifact_requires_single_classifier_stage(tmp_path: Path):
    classifier_path = tmp_path / "llm_predictions_test_colab_local_classifier.parquet"
    deberta_path = tmp_path / "deberta_predictions_test.parquet"
    _classifier_frame().assign(llm_stages_run=2).to_parquet(classifier_path, index=False)
    _deberta_frame().to_parquet(deberta_path, index=False)

    with pytest.raises(ValueError, match="llm_stages_run == 1"):
        validate_colab_handoff.validate_artifact_pair(
            name="test",
            classifier_path=classifier_path,
            deberta_path=deberta_path,
        )


def test_validate_artifact_rejects_lossy_join(tmp_path: Path):
    classifier_path = tmp_path / "llm_predictions_test_colab_local_classifier.parquet"
    deberta_path = tmp_path / "deberta_predictions_test.parquet"
    _classifier_frame(sample_ids=("s1", "s2")).to_parquet(classifier_path, index=False)
    _deberta_frame(sample_ids=("s1", "s3")).to_parquet(deberta_path, index=False)

    with pytest.raises(ValueError, match="join mismatch"):
        validate_colab_handoff.validate_artifact_pair(
            name="test",
            classifier_path=classifier_path,
            deberta_path=deberta_path,
        )


def test_main_writes_validation_report_for_main_and_external_targets(tmp_path: Path):
    predictions_dir = tmp_path / "predictions"
    external_dir = tmp_path / "predictions_external"
    report_path = tmp_path / "reports" / "colab_handoff_validation.json"
    predictions_dir.mkdir()
    external_dir.mkdir()

    for split in validate_colab_handoff.DEFAULT_MAIN_SPLITS:
        _classifier_frame().to_parquet(
            predictions_dir / f"llm_predictions_{split}_colab_local_classifier.parquet",
            index=False,
        )
        _deberta_frame().to_parquet(
            predictions_dir / f"deberta_predictions_{split}.parquet",
            index=False,
        )

    _classifier_frame().to_parquet(
        external_dir / "llm_predictions_external_deepset_colab_local_classifier.parquet",
        index=False,
    )
    _deberta_frame().to_parquet(
        external_dir / "deberta_predictions_external_deepset.parquet",
        index=False,
    )

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "external_datasets:\n"
        "  deepset:\n"
        "    name: deepset/prompt-injections\n"
    )

    validate_colab_handoff.main(
        [
            "--config",
            str(config_path),
            "--predictions-dir",
            str(predictions_dir),
            "--predictions-external-dir",
            str(external_dir),
            "--output",
            str(report_path),
        ]
    )

    payload = json.loads(report_path.read_text())
    assert payload["ok"] is True
    assert [item["name"] for item in payload["artifacts"]] == [
        "val",
        "test",
        "unseen_val",
        "unseen_test",
        "safeguard_test",
        "external_deepset",
    ]

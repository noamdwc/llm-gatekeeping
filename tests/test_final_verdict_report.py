from pathlib import Path

import pandas as pd
import pytest

from src.cli import final_verdict_report


def _judged_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "sample_id": "a1",
                "label_binary": "adversarial",
                "llm_pred_binary": "benign",
                "llm_pred_category": "benign",
                "llm_conf_binary": 0.6,
                "judge_ran": True,
                "judge_final_pred_binary": "adversarial",
                "judge_final_category": "unicode_attack",
                "judge_final_confidence": 0.95,
            },
            {
                "sample_id": "b1",
                "label_binary": "benign",
                "llm_pred_binary": "benign",
                "llm_pred_category": "benign",
                "llm_conf_binary": 0.8,
                "judge_ran": None,
                "judge_final_pred_binary": None,
                "judge_final_category": None,
                "judge_final_confidence": None,
            },
            {
                "sample_id": "c1",
                "label_binary": "benign",
                "llm_pred_binary": "benign",
                "llm_pred_category": "benign",
                "llm_conf_binary": 0.7,
                "judge_ran": True,
                "judge_final_pred_binary": None,
                "judge_final_category": None,
                "judge_final_confidence": None,
            },
        ]
    )


def test_apply_final_verdict_uses_judge_output_only_when_present():
    out = final_verdict_report.apply_final_verdict(_judged_frame())

    assert list(out["final_pred_binary"]) == ["adversarial", "benign", "benign"]
    assert list(out["final_pred_category"]) == ["unicode_attack", "benign", "benign"]
    assert list(out["final_source"]) == ["judge", "cheap_classifier", "cheap_classifier"]
    assert list(out["final_confidence"]) == [0.95, 0.8, 0.7]


def test_render_report_includes_threshold_and_external_canonical_status():
    internal = final_verdict_report.DatasetResult(
        name="test",
        kind="internal",
        frame=final_verdict_report.apply_final_verdict(_judged_frame()),
    )
    external = final_verdict_report.DatasetResult(
        name="external_deepset",
        kind="external",
        frame=final_verdict_report.apply_final_verdict(_judged_frame().head(2)),
    )

    report = final_verdict_report.render_report(
        [internal, external],
        threshold=0.5,
        calibration_method="sigmoid",
        model_path="data/processed/models/escalating_model.pkl",
    )

    assert "Escalation gate threshold" in report
    assert "`0.5`" in report
    assert "External escalation is canonical for datasets with judged artifacts" in report
    assert "external_deepset" in report
    assert "Judge calls" in report


def test_main_writes_report_from_internal_and_external_inputs(tmp_path: Path):
    internal_path = tmp_path / "llm_predictions_test_colab_local_judged.parquet"
    external_path = tmp_path / "llm_predictions_external_deepset_colab_local_judged.parquet"
    output_path = tmp_path / "pipeline_final_verdict_report.md"
    config_path = tmp_path / "config.yaml"

    _judged_frame().to_parquet(internal_path, index=False)
    _judged_frame().head(2).to_parquet(external_path, index=False)
    config_path.write_text(
        "\n".join(
            [
                "hybrid:",
                "  escalating_model:",
                "    judge_threshold: 0.5",
                "    calibration_method: sigmoid",
                "    model_path: data/processed/models/escalating_model.pkl",
            ]
        )
    )

    final_verdict_report.main(
        [
            "--config",
            str(config_path),
            "--output",
            str(output_path),
            "--internal",
            f"test={internal_path}",
            "--external",
            f"external_deepset={external_path}",
        ]
    )

    text = output_path.read_text()
    assert "# Pipeline Final-Verdict Report" in text
    assert "test" in text
    assert "external_deepset" in text


def test_main_requires_configured_external_judged_artifact_by_default(tmp_path: Path, monkeypatch):
    internal_paths = {}
    for split in final_verdict_report.DEFAULT_INTERNAL_SPLITS:
        path = tmp_path / f"llm_predictions_{split}_colab_local_judged.parquet"
        _judged_frame().to_parquet(path, index=False)
        internal_paths[split] = path

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "hybrid:",
                "  escalating_model:",
                "    judge_threshold: 0.5",
                "    calibration_method: sigmoid",
                "    model_path: data/processed/models/escalating_model.pkl",
                "external_datasets:",
                "  deepset:",
                "    name: deepset/prompt-injections",
            ]
        )
    )

    monkeypatch.setattr(
        final_verdict_report,
        "default_internal_path",
        lambda split: internal_paths[split],
    )
    monkeypatch.setattr(
        final_verdict_report,
        "default_external_path",
        lambda dataset: tmp_path / f"missing_{dataset}_judged.parquet",
    )

    with pytest.raises(FileNotFoundError, match="Missing judged final-verdict input"):
        final_verdict_report.main([
            "--config",
            str(config_path),
            "--output",
            str(tmp_path / "pipeline_final_verdict_report.md"),
        ])

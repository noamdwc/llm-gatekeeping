"""Tests for the lightweight escalation inference path."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from src.cli import infer_split, score_escalation
from src.escalating_model import (
    ESCALATING_FEATURE_COLS,
    EscalatingDataset,
    EscalatingModel,
)


def _classifier_predictions(n: int = 12) -> pd.DataFrame:
    rows = []
    for i in range(n):
        label = "adversarial" if i % 2 == 0 else "benign"
        pred = "benign" if i % 4 == 0 else label
        rows.append({
            "sample_id": f"s{i}",
            "modified_sample": f"text {i}",
            "label_binary": label,
            "llm_pred_binary": pred,
            "llm_pred_raw": pred,
            "llm_pred_category": "benign" if pred == "benign" else "unicode_attack",
            "llm_conf_binary": 0.6 + (i % 4) * 0.05,
            "llm_stages_run": 1,
            "llm_provider_name": "transformers-local",
            "llm_model_name": "meta/llama-3.1-8b-instruct",
            "llm_raw_response_text": "{}",
            "llm_parse_success": True,
            "clf_label": pred,
            "clf_category": "benign" if pred == "benign" else "unicode_attack",
            "clf_confidence": 0.5 + (i % 5) * 0.03,
            "clf_evidence": "",
            "clf_nlp_attack_type": "none",
            "clf_provider_name": "transformers-local",
            "clf_model_name": "meta/llama-3.1-8b-instruct",
            "clf_raw_response_text": "{}",
            "clf_parse_success": True,
            "clf_token_logprobs": "null",
            "prompt_hash": f"h{i // 2}",
            "attack_name": "Homoglyphs" if label == "adversarial" else None,
        })
    return pd.DataFrame(rows)


def _deberta_predictions(n: int = 12) -> pd.DataFrame:
    return pd.DataFrame([
        {
            "sample_id": f"s{i}",
            "deberta_proba_binary_adversarial": 0.2 + (i % 6) * 0.1,
        }
        for i in range(n)
    ])


def _train_minimal_model(tmp_path: Path) -> Path:
    classifier = _classifier_predictions()
    deberta = _deberta_predictions()
    ds = EscalatingDataset(classifier, deberta)
    model = EscalatingModel.train(ds.X, ds.y, list(ESCALATING_FEATURE_COLS))
    model_path = tmp_path / "escalating_model.pkl"
    model.save(model_path)
    return model_path


def test_score_split_writes_canonical_eval_columns(tmp_path):
    classifier = _classifier_predictions()
    deberta = _deberta_predictions()
    model_path = _train_minimal_model(tmp_path)
    model = EscalatingModel.load(model_path)

    scored, summary = score_escalation.score_split(
        "test",
        classifier_df=classifier,
        deberta_df=deberta,
        model=model,
    )

    assert "escalation_score" in scored.columns
    assert summary["split"] == "test"
    assert summary["rows_joined"] == len(classifier)


def test_score_escalation_cli_writes_output_parquet(tmp_path):
    classifier = _classifier_predictions()
    deberta = _deberta_predictions()
    classifier_path = tmp_path / "clf.parquet"
    deberta_path = tmp_path / "deberta.parquet"
    classifier.to_parquet(classifier_path, index=False)
    deberta.to_parquet(deberta_path, index=False)
    model_path = _train_minimal_model(tmp_path)
    output_path = tmp_path / "scores.parquet"

    score_escalation.main([
        "--split", "test",
        "--classifier-predictions", str(classifier_path),
        "--deberta-predictions", str(deberta_path),
        "--model", str(model_path),
        "--output", str(output_path),
    ])

    out = pd.read_parquet(output_path)
    assert "escalation_score" in out.columns
    assert len(out) == len(classifier)


def test_run_escalation_split_orchestrates_score_judge_and_verdict(tmp_path, sample_config, monkeypatch):
    """End-to-end: --mode escalation reads classifier+deberta, scores, judges
    only escalated rows, and writes a final-verdict report."""
    classifier = _classifier_predictions()
    deberta = _deberta_predictions()
    classifier_path = tmp_path / "predictions" / "llm_predictions_test_colab_local_classifier.parquet"
    deberta_path = tmp_path / "predictions" / "deberta_predictions_test.parquet"
    classifier_path.parent.mkdir(parents=True, exist_ok=True)
    classifier.to_parquet(classifier_path, index=False)
    deberta.to_parquet(deberta_path, index=False)

    model_path = _train_minimal_model(tmp_path)
    report_path = tmp_path / "inference_escalation_test.md"
    judged_path = tmp_path / "predictions" / "llm_predictions_test_colab_local_judged.parquet"
    scores_path = tmp_path / "research" / "escalating_model_eval_test.parquet"
    scores_path.parent.mkdir(parents=True, exist_ok=True)

    cfg = dict(sample_config)
    cfg["hybrid"] = {
        "escalating_model": {
            "model_path": str(model_path),
            "judge_threshold": 0.0,  # force judge on all joined rows
            "calibration_method": "sigmoid",
        }
    }

    judge_result = {
        "independent_label": "adversarial",
        "independent_confidence": 95,
        "independent_evidence": "",
        "final_label": "adversarial",
        "final_confidence": 95,
        "nlp_attack_type": "none",
        "computed_decision": "ok",
        "judge_benign_task_override": False,
        "judge_override_reason": None,
        "_provider_name": "test",
        "_model_name": "test",
        "_raw_response_text": "{}",
        "_parse_success": True,
        "_token_logprobs": None,
    }

    monkeypatch.setattr(infer_split, "load_config", lambda _path: cfg)
    monkeypatch.setattr(
        infer_split.score_escalation,
        "default_classifier_path",
        lambda split: classifier_path,
    )
    monkeypatch.setattr(
        infer_split.score_escalation,
        "default_deberta_path",
        lambda split: deberta_path,
    )
    monkeypatch.setattr(
        infer_split.score_escalation,
        "default_output_path",
        lambda split: scores_path,
    )
    monkeypatch.setattr(
        infer_split.judge_colab_local_predictions,
        "default_input_path",
        lambda split: classifier_path,
    )
    monkeypatch.setattr(
        infer_split.judge_colab_local_predictions,
        "default_output_path",
        lambda split: judged_path,
    )

    with patch.object(
        infer_split.HierarchicalLLMClassifier,
        "__init__",
        lambda self, cfg: setattr(self, "cfg", cfg) or setattr(self, "usage", None),
    ), patch.object(
        infer_split.HierarchicalLLMClassifier,
        "judge",
        return_value=judge_result,
    ) as judge_mock:
        out = infer_split.run_escalation_split("test", config_path=None, output=report_path)

    assert out == report_path
    assert report_path.exists()
    body = report_path.read_text()
    assert "Pipeline Final-Verdict Report" in body
    assert scores_path.exists()
    assert judged_path.exists()
    judged = pd.read_parquet(judged_path)
    # threshold=0.0 means every joined row is judged
    assert judge_mock.call_count == len(classifier)
    assert judged["judge_ran"].fillna(False).all()

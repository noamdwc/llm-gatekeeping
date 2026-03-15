"""Tests for DeBERTa classifier debug helpers, label validation, and lifecycle guards."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from src.models.debug_numerics import (
    DebugConfig,
    TensorSummary,
    check_tensor_finite,
    dump_bad_batch,
    find_nonfinite_params,
    summarize_tensor,
    validate_labels,
)
from src.models.deberta_classifier import DeBERTaClassifier, TrainingResult


# ── validate_labels ──────────────────────────────────────────────────────────


class TestValidateLabels:
    def test_valid_labels(self):
        assert validate_labels([0, 1, 0, 1], num_labels=2) == []

    def test_out_of_range(self):
        problems = validate_labels([0, 2], num_labels=2)
        assert len(problems) == 1
        assert "out of range" in problems[0]

    def test_nan_label(self):
        problems = validate_labels([0, float("nan")], num_labels=2)
        assert len(problems) == 1
        assert "NaN" in problems[0]

    def test_negative_label(self):
        problems = validate_labels([-1, 0], num_labels=2)
        assert len(problems) == 1
        assert "negative" in problems[0]

    def test_non_integer_float(self):
        problems = validate_labels([0.5], num_labels=2)
        assert len(problems) == 1
        assert "non-integer" in problems[0]


# ── check_tensor_finite ─────────────────────────────────────────────────────


class TestCheckTensorFinite:
    def test_finite_ok(self):
        t = torch.tensor([1.0, 2.0, 3.0])
        assert check_tensor_finite("test", t) == []

    def test_nan_detected(self):
        t = torch.tensor([1.0, float("nan"), 3.0])
        problems = check_tensor_finite("test", t)
        assert len(problems) == 1
        assert "NaN" in problems[0]

    def test_inf_detected(self):
        t = torch.tensor([1.0, float("inf"), 3.0])
        problems = check_tensor_finite("test", t)
        assert len(problems) == 1
        assert "Inf" in problems[0]

    def test_both_nan_and_inf(self):
        t = torch.tensor([float("nan"), float("inf")])
        problems = check_tensor_finite("test", t)
        assert len(problems) == 2


# ── find_nonfinite_params ────────────────────────────────────────────────────


class TestFindNonfiniteParams:
    def test_clean_model(self):
        model = nn.Linear(3, 2)
        assert find_nonfinite_params(model) == []

    def test_poisoned_model(self):
        model = nn.Linear(3, 2)
        with torch.no_grad():
            model.weight[0, 0] = float("nan")
        bad = find_nonfinite_params(model)
        assert len(bad) == 1
        assert bad[0][0] == "weight"
        assert bad[0][1].has_nan


# ── summarize_tensor ─────────────────────────────────────────────────────────


class TestSummarizeTensor:
    def test_basic(self):
        t = torch.tensor([1.0, 2.0, 3.0])
        s = summarize_tensor("test", t)
        assert isinstance(s, TensorSummary)
        assert s.name == "test"
        assert s.has_nan is False
        assert s.has_inf is False
        assert s.min == 1.0
        assert s.max == 3.0

    def test_nan_tensor(self):
        t = torch.tensor([1.0, float("nan")])
        s = summarize_tensor("test", t)
        assert s.has_nan is True


# ── TrainingResult ───────────────────────────────────────────────────────────


class TestTrainingResult:
    def test_success(self):
        r = TrainingResult(success=True)
        assert r.success
        assert r.failed_reason is None
        assert r.debug_artifact_paths == []

    def test_failure(self):
        r = TrainingResult(
            success=False,
            failed_reason="NaN loss",
            first_bad_epoch=0,
            first_bad_step=5,
            first_bad_stage="forward",
        )
        assert not r.success
        assert r.first_bad_stage == "forward"


# ── dump_bad_batch ───────────────────────────────────────────────────────────


class TestDumpBadBatch:
    def test_creates_artifacts(self, tmp_path):
        batch = {"input_ids": torch.tensor([[1, 2, 3]])}
        loss = torch.tensor(float("nan"))
        logits = torch.tensor([[0.5, -0.3]])

        dump_dir = dump_bad_batch(
            tmp_path, epoch=0, step=1, stage="forward",
            batch=batch, loss=loss, logits=logits,
            texts=["hello world"],
        )

        assert dump_dir.exists()
        assert (dump_dir / "metadata.json").exists()
        assert (dump_dir / "batch.pt").exists()
        assert (dump_dir / "loss.pt").exists()
        assert (dump_dir / "logits.pt").exists()

        metadata = json.loads((dump_dir / "metadata.json").read_text())
        assert metadata["epoch"] == 0
        assert metadata["step"] == 1
        assert metadata["stage"] == "forward"
        assert metadata["texts"] == ["hello world"]


# ── Save protection ─────────────────────────────────────────────────────────


class TestSaveProtection:
    def test_save_raises_on_nonfinite(self, tmp_path, sample_config_with_deberta):
        clf = DeBERTaClassifier(sample_config_with_deberta)
        clf.label2id = {"benign": 0, "adversarial": 1}
        clf.id2label = {0: "benign", 1: "adversarial"}
        clf.tokenizer = MagicMock()
        clf.model = nn.Linear(3, 2)
        with torch.no_grad():
            clf.model.weight[0, 0] = float("nan")

        with pytest.raises(ValueError, match="non-finite"):
            clf.save(tmp_path / "model_out")


# ── CLI lifecycle ────────────────────────────────────────────────────────────


class TestCLILifecycle:
    """Verify that failed training prevents save and predict."""

    @patch("src.cli.deberta_classifier.DeBERTaClassifier")
    @patch("src.cli.deberta_classifier.pd.read_parquet")
    def test_no_save_on_failure(self, mock_read, MockClf):
        """When train() returns failure, save() must not be called and sys.exit(1)."""
        mock_read.return_value = MagicMock()

        mock_clf = MagicMock()
        MockClf.return_value = mock_clf
        mock_clf.train.return_value = TrainingResult(
            success=False,
            failed_reason="NaN loss",
            first_bad_epoch=0,
            first_bad_step=0,
            first_bad_stage="forward",
        )

        from src.cli.deberta_classifier import main

        with patch("sys.argv", ["prog", "--train-only", "--cpu"]):
            with pytest.raises(SystemExit) as exc_info:
                main()

        assert exc_info.value.code == 1
        mock_clf.save.assert_not_called()


# ── DebugConfig ──────────────────────────────────────────────────────────────


class TestDebugConfig:
    def test_defaults(self):
        dc = DebugConfig()
        assert dc.enabled is False
        assert dc.first_n_batches == 0
        assert dc.save_bad_batch is False
        assert dc.sanity_forward_only is False
        assert dc.sanity_batches == 3

    def test_enabled(self):
        dc = DebugConfig(enabled=True, first_n_batches=5, save_bad_batch=True)
        assert dc.enabled is True
        assert dc.first_n_batches == 5

"""Tests for DeBERTa classifier debug helpers, label validation, and lifecycle guards."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
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
from src.models.deberta_classifier import DeBERTaClassifier, PromptDataset, TrainingResult


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

    def test_save_writes_best_checkpoint_artifacts(self, tmp_path, sample_config_with_deberta):
        clf = DeBERTaClassifier(sample_config_with_deberta)
        clf.label2id = {"benign": 0, "adversarial": 1}
        clf.id2label = {0: "benign", 1: "adversarial"}
        clf.tokenizer = FakeTokenizer()
        clf.model = FakeHFModel()
        clf.train_history = [{"epoch": 1, "train_loss": 0.1}]
        clf.best_checkpoint = {
            "epoch": 1,
            "metric_name": "f1",
            "metric_value": 0.9,
            "model_state_dict": {
                name: tensor.detach().cpu().clone()
                for name, tensor in clf.model.state_dict().items()
            },
        }

        clf.save(tmp_path / "model_out")

        assert (tmp_path / "model_out" / "best_checkpoint.pt").exists()
        metadata = json.loads((tmp_path / "model_out" / "best_checkpoint.json").read_text())
        assert metadata == {"epoch": 1, "metric_name": "f1", "metric_value": 0.9}


class FakeTokenizer:
    def __call__(self, texts, truncation=True, max_length=128, padding=False):
        n = len(texts)
        return {
            "input_ids": [[n - i, 2] for i in range(n)],
            "attention_mask": [[1, 1] for _ in range(n)],
        }

    def batch_decode(self, input_ids, skip_special_tokens=True):
        return ["decoded"] * len(input_ids)

    def save_pretrained(self, output_dir):
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        (Path(output_dir) / "tokenizer.json").write_text("{}")


class FakeCollator:
    def __call__(self, items):
        batch = {
            "input_ids": torch.tensor([item["input_ids"] for item in items], dtype=torch.long),
            "attention_mask": torch.tensor([item["attention_mask"] for item in items], dtype=torch.long),
        }
        if "labels" in items[0]:
            batch["labels"] = torch.stack([item["labels"] for item in items])
        return batch


class FakeScheduler:
    def step(self):
        return None

    def get_last_lr(self):
        return [1e-3]


class FakeHFModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(0.0))

    def forward(self, input_ids, attention_mask=None, labels=None):
        batch_size = input_ids.shape[0]
        scale = input_ids[:, 0].float()
        logits = torch.stack(
            [self.weight.expand(batch_size) * scale, (-self.weight).expand(batch_size) * scale],
            dim=1,
        )
        loss = (self.weight - 1.0) ** 2
        return type("HFOutput", (), {"loss": loss, "logits": logits})()

    def save_pretrained(self, output_dir):
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), Path(output_dir) / "weights.pt")


class TestTrainingLifecycle:
    @patch("src.models.deberta_classifier.get_linear_schedule_with_warmup", return_value=FakeScheduler())
    @patch("src.models.deberta_classifier.AdamW", side_effect=lambda params, lr, weight_decay: torch.optim.SGD(params, lr=lr))
    @patch("src.models.deberta_classifier.DataCollatorWithPadding", side_effect=lambda tokenizer: FakeCollator())
    @patch("src.models.deberta_classifier.AutoModelForSequenceClassification.from_pretrained", return_value=FakeHFModel())
    @patch("src.models.deberta_classifier.AutoTokenizer.from_pretrained", return_value=FakeTokenizer())
    def test_train_restores_best_checkpoint_and_stops_early(
        self,
        _mock_tokenizer,
        _mock_model,
        _mock_collator,
        _mock_optimizer,
        _mock_scheduler,
        sample_config_with_deberta,
    ):
        cfg = sample_config_with_deberta.copy()
        cfg["deberta"] = cfg["deberta"].copy()
        cfg["deberta"]["num_epochs"] = 5
        cfg["deberta"]["early_stopping_patience"] = 2
        cfg["deberta"]["metric_for_best_model"] = "f1"
        cfg["deberta"]["learning_rate"] = 0.2

        df_train = pd.DataFrame({
            "modified_sample": ["a", "b", "c", "d"],
            "label_binary": ["benign", "adversarial", "benign", "adversarial"],
        })
        df_val = pd.DataFrame({
            "modified_sample": ["e", "f"],
            "label_binary": ["benign", "adversarial"],
        })

        clf = DeBERTaClassifier(cfg)

        metrics = [
            {
                "accuracy": 0.8, "f1": 0.7, "macro_f1": 0.75, "precision": 0.7, "recall": 0.7,
                "f1_benign": 0.8, "f1_adversarial": 0.7,
            },
            {
                "accuracy": 0.7, "f1": 0.6, "macro_f1": 0.65, "precision": 0.6, "recall": 0.6,
                "f1_benign": 0.7, "f1_adversarial": 0.6,
            },
            {
                "accuracy": 0.6, "f1": 0.5, "macro_f1": 0.55, "precision": 0.5, "recall": 0.5,
                "f1_benign": 0.6, "f1_adversarial": 0.5,
            },
        ]

        with patch.object(clf, "_evaluate", side_effect=metrics):
            result = clf.train(df_train, df_val, text_col="modified_sample", force_cpu=True)

        assert result.success is True
        assert result.stopped_early is True
        assert result.best_epoch == 1
        assert result.best_metric_name == "f1"
        assert result.best_metric_value == 0.7
        assert len(result.train_history) == 3
        assert clf.best_checkpoint["epoch"] == 1
        assert torch.isclose(clf.model.weight.detach().cpu(), torch.tensor(0.1), atol=1e-6)

    @patch("src.models.deberta_classifier.get_linear_schedule_with_warmup", return_value=FakeScheduler())
    @patch("src.models.deberta_classifier.AdamW", side_effect=lambda params, lr, weight_decay: torch.optim.SGD(params, lr=lr))
    @patch("src.models.deberta_classifier.DataCollatorWithPadding", side_effect=lambda tokenizer: FakeCollator())
    @patch("src.models.deberta_classifier.AutoModelForSequenceClassification.from_pretrained", return_value=FakeHFModel())
    @patch("src.models.deberta_classifier.AutoTokenizer.from_pretrained", return_value=FakeTokenizer())
    def test_train_records_unseen_monitor_metrics_without_selecting_on_them(
        self,
        _mock_tokenizer,
        _mock_model,
        _mock_collator,
        _mock_optimizer,
        _mock_scheduler,
        sample_config_with_deberta,
    ):
        cfg = sample_config_with_deberta.copy()
        cfg["deberta"] = cfg["deberta"].copy()
        cfg["deberta"]["num_epochs"] = 1
        cfg["deberta"]["metric_for_best_model"] = "f1"

        df_train = pd.DataFrame({
            "modified_sample": ["a", "b", "c", "d"],
            "label_binary": ["benign", "adversarial", "benign", "adversarial"],
        })
        df_val = pd.DataFrame({
            "modified_sample": ["e", "f"],
            "label_binary": ["benign", "adversarial"],
        })
        monitor_dfs = {
            "unseen_val": pd.DataFrame({
                "modified_sample": ["g", "h"],
                "label_binary": ["benign", "adversarial"],
            }),
            "unseen_test": pd.DataFrame({
                "modified_sample": ["i", "j"],
                "label_binary": ["benign", "adversarial"],
            }),
        }

        clf = DeBERTaClassifier(cfg)
        metrics = [
            {
                "accuracy": 0.8, "f1": 0.7, "macro_f1": 0.75, "precision": 0.7, "recall": 0.7,
                "f1_benign": 0.8, "f1_adversarial": 0.7,
            },
            {
                "accuracy": 0.4, "f1": 0.9, "macro_f1": 0.5, "precision": 0.91, "recall": 0.92,
                "f1_benign": 0.2, "f1_adversarial": 0.9,
            },
            {
                "accuracy": 0.3, "f1": 0.95, "macro_f1": 0.4, "precision": 0.96, "recall": 0.97,
                "f1_benign": 0.1, "f1_adversarial": 0.95,
            },
        ]

        with patch.object(clf, "_evaluate", side_effect=metrics):
            result = clf.train(
                df_train,
                df_val,
                text_col="modified_sample",
                force_cpu=True,
                monitor_dfs=monitor_dfs,
            )

        assert result.best_metric_value == 0.7
        assert result.train_history[0]["unseen_val_f1"] == 0.9
        assert result.train_history[0]["unseen_val_precision"] == 0.91
        assert result.train_history[0]["unseen_val_recall"] == 0.92
        assert result.train_history[0]["unseen_test_f1"] == 0.95
        assert result.train_history[0]["unseen_test_precision"] == 0.96
        assert result.train_history[0]["unseen_test_recall"] == 0.97

    @patch("src.models.deberta_classifier.DataCollatorWithPadding", side_effect=lambda tokenizer: FakeCollator())
    @patch("src.models.deberta_classifier.AutoModelForSequenceClassification.from_pretrained", return_value=FakeHFModel())
    @patch("src.models.deberta_classifier.AutoTokenizer.from_pretrained", return_value=FakeTokenizer())
    def test_evaluate_returns_per_label_f1(
        self,
        _mock_tokenizer,
        _mock_model,
        _mock_collator,
        sample_config_with_deberta,
    ):
        clf = DeBERTaClassifier(sample_config_with_deberta)
        clf.label2id = {"benign": 0, "adversarial": 1}
        clf.id2label = {0: "benign", 1: "adversarial"}
        clf.tokenizer = FakeTokenizer()
        clf.model = FakeHFModel()

        ds = PromptDataset(
            clf.tokenizer,
            texts=["a", "b"],
            labels=[0, 1],
            max_length=clf.max_length,
        )
        val_loader = torch.utils.data.DataLoader(
            ds,
            batch_size=2,
            shuffle=False,
            collate_fn=FakeCollator(),
        )

        metrics = clf._evaluate(val_loader, torch.device("cpu"))

        assert metrics["macro_f1"] == pytest.approx(1 / 3)
        assert metrics["f1_benign"] == pytest.approx(2 / 3)
        assert metrics["f1_adversarial"] == 0.0


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

        with patch("sys.argv", ["prog", "--train-only", "--cpu", "--no-wandb"]):
            with pytest.raises(SystemExit) as exc_info:
                main()

        assert exc_info.value.code == 1
        mock_clf.save.assert_not_called()

    @patch("src.cli.deberta_classifier.wandb")
    @patch("src.cli.deberta_classifier.DeBERTaClassifier")
    @patch("src.cli.deberta_classifier.pd.read_parquet")
    def test_wandb_logs_training_history_and_finishes(self, mock_read, MockClf, mock_wandb):
        """Successful training should initialize wandb, log history, and finish the run."""
        train_df = MagicMock()
        val_df = MagicMock()
        unseen_val_df = MagicMock()
        unseen_test_df = MagicMock()
        train_df.__len__.return_value = 10
        val_df.__len__.return_value = 4
        unseen_val_df.__len__.return_value = 3
        unseen_test_df.__len__.return_value = 2
        mock_read.side_effect = [train_df, val_df, unseen_val_df, unseen_test_df]

        mock_clf = MagicMock()
        MockClf.return_value = mock_clf
        mock_clf.train.return_value = TrainingResult(
            success=True,
            train_history=[{
                "epoch": 1,
                "train_loss": 0.25,
                "eval_accuracy": 0.9,
                "eval_f1": 0.8,
                "eval_macro_f1": 0.8,
                "eval_f1_adversarial": 0.82,
                "eval_f1_benign": 0.78,
                "eval_precision": 0.85,
                "eval_recall": 0.75,
                "unseen_val_f1": 0.71,
                "unseen_val_precision": 0.72,
                "unseen_val_recall": 0.73,
                "unseen_test_f1": 0.61,
                "unseen_test_precision": 0.62,
                "unseen_test_recall": 0.63,
            }],
        )
        mock_wandb.run = object()

        from src.cli.deberta_classifier import main

        with patch("sys.argv", ["prog", "--train-only", "--cpu"]):
            main()

        mock_wandb.init.assert_called_once()
        train_kwargs = mock_clf.train.call_args.kwargs
        assert train_kwargs["monitor_dfs"] == {
            "unseen_val": unseen_val_df,
            "unseen_test": unseen_test_df,
        }
        assert mock_wandb.log.call_count >= 3
        assert any(
            call.kwargs.get("step") == 1
            and call.args
            and call.args[0].get("eval/f1_adversarial") == 0.82
            and call.args[0].get("eval/f1_benign") == 0.78
            for call in mock_wandb.log.call_args_list
        )
        assert any(
            call.kwargs.get("step") == 1
            and call.args
            and call.args[0].get("monitor/unseen_val/f1") == 0.71
            and call.args[0].get("monitor/unseen_val/precision") == 0.72
            and call.args[0].get("monitor/unseen_val/recall") == 0.73
            and call.args[0].get("monitor/unseen_test/f1") == 0.61
            and call.args[0].get("monitor/unseen_test/precision") == 0.62
            and call.args[0].get("monitor/unseen_test/recall") == 0.63
            for call in mock_wandb.log.call_args_list
        )
        mock_wandb.log_artifact.assert_not_called()
        mock_wandb.finish.assert_called_once_with()

    @patch("src.cli.deberta_classifier.wandb")
    @patch("src.cli.deberta_classifier.DeBERTaClassifier")
    @patch("src.cli.deberta_classifier.pd.read_parquet")
    def test_wandb_finishes_with_error_on_training_failure(self, mock_read, MockClf, mock_wandb):
        """Failed training should mark the wandb run failed before exiting."""
        train_df = MagicMock()
        val_df = MagicMock()
        unseen_val_df = MagicMock()
        unseen_test_df = MagicMock()
        train_df.__len__.return_value = 10
        val_df.__len__.return_value = 4
        unseen_val_df.__len__.return_value = 3
        unseen_test_df.__len__.return_value = 2
        mock_read.side_effect = [train_df, val_df, unseen_val_df, unseen_test_df]

        mock_clf = MagicMock()
        MockClf.return_value = mock_clf
        mock_clf.train.return_value = TrainingResult(
            success=False,
            failed_reason="NaN loss",
            first_bad_epoch=0,
            first_bad_step=2,
            first_bad_stage="forward",
        )
        mock_wandb.run = object()

        from src.cli.deberta_classifier import main

        with patch("sys.argv", ["prog", "--train-only", "--cpu"]):
            with pytest.raises(SystemExit):
                main()

        mock_wandb.init.assert_called_once()
        mock_wandb.finish.assert_called_once_with(exit_code=1)


class TestCLIReporting:
    def test_summary_includes_unseen_precision_recall_and_f1(self):
        from src.cli.deberta_classifier import compute_split_metrics, generate_summary

        split_df = pd.DataFrame({
            "label_binary": ["adversarial", "benign", "adversarial", "benign"],
            "deberta_pred_binary": ["adversarial", "benign", "benign", "benign"],
            "deberta_proba_binary_adversarial": [0.95, 0.05, 0.40, 0.10],
        })

        metrics = {
            "unseen_val": compute_split_metrics(split_df),
            "unseen_test": compute_split_metrics(split_df),
        }
        summary = generate_summary(metrics)

        assert metrics["unseen_val"]["precision"] == 1.0
        assert metrics["unseen_val"]["recall"] == 0.5
        assert metrics["unseen_val"]["f1"] == pytest.approx(2 / 3)
        assert "| unseen_val |" in summary
        assert "| unseen_test |" in summary
        assert "| Split | Accuracy | Precision | Recall | F1 |" in summary


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

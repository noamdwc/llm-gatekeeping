"""Tests for Colab-oriented DeBERTa CLI overrides and validation."""

from pathlib import Path

import pandas as pd
import pytest

from src.cli import deberta_classifier as cli
from src.models.deberta_classifier import TrainingResult


def _split_df(labels=None):
    labels = labels or ["benign", "adversarial"]
    return pd.DataFrame({
        "modified_sample": [f"text {i}" for i in range(len(labels))],
        "label_binary": labels,
    })


def test_runtime_paths_accept_output_dir_alias(tmp_path):
    args = cli.parse_args([
        "--splits-dir", str(tmp_path / "splits"),
        "--output-dir", str(tmp_path / "artifacts"),
        "--predictions-dir", str(tmp_path / "predictions"),
        "--reports-dir", str(tmp_path / "reports"),
    ])

    paths = cli.resolve_runtime_paths(args)

    assert paths.splits_dir == tmp_path / "splits"
    assert paths.artifacts_dir == tmp_path / "artifacts"
    assert paths.predictions_dir == tmp_path / "predictions"
    assert paths.reports_dir == tmp_path / "reports"


def test_cli_overrides_update_wandb_and_training_config(sample_config_with_deberta):
    args = cli.parse_args([
        "--wandb-project", "colab-project",
        "--wandb-run-name", "colab-run",
        "--num-epochs", "3",
        "--batch-size", "16",
        "--learning-rate", "1e-5",
    ])

    wandb_project, wandb_run_name = cli.resolve_wandb_settings(args)
    cfg = cli.apply_training_overrides(sample_config_with_deberta, args)

    assert wandb_project == "colab-project"
    assert wandb_run_name == "colab-run"
    assert cfg["deberta"]["num_epochs"] == 3
    assert cfg["deberta"]["batch_size"] == 16
    assert cfg["deberta"]["learning_rate"] == 1e-5


def test_requested_unavailable_device_fails_fast(monkeypatch):
    monkeypatch.setattr(cli.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(cli.torch.backends.mps, "is_available", lambda: False)

    with pytest.raises(SystemExit, match="Requested device 'cuda' is not available"):
        cli.resolve_device("cuda", force_cpu=False)


def test_split_validation_requires_required_files_and_columns(tmp_path):
    splits_dir = tmp_path / "splits"
    splits_dir.mkdir()
    _split_df().to_parquet(splits_dir / "train.parquet")

    with pytest.raises(SystemExit, match="Missing required split"):
        cli.validate_split_inputs(splits_dir, text_col="modified_sample",
                                  label_order=["benign", "adversarial"])

    _split_df().drop(columns=["modified_sample"]).to_parquet(splits_dir / "val.parquet")

    with pytest.raises(SystemExit, match="missing required columns"):
        cli.validate_split_inputs(splits_dir, text_col="modified_sample",
                                  label_order=["benign", "adversarial"])


def test_split_validation_rejects_unknown_labels(tmp_path):
    splits_dir = tmp_path / "splits"
    splits_dir.mkdir()
    _split_df().to_parquet(splits_dir / "train.parquet")
    _split_df(["benign", "unknown"]).to_parquet(splits_dir / "val.parquet")

    with pytest.raises(SystemExit, match="invalid label_binary values"):
        cli.validate_split_inputs(splits_dir, text_col="modified_sample",
                                  label_order=["benign", "adversarial"])


def test_run_deberta_accepts_parsed_args(monkeypatch, tmp_path, sample_config_with_deberta):
    args = cli.parse_args([
        "--train-only",
        "--no-wandb",
        "--splits-dir", str(tmp_path / "splits"),
        "--artifacts-dir", str(tmp_path / "artifacts"),
        "--device", "cpu",
    ])
    calls = {}

    class FakeClassifier:
        def __init__(self, cfg):
            calls["cfg"] = cfg

        def train(self, df_train, df_val, **kwargs):
            calls["train_kwargs"] = kwargs
            return TrainingResult(success=True)

        def save(self, output_dir):
            calls["save_dir"] = output_dir

    split_dfs = {
        "train": _split_df(),
        "val": _split_df(),
    }
    monkeypatch.setattr(cli, "load_config", lambda path=None: sample_config_with_deberta)
    monkeypatch.setattr(cli, "validate_split_inputs", lambda *a, **k: split_dfs)
    monkeypatch.setattr(cli, "ensure_writable_dirs", lambda paths: None)
    monkeypatch.setattr(cli, "DeBERTaClassifier", FakeClassifier)

    cli.run_deberta(args)

    assert calls["cfg"]["deberta"]["batch_size"] == sample_config_with_deberta["deberta"]["batch_size"]
    assert calls["train_kwargs"]["device"] == "cpu"
    assert calls["save_dir"] == tmp_path / "artifacts"

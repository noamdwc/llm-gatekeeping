"""Tests for src.build_splits — grouped splitting logic."""

import yaml
import pandas as pd
import pytest

from src.build_splits import build_splits


@pytest.fixture
def splits_input(tmp_path, sample_config, sample_dataframe):
    """
    Write a config YAML and a parquet file to tmp_path for build_splits.
    Returns (config_path, input_path).
    """
    # Override data dir: build_splits writes output to ROOT/data/processed,
    # so we pass input_path directly and patch the save.
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.dump(sample_config))

    input_path = tmp_path / "full_dataset.parquet"
    sample_dataframe.to_parquet(input_path, index=False)

    return str(config_path), str(input_path)


class TestBuildSplits:
    """Tests for build_splits()."""

    def test_no_prompt_hash_overlap(self, splits_input, monkeypatch, tmp_path):
        """No prompt_hash appears in more than one of train/val/test."""
        config_path, input_path = splits_input

        # Patch the save path so parquet files go to tmp_path
        import src.build_splits as mod
        splits_dir = tmp_path / "data" / "processed" / "splits"
        splits_dir.mkdir(parents=True, exist_ok=True)
        monkeypatch.setattr(mod, "DATA_DIR", tmp_path / "data" / "processed")
        monkeypatch.setattr(mod, "SPLITS_DIR", splits_dir)

        splits = build_splits(config_path, input_path)

        train_hashes = set(splits["train"]["prompt_hash"].unique())
        val_hashes = set(splits["val"]["prompt_hash"].unique())
        test_hashes = set(splits["test"]["prompt_hash"].unique())

        assert train_hashes.isdisjoint(val_hashes), "train/val overlap"
        assert train_hashes.isdisjoint(test_hashes), "train/test overlap"
        assert val_hashes.isdisjoint(test_hashes), "val/test overlap"

    def test_held_out_in_test_unseen(self, splits_input, monkeypatch, tmp_path):
        """Held-out attack types end up entirely in test_unseen."""
        config_path, input_path = splits_input

        import src.build_splits as mod
        splits_dir = tmp_path / "data" / "processed" / "splits"
        splits_dir.mkdir(parents=True, exist_ok=True)
        monkeypatch.setattr(mod, "DATA_DIR", tmp_path / "data" / "processed")
        monkeypatch.setattr(mod, "SPLITS_DIR", splits_dir)

        splits = build_splits(config_path, input_path)

        # Homoglyphs is the held-out attack in sample_config
        for name in ["train", "val", "test"]:
            attacks_in_split = splits[name]["attack_name"].unique()
            assert "Homoglyphs" not in attacks_in_split, (
                f"Held-out 'Homoglyphs' leaked into {name}"
            )

        assert "Homoglyphs" in splits["test_unseen"]["attack_name"].values

    def test_all_samples_accounted_for(self, splits_input, monkeypatch, tmp_path, sample_dataframe):
        """Total rows across all splits equals total input rows."""
        config_path, input_path = splits_input

        import src.build_splits as mod
        splits_dir = tmp_path / "data" / "processed" / "splits"
        splits_dir.mkdir(parents=True, exist_ok=True)
        monkeypatch.setattr(mod, "DATA_DIR", tmp_path / "data" / "processed")
        monkeypatch.setattr(mod, "SPLITS_DIR", splits_dir)

        splits = build_splits(config_path, input_path)
        total = sum(len(s) for s in splits.values())
        assert total == len(sample_dataframe)

    def test_reproducible_with_same_seed(self, splits_input, monkeypatch, tmp_path):
        """Running twice with the same seed gives identical splits."""
        config_path, input_path = splits_input

        import src.build_splits as mod
        splits_dir = tmp_path / "data" / "processed" / "splits"
        splits_dir.mkdir(parents=True, exist_ok=True)
        monkeypatch.setattr(mod, "DATA_DIR", tmp_path / "data" / "processed")
        monkeypatch.setattr(mod, "SPLITS_DIR", splits_dir)

        splits1 = build_splits(config_path, input_path)
        splits2 = build_splits(config_path, input_path)

        for name in ["train", "val", "test", "test_unseen"]:
            pd.testing.assert_frame_equal(
                splits1[name].reset_index(drop=True),
                splits2[name].reset_index(drop=True),
            )

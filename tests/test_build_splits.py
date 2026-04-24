"""Tests for src.build_splits — grouped splitting logic with unseen_val + unseen_test."""

import yaml
import pandas as pd
import pytest

from src.build_splits import build_splits


@pytest.fixture
def splits_input(tmp_path, sample_config, sample_dataframe):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.dump(sample_config))
    input_path = tmp_path / "full_dataset.parquet"
    sample_dataframe.to_parquet(input_path, index=False)
    return str(config_path), str(input_path)


def _patch_splits_dir(monkeypatch, tmp_path):
    import src.build_splits as mod
    splits_dir = tmp_path / "data" / "processed" / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(mod, "DATA_DIR", tmp_path / "data" / "processed")
    monkeypatch.setattr(mod, "SPLITS_DIR", splits_dir)


class TestBuildSplits:
    """Tests for build_splits() with unseen_val + unseen_test."""

    def test_produces_five_splits(self, splits_input, monkeypatch, tmp_path):
        _patch_splits_dir(monkeypatch, tmp_path)
        config_path, input_path = splits_input
        splits = build_splits(config_path, input_path)
        assert set(splits.keys()) == {"train", "val", "test", "unseen_val", "unseen_test"}

    def test_no_prompt_hash_overlap_within_groups(self, splits_input, monkeypatch, tmp_path):
        """Disjoint within main (train/val/test) and within unseen (unseen_val/unseen_test).

        Cross-group overlap (main <-> unseen) is allowed by design: the same
        prompt_hash can appear in train as a non-held-out attack variant and
        in unseen_val as a held-out attack variant. This matches the
        pre-existing test_unseen behavior.
        """
        _patch_splits_dir(monkeypatch, tmp_path)
        config_path, input_path = splits_input
        splits = build_splits(config_path, input_path)

        for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
            ha = set(splits[a]["prompt_hash"].unique())
            hb = set(splits[b]["prompt_hash"].unique())
            assert ha.isdisjoint(hb), f"{a}/{b} prompt_hash overlap (main group)"

        hv = set(splits["unseen_val"]["prompt_hash"].unique())
        ht = set(splits["unseen_test"]["prompt_hash"].unique())
        assert hv.isdisjoint(ht), "unseen_val/unseen_test prompt_hash overlap"

    def test_held_out_attacks_only_in_unseen_splits(self, splits_input, monkeypatch, tmp_path):
        _patch_splits_dir(monkeypatch, tmp_path)
        config_path, input_path = splits_input
        splits = build_splits(config_path, input_path)

        held_out = {"BAE"}
        for name in ["train", "val", "test"]:
            attacks = set(splits[name]["attack_name"].unique())
            assert attacks.isdisjoint(held_out), (
                f"held-out attack leaked into {name}: {attacks & held_out}"
            )
        for name in ["unseen_val", "unseen_test"]:
            adv_rows = splits[name][splits[name]["label_binary"] == "adversarial"]
            adv_attacks = set(adv_rows["attack_name"].unique())
            assert adv_attacks.issubset(held_out), (
                f"non-held-out attack in {name}: {adv_attacks - held_out}"
            )

    def test_unseen_splits_stratified_per_attack(self, splits_input, monkeypatch, tmp_path):
        _patch_splits_dir(monkeypatch, tmp_path)
        config_path, input_path = splits_input
        splits = build_splits(config_path, input_path)

        for attack in {"BAE"}:
            v_hashes = set(splits["unseen_val"].query("attack_name == @attack")["prompt_hash"])
            t_hashes = set(splits["unseen_test"].query("attack_name == @attack")["prompt_hash"])
            assert len(v_hashes) >= 1, f"{attack} missing from unseen_val"
            assert len(t_hashes) >= 1, f"{attack} missing from unseen_test"

    def test_unseen_splits_contain_benigns(self, splits_input, monkeypatch, tmp_path):
        _patch_splits_dir(monkeypatch, tmp_path)
        config_path, input_path = splits_input
        splits = build_splits(config_path, input_path)
        for name in ["unseen_val", "unseen_test"]:
            n_benign = (splits[name]["label_binary"] == "benign").sum()
            assert n_benign >= 1, f"{name} has no benign rows"

    def test_benign_rows_do_not_overlap_across_splits(self, splits_input, monkeypatch, tmp_path):
        """Every benign prompt_hash lives in exactly one split.

        Unlike adversarial rows (whose hashes may cross main <-> unseen by
        design), benign rows are assigned to a single split so FPR is
        measurable without double-counting.
        """
        _patch_splits_dir(monkeypatch, tmp_path)
        config_path, input_path = splits_input
        splits = build_splits(config_path, input_path)
        b_hashes = {}
        for name in ["train", "val", "test", "unseen_val", "unseen_test"]:
            b = splits[name][splits[name]["label_binary"] == "benign"]
            b_hashes[name] = set(b["prompt_hash"].unique())
        names = list(b_hashes)
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                assert b_hashes[a].isdisjoint(b_hashes[b]), f"benign overlap {a}/{b}"

    def test_all_samples_accounted_for(self, splits_input, monkeypatch, tmp_path, sample_dataframe):
        _patch_splits_dir(monkeypatch, tmp_path)
        config_path, input_path = splits_input
        splits = build_splits(config_path, input_path)
        total = sum(len(s) for s in splits.values())
        assert total == len(sample_dataframe)

    def test_reproducible_with_same_seed(self, splits_input, monkeypatch, tmp_path):
        _patch_splits_dir(monkeypatch, tmp_path)
        config_path, input_path = splits_input
        splits1 = build_splits(config_path, input_path)
        splits2 = build_splits(config_path, input_path)
        for name in ["train", "val", "test", "unseen_val", "unseen_test"]:
            pd.testing.assert_frame_equal(
                splits1[name].reset_index(drop=True),
                splits2[name].reset_index(drop=True),
            )

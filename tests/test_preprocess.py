"""Tests for src.preprocess — label hierarchy, prompt hashing, benign set."""

import pandas as pd
import pytest

from src.preprocess import (
    add_hierarchical_labels,
    add_hierarchical_labels_benign,
    build_benign_set,
    build_prompt_hash,
)


# ---------------------------------------------------------------------------
# build_prompt_hash
# ---------------------------------------------------------------------------
class TestBuildPromptHash:
    """Tests for build_prompt_hash()."""

    def test_deterministic(self):
        """Same input always produces the same hash."""
        h1 = build_prompt_hash("hello world")
        h2 = build_prompt_hash("hello world")
        assert h1 == h2

    def test_case_insensitive(self):
        """Hash is case-insensitive."""
        assert build_prompt_hash("Hello World") == build_prompt_hash("hello world")

    def test_whitespace_stripped(self):
        """Leading/trailing whitespace is stripped before hashing."""
        assert build_prompt_hash("  hello world  ") == build_prompt_hash("hello world")

    def test_different_texts_differ(self):
        """Different texts produce different hashes."""
        assert build_prompt_hash("hello") != build_prompt_hash("world")

    def test_returns_12_char_hex(self):
        """Hash is a 12-character hex string."""
        h = build_prompt_hash("test")
        assert len(h) == 12
        assert all(c in "0123456789abcdef" for c in h)

    def test_empty_string(self):
        """Empty string produces a valid hash."""
        h = build_prompt_hash("")
        assert len(h) == 12


# ---------------------------------------------------------------------------
# add_hierarchical_labels
# ---------------------------------------------------------------------------
class TestAddHierarchicalLabels:
    """Tests for add_hierarchical_labels()."""

    def _make_df(self, attack_names):
        return pd.DataFrame({"attack_name": attack_names})

    def test_unicode_attack_labels(self, sample_config):
        """Unicode attacks get correct binary/category/type labels."""
        df = self._make_df(["Diacritcs", "Zero Width"])
        result = add_hierarchical_labels(df, sample_config)

        assert (result["label_binary"] == "adversarial").all()
        assert (result["label_category"] == "unicode_attack").all()
        assert list(result["label_type"]) == ["Diacritcs", "Zero Width"]

    def test_nlp_attack_labels(self, sample_config):
        """NLP attacks get correct labels with collapsed type."""
        df = self._make_df(["BAE", "TextFooler"])
        result = add_hierarchical_labels(df, sample_config)

        assert (result["label_binary"] == "adversarial").all()
        assert (result["label_category"] == "nlp_attack").all()
        assert (result["label_type"] == "nlp_attack").all()

    def test_unknown_attack_category(self, sample_config):
        """An attack not in unicode or nlp lists gets 'unknown' category."""
        df = self._make_df(["SomeNewAttack"])
        result = add_hierarchical_labels(df, sample_config)

        assert result["label_binary"].iloc[0] == "adversarial"
        assert result["label_category"].iloc[0] == "unknown"

    def test_does_not_modify_original(self, sample_config):
        """Original DataFrame is not modified in place."""
        df = self._make_df(["Diacritcs"])
        _ = add_hierarchical_labels(df, sample_config)
        assert "label_binary" not in df.columns


# ---------------------------------------------------------------------------
# add_hierarchical_labels_benign
# ---------------------------------------------------------------------------
class TestAddHierarchicalLabelsBenign:
    """Tests for add_hierarchical_labels_benign()."""

    def test_all_columns_set_to_benign(self):
        """All label columns are set to 'benign'."""
        df = pd.DataFrame({"text": ["hello"], "label_binary": ["adversarial"]})
        result = add_hierarchical_labels_benign(df)

        assert result["label_binary"].iloc[0] == "benign"
        assert result["label_category"].iloc[0] == "benign"
        assert result["label_type"].iloc[0] == "benign"

    def test_does_not_modify_original(self):
        """Original DataFrame is not modified."""
        df = pd.DataFrame({"label_binary": ["adversarial"]})
        _ = add_hierarchical_labels_benign(df)
        assert df["label_binary"].iloc[0] == "adversarial"


# ---------------------------------------------------------------------------
# build_benign_set
# ---------------------------------------------------------------------------
class TestBuildBenignSet:
    """Tests for build_benign_set()."""

    def test_returns_target_count(self, sample_config):
        """Output has at most target_count rows."""
        df = pd.DataFrame({
            "modified_sample": ["a", "b", "c"],
            "original_sample": ["x", "y", "z"],
            "attack_name": ["atk"] * 3,
        })
        result = build_benign_set(df, sample_config)
        assert len(result) == sample_config["benign"]["target_count"]

    def test_deduplicates_originals(self, sample_config):
        """Duplicate original_sample values are collapsed."""
        df = pd.DataFrame({
            "modified_sample": ["a", "b", "c"],
            "original_sample": ["same", "same", "same"],
            "attack_name": ["atk"] * 3,
        })
        result = build_benign_set(df, sample_config)
        # All rows derive from the single unique original
        assert result["original_sample"].iloc[0] == "same"

    def test_benign_labels_assigned(self, sample_config):
        """All rows have label_binary = 'benign'."""
        df = pd.DataFrame({
            "modified_sample": ["a", "b"],
            "original_sample": ["x", "y"],
            "attack_name": ["atk"] * 2,
        })
        result = build_benign_set(df, sample_config)
        assert (result["label_binary"] == "benign").all()
        assert (result["label_category"] == "benign").all()
        assert (result["label_type"] == "benign").all()

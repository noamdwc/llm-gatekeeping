"""Tests for src.llm_classifier — LLM classification with mocked API calls."""

import json
from collections import defaultdict
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.llm_classifier import (
    HierarchicalLLMClassifier,
    UsageStats,
    build_few_shot_examples,
)


# ---------------------------------------------------------------------------
# UsageStats
# ---------------------------------------------------------------------------
class TestUsageStats:
    """Tests for the UsageStats dataclass."""

    def test_initial_values(self):
        stats = UsageStats()
        assert stats.total_calls == 0
        assert stats.total_tokens == 0
        assert stats.avg_latency_s == 0.0

    def test_total_tokens(self):
        stats = UsageStats(prompt_tokens=100, completion_tokens=50)
        assert stats.total_tokens == 150

    def test_avg_latency(self):
        stats = UsageStats(total_calls=4, total_latency_s=2.0)
        assert stats.avg_latency_s == 0.5

    def test_avg_latency_zero_calls(self):
        stats = UsageStats(total_calls=0, total_latency_s=0.0)
        assert stats.avg_latency_s == 0.0

    def test_to_dict_keys(self):
        stats = UsageStats(total_calls=1, prompt_tokens=10, completion_tokens=5)
        d = stats.to_dict()
        assert "total_calls" in d
        assert "total_tokens" in d
        assert "avg_latency_s" in d
        assert "calls_by_stage" in d
        assert d["total_tokens"] == 15


# ---------------------------------------------------------------------------
# build_few_shot_examples
# ---------------------------------------------------------------------------
class TestBuildFewShotExamples:
    """Tests for build_few_shot_examples()."""

    def test_returns_dict_and_used_ids(self, sample_config, sample_dataframe):
        few_shot, used_ids = build_few_shot_examples(sample_dataframe, sample_config)
        assert isinstance(few_shot, dict)
        assert isinstance(used_ids, list)

    def test_samples_per_type(self, sample_config, sample_dataframe):
        """Each type gets at most n_unicode or n_nlp samples."""
        few_shot, _ = build_few_shot_examples(sample_dataframe, sample_config)
        n_unicode = sample_config["llm"]["few_shot"]["unicode"]
        n_nlp = sample_config["llm"]["few_shot"]["nlp"]

        for attack_type, samples in few_shot.items():
            if attack_type in sample_config["labels"]["unicode_attacks"]:
                assert len(samples) <= n_unicode
            else:
                assert len(samples) <= n_nlp

    def test_empty_pool_skipped(self, sample_config):
        """Attack types with no training data are skipped."""
        df = pd.DataFrame({
            "modified_sample": ["hello"],
            "attack_name": ["Diacritcs"],
        })
        few_shot, _ = build_few_shot_examples(df, sample_config)
        # "BAE" and "TextFooler" have no rows → should not appear
        assert "BAE" not in few_shot or len(few_shot.get("BAE", [])) == 0


# ---------------------------------------------------------------------------
# HierarchicalLLMClassifier (mocked)
# ---------------------------------------------------------------------------
def _make_classifier(cfg, few_shot=None):
    """Create a classifier with a mocked OpenAI client."""
    with patch("src.llm_classifier.openai.OpenAI"):
        classifier = HierarchicalLLMClassifier(
            cfg, few_shot_examples=few_shot or {}
        )
    return classifier


def _mock_call_llm_response(label, confidence=0.95):
    """Return a dict as if parsed from LLM JSON response."""
    return {"label": label, "confidence": confidence}


class TestClassifyBinary:
    """Tests for classify_binary() with mocked _call_llm."""

    def test_adversarial(self, sample_config):
        clf = _make_classifier(sample_config)
        clf._call_llm = MagicMock(
            return_value=_mock_call_llm_response("adversarial", 0.95)
        )
        result = clf.classify_binary("some text")
        assert result["label"] == "adversarial"
        assert result["confidence"] == 0.95

    def test_benign(self, sample_config):
        clf = _make_classifier(sample_config)
        clf._call_llm = MagicMock(
            return_value=_mock_call_llm_response("benign", 0.99)
        )
        result = clf.classify_binary("normal text")
        assert result["label"] == "benign"
        assert result["confidence"] == 0.99

    def test_empty_response_defaults(self, sample_config):
        """Empty LLM response defaults to adversarial with 0.5 confidence."""
        clf = _make_classifier(sample_config)
        clf._call_llm = MagicMock(return_value={})
        result = clf.classify_binary("text")
        assert result["label"] == "adversarial"
        assert result["confidence"] == 0.5


class TestClassifyCategory:
    """Tests for classify_category() with mocked _call_llm."""

    def test_unicode_attack(self, sample_config):
        clf = _make_classifier(sample_config)
        clf._call_llm = MagicMock(
            return_value=_mock_call_llm_response("unicode_attack", 0.9)
        )
        result = clf.classify_category("text")
        assert result["label"] == "unicode_attack"

    def test_nlp_attack(self, sample_config):
        clf = _make_classifier(sample_config)
        clf._call_llm = MagicMock(
            return_value=_mock_call_llm_response("nlp_attack", 0.85)
        )
        result = clf.classify_category("text")
        assert result["label"] == "nlp_attack"

    def test_invalid_label_defaults_to_nlp(self, sample_config):
        """Invalid label falls back to nlp_attack."""
        clf = _make_classifier(sample_config)
        clf._call_llm = MagicMock(
            return_value=_mock_call_llm_response("something_else", 0.5)
        )
        result = clf.classify_category("text")
        assert result["label"] == "nlp_attack"


class TestClassifyType:
    """Tests for classify_type() with mocked _call_llm."""

    def test_valid_unicode_type(self, sample_config):
        clf = _make_classifier(sample_config)
        clf._call_llm = MagicMock(
            return_value=_mock_call_llm_response("Diacritcs", 0.9)
        )
        result = clf.classify_type("text")
        assert result["label"] == "Diacritcs"

    def test_unknown_type_defaults(self, sample_config):
        """Invalid type label falls back to 'unknown'."""
        clf = _make_classifier(sample_config)
        clf._call_llm = MagicMock(
            return_value=_mock_call_llm_response("not_a_real_type", 0.5)
        )
        result = clf.classify_type("text")
        assert result["label"] == "unknown"


class TestPredict:
    """Tests for the full predict() pipeline with mocked stages."""

    def test_benign_skips_stages_1_2(self, sample_config):
        """Benign classification skips category and type stages."""
        clf = _make_classifier(sample_config)
        clf.classify_binary = MagicMock(
            return_value={"label": "benign", "confidence": 0.99}
        )
        clf.classify_category = MagicMock()
        clf.classify_type = MagicMock()

        result = clf.predict("normal text")

        assert result["label_binary"] == "benign"
        assert result["label_category"] == "benign"
        assert result["label_type"] == "benign"
        clf.classify_category.assert_not_called()
        clf.classify_type.assert_not_called()

    def test_adversarial_unicode_full_pipeline(self, sample_config):
        """Adversarial + unicode_attack → all three stages called."""
        clf = _make_classifier(sample_config)
        clf.classify_binary = MagicMock(
            return_value={"label": "adversarial", "confidence": 0.95}
        )
        clf.classify_category = MagicMock(
            return_value={"label": "unicode_attack", "confidence": 0.9}
        )
        clf.classify_type = MagicMock(
            return_value={"label": "Diacritcs", "confidence": 0.85}
        )

        result = clf.predict("héllö")

        assert result["label_binary"] == "adversarial"
        assert result["label_category"] == "unicode_attack"
        assert result["label_type"] == "Diacritcs"
        clf.classify_binary.assert_called_once()
        clf.classify_category.assert_called_once()
        clf.classify_type.assert_called_once()

    def test_adversarial_nlp_skips_type(self, sample_config):
        """Adversarial + nlp_attack → type stage skipped, type='nlp_attack'."""
        clf = _make_classifier(sample_config)
        clf.classify_binary = MagicMock(
            return_value={"label": "adversarial", "confidence": 0.95}
        )
        clf.classify_category = MagicMock(
            return_value={"label": "nlp_attack", "confidence": 0.88}
        )
        clf.classify_type = MagicMock()

        result = clf.predict("greetings earth")

        assert result["label_binary"] == "adversarial"
        assert result["label_category"] == "nlp_attack"
        assert result["label_type"] == "nlp_attack"
        clf.classify_type.assert_not_called()

    def test_confidence_propagation(self, sample_config):
        """Confidence values from each stage appear in final result."""
        clf = _make_classifier(sample_config)
        clf.classify_binary = MagicMock(
            return_value={"label": "adversarial", "confidence": 0.91}
        )
        clf.classify_category = MagicMock(
            return_value={"label": "unicode_attack", "confidence": 0.82}
        )
        clf.classify_type = MagicMock(
            return_value={"label": "Zero Width", "confidence": 0.73}
        )

        result = clf.predict("text")
        assert result["confidence_binary"] == 0.91
        assert result["confidence_category"] == 0.82
        assert result["confidence_type"] == 0.73


class TestDynamicFewShot:
    """Tests for dynamic=True requiring an ExemplarBank."""

    def test_dynamic_without_bank_raises(self, sample_config):
        """Creating classifier with dynamic=True but no bank raises ValueError."""
        with patch("src.llm_classifier.openai.OpenAI"):
            with pytest.raises(ValueError, match="ExemplarBank required"):
                HierarchicalLLMClassifier(
                    sample_config, dynamic=True, exemplar_bank=None
                )

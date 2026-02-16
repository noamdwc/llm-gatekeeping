"""Tests for src.llm_classifier — LLM classification with mocked API calls."""

import json
from collections import defaultdict
from unittest.mock import MagicMock, patch, call

import pandas as pd
import pytest

from src.llm_classifier.llm_classifier import (
    HierarchicalLLMClassifier,
    UsageStats,
    build_few_shot_examples,
)
from src.llm_classifier.constants import UNICODE_TYPES


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
# Helper: create classifier with mocked OpenAI client
# ---------------------------------------------------------------------------
def _make_classifier(cfg, few_shot=None):
    """Create a classifier with a mocked OpenAI client."""
    with patch("src.llm_classifier.llm_classifier.openai.OpenAI"):
        classifier = HierarchicalLLMClassifier(
            cfg, few_shot_examples=few_shot or {}
        )
    return classifier


def _mock_call_llm_response(label, confidence=0.95, reasoning=""):
    """Return a dict as if parsed from LLM JSON response."""
    d = {"label": label, "confidence": confidence}
    if reasoning:
        d["reasoning"] = reasoning
    return d


# ---------------------------------------------------------------------------
# TestClassify
# ---------------------------------------------------------------------------
class TestClassify:
    """Tests for classify() — single-call classifier."""

    def test_benign(self, sample_config):
        clf = _make_classifier(sample_config)
        clf._call_llm = MagicMock(
            return_value=_mock_call_llm_response("benign", 0.99)
        )
        result = clf.classify("normal text")
        assert result["label"] == "benign"
        assert result["confidence"] == 0.99

    def test_unicode_type(self, sample_config):
        clf = _make_classifier(sample_config)
        clf._call_llm = MagicMock(
            return_value=_mock_call_llm_response("Diacritcs", 0.9)
        )
        result = clf.classify("héllö wörld")
        assert result["label"] == "Diacritcs"
        assert result["confidence"] == 0.9

    def test_nlp_attack(self, sample_config):
        clf = _make_classifier(sample_config)
        clf._call_llm = MagicMock(
            return_value=_mock_call_llm_response("nlp_attack", 0.85)
        )
        result = clf.classify("greetings earth")
        assert result["label"] == "nlp_attack"
        assert result["confidence"] == 0.85

    def test_unknown_label_defaults_to_nlp_attack(self, sample_config):
        """Unknown label from LLM defaults to nlp_attack."""
        clf = _make_classifier(sample_config)
        clf._call_llm = MagicMock(
            return_value=_mock_call_llm_response("something_weird", 0.6)
        )
        result = clf.classify("text")
        assert result["label"] == "nlp_attack"

    def test_empty_response_defaults(self, sample_config):
        """Empty LLM response defaults to nlp_attack with 0.5 confidence."""
        clf = _make_classifier(sample_config)
        clf._call_llm = MagicMock(return_value={})
        result = clf.classify("text")
        assert result["label"] == "nlp_attack"
        assert result["confidence"] == 0.5

    def test_calls_with_classifier_stage(self, sample_config):
        """Verify _call_llm is called with stage='classifier'."""
        clf = _make_classifier(sample_config)
        clf._call_llm = MagicMock(
            return_value=_mock_call_llm_response("benign", 0.9)
        )
        clf.classify("text")
        args, kwargs = clf._call_llm.call_args
        assert args[2] == "classifier"  # stage parameter


# ---------------------------------------------------------------------------
# TestJudge
# ---------------------------------------------------------------------------
class TestJudge:
    """Tests for judge() — conditional higher-quality review."""

    def test_overrides_classifier(self, sample_config):
        """Judge can change the classifier's prediction."""
        clf = _make_classifier(sample_config)
        clf._call_llm = MagicMock(
            return_value=_mock_call_llm_response("Homoglyphs", 0.95, "Cyrillic chars detected")
        )
        classifier_output = {"label": "nlp_attack", "confidence": 0.5}
        result = clf.judge("hеllo", classifier_output)
        assert result["label"] == "Homoglyphs"
        assert result["confidence"] == 0.95
        assert result["reasoning"] == "Cyrillic chars detected"

    def test_invalid_label_falls_back_to_classifier(self, sample_config):
        """Invalid judge label falls back to classifier's prediction."""
        clf = _make_classifier(sample_config)
        clf._call_llm = MagicMock(
            return_value=_mock_call_llm_response("garbage_label", 0.8)
        )
        classifier_output = {"label": "Diacritcs", "confidence": 0.6}
        result = clf.judge("text", classifier_output)
        assert result["label"] == "Diacritcs"
        assert result["confidence"] == 0.6

    def test_uses_quality_model(self, sample_config):
        """Judge calls _call_llm with model_quality."""
        clf = _make_classifier(sample_config)
        clf._call_llm = MagicMock(
            return_value=_mock_call_llm_response("benign", 0.95)
        )
        classifier_output = {"label": "nlp_attack", "confidence": 0.5}
        clf.judge("text", classifier_output)
        _, kwargs = clf._call_llm.call_args
        assert kwargs["model"] == "gpt-4o"

    def test_empty_response_falls_back(self, sample_config):
        """Empty judge response falls back to classifier prediction."""
        clf = _make_classifier(sample_config)
        clf._call_llm = MagicMock(return_value={})
        classifier_output = {"label": "Zero Width", "confidence": 0.65}
        result = clf.judge("text", classifier_output)
        # Empty response → label="" which is invalid → falls back
        assert result["label"] == "Zero Width"
        assert result["confidence"] == 0.65

    def test_calls_with_judge_stage(self, sample_config):
        """Verify _call_llm is called with stage='judge'."""
        clf = _make_classifier(sample_config)
        clf._call_llm = MagicMock(
            return_value=_mock_call_llm_response("benign", 0.9)
        )
        clf.judge("text", {"label": "nlp_attack", "confidence": 0.5})
        args, kwargs = clf._call_llm.call_args
        assert args[2] == "judge"


# ---------------------------------------------------------------------------
# TestDeriveCategory
# ---------------------------------------------------------------------------
class TestDeriveCategory:
    """Tests for _derive_category() — pure function."""

    def test_benign(self):
        assert HierarchicalLLMClassifier._derive_category("benign", "benign") == "benign"

    def test_unicode_type(self):
        for utype in UNICODE_TYPES:
            assert HierarchicalLLMClassifier._derive_category("adversarial", utype) == "unicode_attack"

    def test_nlp_attack(self):
        assert HierarchicalLLMClassifier._derive_category("adversarial", "nlp_attack") == "nlp_attack"

    def test_unknown_adversarial_type(self):
        """Unknown adversarial type defaults to nlp_attack category."""
        assert HierarchicalLLMClassifier._derive_category("adversarial", "unknown") == "nlp_attack"


# ---------------------------------------------------------------------------
# TestPredict
# ---------------------------------------------------------------------------
class TestPredict:
    """Tests for the full predict() pipeline with mocked classify/judge."""

    def test_high_confidence_skips_judge(self, sample_config):
        """High-confidence classifier result skips judge."""
        clf = _make_classifier(sample_config)
        clf.classify = MagicMock(
            return_value={"label": "Diacritcs", "confidence": 0.95}
        )
        clf.judge = MagicMock()

        result = clf.predict("héllö")

        clf.classify.assert_called_once()
        clf.judge.assert_not_called()
        assert result["llm_stages_run"] == 1

    def test_low_confidence_triggers_judge(self, sample_config):
        """Low-confidence classifier result triggers judge."""
        clf = _make_classifier(sample_config)
        clf.classify = MagicMock(
            return_value={"label": "nlp_attack", "confidence": 0.5}
        )
        clf.judge = MagicMock(
            return_value={"label": "Homoglyphs", "confidence": 0.9, "reasoning": ""}
        )

        result = clf.predict("text")

        clf.classify.assert_called_once()
        clf.judge.assert_called_once()
        assert result["llm_stages_run"] == 2
        assert result["label_type"] == "Homoglyphs"
        assert result["label_binary"] == "adversarial"
        assert result["label_category"] == "unicode_attack"

    def test_force_all_stages(self, sample_config):
        """force_all_stages=True always runs judge even with high confidence."""
        clf = _make_classifier(sample_config)
        clf.classify = MagicMock(
            return_value={"label": "benign", "confidence": 0.99}
        )
        clf.judge = MagicMock(
            return_value={"label": "benign", "confidence": 0.98, "reasoning": ""}
        )

        result = clf.predict("normal text", force_all_stages=True)

        clf.judge.assert_called_once()
        assert result["llm_stages_run"] == 2

    def test_output_contract_keys(self, sample_config):
        """Predict output has all required keys."""
        clf = _make_classifier(sample_config)
        clf.classify = MagicMock(
            return_value={"label": "benign", "confidence": 0.95}
        )

        result = clf.predict("text")

        expected_keys = {
            "label_binary", "label_category", "label_type",
            "confidence_binary", "confidence_category", "confidence_type",
            "llm_stages_run",
        }
        assert set(result.keys()) == expected_keys

    def test_benign_prediction(self, sample_config):
        """Benign classifier output produces correct labels."""
        clf = _make_classifier(sample_config)
        clf.classify = MagicMock(
            return_value={"label": "benign", "confidence": 0.95}
        )

        result = clf.predict("hello world")

        assert result["label_binary"] == "benign"
        assert result["label_category"] == "benign"
        assert result["label_type"] == "benign"

    def test_category_derived_correctly_unicode(self, sample_config):
        """Unicode type → category = unicode_attack."""
        clf = _make_classifier(sample_config)
        clf.classify = MagicMock(
            return_value={"label": "Zero Width", "confidence": 0.9}
        )

        result = clf.predict("text")

        assert result["label_binary"] == "adversarial"
        assert result["label_category"] == "unicode_attack"
        assert result["label_type"] == "Zero Width"

    def test_category_derived_correctly_nlp(self, sample_config):
        """nlp_attack type → category = nlp_attack."""
        clf = _make_classifier(sample_config)
        clf.classify = MagicMock(
            return_value={"label": "nlp_attack", "confidence": 0.85}
        )

        result = clf.predict("greetings earth")

        assert result["label_binary"] == "adversarial"
        assert result["label_category"] == "nlp_attack"
        assert result["label_type"] == "nlp_attack"

    def test_judge_result_used_when_triggered(self, sample_config):
        """When judge is triggered, its label/confidence are used."""
        clf = _make_classifier(sample_config)
        clf.classify = MagicMock(
            return_value={"label": "nlp_attack", "confidence": 0.4}
        )
        clf.judge = MagicMock(
            return_value={"label": "benign", "confidence": 0.92, "reasoning": ""}
        )

        result = clf.predict("text")

        assert result["label_binary"] == "benign"
        assert result["label_category"] == "benign"
        assert result["label_type"] == "benign"
        assert result["confidence_binary"] == 0.92


# ---------------------------------------------------------------------------
# TestDynamicFewShot
# ---------------------------------------------------------------------------
class TestDynamicFewShot:
    """Tests for dynamic=True requiring an ExemplarBank."""

    def test_dynamic_without_bank_raises(self, sample_config):
        """Creating classifier with dynamic=True but no bank raises ValueError."""
        with patch("src.llm_classifier.llm_classifier.openai.OpenAI"):
            with pytest.raises(ValueError, match="ExemplarBank required"):
                HierarchicalLLMClassifier(
                    sample_config, dynamic=True, exemplar_bank=None
                )

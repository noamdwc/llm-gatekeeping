"""Tests for src.hybrid_router — routing logic with mocked ML + LLM."""

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from src.hybrid_router import HybridRouter, RouterStats, threshold_sweep


# ---------------------------------------------------------------------------
# RouterStats
# ---------------------------------------------------------------------------
class TestRouterStats:
    """Tests for the RouterStats dataclass."""

    def test_initial_rates_zero(self):
        stats = RouterStats()
        assert stats.ml_rate == 0.0
        assert stats.llm_rate == 0.0
        assert stats.abstain_rate == 0.0

    def test_rates_sum_to_one(self):
        # ml_handled + llm_escalated + abstained must equal total (mutually exclusive)
        stats = RouterStats(total=10, ml_handled=5, llm_escalated=3, abstained=2)
        total = stats.ml_rate + stats.llm_rate + stats.abstain_rate
        assert abs(total - 1.0) < 1e-9

    def test_rates_sum_to_one_with_all_abstained(self):
        """All escalated samples abstaining: llm_escalated=0, abstained=5."""
        stats = RouterStats(total=10, ml_handled=5, llm_escalated=0, abstained=5)
        total = stats.ml_rate + stats.llm_rate + stats.abstain_rate
        assert abs(total - 1.0) < 1e-9

    def test_to_dict_keys(self):
        stats = RouterStats(total=1, ml_handled=1)
        d = stats.to_dict()
        expected_keys = {
            "total", "ml_handled", "llm_escalated", "abstained",
            "ml_rate", "llm_rate", "abstain_rate",
        }
        assert expected_keys == set(d.keys())


# ---------------------------------------------------------------------------
# HybridRouter.predict_single
# ---------------------------------------------------------------------------
class TestPredictSingle:
    """Tests for HybridRouter.predict_single()."""

    @pytest.fixture
    def router(self, sample_config):
        """Router with mocked ML model and LLM classifier."""
        ml_mock = MagicMock()
        llm_mock = MagicMock()
        return HybridRouter(ml_mock, llm_mock, sample_config)

    def test_high_ml_confidence_uses_ml(self, router):
        """High-confidence ML adversarial prediction uses ML fast path."""
        ml_pred = {
            "pred_label_binary": "adversarial",
            "confidence_label_binary": 0.95,
            "confidence_label_binary_cal": 0.95,
            "pred_label_category": "unicode_attack",
            "confidence_label_category": 0.90,
            "pred_label_type": "Diacritcs",
            "confidence_label_type": 0.85,
        }
        result = router.predict_single("text", ml_pred)

        assert result["routed_to"] == "ml"
        assert result["route_reason"] == "ml_adv_high_conf_unicode_finalize"
        assert result["label_binary"] == "adversarial"
        assert result["label_category"] == "unicode_attack"
        assert result["label_type"] == "Diacritcs"
        router.llm.predict.assert_not_called()

    def test_adversarial_label_matching_is_case_insensitive(self, router):
        """Adversarial label variants should still be treated as adversarial."""
        ml_pred = {
            "pred_label_binary": "AdVerSarial",
            "confidence_label_binary": 0.95,
            "pred_label_category": "unicode_attack",
        }
        result = router.predict_single("text", ml_pred)
        assert result["routed_to"] == "ml"
        assert result["route_reason"] == "ml_adv_high_conf_unicode_finalize"

    def test_low_ml_confidence_escalates_to_llm(self, router):
        """Low-confidence ML adversarial prediction escalates to LLM."""
        ml_pred = {
            "pred_label_binary": "adversarial",
            "confidence_label_binary": 0.5,  # below 0.85 threshold
            "pred_label_category": "unicode_attack",
        }
        router.llm.predict.return_value = {
            "label": "adversarial",
            "label_category": "nlp_attack",
            "confidence": 0.9,
        }

        result = router.predict_single("text", ml_pred)

        assert result["routed_to"] == "llm"
        assert result["route_reason"] == "ml_adv_low_conf_escalate"
        assert result["label_binary"] == "adversarial"
        assert result["label_category"] == "nlp_attack"  # LLM category preferred over ML's
        router.llm.predict.assert_called_once_with("text")

    def test_low_llm_confidence_abstains(self, router):
        """Low ML + low LLM confidence → routed_to='abstain'."""
        ml_pred = {
            "pred_label_binary": "adversarial",
            "confidence_label_binary": 0.5,
            "pred_label_category": "unicode_attack",
        }
        router.llm.predict.return_value = {
            "label": "benign",
            "confidence": 0.3,  # below 0.7 threshold
        }

        result = router.predict_single("text", ml_pred)
        assert result["routed_to"] == "abstain"
        assert result["route_reason"] == "ml_adv_low_conf_escalate"
        assert result["label"] == "uncertain"
        assert result["label_binary"] == "adversarial"

    def test_benign_high_confidence_always_escalates(self, router):
        """Benign predictions are never finalized by ML."""
        ml_pred = {
            "pred_label_binary": "benign",
            "confidence_label_binary": 0.99,
            "confidence_label_binary_cal": 0.99,
        }
        router.llm.predict.return_value = {
            "label_binary": "benign",
            "confidence": 0.92,
        }

        result = router.predict_single("text", ml_pred)

        assert result["routed_to"] == "llm"
        assert result["route_reason"] == "ml_benign_escalate"
        router.llm.predict.assert_called_once_with("text")

    def test_stats_tracking(self, router):
        """Stats are incremented correctly for each routing path."""
        router.llm.predict.return_value = {
            "label": "benign",
            "confidence": 0.9,
        }
        # Benign prediction escalates to LLM
        router.predict_single("t1", {
            "pred_label_binary": "benign",
            "confidence_label_binary": 0.99,
        })

        # Route via LLM
        router.llm.predict.return_value = {
            "label": "adversarial",
            "confidence": 0.9,
        }
        router.predict_single("t2", {
            "pred_label_binary": "adversarial",
            "confidence_label_binary": 0.4,
            "pred_label_category": "unicode_attack",
        })

        assert router.stats.total == 2
        assert router.stats.ml_handled == 0
        assert router.stats.llm_escalated == 2

    def test_ml_confidence_at_threshold_uses_ml(self, router):
        """Adversarial confidence == threshold uses ML (>= comparison)."""
        ml_pred = {
            "pred_label_binary": "adversarial",
            "confidence_label_binary": 0.85,  # exactly at threshold
            "pred_label_category": "unicode_attack",
        }
        result = router.predict_single("text", ml_pred)
        assert result["routed_to"] == "ml"

    def test_router_prefers_calibrated_binary_confidence(self, router):
        """Calibrated binary confidence should override raw confidence for routing."""
        ml_pred = {
            "pred_label_binary": "adversarial",
            "confidence_label_binary": 0.99,
            "confidence_label_binary_cal": 0.20,  # force escalation
            "pred_label_category": "unicode_attack",
        }
        router.llm.predict.return_value = {
            "label_binary": "adversarial",
            "confidence": 0.9,
        }
        result = router.predict_single("text", ml_pred)
        assert result["routed_to"] == "llm"
        assert result["route_reason"] == "ml_adv_low_conf_escalate"

    def test_non_unicode_adversarial_high_confidence_escalates(self, router):
        """High-confidence non-unicode adversarial should not finalize via ML."""
        ml_pred = {
            "pred_label_binary": "adversarial",
            "confidence_label_binary": 0.99,
            "pred_label_category": "nlp_attack",
            "pred_label_type": "TextFooler",
        }
        router.llm.predict.return_value = {
            "label_binary": "adversarial",
            "confidence": 0.95,
        }
        result = router.predict_single("text", ml_pred)
        assert result["routed_to"] == "llm"
        assert result["route_reason"] == "ml_adv_non_unicode_escalate"

    def test_missing_lane_signal_escalates_safely(self, router):
        """Missing category/type should default to escalation."""
        ml_pred = {
            "pred_label_binary": "adversarial",
            "confidence_label_binary": 0.99,
        }
        router.llm.predict.return_value = {
            "label_binary": "adversarial",
            "confidence": 0.95,
        }
        result = router.predict_single("text", ml_pred)
        assert result["routed_to"] == "llm"
        assert result["route_reason"] == "ml_adv_unknown_lane_escalate"

    def test_supports_research_style_ml_keys(self, router):
        """Router accepts ml_* keys in addition to pred_* keys."""
        ml_pred = {
            "ml_pred_binary": "adversarial",
            "ml_conf_binary_cal": 0.95,
            "ml_pred_category": "unicode_attack",
            "ml_pred_type": "Diacritcs",
            "ml_conf_category": 0.9,
            "ml_conf_type": 0.85,
        }
        result = router.predict_single("text", ml_pred)
        assert result["routed_to"] == "ml"
        assert result["label_binary"] == "adversarial"
        assert result["label_category"] == "unicode_attack"
        assert result["label_type"] == "Diacritcs"


# ---------------------------------------------------------------------------
# threshold_sweep
# ---------------------------------------------------------------------------
class TestThresholdSweep:
    """Tests for threshold_sweep()."""

    @pytest.fixture
    def sweep_data(self):
        """DataFrame + ML predictions for threshold sweep."""
        df = pd.DataFrame({
            "label_binary": ["adversarial", "adversarial", "benign", "benign"],
        })
        ml_preds = pd.DataFrame({
            "pred_label_binary": ["adversarial", "adversarial", "benign", "benign"],
            "confidence_label_binary": [0.99, 0.60, 0.95, 0.55],
            "pred_label_category": ["unicode_attack", "unicode_attack", "benign", "benign"],
            "pred_label_type": ["Diacritcs", "Zero Width", "benign", "benign"],
        })
        return df, ml_preds

    def test_returns_dataframe(self, sweep_data):
        df, ml_preds = sweep_data
        result = threshold_sweep(df, ml_preds, [0.5, 0.8])
        assert isinstance(result, pd.DataFrame)

    def test_expected_columns(self, sweep_data):
        df, ml_preds = sweep_data
        result = threshold_sweep(df, ml_preds, [0.5])
        expected = {"threshold", "ml_handled", "llm_escalated", "ml_rate", "ml_accuracy_on_handled"}
        assert expected == set(result.columns)

    def test_threshold_zero_all_ml(self, sweep_data):
        """threshold=0.0 still escalates benign predictions."""
        df, ml_preds = sweep_data
        result = threshold_sweep(df, ml_preds, [0.0])
        assert result.iloc[0]["ml_handled"] == 2
        assert result.iloc[0]["llm_escalated"] == 2

    def test_threshold_one_all_llm(self, sweep_data):
        """threshold=1.0 (above all confidences) → all escalated to LLM."""
        df, ml_preds = sweep_data
        result = threshold_sweep(df, ml_preds, [1.01])
        assert result.iloc[0]["ml_handled"] == 0
        assert result.iloc[0]["llm_escalated"] == len(df)

    def test_partial_split(self, sweep_data):
        """Intermediate threshold splits samples between ML and LLM."""
        df, ml_preds = sweep_data
        result = threshold_sweep(df, ml_preds, [0.90])
        row = result.iloc[0]
        # Only adversarial prediction 0.99 is eligible for ML fast path.
        assert row["ml_handled"] == 1
        assert row["llm_escalated"] == 3

    def test_uses_calibrated_confidence_when_available(self, sweep_data):
        """Threshold sweep should prefer calibrated confidence column when present."""
        df, ml_preds = sweep_data
        ml_preds = ml_preds.copy()
        ml_preds["confidence_label_binary_cal"] = [0.2, 0.95, 0.99, 0.7]
        result = threshold_sweep(df, ml_preds, [0.90])
        row = result.iloc[0]
        assert row["ml_handled"] == 1
        assert row["llm_escalated"] == 3

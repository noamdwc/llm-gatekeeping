"""Tests for src.research — research mode pipeline."""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock, patch

from src.ml_classifier.ml_baseline import MLBaseline
from src.research import (
    compute_hybrid_routing,
    build_research_dataframe,
)
from src.utils import build_sample_id


# ---------------------------------------------------------------------------
# predict_full()
# ---------------------------------------------------------------------------
class TestPredictFull:
    """Tests for MLBaseline.predict_full()."""

    def test_columns_present(self, fitted_ml_model, sample_dataframe):
        """predict_full() returns ml_pred, ml_conf, and ml_proba columns."""
        result = fitted_ml_model.predict_full(sample_dataframe, "modified_sample")
        for short in ["binary", "category", "type"]:
            assert f"ml_pred_{short}" in result.columns
            assert f"ml_conf_{short}" in result.columns

    def test_proba_columns_per_class(self, fitted_ml_model, sample_dataframe):
        """One ml_proba column per class per level."""
        result = fitted_ml_model.predict_full(sample_dataframe, "modified_sample")
        for level in ["label_binary", "label_category", "label_type"]:
            le = fitted_ml_model.label_encoders[level]
            short = level.replace("label_", "")
            for cls in le.classes_:
                col = f"ml_proba_{short}_{cls}"
                assert col in result.columns, f"Missing column: {col}"

    def test_probabilities_sum_to_one(self, fitted_ml_model, sample_dataframe):
        """Probabilities for each level sum to ~1.0."""
        result = fitted_ml_model.predict_full(sample_dataframe, "modified_sample")
        for level in ["label_binary", "label_category", "label_type"]:
            short = level.replace("label_", "")
            proba_cols = [c for c in result.columns if c.startswith(f"ml_proba_{short}_")]
            row_sums = result[proba_cols].sum(axis=1)
            np.testing.assert_allclose(row_sums, 1.0, atol=1e-6)

    def test_confidence_equals_max_proba(self, fitted_ml_model, sample_dataframe):
        """ml_conf_{level} == max of the corresponding probabilities."""
        result = fitted_ml_model.predict_full(sample_dataframe, "modified_sample")
        for level in ["label_binary", "label_category", "label_type"]:
            short = level.replace("label_", "")
            proba_cols = [c for c in result.columns if c.startswith(f"ml_proba_{short}_")]
            max_proba = result[proba_cols].max(axis=1)
            np.testing.assert_allclose(result[f"ml_conf_{short}"], max_proba, atol=1e-9)

    def test_pred_equals_argmax(self, fitted_ml_model, sample_dataframe):
        """ml_pred_{level} matches the class with the highest probability."""
        result = fitted_ml_model.predict_full(sample_dataframe, "modified_sample")
        for level in ["label_binary", "label_category", "label_type"]:
            le = fitted_ml_model.label_encoders[level]
            short = level.replace("label_", "")
            proba_cols = [f"ml_proba_{short}_{cls}" for cls in le.classes_]
            argmax_idx = result[proba_cols].values.argmax(axis=1)
            expected_labels = le.classes_[argmax_idx]
            np.testing.assert_array_equal(result[f"ml_pred_{short}"].values, expected_labels)

    def test_row_count(self, fitted_ml_model, sample_dataframe):
        """predict_full() returns one row per input."""
        result = fitted_ml_model.predict_full(sample_dataframe, "modified_sample")
        assert len(result) == len(sample_dataframe)

    def test_probabilities_in_range(self, fitted_ml_model, sample_dataframe):
        """All probability values are in [0, 1]."""
        result = fitted_ml_model.predict_full(sample_dataframe, "modified_sample")
        proba_cols = [c for c in result.columns if c.startswith("ml_proba_")]
        for col in proba_cols:
            assert (result[col] >= 0).all(), f"{col} has negative values"
            assert (result[col] <= 1).all(), f"{col} has values > 1"


# ---------------------------------------------------------------------------
# force_all_stages (LLM)
# ---------------------------------------------------------------------------
class TestForceAllStages:
    """Tests for force_all_stages parameter on LLM classifier."""

    def _make_classifier(self, cfg):
        with patch("src.llm_classifier.llm_classifier.openai.OpenAI"):
            from src.llm_classifier.llm_classifier import HierarchicalLLMClassifier
            return HierarchicalLLMClassifier(cfg, few_shot_examples={})

    def test_default_high_confidence_skips_judge(self, sample_config):
        """Default (force_all_stages=False): high-confidence skips judge."""
        clf = self._make_classifier(sample_config)
        clf.classify = MagicMock(
            return_value={"label": "benign", "confidence": 0.99}
        )
        clf.judge = MagicMock()

        result = clf.predict("normal text", force_all_stages=False)

        assert result["llm_stages_run"] == 1
        clf.judge.assert_not_called()

    def test_forced_benign_runs_judge(self, sample_config):
        """force_all_stages=True: always runs judge even for high confidence."""
        clf = self._make_classifier(sample_config)
        clf.classify = MagicMock(
            return_value={"label": "benign", "confidence": 0.99}
        )
        clf.judge = MagicMock(
            return_value={"label": "benign", "confidence": 0.98, "reasoning": ""}
        )

        result = clf.predict("normal text", force_all_stages=True)

        assert result["llm_stages_run"] == 2
        clf.judge.assert_called_once()

    def test_low_confidence_triggers_judge(self, sample_config):
        """Low-confidence classifier result triggers judge."""
        clf = self._make_classifier(sample_config)
        clf.classify = MagicMock(
            return_value={"label": "nlp_attack", "confidence": 0.5}
        )
        clf.judge = MagicMock(
            return_value={"label": "Homoglyphs", "confidence": 0.9, "reasoning": ""}
        )

        result = clf.predict("some text", force_all_stages=False)

        assert result["llm_stages_run"] == 2
        clf.judge.assert_called_once()

    def test_high_confidence_nlp_skips_judge(self, sample_config):
        """High-confidence NLP prediction skips judge, stages_run=1."""
        clf = self._make_classifier(sample_config)
        clf.classify = MagicMock(
            return_value={"label": "nlp_attack", "confidence": 0.88}
        )
        clf.judge = MagicMock()

        result = clf.predict("some text", force_all_stages=False)

        assert result["llm_stages_run"] == 1
        clf.judge.assert_not_called()

    def test_stages_run_in_result(self, sample_config):
        """llm_stages_run is always present in result dict."""
        clf = self._make_classifier(sample_config)
        clf.classify = MagicMock(
            return_value={"label": "Diacritcs", "confidence": 0.95}
        )

        result = clf.predict("text")
        assert "llm_stages_run" in result
        assert result["llm_stages_run"] == 1

    def test_predict_batch_passes_force_flag(self, sample_config):
        """predict_batch forwards force_all_stages to predict."""
        clf = self._make_classifier(sample_config)
        clf.predict = MagicMock(return_value={"label_binary": "benign"})

        clf.predict_batch(["a", "b"], force_all_stages=True)

        for call in clf.predict.call_args_list:
            assert call.kwargs["force_all_stages"] is True


# ---------------------------------------------------------------------------
# compute_hybrid_routing
# ---------------------------------------------------------------------------
class TestComputeHybridRouting:
    """Tests for hybrid routing logic."""

    def _make_ml_df(self, confs, preds_binary=None):
        n = len(confs)
        if preds_binary is None:
            preds_binary = ["adversarial"] * n
        samples = [f"sample_{i}" for i in range(n)]
        return pd.DataFrame({
            "sample_id": [build_sample_id(s) for s in samples],
            "ml_pred_binary": preds_binary,
            "ml_pred_category": ["unicode_attack"] * n,
            "ml_pred_type": ["Diacritcs"] * n,
            "ml_conf_binary": confs,
            "ml_conf_category": [0.9] * n,
            "ml_conf_type": [0.85] * n,
        })

    def _make_llm_df(self, n):
        samples = [f"sample_{i}" for i in range(n)]
        return pd.DataFrame({
            "sample_id": [build_sample_id(s) for s in samples],
            "llm_pred_binary": ["adversarial"] * n,
            "llm_pred_category": ["nlp_attack"] * n,
            "llm_conf_binary": [0.95] * n,
            "llm_stages_run": [2] * n,
        })

    def test_high_confidence_routes_to_ml(self):
        """ML confidence >= threshold → routed to ML."""
        ml_df = self._make_ml_df([0.95, 0.90])
        result = compute_hybrid_routing(ml_df, None, threshold=0.85)
        assert (result["hybrid_routed_to"] == "ml").all()
        assert list(result["hybrid_pred_binary"]) == ["adversarial", "adversarial"]

    def test_low_confidence_routes_to_llm(self):
        """ML confidence < threshold with LLM available → routed to LLM."""
        ml_df = self._make_ml_df([0.5, 0.6])
        llm_df = self._make_llm_df(2)
        result = compute_hybrid_routing(ml_df, llm_df, threshold=0.85)
        assert (result["hybrid_routed_to"] == "llm").all()
        # Binary from LLM; category from LLM's llm_pred_category
        assert list(result["hybrid_pred_binary"]) == ["adversarial", "adversarial"]
        assert list(result["hybrid_pred_category"]) == ["nlp_attack", "nlp_attack"]
        # Type stays as ML's prediction (LLM doesn't provide type)
        assert list(result["hybrid_pred_type"]) == ["Diacritcs", "Diacritcs"]

    def test_mixed_routing(self):
        """Mix of ML and LLM routing."""
        ml_df = self._make_ml_df([0.95, 0.50, 0.85, 0.40])
        llm_df = self._make_llm_df(4)
        result = compute_hybrid_routing(ml_df, llm_df, threshold=0.85)
        assert list(result["hybrid_routed_to"]) == ["ml", "llm", "ml", "llm"]

    def test_skip_llm_fallback(self):
        """When LLM is None, escalated samples fall back to ML predictions."""
        ml_df = self._make_ml_df([0.50])
        result = compute_hybrid_routing(ml_df, None, threshold=0.85)
        assert result.iloc[0]["hybrid_routed_to"] == "ml"
        assert result.iloc[0]["hybrid_pred_binary"] == "adversarial"
        assert result.iloc[0]["hybrid_pred_type"] == "Diacritcs"

    def test_partial_llm_coverage_falls_back_to_ml(self):
        """Escalated samples missing from llm_df fall back to ML predictions."""
        # 4 samples: 0 confident, 1-3 escalated; LLM only covers samples 1 & 2
        ml_df = self._make_ml_df([0.95, 0.50, 0.40, 0.30])
        llm_df = self._make_llm_df(3)  # covers sample_0, sample_1, sample_2
        # sample_3 is escalated but has no LLM prediction
        result = compute_hybrid_routing(ml_df, llm_df, threshold=0.85)

        # sample_0: confident → ml
        assert result.iloc[0]["hybrid_routed_to"] == "ml"
        assert result.iloc[0]["hybrid_pred_category"] == "unicode_attack"  # ML pred

        # sample_1, sample_2: escalated + LLM available → llm
        assert result.iloc[1]["hybrid_routed_to"] == "llm"
        assert result.iloc[1]["hybrid_pred_category"] == "nlp_attack"  # LLM pred_category
        assert result.iloc[2]["hybrid_routed_to"] == "llm"
        assert result.iloc[2]["hybrid_pred_category"] == "nlp_attack"  # LLM pred_category

        # sample_3: escalated but missing from llm_df → falls back to ml
        assert result.iloc[3]["hybrid_routed_to"] == "ml"
        assert result.iloc[3]["hybrid_pred_binary"] == "adversarial"  # ML pred
        assert result.iloc[3]["hybrid_pred_category"] == "unicode_attack"  # ML pred
        assert result.iloc[3]["hybrid_pred_type"] == "Diacritcs"  # ML pred

    def test_output_columns(self):
        """Output has exactly the expected columns."""
        ml_df = self._make_ml_df([0.5])
        result = compute_hybrid_routing(ml_df, None, threshold=0.85)
        expected = {"sample_id", "hybrid_routed_to", "hybrid_pred_binary", "hybrid_pred_category", "hybrid_pred_type"}
        assert set(result.columns) == expected

    def test_benign_predictions_always_escalate_when_llm_available(self):
        """High-confidence benign ML predictions should route to LLM."""
        ml_df = self._make_ml_df([0.99, 0.95], preds_binary=["benign", "benign"])
        llm_df = self._make_llm_df(2)
        result = compute_hybrid_routing(ml_df, llm_df, threshold=0.85)
        assert (result["hybrid_routed_to"] == "llm").all()
        assert list(result["hybrid_pred_binary"]) == ["adversarial", "adversarial"]


# ---------------------------------------------------------------------------
# build_research_dataframe
# ---------------------------------------------------------------------------
class TestBuildResearchDataframe:
    """Tests for merging into the wide research DataFrame."""

    def _make_ml_df(self, n):
        """ML predictions parquet includes ground truth + predictions."""
        samples = [f"text_{i}" for i in range(n)]
        return pd.DataFrame({
            "sample_id": [build_sample_id(s) for s in samples],
            "modified_sample": samples,
            "label_binary": ["adversarial"] * n,
            "label_category": ["unicode_attack"] * n,
            "label_type": ["Diacritcs"] * n,
            "ml_pred_binary": ["adversarial"] * n,
            "ml_conf_binary": [0.9] * n,
        })

    def test_without_llm(self):
        """Without LLM, output has ML + hybrid columns."""
        n = 5
        ml_df = self._make_ml_df(n)
        samples = [f"text_{i}" for i in range(n)]
        hybrid_df = pd.DataFrame({
            "sample_id": [build_sample_id(s) for s in samples],
            "hybrid_routed_to": ["ml"] * n,
            "hybrid_pred_binary": ["adversarial"] * n,
        })
        result = build_research_dataframe(ml_df, hybrid_df)
        assert len(result) == n
        assert "ml_pred_binary" in result.columns
        assert "hybrid_routed_to" in result.columns
        assert "label_binary" in result.columns

    def test_with_llm(self):
        """With LLM, output also includes llm columns."""
        n = 5
        ml_df = self._make_ml_df(n)
        samples = [f"text_{i}" for i in range(n)]
        llm_df = pd.DataFrame({
            "sample_id": [build_sample_id(s) for s in samples],
            "llm_pred_binary": ["benign"] * n,
            "llm_conf_binary": [0.8] * n,
        })
        hybrid_df = pd.DataFrame({
            "sample_id": [build_sample_id(s) for s in samples],
            "hybrid_routed_to": ["ml"] * n,
        })
        result = build_research_dataframe(ml_df, hybrid_df, llm_df=llm_df)
        assert "llm_pred_binary" in result.columns

    def test_row_count_preserved(self):
        """Output row count matches input."""
        n = 5
        ml_df = self._make_ml_df(n)
        samples = [f"text_{i}" for i in range(n)]
        hybrid_df = pd.DataFrame({
            "sample_id": [build_sample_id(s) for s in samples],
            "hybrid_routed_to": ["ml"] * n,
        })
        result = build_research_dataframe(ml_df, hybrid_df)
        assert len(result) == n


# ---------------------------------------------------------------------------
# run_ml_full (integration with fitted model via predict_full)
# ---------------------------------------------------------------------------
class TestRunMlFull:
    """Integration test for MLBaseline.predict_full() used by research pipeline."""

    def test_returns_correct_shape(self, fitted_ml_model, sample_dataframe):
        """predict_full returns one row per input with expected columns."""
        result = fitted_ml_model.predict_full(sample_dataframe, "modified_sample")
        assert len(result) == len(sample_dataframe)
        assert "ml_pred_binary" in result.columns
        assert "ml_conf_binary" in result.columns

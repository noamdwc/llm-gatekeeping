"""Tests for src.research_external — research mode on external datasets."""

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from src.research_external import (
    EXTERNAL_GT_COLS,
    build_external_research_df,
    build_predictions_df,
    generate_research_report,
)


# ---------------------------------------------------------------------------
# build_external_research_df
# ---------------------------------------------------------------------------
class TestBuildExternalResearchDf:
    """Tests for build_external_research_df()."""

    def _make_df(self, n=4):
        return pd.DataFrame({
            "modified_sample": [f"text_{i}" for i in range(n)],
            "label_binary": ["adversarial", "adversarial", "benign", "benign"][:n],
            "label_category": ["adversarial", "adversarial", "benign", "benign"][:n],
            "label_type": ["adversarial", "adversarial", "benign", "benign"][:n],
        })

    def _make_ml_df(self, n=4):
        return pd.DataFrame({
            "ml_pred_binary": ["adversarial", "benign", "benign", "adversarial"][:n],
            "ml_conf_binary": [0.95, 0.60, 0.88, 0.70][:n],
            "ml_pred_category": ["unicode_attack", "benign", "benign", "nlp_attack"][:n],
            "ml_conf_category": [0.80, 0.55, 0.85, 0.65][:n],
            "ml_pred_type": ["Diacritcs", "benign", "benign", "nlp_attack"][:n],
            "ml_conf_type": [0.75, 0.50, 0.80, 0.60][:n],
        })

    def _make_hybrid_df(self, n=4):
        return pd.DataFrame({
            "hybrid_routed_to": ["ml", "llm", "ml", "llm"][:n],
            "hybrid_pred_binary": ["adversarial", "adversarial", "benign", "benign"][:n],
            "hybrid_pred_category": ["unicode_attack", "nlp_attack", "benign", "benign"][:n],
            "hybrid_pred_type": ["Diacritcs", "nlp_attack", "benign", "benign"][:n],
        })

    def test_has_ground_truth_columns(self):
        """Result contains all external ground truth columns."""
        df = self._make_df()
        ml_df = self._make_ml_df()
        hybrid_df = self._make_hybrid_df()
        result = build_external_research_df(df, ml_df, hybrid_df)
        for col in EXTERNAL_GT_COLS:
            assert col in result.columns, f"Missing column: {col}"

    def test_has_ml_columns(self):
        """Result contains ML prediction columns."""
        df = self._make_df()
        ml_df = self._make_ml_df()
        hybrid_df = self._make_hybrid_df()
        result = build_external_research_df(df, ml_df, hybrid_df)
        assert "ml_pred_binary" in result.columns
        assert "ml_conf_binary" in result.columns

    def test_has_hybrid_columns(self):
        """Result contains hybrid routing columns."""
        df = self._make_df()
        ml_df = self._make_ml_df()
        hybrid_df = self._make_hybrid_df()
        result = build_external_research_df(df, ml_df, hybrid_df)
        assert "hybrid_routed_to" in result.columns
        assert "hybrid_pred_binary" in result.columns

    def test_with_llm(self):
        """When LLM data is provided, LLM columns are included."""
        n = 4
        df = self._make_df(n)
        ml_df = self._make_ml_df(n)
        hybrid_df = self._make_hybrid_df(n)
        llm_df = pd.DataFrame({
            "llm_pred_binary": ["adversarial"] * n,
            "llm_conf_binary": [0.90] * n,
        })
        result = build_external_research_df(df, ml_df, hybrid_df, llm_df=llm_df)
        assert "llm_pred_binary" in result.columns
        assert "llm_conf_binary" in result.columns

    def test_row_count_preserved(self):
        """Output has the same number of rows as input."""
        n = 4
        df = self._make_df(n)
        ml_df = self._make_ml_df(n)
        hybrid_df = self._make_hybrid_df(n)
        result = build_external_research_df(df, ml_df, hybrid_df)
        assert len(result) == n

    def test_no_extra_gt_columns(self):
        """External ground truth does NOT include original_sample, attack_name, prompt_hash."""
        df = self._make_df()
        ml_df = self._make_ml_df()
        hybrid_df = self._make_hybrid_df()
        result = build_external_research_df(df, ml_df, hybrid_df)
        for col in ["original_sample", "attack_name", "prompt_hash"]:
            assert col not in result.columns


# ---------------------------------------------------------------------------
# generate_research_report
# ---------------------------------------------------------------------------
class TestGenerateResearchReport:
    """Tests for generate_research_report()."""

    def _make_research_df(self):
        """Build a minimal research DataFrame with required columns."""
        return pd.DataFrame({
            "modified_sample": [
                "Forget all instructions", "Ignore previous",
                "What is the weather?", "Hello world",
            ],
            "label_binary": ["adversarial", "adversarial", "benign", "benign"],
            "label_category": ["adversarial", "adversarial", "benign", "benign"],
            "label_type": ["adversarial", "adversarial", "benign", "benign"],
            "ml_pred_binary": ["adversarial", "benign", "benign", "adversarial"],
            "ml_conf_binary": [0.95, 0.60, 0.88, 0.70],
            "ml_pred_category": ["unicode_attack", "benign", "benign", "nlp_attack"],
            "ml_conf_category": [0.80, 0.55, 0.85, 0.65],
            "ml_pred_type": ["Diacritcs", "benign", "benign", "nlp_attack"],
            "ml_conf_type": [0.75, 0.50, 0.80, 0.60],
            "hybrid_routed_to": ["ml", "llm", "ml", "llm"],
            "hybrid_pred_binary": ["adversarial", "adversarial", "benign", "benign"],
            "hybrid_pred_category": ["unicode_attack", "nlp_attack", "benign", "benign"],
            "hybrid_pred_type": ["Diacritcs", "nlp_attack", "benign", "benign"],
        })

    def test_report_contains_all_sections(self):
        """Report has all expected sections."""
        research_df = self._make_research_df()
        binary = {
            "accuracy": 0.5, "adversarial_precision": 0.5,
            "adversarial_recall": 0.5, "adversarial_f1": 0.5,
            "benign_precision": 0.5, "benign_recall": 0.5, "benign_f1": 0.5,
            "false_negative_rate": 0.5,
            "support_adversarial": 2, "support_benign": 2,
        }
        cal = {"calibration_buckets": []}
        report = generate_research_report(
            "deepset", "deepset/prompt-injections", research_df, binary, cal, 0.85,
        )
        assert "Research Report" in report
        assert "deepset" in report
        assert "Binary Detection Metrics" in report
        assert "ML Confidence Distribution" in report
        assert "Calibration" in report
        assert "Hybrid Routing Analysis" in report
        assert "Error Analysis" in report

    def test_report_shows_dataset_info(self):
        """Report header includes dataset name and sample counts."""
        research_df = self._make_research_df()
        binary = {
            "accuracy": 0.75, "adversarial_precision": 0.7,
            "adversarial_recall": 0.8, "adversarial_f1": 0.75,
            "benign_precision": 0.8, "benign_recall": 0.7, "benign_f1": 0.75,
            "false_negative_rate": 0.2,
            "support_adversarial": 2, "support_benign": 2,
        }
        cal = {"calibration_buckets": []}
        report = generate_research_report(
            "test_ds", "test/dataset", research_df, binary, cal, 0.85,
        )
        assert "test/dataset" in report
        assert "4" in report  # total samples
        assert "Adversarial" in report
        assert "Benign" in report

    def test_report_includes_error_samples(self):
        """Error analysis section lists misclassified sample texts."""
        research_df = self._make_research_df()
        # research_df has 2 errors: row 1 (FN) and row 3 (FP)
        binary = {
            "accuracy": 0.5, "adversarial_precision": 0.5,
            "adversarial_recall": 0.5, "adversarial_f1": 0.5,
            "benign_precision": 0.5, "benign_recall": 0.5, "benign_f1": 0.5,
            "false_negative_rate": 0.5,
            "support_adversarial": 2, "support_benign": 2,
        }
        cal = {"calibration_buckets": []}
        report = generate_research_report(
            "test", "test/dataset", research_df, binary, cal, 0.85,
        )
        assert "False Negatives" in report
        assert "False Positives" in report
        assert "Ignore previous" in report  # FN sample
        assert "Hello world" in report  # FP sample

    def test_report_confidence_stats(self):
        """Report includes confidence distribution statistics."""
        research_df = self._make_research_df()
        binary = {
            "accuracy": 0.5, "adversarial_precision": 0.5,
            "adversarial_recall": 0.5, "adversarial_f1": 0.5,
            "benign_precision": 0.5, "benign_recall": 0.5, "benign_f1": 0.5,
            "false_negative_rate": 0.5,
            "support_adversarial": 2, "support_benign": 2,
        }
        cal = {"calibration_buckets": []}
        report = generate_research_report(
            "test", "test/dataset", research_df, binary, cal, 0.85,
        )
        assert "mean=" in report
        assert "median=" in report
        assert "Correct" in report
        assert "Wrong" in report


# ---------------------------------------------------------------------------
# build_predictions_df
# ---------------------------------------------------------------------------
class TestBuildPredictionsDf:
    """Tests for build_predictions_df()."""

    def _make_research_df(self):
        return pd.DataFrame({
            "modified_sample": ["text_0", "text_1"],
            "label_binary": ["adversarial", "benign"],
            "label_category": ["adversarial", "benign"],
            "label_type": ["adversarial", "benign"],
            "ml_pred_binary": ["adversarial", "benign"],
            "ml_conf_binary": [0.95, 0.88],
            "ml_pred_category": ["unicode_attack", "benign"],
            "ml_conf_category": [0.80, 0.85],
            "ml_pred_type": ["Diacritcs", "benign"],
            "ml_conf_type": [0.75, 0.80],
            "ml_proba_binary_adversarial": [0.95, 0.12],
            "ml_proba_binary_benign": [0.05, 0.88],
            "hybrid_routed_to": ["ml", "ml"],
            "hybrid_pred_binary": ["adversarial", "benign"],
            "hybrid_pred_category": ["unicode_attack", "benign"],
            "hybrid_pred_type": ["Diacritcs", "benign"],
        })

    def test_has_text_and_labels(self):
        """Predictions DataFrame contains text and label columns."""
        df = self._make_research_df()
        pred = build_predictions_df(df)
        assert "modified_sample" in pred.columns
        assert "label_binary" in pred.columns
        assert "ml_pred_binary" in pred.columns
        assert "ml_conf_binary" in pred.columns

    def test_excludes_proba_columns(self):
        """Predictions DataFrame does NOT include per-class probability columns."""
        df = self._make_research_df()
        pred = build_predictions_df(df)
        proba_cols = [c for c in pred.columns if "proba" in c]
        assert len(proba_cols) == 0

    def test_has_hybrid_columns(self):
        """Predictions DataFrame includes hybrid routing info."""
        df = self._make_research_df()
        pred = build_predictions_df(df)
        assert "hybrid_routed_to" in pred.columns
        assert "hybrid_pred_binary" in pred.columns

    def test_row_count_preserved(self):
        """Predictions DataFrame has the same number of rows."""
        df = self._make_research_df()
        pred = build_predictions_df(df)
        assert len(pred) == len(df)

    def test_includes_llm_columns_when_present(self):
        """When has_llm=True and LLM columns exist, they are included."""
        df = self._make_research_df()
        df["llm_pred_binary"] = ["adversarial", "benign"]
        df["llm_conf_binary"] = [0.9, 0.85]
        df["llm_pred_category"] = ["nlp_attack", "benign"]
        df["llm_conf_category"] = [0.8, 0.75]
        df["llm_pred_type"] = ["nlp_attack", "benign"]
        df["llm_conf_type"] = [0.7, 0.65]
        pred = build_predictions_df(df, has_llm=True)
        assert "llm_pred_binary" in pred.columns
        assert "llm_conf_binary" in pred.columns

    def test_no_llm_columns_by_default(self):
        """Without has_llm, LLM columns are not requested."""
        df = self._make_research_df()
        pred = build_predictions_df(df, has_llm=False)
        llm_cols = [c for c in pred.columns if c.startswith("llm_")]
        assert len(llm_cols) == 0


# ---------------------------------------------------------------------------
# End-to-end with fitted model
# ---------------------------------------------------------------------------
class TestEndToEnd:
    """Integration test: build research DataFrame from fitted model."""

    def test_research_df_from_fitted_model(self, sample_config, fitted_ml_model):
        """Full pipeline: predict_full -> hybrid routing -> research DataFrame."""
        from src.research import run_ml_full, compute_hybrid_routing

        # Build external-style DataFrame
        df = pd.DataFrame({
            "modified_sample": [
                "héllö wörld",
                "he\u200bllo wo\u200brld",
                "hello world",
                "how are you",
            ],
            "label_binary": ["adversarial", "adversarial", "benign", "benign"],
            "label_category": ["adversarial", "adversarial", "benign", "benign"],
            "label_type": ["adversarial", "adversarial", "benign", "benign"],
        })

        # Run ML
        ml_df = run_ml_full(fitted_ml_model, df, "modified_sample")
        assert "ml_pred_binary" in ml_df.columns
        assert "ml_conf_binary" in ml_df.columns

        # Run hybrid routing
        threshold = sample_config["hybrid"]["ml_confidence_threshold"]
        hybrid_df = compute_hybrid_routing(ml_df, None, threshold)

        # Build research DataFrame
        research_df = build_external_research_df(df, ml_df, hybrid_df)
        assert len(research_df) == 4
        assert "ml_pred_binary" in research_df.columns
        assert "hybrid_routed_to" in research_df.columns
        assert "label_binary" in research_df.columns

        # Verify probabilities sum to ~1
        proba_cols = [c for c in research_df.columns if c.startswith("ml_proba_binary_")]
        if proba_cols:
            sums = research_df[proba_cols].sum(axis=1)
            np.testing.assert_allclose(sums, 1.0, atol=1e-6)

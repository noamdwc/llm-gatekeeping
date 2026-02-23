"""Tests for src.ml_baseline — feature extraction, entropy, MLBaseline."""

import math
import pickle

import numpy as np
import pandas as pd
import pytest

from src.ml_classifier.ml_baseline import (
    MLBaseline,
)
from src.ml_classifier.utils import char_entropy, extract_features_df, unicode_features
from src.ml_classifier.ml_baseline import MLBaseline


# ---------------------------------------------------------------------------
# char_entropy
# ---------------------------------------------------------------------------
class TestCharEntropy:
    """Tests for char_entropy()."""

    def test_empty_string_returns_zero(self):
        assert char_entropy("") == 0.0

    def test_single_char_returns_zero(self):
        """A single repeated character has zero entropy."""
        assert char_entropy("aaaa") == 0.0

    def test_two_equally_likely(self):
        """Two equally frequent characters → entropy = 1.0 bit."""
        assert abs(char_entropy("ab") - 1.0) < 1e-9

    def test_uniform_distribution(self):
        """All characters equally likely → entropy = log2(n_unique)."""
        text = "abcdefgh"
        expected = math.log2(8)
        assert abs(char_entropy(text) - expected) < 1e-9

    def test_positive_for_real_text(self):
        """Normal English text has positive entropy."""
        assert char_entropy("hello world") > 0


# ---------------------------------------------------------------------------
# unicode_features
# ---------------------------------------------------------------------------
class TestUnicodeFeatures:
    """Tests for unicode_features()."""

    def test_empty_string(self):
        """Empty string returns all-zero features."""
        feats = unicode_features("")
        assert feats["non_ascii_ratio"] == 0
        assert feats["zero_width_count"] == 0
        assert feats["text_length"] == 0

    def test_ascii_only(self):
        """Pure ASCII text has zero non-ASCII ratio and zero special counts."""
        feats = unicode_features("hello world")
        assert feats["non_ascii_ratio"] == 0.0
        assert feats["zero_width_count"] == 0
        assert feats["bidi_count"] == 0
        assert feats["fullwidth_count"] == 0
        assert feats["combining_count"] == 0
        assert feats["tag_count"] == 0

    def test_zero_width_detected(self):
        """Zero-width characters are counted."""
        text = "he\u200bllo"  # ZWSP inserted
        feats = unicode_features(text)
        assert feats["zero_width_count"] == 1
        assert feats["zero_width_ratio"] > 0

    def test_bidi_detected(self):
        """BiDi override characters are counted."""
        text = "\u202ehello"  # RLO prefix
        feats = unicode_features(text)
        assert feats["bidi_count"] == 1

    def test_fullwidth_detected(self):
        """Full-width characters are counted."""
        text = "\uff41\uff42\uff43"  # ａｂｃ
        feats = unicode_features(text)
        assert feats["fullwidth_count"] == 3

    def test_combining_detected(self):
        """Combining diacritics are counted."""
        text = "t\u0332e\u0332s\u0332t"  # combining low line
        feats = unicode_features(text)
        assert feats["combining_count"] == 3

    def test_text_length(self):
        """text_length matches actual len()."""
        text = "abc"
        feats = unicode_features(text)
        assert feats["text_length"] == 3

    def test_non_ascii_ratio(self):
        """Non-ASCII ratio is correct for mixed text."""
        text = "aé"  # 1 ASCII + 1 non-ASCII
        feats = unicode_features(text)
        assert abs(feats["non_ascii_ratio"] - 0.5) < 1e-9

    def test_entropy_included(self):
        """char_entropy is embedded in the features."""
        feats = unicode_features("ab")
        assert abs(feats["char_entropy"] - 1.0) < 1e-9


# ---------------------------------------------------------------------------
# extract_features_df
# ---------------------------------------------------------------------------
class TestExtractFeaturesDf:
    """Tests for extract_features_df()."""

    def test_returns_dataframe(self):
        texts = pd.Series(["hello", "world"])
        result = extract_features_df(texts)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2

    def test_expected_columns(self):
        result = extract_features_df(pd.Series(["test"]))
        expected_cols = {
            "non_ascii_ratio", "zero_width_count", "zero_width_ratio",
            "bidi_count", "control_count", "tag_count",
            "fullwidth_count", "combining_count", "combining_ratio",
            "char_entropy", "unique_scripts", "text_length",
            "cat_Lu", "cat_Ll", "cat_Mn", "cat_Cf", "cat_So",
        }
        assert expected_cols == set(result.columns)


# ---------------------------------------------------------------------------
# MLBaseline
# ---------------------------------------------------------------------------
class TestMLBaseline:
    """Tests for the MLBaseline class."""

    def test_fit_creates_models(self, fitted_ml_model):
        """After fit(), models exist for all three hierarchy levels."""
        for level in ["label_binary", "label_category", "label_type"]:
            assert level in fitted_ml_model.models
            assert level in fitted_ml_model.label_encoders

    def test_predict_columns(self, fitted_ml_model, sample_dataframe):
        """predict() returns expected columns."""
        preds = fitted_ml_model.predict(sample_dataframe, "modified_sample")
        expected = {
            "pred_label_binary", "confidence_label_binary",
            "confidence_label_binary_cal",
            "pred_label_category", "confidence_label_category",
            "pred_label_type", "confidence_label_type",
        }
        assert expected == set(preds.columns)

    def test_predict_row_count(self, fitted_ml_model, sample_dataframe):
        """predict() returns one row per input."""
        preds = fitted_ml_model.predict(sample_dataframe, "modified_sample")
        assert len(preds) == len(sample_dataframe)

    def test_confidence_in_range(self, fitted_ml_model, sample_dataframe):
        """Confidence scores are in [0, 1]."""
        preds = fitted_ml_model.predict(sample_dataframe, "modified_sample")
        for col in preds.columns:
            if col.startswith("confidence"):
                assert (preds[col] >= 0).all()
                assert (preds[col] <= 1).all()

    def test_predict_proba_binary_shape(self, fitted_ml_model, sample_dataframe):
        """predict_proba_binary returns (n, 2) array with values in [0,1]."""
        proba = fitted_ml_model.predict_proba_binary(sample_dataframe, "modified_sample")
        assert proba.shape == (len(sample_dataframe), 2)
        assert np.all(proba >= 0)
        assert np.all(proba <= 1)
        # Rows should sum to ~1
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)

    def test_save_load_roundtrip(self, fitted_ml_model, sample_dataframe, tmp_path):
        """save() then load() reproduces identical predictions."""
        path = str(tmp_path / "model.pkl")
        fitted_ml_model.save(path)

        loaded = MLBaseline(fitted_ml_model.cfg)
        loaded.load(path)

        preds_orig = fitted_ml_model.predict(sample_dataframe, "modified_sample")
        preds_loaded = loaded.predict(sample_dataframe, "modified_sample")

        pd.testing.assert_frame_equal(preds_orig, preds_loaded)

    def test_load_legacy_artifact_without_calibrator(self, fitted_ml_model, sample_dataframe, tmp_path):
        """Old model artifacts without binary_calibrator still load and predict."""
        legacy_path = tmp_path / "legacy_model.pkl"
        with open(legacy_path, "wb") as f:
            pickle.dump(
                {
                    "tfidf": fitted_ml_model.tfidf,
                    "models": fitted_ml_model.models,
                    "le": fitted_ml_model.label_encoders,
                },
                f,
            )

        loaded = MLBaseline(fitted_ml_model.cfg)
        loaded.load(str(legacy_path))
        preds = loaded.predict(sample_dataframe, "modified_sample")
        assert "confidence_label_binary_cal" in preds.columns
        np.testing.assert_allclose(
            preds["confidence_label_binary"].values,
            preds["confidence_label_binary_cal"].values,
            atol=1e-9,
        )

    def test_predicted_labels_are_known(self, fitted_ml_model, sample_dataframe):
        """All predicted labels were seen during training."""
        preds = fitted_ml_model.predict(sample_dataframe, "modified_sample")
        for level in ["label_binary", "label_category", "label_type"]:
            known = set(fitted_ml_model.label_encoders[level].classes_)
            predicted = set(preds[f"pred_{level}"].unique())
            assert predicted.issubset(known), (
                f"Unknown labels in pred_{level}: {predicted - known}"
            )

"""Tests for src.evaluate — metric functions and report generation."""

import numpy as np
import pandas as pd
import pytest

from src.evaluate import (
    binary_metrics,
    calibration_metrics,
    category_metrics,
    generate_report,
    type_metrics,
)


# ---------------------------------------------------------------------------
# binary_metrics
# ---------------------------------------------------------------------------
class TestBinaryMetrics:
    """Tests for binary_metrics()."""

    def test_perfect_predictions(self):
        """Perfect predictions yield accuracy=1 and F1=1 for both classes."""
        y_true = pd.Series(["adversarial", "adversarial", "benign", "benign"])
        y_pred = pd.Series(["adversarial", "adversarial", "benign", "benign"])

        m = binary_metrics(y_true, y_pred)
        assert m["accuracy"] == 1.0
        assert m["adversarial_f1"] == 1.0
        assert m["benign_f1"] == 1.0
        assert m["false_negative_rate"] == 0.0

    def test_all_wrong(self):
        """All wrong predictions yield accuracy=0."""
        y_true = pd.Series(["adversarial", "adversarial", "benign", "benign"])
        y_pred = pd.Series(["benign", "benign", "adversarial", "adversarial"])

        m = binary_metrics(y_true, y_pred)
        assert m["accuracy"] == 0.0

    def test_false_negative_rate(self):
        """FNR = fraction of adversarial samples predicted benign."""
        y_true = pd.Series(["adversarial", "adversarial", "adversarial", "benign"])
        y_pred = pd.Series(["adversarial", "benign", "benign", "benign"])

        m = binary_metrics(y_true, y_pred)
        assert abs(m["false_negative_rate"] - 2 / 3) < 1e-9

    def test_support_counts(self):
        """Support counts match actual class distribution."""
        y_true = pd.Series(["adversarial", "adversarial", "benign"])
        y_pred = pd.Series(["adversarial", "adversarial", "benign"])

        m = binary_metrics(y_true, y_pred)
        assert m["support_adversarial"] == 2
        assert m["support_benign"] == 1

    def test_no_adversarial_samples(self):
        """FNR is 0.0 when there are no adversarial samples."""
        y_true = pd.Series(["benign", "benign"])
        y_pred = pd.Series(["benign", "benign"])

        m = binary_metrics(y_true, y_pred)
        assert m["false_negative_rate"] == 0.0


# ---------------------------------------------------------------------------
# category_metrics
# ---------------------------------------------------------------------------
class TestCategoryMetrics:
    """Tests for category_metrics()."""

    def test_perfect_category(self):
        """Perfect category predictions on adversarial samples."""
        y_true = pd.Series(["unicode_attack", "nlp_attack", "benign"])
        y_pred = pd.Series(["unicode_attack", "nlp_attack", "benign"])

        m = category_metrics(y_true, y_pred)
        assert m["category_accuracy"] == 1.0

    def test_filters_benign(self):
        """Benign samples are excluded — only adversarial pairs considered."""
        y_true = pd.Series(["unicode_attack", "benign"])
        y_pred = pd.Series(["unicode_attack", "benign"])

        m = category_metrics(y_true, y_pred)
        # Only 1 non-benign pair → accuracy is 1.0
        assert m["category_accuracy"] == 1.0

    def test_empty_adversarial_set(self):
        """All benign → returns default zeros."""
        y_true = pd.Series(["benign", "benign"])
        y_pred = pd.Series(["benign", "benign"])

        m = category_metrics(y_true, y_pred)
        assert m["category_accuracy"] == 0.0
        assert m["category_f1_macro"] == 0.0

    def test_confusion_matrix_present(self):
        """Result contains a confusion matrix when non-empty."""
        y_true = pd.Series(["unicode_attack", "nlp_attack"])
        y_pred = pd.Series(["unicode_attack", "nlp_attack"])

        m = category_metrics(y_true, y_pred)
        assert "confusion_matrix" in m
        assert len(m["confusion_matrix"]) == 2
        assert len(m["confusion_matrix"][0]) == 2


# ---------------------------------------------------------------------------
# type_metrics
# ---------------------------------------------------------------------------
class TestTypeMetrics:
    """Tests for type_metrics()."""

    def test_perfect_type(self):
        """Perfect type predictions on unicode samples."""
        y_true = pd.Series(["Diacritcs", "Zero Width", "nlp_attack", "benign"])
        y_pred = pd.Series(["Diacritcs", "Zero Width", "nlp_attack", "benign"])

        m = type_metrics(y_true, y_pred)
        assert m["type_accuracy"] == 1.0

    def test_filters_benign_and_nlp(self):
        """Only unicode types (not benign, not nlp_attack) are evaluated."""
        y_true = pd.Series(["Diacritcs", "nlp_attack", "benign"])
        y_pred = pd.Series(["Diacritcs", "nlp_attack", "benign"])

        m = type_metrics(y_true, y_pred)
        # Only "Diacritcs" row is included
        assert m["type_accuracy"] == 1.0

    def test_empty_unicode_set(self):
        """No unicode types → returns default zeros."""
        y_true = pd.Series(["nlp_attack", "benign"])
        y_pred = pd.Series(["nlp_attack", "benign"])

        m = type_metrics(y_true, y_pred)
        assert m["type_accuracy"] == 0.0
        assert m["type_f1_macro"] == 0.0

    def test_type_report_present(self):
        """Result includes a per-type classification report."""
        y_true = pd.Series(["Diacritcs", "Zero Width"])
        y_pred = pd.Series(["Diacritcs", "Zero Width"])

        m = type_metrics(y_true, y_pred)
        assert "type_report" in m


# ---------------------------------------------------------------------------
# calibration_metrics
# ---------------------------------------------------------------------------
class TestCalibrationMetrics:
    """Tests for calibration_metrics()."""

    def test_all_correct_high_confidence(self):
        """All correct with confidence=0.95 lands in the 0.9-1.0 bucket."""
        y_true = pd.Series(["adversarial", "adversarial"])
        y_pred = pd.Series(["adversarial", "adversarial"])
        conf = pd.Series([0.95, 0.95])

        m = calibration_metrics(y_true, y_pred, conf)
        buckets = m["calibration_buckets"]
        assert len(buckets) >= 1
        # Find the bucket containing our samples
        high_bucket = [b for b in buckets if b["count"] > 0][0]
        assert high_bucket["accuracy"] == 1.0

    def test_mixed_confidence_bins(self):
        """Multiple confidence levels create multiple bins."""
        y_true = pd.Series(["a", "a", "a", "a"])
        y_pred = pd.Series(["a", "a", "b", "b"])
        conf = pd.Series([0.1, 0.2, 0.8, 0.9])

        m = calibration_metrics(y_true, y_pred, conf, n_bins=5)
        buckets = m["calibration_buckets"]
        assert len(buckets) >= 2

    def test_nan_confidence_filled(self):
        """NaN confidences are filled with 0.5."""
        y_true = pd.Series(["a"])
        y_pred = pd.Series(["a"])
        conf = pd.Series([float("nan")])

        m = calibration_metrics(y_true, y_pred, conf)
        buckets = m["calibration_buckets"]
        assert len(buckets) >= 1


# ---------------------------------------------------------------------------
# generate_report
# ---------------------------------------------------------------------------
class TestGenerateReport:
    """Tests for generate_report()."""

    @pytest.fixture
    def report_inputs(self):
        df = pd.DataFrame({"x": [1, 2]})
        binary = {
            "accuracy": 0.9, "adversarial_precision": 0.8,
            "adversarial_recall": 0.85, "adversarial_f1": 0.82,
            "benign_precision": 0.95, "benign_recall": 0.9,
            "benign_f1": 0.92, "false_negative_rate": 0.15,
            "support_adversarial": 100, "support_benign": 100,
        }
        category = {"category_accuracy": 0.88, "category_f1_macro": 0.87}
        types = {"type_accuracy": 0.75, "type_f1_macro": 0.70}
        cal = {"calibration_buckets": [
            {"bin": "0.8-0.9", "count": 50, "avg_confidence": 0.85, "accuracy": 0.82}
        ]}
        return df, binary, category, types, cal

    def test_contains_all_sections(self, report_inputs):
        """Report has sections for binary, category, type, calibration."""
        df, binary, category, types, cal = report_inputs
        report = generate_report(df, binary, category, types, cal)

        assert "## Binary Detection" in report
        assert "## Category Classification" in report
        assert "## Per-Type Classification" in report
        assert "## Calibration" in report

    def test_usage_section_optional(self, report_inputs):
        """Usage section only appears when usage dict is provided."""
        df, binary, category, types, cal = report_inputs

        report_no_usage = generate_report(df, binary, category, types, cal)
        assert "## Cost / Usage" not in report_no_usage

        report_with_usage = generate_report(
            df, binary, category, types, cal,
            usage={"total_tokens": 1000, "total_calls": 50},
        )
        assert "## Cost / Usage" in report_with_usage
        assert "total_tokens" in report_with_usage

    def test_returns_string(self, report_inputs):
        df, binary, category, types, cal = report_inputs
        report = generate_report(df, binary, category, types, cal)
        assert isinstance(report, str)
        assert len(report) > 0

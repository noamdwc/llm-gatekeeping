"""Tests for src.cli.infer_split scope breakdown helpers."""

import pandas as pd

from src.cli.infer_split import compute_scope_breakdown, format_scope_breakdown_markdown


class TestScopeBreakdown:
    def test_scope_breakdown_full_vs_ml_vs_nlp(self):
        df = pd.DataFrame({
            "label_binary": ["adversarial", "benign", "adversarial", "adversarial"],
            "label_category": ["unicode_attack", "benign", "nlp_attack", "nlp_attack"],
            "label_type": ["Diacritcs", "benign", "nlp_attack", "nlp_attack"],
        })
        preds = pd.DataFrame({
            "pred_label_binary": ["adversarial", "benign", "benign", "benign"],
            "pred_label_category": ["unicode_attack", "benign", "benign", "benign"],
            "pred_label_type": ["Diacritcs", "benign", "benign", "benign"],
            "confidence_label_binary": [0.98, 0.99, 0.80, 0.85],
        })

        breakdown = compute_scope_breakdown(df, preds)

        assert breakdown["full"]["rows"] == 4
        assert breakdown["ml_scope_no_nlp"]["rows"] == 2
        assert breakdown["nlp_only"]["rows"] == 2

        assert breakdown["full"]["binary"]["accuracy"] == 0.5
        assert breakdown["ml_scope_no_nlp"]["binary"]["accuracy"] == 1.0
        assert breakdown["nlp_only"]["binary"]["accuracy"] == 0.0

        assert breakdown["full"]["binary"]["false_negative_rate"] == 2 / 3
        assert breakdown["ml_scope_no_nlp"]["binary"]["false_negative_rate"] == 0.0
        assert breakdown["nlp_only"]["binary"]["false_negative_rate"] == 1.0

    def test_scope_breakdown_without_nlp_rows(self):
        df = pd.DataFrame({
            "label_binary": ["adversarial", "benign"],
            "label_category": ["unicode_attack", "benign"],
            "label_type": ["Diacritcs", "benign"],
        })
        preds = pd.DataFrame({
            "pred_label_binary": ["adversarial", "benign"],
            "pred_label_category": ["unicode_attack", "benign"],
            "pred_label_type": ["Diacritcs", "benign"],
            "confidence_label_binary": [0.99, 0.99],
        })

        breakdown = compute_scope_breakdown(df, preds)

        assert breakdown["full"]["rows"] == 2
        assert breakdown["ml_scope_no_nlp"]["rows"] == 2
        assert breakdown["nlp_only"]["rows"] == 0
        assert breakdown["nlp_only"]["binary"] is None

    def test_scope_breakdown_markdown_format(self):
        breakdown = {
            "full": {"rows": 4, "binary": {"accuracy": 0.5, "false_negative_rate": 2 / 3}},
            "ml_scope_no_nlp": {"rows": 2, "binary": {"accuracy": 1.0, "false_negative_rate": 0.0}},
            "nlp_only": {"rows": 2, "binary": {"accuracy": 0.0, "false_negative_rate": 1.0}},
        }

        md = format_scope_breakdown_markdown(breakdown)

        assert "## Scope Breakdown" in md
        assert "| full | 4 | 0.5000 | 0.6667 |" in md
        assert "| ml_scope_no_nlp | 2 | 1.0000 | 0.0000 |" in md
        assert "| nlp_only | 2 | 0.0000 | 1.0000 |" in md

"""Tests for the post-hoc benign risk model."""

import numpy as np
import pandas as pd
import pytest

from src.benign_risk_model import (
    BenignRiskDataset,
    CrossFittedEvaluator,
    PolicySimulator,
    TrainTestEvaluator,
    compute_calibration_metrics,
    logistic_risk_factory,
    margin_isotonic_factory,
)


def _make_trace_df(n_adv: int = 40, n_ben: int = 10, seed: int = 42) -> pd.DataFrame:
    """Synthetic trace DataFrame with the columns BenignRiskDataset expects."""
    rng = np.random.RandomState(seed)
    n = n_adv + n_ben
    true_labels = ["adversarial"] * n_adv + ["benign"] * n_ben
    rows = {
        "sample_id": [f"s{i}" for i in range(n)],
        "true_label": true_labels,
        "predicted_label": ["benign"] * n,  # all predicted benign (eligible)
        "route": ["llm"] * n,               # all LLM-path
        "margin": rng.uniform(0.0, 6.0, n),
        "top1_logprob": rng.uniform(-3.0, 0.0, n),
        "top2_logprob": rng.uniform(-6.0, -1.0, n),
        "self_reported_confidence": rng.uniform(50, 100, n),
        "margin_source_stage": rng.choice(["clf", "judge"], n),
    }
    return pd.DataFrame(rows)


@pytest.fixture
def trace_df():
    return _make_trace_df()


class TestBenignRiskDataset:
    def test_filters_to_eligible(self, trace_df):
        # Add some rows that should be filtered out
        extra = pd.DataFrame([
            {"sample_id": "ml1", "true_label": "adversarial", "predicted_label": "adversarial",
             "route": "ml", "margin": 1.0, "top1_logprob": -0.5, "top2_logprob": -2.0,
             "self_reported_confidence": 80, "margin_source_stage": "clf"},
            {"sample_id": "adv1", "true_label": "adversarial", "predicted_label": "adversarial",
             "route": "llm", "margin": 1.0, "top1_logprob": -0.5, "top2_logprob": -2.0,
             "self_reported_confidence": 80, "margin_source_stage": "clf"},
        ])
        combined = pd.concat([trace_df, extra], ignore_index=True)
        ds = BenignRiskDataset(combined)
        # Only the original 50 rows are eligible (LLM-path + predicted benign)
        assert len(ds.df) == 50
        assert len(ds.X) == 50
        assert len(ds.y) == 50

    def test_feature_columns(self, trace_df):
        ds = BenignRiskDataset(trace_df)
        expected_cols = ["margin", "top1_logprob", "top2_logprob",
                         "self_reported_confidence", "is_judge_stage"]
        assert list(ds.X.columns) == expected_cols

    def test_target_encoding(self, trace_df):
        ds = BenignRiskDataset(trace_df)
        assert ds.y.sum() == 40  # 40 adversarial
        assert (ds.y == 0).sum() == 10  # 10 benign

    def test_is_judge_stage_binary(self, trace_df):
        ds = BenignRiskDataset(trace_df)
        assert set(ds.X["is_judge_stage"].unique()).issubset({0, 1})

    def test_summary(self, trace_df):
        ds = BenignRiskDataset(trace_df)
        s = ds.summary()
        assert s["n_eligible"] == 50
        assert s["n_adversarial"] == 40
        assert s["n_benign"] == 10
        assert 0.0 <= s["base_rate_adversarial"] <= 1.0


class TestCrossFittedEvaluator:
    def test_evaluate_returns_correct_keys(self, trace_df):
        ds = BenignRiskDataset(trace_df)
        ev = CrossFittedEvaluator(n_splits=3, random_state=42)
        result = ev.evaluate(ds.X, ds.y, logistic_risk_factory)
        assert "fold_metrics" in result
        assert "aggregate" in result
        assert "predictions" in result
        assert len(result["fold_metrics"]) == 3
        assert len(result["predictions"]) == len(ds.y)

    def test_predictions_are_probabilities(self, trace_df):
        ds = BenignRiskDataset(trace_df)
        ev = CrossFittedEvaluator(n_splits=3, random_state=42)
        result = ev.evaluate(ds.X, ds.y, logistic_risk_factory)
        preds = result["predictions"]
        assert np.all(preds >= 0.0)
        assert np.all(preds <= 1.0)

    def test_isotonic_on_margin(self, trace_df):
        ds = BenignRiskDataset(trace_df)
        ev = CrossFittedEvaluator(n_splits=3, random_state=42)
        result = ev.evaluate(ds.X[["margin"]], ds.y, margin_isotonic_factory)
        assert len(result["predictions"]) == len(ds.y)
        assert all(0.0 <= p <= 1.0 for p in result["predictions"])

    def test_aggregate_metrics_structure(self, trace_df):
        ds = BenignRiskDataset(trace_df)
        ev = CrossFittedEvaluator(n_splits=3, random_state=42)
        result = ev.evaluate(ds.X, ds.y, logistic_risk_factory)
        for metric in ["roc_auc", "pr_auc", "brier"]:
            assert "mean" in result["aggregate"][metric]
            assert "std" in result["aggregate"][metric]

    def test_threshold_sweep(self, trace_df):
        ds = BenignRiskDataset(trace_df)
        ev = CrossFittedEvaluator(n_splits=3, random_state=42)
        result = ev.evaluate_threshold_sweep(ds.X["margin"], ds.y, [1.0, 2.0, 3.0])
        assert len(result["per_threshold"]) == 3
        for row in result["per_threshold"]:
            assert "tpr" in row
            assert "fpr" in row
            assert "n_flipped" in row


class TestTrainTestEvaluator:
    def test_evaluate_returns_correct_keys(self):
        train_df = _make_trace_df(n_adv=30, n_ben=10, seed=1)
        test_df = _make_trace_df(n_adv=20, n_ben=5, seed=2)
        train_ds = BenignRiskDataset(train_df)
        test_ds = BenignRiskDataset(test_df)
        ev = TrainTestEvaluator()
        result = ev.evaluate(train_ds.X, train_ds.y, test_ds.X, test_ds.y, logistic_risk_factory)
        assert "fold_metrics" in result
        assert "aggregate" in result
        assert "predictions" in result
        assert len(result["fold_metrics"]) == 1
        assert len(result["predictions"]) == len(test_ds.y)

    def test_predictions_are_probabilities(self):
        train_df = _make_trace_df(n_adv=30, n_ben=10, seed=1)
        test_df = _make_trace_df(n_adv=20, n_ben=5, seed=2)
        train_ds = BenignRiskDataset(train_df)
        test_ds = BenignRiskDataset(test_df)
        ev = TrainTestEvaluator()
        result = ev.evaluate(train_ds.X, train_ds.y, test_ds.X, test_ds.y, logistic_risk_factory)
        preds = result["predictions"]
        assert np.all(preds >= 0.0)
        assert np.all(preds <= 1.0)

    def test_isotonic_on_margin(self):
        train_df = _make_trace_df(n_adv=30, n_ben=10, seed=1)
        test_df = _make_trace_df(n_adv=20, n_ben=5, seed=2)
        train_ds = BenignRiskDataset(train_df)
        test_ds = BenignRiskDataset(test_df)
        ev = TrainTestEvaluator()
        result = ev.evaluate(
            train_ds.X[["margin"]], train_ds.y,
            test_ds.X[["margin"]], test_ds.y,
            margin_isotonic_factory,
        )
        assert len(result["predictions"]) == len(test_ds.y)
        assert all(0.0 <= p <= 1.0 for p in result["predictions"])

    def test_aggregate_std_is_zero(self):
        train_df = _make_trace_df(n_adv=30, n_ben=10, seed=1)
        test_df = _make_trace_df(n_adv=20, n_ben=5, seed=2)
        train_ds = BenignRiskDataset(train_df)
        test_ds = BenignRiskDataset(test_df)
        ev = TrainTestEvaluator()
        result = ev.evaluate(train_ds.X, train_ds.y, test_ds.X, test_ds.y, logistic_risk_factory)
        for metric in ["roc_auc", "pr_auc", "brier"]:
            assert result["aggregate"][metric]["std"] == 0.0


class TestPolicySimulator:
    def test_two_zone_shape(self):
        y = np.array([1, 1, 1, 0, 0])
        risk = np.array([0.9, 0.8, 0.3, 0.2, 0.7])
        df = PolicySimulator.simulate_two_zone(y, risk, [0.5, 0.7])
        assert len(df) == 2
        assert "tpr" in df.columns
        assert "fpr" in df.columns
        assert "n_flipped" in df.columns

    def test_two_zone_perfect(self):
        y = np.array([1, 1, 0, 0])
        risk = np.array([0.9, 0.8, 0.1, 0.2])
        df = PolicySimulator.simulate_two_zone(y, risk, [0.5])
        assert df.iloc[0]["tpr"] == 1.0
        assert df.iloc[0]["fpr"] == 0.0

    def test_three_zone_shape(self):
        y = np.array([1, 1, 1, 0, 0])
        risk = np.array([0.9, 0.5, 0.3, 0.2, 0.7])
        df = PolicySimulator.simulate_three_zone(y, risk, [(0.3, 0.7)])
        assert len(df) == 1
        assert "benign_zone_coverage" in df.columns
        assert "uncertain_zone_coverage" in df.columns
        assert "adversarial_zone_coverage" in df.columns

    def test_three_zone_coverages_sum_to_one(self):
        rng = np.random.RandomState(0)
        y = rng.randint(0, 2, 100)
        risk = rng.uniform(0, 1, 100)
        df = PolicySimulator.simulate_three_zone(y, risk, [(0.3, 0.7)])
        row = df.iloc[0]
        total = row["benign_zone_coverage"] + row["uncertain_zone_coverage"] + row["adversarial_zone_coverage"]
        assert abs(total - 1.0) < 0.01


class TestCalibrationMetrics:
    def test_perfect_calibration(self):
        # Perfectly calibrated: predicted probability matches actual rate
        y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        y_prob = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        result = compute_calibration_metrics(y_true, y_prob, n_bins=5)
        assert result["ece"] < 0.01

    def test_ece_bounded(self):
        rng = np.random.RandomState(42)
        y_true = rng.randint(0, 2, 200)
        y_prob = rng.uniform(0, 1, 200)
        result = compute_calibration_metrics(y_true, y_prob, n_bins=10)
        assert 0.0 <= result["ece"] <= 1.0
        assert len(result["bins"]) == 10

    def test_bins_have_required_keys(self):
        y_true = np.array([0, 1, 0, 1])
        y_prob = np.array([0.1, 0.9, 0.2, 0.8])
        result = compute_calibration_metrics(y_true, y_prob, n_bins=5)
        for b in result["bins"]:
            assert "bin_lo" in b
            assert "bin_hi" in b
            assert "count" in b

import numpy as np
import pandas as pd
import pytest
import json

from src.escalating_model import (
    ESCALATING_FEATURE_COLS,
    EVAL_SUMMARY_COLS,
    EscalatingDataset,
    EscalatingModel,
    evaluate_escalating_split,
)
from src.cli.train_escalating_model import main as train_escalating_main


def _make_colab_df() -> pd.DataFrame:
    return pd.DataFrame([
        {
            "sample_id": "s1",
            "label_binary": "adversarial",
            "llm_pred_binary": "benign",
            "llm_conf_binary": 0.80,
            "clf_confidence": 0.70,
            "clf_token_logprobs": "unused",
            "attack_name": "A",
        },
        {
            "sample_id": "s2",
            "label_binary": "benign",
            "llm_pred_binary": "benign",
            "llm_conf_binary": 0.60,
            "clf_confidence": 0.90,
            "clf_token_logprobs": "unused",
            "attack_name": "B",
        },
        {
            "sample_id": "s3",
            "label_binary": "adversarial",
            "llm_pred_binary": "adversarial",
            "llm_conf_binary": 0.40,
            "clf_confidence": 0.30,
            "clf_token_logprobs": "unused",
            "attack_name": "C",
        },
    ])


def _make_deberta_df() -> pd.DataFrame:
    return pd.DataFrame([
        {"sample_id": "s1", "deberta_proba_binary_adversarial": 0.90},
        {"sample_id": "s2", "deberta_proba_binary_adversarial": 0.20},
        {"sample_id": "s4", "deberta_proba_binary_adversarial": 0.75},
    ])


def _make_training_dataset(n: int = 20) -> EscalatingDataset:
    rows = []
    deberta_rows = []
    for i in range(n):
        label = "adversarial" if i % 2 == 0 else "benign"
        pred = "benign" if i % 4 == 0 else label
        rows.append({
            "sample_id": f"s{i}",
            "label_binary": label,
            "llm_pred_binary": pred,
            "llm_conf_binary": 0.55 + (i % 5) * 0.04,
            "clf_confidence": 0.50 + (i % 7) * 0.03,
            "clf_token_logprobs": "unused",
        })
        deberta_rows.append({
            "sample_id": f"s{i}",
            "deberta_proba_binary_adversarial": 0.15 + (i % 8) * 0.09,
        })
    return EscalatingDataset(pd.DataFrame(rows), pd.DataFrame(deberta_rows))


def _make_prediction_frames(n: int = 20) -> tuple[pd.DataFrame, pd.DataFrame]:
    colab_rows = []
    deberta_rows = []
    for i in range(n):
        label = "adversarial" if i % 2 == 0 else "benign"
        pred = "benign" if i % 4 == 0 else label
        colab_rows.append({
            "sample_id": f"s{i}",
            "label_binary": label,
            "llm_pred_binary": pred,
            "llm_conf_binary": 0.55 + (i % 5) * 0.04,
            "clf_confidence": 0.50 + (i % 7) * 0.03,
            "clf_token_logprobs": "unused",
        })
        deberta_rows.append({
            "sample_id": f"s{i}",
            "deberta_proba_binary_adversarial": 0.15 + (i % 8) * 0.09,
        })
    return pd.DataFrame(colab_rows), pd.DataFrame(deberta_rows)


class TestEscalatingDataset:
    def test_joined_dataset_contains_shared_sample_ids_only(self):
        ds = EscalatingDataset(_make_colab_df(), _make_deberta_df())

        assert list(ds.df["sample_id"]) == ["s1", "s2"]
        assert ds.rows_colab == 3
        assert ds.rows_deberta == 3
        assert ds.rows_joined == 2
        assert ds.rows_dropped_colab_only == 1
        assert ds.rows_dropped_deberta_only == 1

    def test_needs_escalation_target_uses_cheap_prediction_error(self):
        ds = EscalatingDataset(_make_colab_df(), _make_deberta_df())

        by_id = ds.df.set_index("sample_id")
        assert by_id.loc["s1", "needs_escalation"] == 1
        assert by_id.loc["s2", "needs_escalation"] == 0
        assert list(ds.y) == [1, 0]

    def test_derived_features(self):
        ds = EscalatingDataset(_make_colab_df(), _make_deberta_df())
        by_id = ds.df.set_index("sample_id")

        assert by_id.loc["s1", "llm_pred_is_adversarial"] == 0
        assert by_id.loc["s1", "deberta_pred_is_adversarial"] == 1
        assert by_id.loc["s1", "deberta_llm_disagree"] == 1
        assert by_id.loc["s1", "llm_distance_from_uncertain"] == pytest.approx(0.30)
        assert by_id.loc["s1", "deberta_distance_from_uncertain"] == pytest.approx(0.40)

        assert by_id.loc["s2", "llm_pred_is_adversarial"] == 0
        assert by_id.loc["s2", "deberta_pred_is_adversarial"] == 0
        assert by_id.loc["s2", "deberta_llm_disagree"] == 0
        assert by_id.loc["s2", "llm_distance_from_uncertain"] == pytest.approx(0.10)
        assert by_id.loc["s2", "deberta_distance_from_uncertain"] == pytest.approx(0.30)

    def test_derives_classifier_logprob_features_from_llm_prediction_output(self):
        colab = _make_colab_df().iloc[:1].copy()
        colab.loc[0, "clf_token_logprobs"] = json.dumps([
            {"token": "{"},
            {"token": "\""},
            {"token": "label"},
            {"token": "\":"},
            {
                "token": " benign",
                "top_logprobs": [
                    {"token": " benign", "logprob": -0.20},
                    {"token": " adversarial", "logprob": -1.30},
                    {"token": " other", "logprob": -5.00},
                ],
            },
        ])
        deberta = pd.DataFrame([
            {"sample_id": "s1", "deberta_proba_binary_adversarial": 0.90},
        ])

        ds = EscalatingDataset(colab, deberta)

        assert ds.df.loc[0, "clf_top1_logprob"] == pytest.approx(-0.20)
        assert ds.df.loc[0, "clf_top2_logprob"] == pytest.approx(-1.30)
        assert ds.df.loc[0, "clf_logprob_diff"] == pytest.approx(1.10)

    def test_uses_selected_label_token_logprob_when_top_logprobs_are_absent(self):
        colab = _make_colab_df().iloc[:1].copy()
        colab.loc[0, "clf_token_logprobs"] = json.dumps([
            {"token": "{\"", "logprob": -0.01},
            {"token": "label", "logprob": -0.02},
            {"token": "\":", "logprob": -0.03},
            {"token": " \"", "logprob": -0.04},
            {"token": "ben", "logprob": -0.25},
            {"token": "ign", "logprob": 0.0},
        ])
        deberta = pd.DataFrame([
            {"sample_id": "s1", "deberta_proba_binary_adversarial": 0.90},
        ])

        ds = EscalatingDataset(colab, deberta)

        assert ds.df.loc[0, "clf_top1_logprob"] == pytest.approx(-0.25)
        assert ds.df.loc[0, "clf_top2_logprob"] == 0.0
        assert ds.df.loc[0, "clf_logprob_diff"] == 0.0

    def test_feature_columns(self):
        ds = EscalatingDataset(_make_colab_df(), _make_deberta_df())

        assert list(ds.X.columns) == ESCALATING_FEATURE_COLS

    def test_duplicate_deberta_rows_are_collapsed_to_one_sample(self):
        colab = _make_colab_df().iloc[:2].copy()
        deberta = pd.DataFrame([
            {"sample_id": "s1", "deberta_proba_binary_adversarial": 0.90},
            {"sample_id": "s1", "deberta_proba_binary_adversarial": 0.90},
            {"sample_id": "s2", "deberta_proba_binary_adversarial": 0.20},
        ])

        ds = EscalatingDataset(colab, deberta)

        assert list(ds.df["sample_id"]) == ["s1", "s2"]
        assert ds.rows_deberta == 3
        assert ds.rows_joined == 2
        assert ds.rows_dropped_deberta_only == 1

    def test_conflicting_duplicate_deberta_rows_are_averaged(self):
        colab = _make_colab_df().iloc[:1].copy()
        deberta = pd.DataFrame([
            {"sample_id": "s1", "deberta_proba_binary_adversarial": 0.90},
            {"sample_id": "s1", "deberta_proba_binary_adversarial": 0.10},
        ])

        ds = EscalatingDataset(colab, deberta)

        assert ds.df.loc[0, "deberta_proba_binary_adversarial"] == pytest.approx(0.50)

    def test_conflicting_duplicate_colab_rows_raise(self):
        colab = pd.concat([_make_colab_df().iloc[:1], _make_colab_df().iloc[:1]], ignore_index=True)
        colab.loc[1, "llm_pred_binary"] = "adversarial"
        deberta = pd.DataFrame([
            {"sample_id": "s1", "deberta_proba_binary_adversarial": 0.90},
        ])

        with pytest.raises(ValueError, match="conflicting duplicate sample_id"):
            EscalatingDataset(colab, deberta)


class TestEscalatingModel:
    def test_train_and_predict_escalation_batch(self):
        ds = _make_training_dataset()
        model = EscalatingModel.train(ds.X, ds.y, list(ESCALATING_FEATURE_COLS))

        assert model.pipeline.named_steps["lgbm"].__class__.__name__ == "LGBMClassifier"

        scores = model.predict_escalation_batch(ds.df)

        assert len(scores) == len(ds.df)
        assert np.all(scores >= 0.0)
        assert np.all(scores <= 1.0)

    def test_save_and_load(self, tmp_path):
        ds = _make_training_dataset()
        model = EscalatingModel.train(ds.X, ds.y, list(ESCALATING_FEATURE_COLS))
        before = model.predict_escalation_batch(ds.df)

        path = tmp_path / "escalating_model.pkl"
        model.save(path)
        loaded = EscalatingModel.load(path)
        after = loaded.predict_escalation_batch(ds.df)

        assert loaded.feature_cols == list(ESCALATING_FEATURE_COLS)
        np.testing.assert_allclose(before, after)


class TestPocEvaluation:
    def test_split_summary_counts_and_metrics(self):
        ds = _make_training_dataset(n=10)
        scores = np.array([0.99, 0.01, 0.20, 0.30, 0.95, 0.40, 0.50, 0.60, 0.90, 0.70])

        scored, summary = evaluate_escalating_split("test", ds, scores)

        assert set(EVAL_SUMMARY_COLS).issubset(summary.keys())
        assert len(scored) == 10
        assert summary["split"] == "test"
        assert summary["rows_colab"] == 10
        assert summary["rows_deberta"] == 10
        assert summary["rows_joined"] == 10
        assert summary["rows_dropped_colab_only"] == 0
        assert summary["rows_dropped_deberta_only"] == 0
        assert summary["cheap_error_rate"] == 0.3
        assert summary["top_10pct_error_rate"] == 1.0
        assert summary["bottom_50pct_error_rate"] == 0.0
        assert summary["top_10pct_adversarial_fn_rate"] == 1.0

    def test_top_10pct_adversarial_fn_rate_is_null_without_true_adversarial_top_bucket(self):
        ds = _make_training_dataset(n=10)
        scores = np.array([0.01, 0.99, 0.20, 0.30, 0.95, 0.40, 0.50, 0.60, 0.90, 0.70])

        _, summary = evaluate_escalating_split("test", ds, scores)

        assert summary["top_10pct_adversarial_fn_rate"] is None


class TestTrainEscalatingModelCli:
    def test_cli_writes_expected_artifacts(self, tmp_path):
        train_colab, train_deberta = _make_prediction_frames(n=24)
        train_colab_path = tmp_path / "llm_predictions_val_colab_local_classifier.parquet"
        train_deberta_path = tmp_path / "deberta_predictions_val.parquet"
        train_colab.to_parquet(train_colab_path, index=False)
        train_deberta.to_parquet(train_deberta_path, index=False)

        eval_args = []
        for split in ["test", "unseen_val", "unseen_test", "safeguard_test"]:
            colab, deberta = _make_prediction_frames(n=12)
            colab_path = tmp_path / f"llm_predictions_{split}_colab_local_classifier.parquet"
            deberta_path = tmp_path / f"deberta_predictions_{split}.parquet"
            colab.to_parquet(colab_path, index=False)
            deberta.to_parquet(deberta_path, index=False)
            eval_args.extend([
                "--eval-split",
                split,
                str(colab_path),
                str(deberta_path),
            ])

        config_path = tmp_path / "default.yaml"
        config_path.write_text(
            "hybrid:\n"
            "  escalating_model:\n"
            "    enabled: false\n"
            "    model_path: data/processed/models/escalating_model.pkl\n"
        )

        model_output = tmp_path / "models" / "escalating_model.pkl"
        research_output_dir = tmp_path / "research"
        summary_output = research_output_dir / "escalating_model_summary.csv"
        report_output = tmp_path / "reports" / "escalating_model_poc.md"

        train_escalating_main([
            "--config",
            str(config_path),
            "--train-colab-predictions",
            str(train_colab_path),
            "--train-deberta-predictions",
            str(train_deberta_path),
            "--model-output",
            str(model_output),
            "--research-output-dir",
            str(research_output_dir),
            "--summary-output",
            str(summary_output),
            "--report-output",
            str(report_output),
            *eval_args,
        ])

        assert model_output.exists()
        for split in ["test", "unseen_val", "unseen_test", "safeguard_test"]:
            eval_output = research_output_dir / f"escalating_model_eval_{split}.parquet"
            assert eval_output.exists()
            assert "escalation_score" in pd.read_parquet(eval_output).columns
        assert summary_output.exists()
        assert report_output.exists()

        summary = pd.read_csv(summary_output)
        assert list(summary["split"]) == ["test", "unseen_val", "unseen_test", "safeguard_test"]
        assert list(summary.columns) == EVAL_SUMMARY_COLS

"""Offline POC model for deciding whether cheap LLM outputs need judge escalation."""

from __future__ import annotations

import math
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline

from src.logprob_margin import (
    extract_margin_features,
    find_label_start_position,
    safe_json_loads,
)


REQUIRED_COLAB_COLS = [
    "sample_id",
    "label_binary",
    "llm_pred_binary",
    "llm_conf_binary",
    "clf_confidence",
    "clf_token_logprobs",
]

REQUIRED_DEBERTA_COLS = [
    "sample_id",
    "deberta_proba_binary_adversarial",
]

ESCALATING_FEATURE_COLS = [
    "llm_conf_binary",
    "clf_confidence",
    "deberta_proba_binary_adversarial",
    "llm_pred_is_adversarial",
    "deberta_pred_is_adversarial",
    "deberta_llm_disagree",
    "llm_distance_from_uncertain",
    "deberta_distance_from_uncertain",
    "clf_top1_logprob",
    "clf_top2_logprob",
    "clf_logprob_diff",
]

EVAL_SUMMARY_COLS = [
    "split",
    "rows_colab",
    "rows_deberta",
    "rows_joined",
    "rows_dropped_colab_only",
    "rows_dropped_deberta_only",
    "cheap_error_rate",
    "roc_auc",
    "pr_auc",
    "top_10pct_error_rate",
    "top_10pct_adversarial_fn_rate",
    "bottom_50pct_error_rate",
]


def _require_columns(df: pd.DataFrame, cols: list[str], name: str) -> None:
    missing = [col for col in cols if col not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def _deduplicate_by_sample_id(
    df: pd.DataFrame,
    cols: list[str],
    name: str,
    average_numeric_conflicts: bool = False,
) -> pd.DataFrame:
    duplicated = df[df["sample_id"].duplicated(keep=False)]
    if duplicated.empty:
        return df.copy()

    compare_cols = [col for col in cols if col != "sample_id"]
    conflicts = (
        duplicated.groupby("sample_id")[compare_cols]
        .nunique(dropna=False)
        .gt(1)
        .any(axis=1)
    )
    conflicting_ids = conflicts[conflicts].index.tolist()
    if conflicting_ids:
        if average_numeric_conflicts and all(
            pd.api.types.is_numeric_dtype(df[col]) for col in compare_cols
        ):
            return df.groupby("sample_id", as_index=False)[cols].mean(numeric_only=True)
        raise ValueError(
            f"{name} has conflicting duplicate sample_id rows: {conflicting_ids[:5]}"
        )
    return df.drop_duplicates(subset=["sample_id"], keep="first").copy()


def _safe_rate(numerator: int | float, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return float(numerator) / denominator


def _safe_auc(y_true: pd.Series | np.ndarray, scores: pd.Series | np.ndarray) -> float | None:
    y_arr = np.asarray(y_true)
    if len(np.unique(y_arr)) < 2:
        return None
    return float(roc_auc_score(y_arr, scores))


def _safe_pr_auc(y_true: pd.Series | np.ndarray, scores: pd.Series | np.ndarray) -> float | None:
    y_arr = np.asarray(y_true)
    if len(y_arr) == 0 or int(y_arr.sum()) == 0:
        return None
    return float(average_precision_score(y_arr, scores))


def _extract_clf_logprob_features(value: object) -> pd.Series:
    token_logprobs = safe_json_loads(value)
    margin = extract_margin_features(token_logprobs, mode="clf", source_stage="clf")
    top1 = margin.top1_logprob if margin.top1_logprob is not None else 0.0
    if margin.top1_logprob is None:
        label_idx = find_label_start_position(token_logprobs, mode="clf")
        if label_idx is not None and token_logprobs is not None and label_idx < len(token_logprobs):
            token_payload = token_logprobs[label_idx] or {}
            token_logprob = token_payload.get("logprob")
            if isinstance(token_logprob, (int, float)):
                top1 = float(token_logprob)
    top2 = margin.top2_logprob if margin.top2_logprob is not None else 0.0
    diff = margin.margin if margin.margin is not None else 0.0
    return pd.Series({
        "clf_top1_logprob": top1,
        "clf_top2_logprob": top2,
        "clf_logprob_diff": diff,
    })


class EscalatingDataset:
    """Join cheap LLM classifier and DeBERTa predictions and build POC features."""

    def __init__(self, colab_df: pd.DataFrame, deberta_df: pd.DataFrame) -> None:
        _require_columns(colab_df, REQUIRED_COLAB_COLS, "colab_df")
        _require_columns(deberta_df, REQUIRED_DEBERTA_COLS, "deberta_df")

        self.rows_colab = len(colab_df)
        self.rows_deberta = len(deberta_df)

        colab_unique = _deduplicate_by_sample_id(
            colab_df,
            REQUIRED_COLAB_COLS,
            "colab_df",
        )
        deberta_unique = _deduplicate_by_sample_id(
            deberta_df,
            REQUIRED_DEBERTA_COLS,
            "deberta_df",
            average_numeric_conflicts=True,
        )

        joined = colab_unique.merge(
            deberta_unique[REQUIRED_DEBERTA_COLS],
            on="sample_id",
            how="inner",
            validate="one_to_one",
        ).copy()

        self.rows_dropped_colab_only = self.rows_colab - len(joined)
        self.rows_dropped_deberta_only = self.rows_deberta - len(joined)

        joined["needs_escalation"] = (
            joined["llm_pred_binary"] != joined["label_binary"]
        ).astype(int)
        joined["llm_pred_is_adversarial"] = (
            joined["llm_pred_binary"] == "adversarial"
        ).astype(int)
        joined["deberta_pred_is_adversarial"] = (
            joined["deberta_proba_binary_adversarial"].astype(float) >= 0.5
        ).astype(int)
        joined["deberta_llm_disagree"] = (
            joined["llm_pred_is_adversarial"]
            != joined["deberta_pred_is_adversarial"]
        ).astype(int)
        joined["llm_distance_from_uncertain"] = (
            joined["llm_conf_binary"].astype(float) - 0.5
        ).abs()
        joined["deberta_distance_from_uncertain"] = (
            joined["deberta_proba_binary_adversarial"].astype(float) - 0.5
        ).abs()
        logprob_features = joined["clf_token_logprobs"].apply(_extract_clf_logprob_features)
        joined = pd.concat([joined, logprob_features], axis=1)

        self._df = joined.reset_index(drop=True)

    @property
    def df(self) -> pd.DataFrame:
        return self._df

    @property
    def X(self) -> pd.DataFrame:
        return self._df[ESCALATING_FEATURE_COLS].astype(float).fillna(0.0)

    @property
    def y(self) -> pd.Series:
        return self._df["needs_escalation"].astype(int)

    @property
    def rows_joined(self) -> int:
        return len(self._df)


class EscalatingModel:
    """Trained escalating model that emits P(cheap path is wrong)."""

    def __init__(self, pipeline: Pipeline, feature_cols: list[str] | None = None) -> None:
        self.pipeline = pipeline
        self.feature_cols = feature_cols or list(ESCALATING_FEATURE_COLS)

    @staticmethod
    def train(
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray,
        feature_cols: list[str] | None = None,
    ) -> "EscalatingModel":
        pipeline = Pipeline([
            ("lgbm", LGBMClassifier(random_state=42, verbosity=-1)),
        ])
        pipeline.fit(X, y)
        return EscalatingModel(pipeline=pipeline, feature_cols=feature_cols)

    def predict_escalation_batch(self, df: pd.DataFrame) -> np.ndarray:
        X = df[self.feature_cols].astype(float).fillna(0.0)
        return self.pipeline.predict_proba(X)[:, 1]

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(
                {"pipeline": self.pipeline, "feature_cols": self.feature_cols},
                f,
            )

    @classmethod
    def load(cls, path: str | Path) -> "EscalatingModel":
        with open(path, "rb") as f:
            obj = pickle.load(f)
        return cls(pipeline=obj["pipeline"], feature_cols=obj["feature_cols"])


def evaluate_escalating_split(
    split: str,
    dataset: EscalatingDataset,
    escalation_scores: np.ndarray | pd.Series,
) -> tuple[pd.DataFrame, dict]:
    """Attach scores to a split and return scored rows plus POC summary metrics."""
    scored = dataset.df.copy()
    scored["escalation_score"] = np.asarray(escalation_scores, dtype=float)
    scored_desc = scored.sort_values(
        "escalation_score",
        ascending=False,
        kind="mergesort",
    )

    n = len(scored)
    top_n = math.ceil(n * 0.10) if n else 0
    bottom_n = math.ceil(n * 0.50) if n else 0
    top = scored_desc.head(top_n)
    bottom = scored_desc.tail(bottom_n)

    top_adv = top[top["label_binary"] == "adversarial"]
    top_adv_fn = (
        (top_adv["llm_pred_binary"] == "benign").sum()
        if len(top_adv) > 0
        else 0
    )

    summary = {
        "split": split,
        "rows_colab": dataset.rows_colab,
        "rows_deberta": dataset.rows_deberta,
        "rows_joined": dataset.rows_joined,
        "rows_dropped_colab_only": dataset.rows_dropped_colab_only,
        "rows_dropped_deberta_only": dataset.rows_dropped_deberta_only,
        "cheap_error_rate": _safe_rate(int(scored["needs_escalation"].sum()), n),
        "roc_auc": _safe_auc(scored["needs_escalation"], scored["escalation_score"]),
        "pr_auc": _safe_pr_auc(scored["needs_escalation"], scored["escalation_score"]),
        "top_10pct_error_rate": _safe_rate(int(top["needs_escalation"].sum()), len(top)),
        "top_10pct_adversarial_fn_rate": _safe_rate(int(top_adv_fn), len(top_adv)),
        "bottom_50pct_error_rate": _safe_rate(
            int(bottom["needs_escalation"].sum()),
            len(bottom),
        ),
    }
    return scored, summary


def write_escalating_report(summary_df: pd.DataFrame, path: str | Path) -> None:
    """Write the compact markdown report for the offline POC."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    report = [
        "# Escalating Model POC",
        "",
        "This offline POC trains a model to estimate `P(cheap path is wrong)` from "
        "Colab/local classifier output and DeBERTa output. It does not choose a "
        "production threshold or integrate with the hybrid router.",
        "",
        "Parsed `clf_token_logprobs` features are intentionally omitted in this "
        "version except for top-1, top-2, and top-1 minus top-2 label-token "
        "logprob features from the cheap/local LLM classifier output.",
        "",
        "## Evaluation Summary",
        "",
        summary_df.to_markdown(index=False),
        "",
    ]
    path.write_text("\n".join(report))

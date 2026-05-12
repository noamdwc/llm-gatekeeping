"""Escalating model for deciding whether cheap LLM outputs need judge escalation."""

from __future__ import annotations

import math
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
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

THRESHOLD_SWEEP_COLS = [
    "threshold",
    "rows",
    "judge_call_rate",
    "judge_calls",
    "trusted_rows",
    "cheap_errors_total",
    "cheap_errors_caught",
    "cheap_errors_missed",
    "cheap_error_catch_rate",
    "non_escalated_error_rate",
]

POSTSCORE_SPLIT_MAP_COLS = [
    "prompt_hash",
    "postscore_split",
    "stratum",
    "rows",
    "cheap_errors",
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


def evaluate_threshold_sweep(
    scored_df: pd.DataFrame,
    thresholds: list[float] | np.ndarray | None = None,
    score_col: str = "escalation_score",
) -> pd.DataFrame:
    """Compute judge-escalation operating points from scored rows."""
    _require_columns(
        scored_df,
        ["needs_escalation", score_col],
        "scored_df",
    )
    if thresholds is None:
        thresholds = np.round(np.arange(0.0, 1.0001, 0.05), 2)

    needs_escalation = scored_df["needs_escalation"].astype(int)
    scores = scored_df[score_col].astype(float)
    rows = len(scored_df)
    cheap_errors_total = int(needs_escalation.sum())

    sweep_rows = []
    for threshold in thresholds:
        threshold = float(threshold)
        escalate = scores >= threshold
        judge_calls = int(escalate.sum())
        trusted_rows = rows - judge_calls
        cheap_errors_caught = int((needs_escalation & escalate).sum())
        cheap_errors_missed = cheap_errors_total - cheap_errors_caught
        sweep_rows.append({
            "threshold": threshold,
            "rows": rows,
            "judge_call_rate": _safe_rate(judge_calls, rows),
            "judge_calls": judge_calls,
            "trusted_rows": trusted_rows,
            "cheap_errors_total": cheap_errors_total,
            "cheap_errors_caught": cheap_errors_caught,
            "cheap_errors_missed": cheap_errors_missed,
            "cheap_error_catch_rate": _safe_rate(cheap_errors_caught, cheap_errors_total),
            "non_escalated_error_rate": _safe_rate(cheap_errors_missed, trusted_rows),
        })

    return pd.DataFrame(sweep_rows, columns=THRESHOLD_SWEEP_COLS)


def _group_label(labels: pd.Series) -> str:
    values = sorted(set(labels.dropna().astype(str)))
    return values[0] if len(values) == 1 else "mixed:" + "+".join(values)


def _group_attack(values: pd.Series) -> str:
    clean = sorted(v for v in set(values.dropna().astype(str)) if v and v != "nan")
    if not clean:
        return "benign"
    return clean[0] if len(clean) == 1 else "mixed:" + "+".join(clean)


def build_postscore_split_map(
    scored_df: pd.DataFrame,
    seed: int = 42,
) -> tuple[pd.DataFrame, dict]:
    """Split scored unseen_val prompt_hash groups into calibration/threshold halves."""
    _require_columns(
        scored_df,
        ["sample_id", "prompt_hash", "label_binary", "attack_name", "needs_escalation"],
        "scored_df",
    )
    grouped = scored_df.groupby("prompt_hash").agg(
        rows=("sample_id", "size"),
        label_group=("label_binary", _group_label),
        attack_group=("attack_name", _group_attack),
        cheap_errors=("needs_escalation", "sum"),
    ).reset_index()
    grouped["has_error"] = grouped["cheap_errors"].astype(int) > 0
    grouped["stratum"] = np.where(
        grouped["label_group"].eq("adversarial"),
        "adv:" + grouped["attack_group"].astype(str),
        "benign",
    ) + np.where(grouped["has_error"], "|err", "|ok")

    rng = np.random.RandomState(seed)
    assignments: list[dict] = []
    for _, part in grouped.groupby("stratum", sort=True):
        part = part.copy()
        part["_rand"] = rng.random(len(part))
        part = part.sort_values(
            ["cheap_errors", "rows", "_rand"],
            ascending=[False, False, True],
        )
        total_rows = int(part["rows"].sum())
        total_errors = int(part["cheap_errors"].sum())
        state = {
            "calibration": {"rows": 0, "cheap_errors": 0, "groups": 0},
            "threshold": {"rows": 0, "cheap_errors": 0, "groups": 0},
        }

        for pos, row in enumerate(part.itertuples(index=False)):
            remaining = len(part) - pos
            if len(part) >= 2 and state["calibration"]["groups"] == 0 and remaining == 1:
                side = "calibration"
            elif len(part) >= 2 and state["threshold"]["groups"] == 0 and remaining == 1:
                side = "threshold"
            else:
                objectives = {}
                for candidate in ("calibration", "threshold"):
                    cal_rows = state["calibration"]["rows"]
                    thr_rows = state["threshold"]["rows"]
                    cal_errors = state["calibration"]["cheap_errors"]
                    thr_errors = state["threshold"]["cheap_errors"]
                    if candidate == "calibration":
                        cal_rows += int(row.rows)
                        cal_errors += int(row.cheap_errors)
                    else:
                        thr_rows += int(row.rows)
                        thr_errors += int(row.cheap_errors)
                    error_denom = max(total_errors, 1)
                    row_denom = max(total_rows, 1)
                    objectives[candidate] = (
                        abs(cal_errors - thr_errors) / error_denom
                        + abs(cal_rows - thr_rows) / row_denom
                    )
                side = min(objectives, key=lambda key: (objectives[key], key != "calibration"))

            state[side]["rows"] += int(row.rows)
            state[side]["cheap_errors"] += int(row.cheap_errors)
            state[side]["groups"] += 1
            assignments.append({
                "prompt_hash": row.prompt_hash,
                "postscore_split": side,
                "stratum": row.stratum,
                "rows": int(row.rows),
                "cheap_errors": int(row.cheap_errors),
            })

    split_map = pd.DataFrame(assignments, columns=POSTSCORE_SPLIT_MAP_COLS)
    cal_hashes = set(split_map.query("postscore_split == 'calibration'")["prompt_hash"])
    threshold_hashes = set(split_map.query("postscore_split == 'threshold'")["prompt_hash"])
    overlap = cal_hashes & threshold_hashes
    if overlap:
        raise AssertionError(f"postscore split prompt_hash overlap: {sorted(overlap)[:5]}")

    assigned = scored_df.merge(
        split_map[["prompt_hash", "postscore_split"]],
        on="prompt_hash",
        how="left",
        validate="many_to_one",
    )
    diagnostics = build_postscore_split_diagnostics(assigned, split_map, len(overlap))
    return split_map, diagnostics


def build_postscore_split_diagnostics(
    assigned_df: pd.DataFrame,
    split_map: pd.DataFrame,
    prompt_hash_overlap: int,
) -> dict:
    """Build report-ready diagnostics for the post-score unseen_val split."""
    summary = assigned_df.groupby("postscore_split").agg(
        rows=("sample_id", "size"),
        prompt_hash_groups=("prompt_hash", "nunique"),
        cheap_errors=("needs_escalation", "sum"),
    ).sort_index()
    summary["error_rate"] = summary["cheap_errors"] / summary["rows"]

    label_counts = pd.crosstab(
        assigned_df["label_binary"],
        assigned_df["postscore_split"],
    )
    report_group = np.where(
        assigned_df["label_binary"].eq("benign"),
        "benign",
        assigned_df["attack_name"].fillna("unknown").astype(str),
    )
    attack_counts = pd.crosstab(report_group, assigned_df["postscore_split"])
    attack_counts.index.name = "attack_or_benign_group"
    stratum_counts = split_map.groupby(["stratum", "postscore_split"]).agg(
        prompt_hash_groups=("prompt_hash", "size"),
        rows=("rows", "sum"),
        cheap_errors=("cheap_errors", "sum"),
    ).reset_index()

    return {
        "summary": summary,
        "label_counts": label_counts,
        "attack_group_counts": attack_counts,
        "stratum_counts": stratum_counts,
        "prompt_hash_overlap": int(prompt_hash_overlap),
    }


class _IdentityCalibrator:
    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        values = np.asarray(X, dtype=float).reshape(-1)
        values = np.clip(values, 0.0, 1.0)
        return np.column_stack([1.0 - values, values])


def fit_score_calibrator(
    calibration_df: pd.DataFrame,
    method: str = "sigmoid",
):
    """Fit post-hoc calibration for escalation scores."""
    _require_columns(calibration_df, ["needs_escalation", "escalation_score"], "calibration_df")
    y = calibration_df["needs_escalation"].astype(int).to_numpy()
    scores = calibration_df[["escalation_score"]].astype(float).to_numpy()
    if len(np.unique(y)) < 2:
        return _IdentityCalibrator()
    method = method.lower()
    if method == "sigmoid":
        calibrator = LogisticRegression(solver="lbfgs")
        calibrator.fit(scores, y)
        return calibrator
    if method == "isotonic":
        calibrator = IsotonicRegression(out_of_bounds="clip")
        calibrator.fit(scores.reshape(-1), y)
        return calibrator
    raise ValueError("Unsupported escalating calibration method: " f"{method}")


def apply_score_calibrator(
    scored_df: pd.DataFrame,
    calibrator,
    output_col: str = "calibrated_escalation_score",
) -> pd.DataFrame:
    """Attach calibrated escalation scores to a scored frame."""
    out = scored_df.copy()
    scores = out[["escalation_score"]].astype(float).to_numpy()
    if isinstance(calibrator, IsotonicRegression):
        calibrated = calibrator.predict(scores.reshape(-1))
    else:
        calibrated = calibrator.predict_proba(scores)[:, 1]
    out[output_col] = np.clip(calibrated, 0.0, 1.0)
    return out


def write_escalating_report(
    summary_df: pd.DataFrame,
    path: str | Path,
    threshold_sweep_df: pd.DataFrame | None = None,
    postscore_split_diagnostics: dict | None = None,
    calibration_method: str = "sigmoid",
) -> None:
    """Write the compact markdown report for the escalating model."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    report = [
        "# Escalating Model",
        "",
        "This model estimates `P(cheap path is wrong)` from Colab/local "
        "classifier output and DeBERTa output. The canonical final-verdict "
        "pipeline uses `hybrid.escalating_model.judge_threshold` to decide "
        "which cheap-classifier rows are escalated to the stronger judge.",
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
    if postscore_split_diagnostics is not None:
        summary = postscore_split_diagnostics["summary"]
        label_counts = postscore_split_diagnostics["label_counts"]
        attack_counts = postscore_split_diagnostics["attack_group_counts"]
        total_errors = int(summary["cheap_errors"].sum())
        calibration_errors = int(summary.loc["calibration", "cheap_errors"])
        threshold_errors = int(summary.loc["threshold", "cheap_errors"])
        one_error_pp = 100.0 / threshold_errors if threshold_errors else 0.0
        report.extend([
            "## Post-score unseen_val Split Diagnostics",
            "",
            f"Calibration method: `{calibration_method}`.",
            "",
            summary.to_markdown(),
            "",
            "### Label Counts",
            "",
            label_counts.to_markdown(),
            "",
            "### Attack / Benign Group Counts",
            "",
            attack_counts.to_markdown(),
            "",
            f"Prompt hash overlap: {postscore_split_diagnostics['prompt_hash_overlap']}",
            "",
            "## Limitations / Statistical Power",
            "",
            f"`unseen_val` has only {total_errors} cheap-path errors total. "
            f"The calibration half has {calibration_errors} cheap-path errors, "
            f"and the threshold-selection half has {threshold_errors} cheap-path errors. "
            "Calibration and threshold estimates are therefore noisy.",
            "",
            f"One missed cheap-path error in the threshold half changes the missed-error "
            f"rate by about {one_error_pp:.1f} percentage points. Per-attack conclusions "
            "are diagnostic only. The selected `0.5` threshold is the frozen "
            "operating point for the current canonical POC path because it sits "
            "on a useful cost/error tradeoff without making judge-everything "
            "the default.",
            "",
        ])
    if threshold_sweep_df is not None:
        report.extend([
            "## Threshold Sweep",
            "",
            "Threshold operating points are selected on `unseen_val` because the "
            "escalating model is trained on `val`.",
            "",
            threshold_sweep_df.to_markdown(index=False),
            "",
        ])
    path.write_text("\n".join(report))

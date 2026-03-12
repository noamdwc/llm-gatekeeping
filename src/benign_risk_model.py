"""Post-hoc benign risk model: P(true_adversarial | LLM predicted benign, trace features).

Estimates the probability that a sample the LLM called "benign" is actually adversarial,
using logprob margin and other trace features. Cross-fitted evaluation avoids overfitting
on the same test set used to generate the traces.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


FEATURE_COLS = [
    "margin",
    "top1_logprob",
    "top2_logprob",
    "self_reported_confidence",
    "is_judge_stage",
]


class BenignRiskDataset:
    """Filter a margin trace to LLM-path benign predictions and build features."""

    def __init__(self, trace_df: pd.DataFrame) -> None:
        mask = (trace_df["route"] != "ml") & (trace_df["predicted_label"] == "benign")
        self._df = trace_df.loc[mask].copy().reset_index(drop=True)
        self._df["is_judge_stage"] = (
            self._df["margin_source_stage"] == "judge"
        ).astype(int)
        self._y = (self._df["true_label"] == "adversarial").astype(int)

    @property
    def df(self) -> pd.DataFrame:
        return self._df

    @property
    def X(self) -> pd.DataFrame:
        return self._df[FEATURE_COLS]

    @property
    def y(self) -> pd.Series:
        return self._y

    def summary(self) -> dict:
        n = len(self._df)
        n_adv = int(self._y.sum())
        return {
            "n_eligible": n,
            "n_adversarial": n_adv,
            "n_benign": n - n_adv,
            "base_rate_adversarial": round(n_adv / max(n, 1), 4),
        }


# ── Model factories ─────────────────────────────────────────────────────────


def margin_isotonic_factory() -> IsotonicRegression:
    """Isotonic regression on margin only (monotonic calibration)."""
    return IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")


def logistic_risk_factory() -> Pipeline:
    """Logistic regression on all features with standard scaling."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(C=1.0, max_iter=1000, random_state=42)),
    ])


# ── Cross-fitted evaluation ─────────────────────────────────────────────────


class CrossFittedEvaluator:
    """StratifiedKFold cross-fitting to produce held-out risk scores."""

    def __init__(self, n_splits: int = 5, random_state: int = 42) -> None:
        self.n_splits = n_splits
        self.random_state = random_state

    def evaluate(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray,
        model_factory: Callable,
    ) -> dict:
        """Cross-fit and return per-fold + aggregated metrics plus held-out predictions."""
        from sklearn.metrics import brier_score_loss, roc_auc_score, average_precision_score

        skf = StratifiedKFold(
            n_splits=self.n_splits, shuffle=True, random_state=self.random_state,
        )
        y_arr = np.asarray(y)
        X_arr = np.asarray(X)

        fold_metrics: list[dict] = []
        held_out_indices: list[np.ndarray] = []
        held_out_probs: list[np.ndarray] = []

        for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X_arr, y_arr)):
            X_train, X_test = X_arr[train_idx], X_arr[test_idx]
            y_train, y_test = y_arr[train_idx], y_arr[test_idx]

            model = model_factory()
            if isinstance(model, IsotonicRegression):
                # Isotonic expects 1-D input
                model.fit(X_train.ravel(), y_train)
                probs = model.predict(X_test.ravel())
            else:
                model.fit(X_train, y_train)
                probs = model.predict_proba(X_test)[:, 1]

            probs = np.clip(probs, 0.0, 1.0)
            held_out_indices.append(test_idx)
            held_out_probs.append(probs)

            fm = {
                "fold": fold_idx + 1,
                "roc_auc": roc_auc_score(y_test, probs),
                "pr_auc": average_precision_score(y_test, probs),
                "brier": brier_score_loss(y_test, probs),
            }
            fold_metrics.append(fm)

        # Concatenate held-out predictions in original order
        all_indices = np.concatenate(held_out_indices)
        all_probs = np.concatenate(held_out_probs)
        order = np.argsort(all_indices)
        predictions = all_probs[order]

        metrics_df = pd.DataFrame(fold_metrics)
        agg = {
            col: {"mean": metrics_df[col].mean(), "std": metrics_df[col].std()}
            for col in ["roc_auc", "pr_auc", "brier"]
        }

        return {
            "fold_metrics": fold_metrics,
            "aggregate": agg,
            "predictions": predictions,
        }

    def evaluate_threshold_sweep(
        self,
        X_margin: np.ndarray | pd.Series,
        y: np.ndarray | pd.Series,
        thresholds: list[float],
    ) -> dict:
        """Baseline A: sweep raw margin thresholds, cross-fitted.

        For each fold, select best threshold on train by Youden's J,
        then evaluate on test. Also returns per-threshold global results.
        """
        y_arr = np.asarray(y)
        margin_arr = np.asarray(X_margin).ravel()

        per_threshold: list[dict] = []
        for t in thresholds:
            # risk = 1 if margin < t (low margin → likely adversarial)
            preds = (margin_arr < t).astype(int)
            tp = int(((y_arr == 1) & (preds == 1)).sum())
            tn = int(((y_arr == 0) & (preds == 0)).sum())
            fp = int(((y_arr == 0) & (preds == 1)).sum())
            fn = int(((y_arr == 1) & (preds == 0)).sum())
            tpr = tp / max(tp + fn, 1)
            fpr = fp / max(fp + tn, 1)
            precision = tp / max(tp + fp, 1)
            recall = tpr
            f1 = 2 * precision * recall / max(precision + recall, 1e-9)
            accuracy = (tp + tn) / max(len(y_arr), 1)
            per_threshold.append({
                "threshold": t,
                "tp": tp, "tn": tn, "fp": fp, "fn": fn,
                "tpr": round(tpr, 4),
                "fpr": round(fpr, 4),
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(f1, 4),
                "accuracy": round(accuracy, 4),
                "n_flipped": tp + fp,
            })

        return {"per_threshold": per_threshold}


class TrainTestEvaluator:
    """Train on one dataset, evaluate on another (proper train/test separation)."""

    def evaluate(
        self,
        X_train: pd.DataFrame | np.ndarray,
        y_train: pd.Series | np.ndarray,
        X_test: pd.DataFrame | np.ndarray,
        y_test: pd.Series | np.ndarray,
        model_factory: Callable,
    ) -> dict:
        """Fit on train, predict on test. Returns same shape as CrossFittedEvaluator."""
        from sklearn.metrics import brier_score_loss, roc_auc_score, average_precision_score

        X_train_arr = np.asarray(X_train)
        X_test_arr = np.asarray(X_test)
        y_train_arr = np.asarray(y_train)
        y_test_arr = np.asarray(y_test)

        model = model_factory()
        if isinstance(model, IsotonicRegression):
            model.fit(X_train_arr.ravel(), y_train_arr)
            probs = model.predict(X_test_arr.ravel())
        else:
            model.fit(X_train_arr, y_train_arr)
            probs = model.predict_proba(X_test_arr)[:, 1]

        probs = np.clip(probs, 0.0, 1.0)

        metrics = {
            "fold": 1,
            "roc_auc": roc_auc_score(y_test_arr, probs),
            "pr_auc": average_precision_score(y_test_arr, probs),
            "brier": brier_score_loss(y_test_arr, probs),
        }

        agg = {
            col: {"mean": metrics[col], "std": 0.0}
            for col in ["roc_auc", "pr_auc", "brier"]
        }

        return {
            "fold_metrics": [metrics],
            "aggregate": agg,
            "predictions": probs,
        }


# ── Calibration ──────────────────────────────────────────────────────────────


def compute_calibration_metrics(
    y_true: np.ndarray | pd.Series,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> dict:
    """Expected calibration error + per-bin reliability."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins: list[dict] = []
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (y_prob >= lo) & (y_prob < hi) if i < n_bins - 1 else (y_prob >= lo) & (y_prob <= hi)
        n_in_bin = int(mask.sum())
        if n_in_bin == 0:
            bins.append({"bin_lo": lo, "bin_hi": hi, "count": 0, "avg_predicted": None, "avg_actual": None})
            continue
        avg_pred = float(y_prob[mask].mean())
        avg_actual = float(y_true[mask].mean())
        ece += abs(avg_pred - avg_actual) * (n_in_bin / len(y_true))
        bins.append({
            "bin_lo": round(lo, 2),
            "bin_hi": round(hi, 2),
            "count": n_in_bin,
            "avg_predicted": round(avg_pred, 4),
            "avg_actual": round(avg_actual, 4),
        })

    return {"ece": round(ece, 4), "bins": bins}


# ── Policy simulation ────────────────────────────────────────────────────────


class PolicySimulator:
    """Simulate decision policies on risk scores."""

    @staticmethod
    def simulate_two_zone(
        y_true: np.ndarray | pd.Series,
        risk_scores: np.ndarray,
        thresholds: list[float],
    ) -> pd.DataFrame:
        """At each threshold, if risk > t → adversarial, else benign."""
        y_true = np.asarray(y_true)
        rows = []
        for t in thresholds:
            preds = (risk_scores > t).astype(int)
            tp = int(((y_true == 1) & (preds == 1)).sum())
            tn = int(((y_true == 0) & (preds == 0)).sum())
            fp = int(((y_true == 0) & (preds == 1)).sum())
            fn = int(((y_true == 1) & (preds == 0)).sum())
            n = len(y_true)
            tpr = tp / max(tp + fn, 1)
            fpr = fp / max(fp + tn, 1)
            precision = tp / max(tp + fp, 1)
            recall = tpr
            f1 = 2 * precision * recall / max(precision + recall, 1e-9)
            rows.append({
                "threshold": t,
                "tpr": round(tpr, 4),
                "fpr": round(fpr, 4),
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(f1, 4),
                "accuracy": round((tp + tn) / max(n, 1), 4),
                "n_flipped": tp + fp,
            })
        return pd.DataFrame(rows)

    @staticmethod
    def simulate_three_zone(
        y_true: np.ndarray | pd.Series,
        risk_scores: np.ndarray,
        zone_pairs: list[tuple[float, float]],
    ) -> pd.DataFrame:
        """Three-zone: risk <= low → benign, risk >= high → adversarial, between → uncertain."""
        y_true = np.asarray(y_true)
        n = len(y_true)
        rows = []
        for low_t, high_t in zone_pairs:
            benign_mask = risk_scores <= low_t
            adv_mask = risk_scores >= high_t
            uncertain_mask = ~benign_mask & ~adv_mask

            def _zone_acc(mask: np.ndarray, expected: int | None) -> float:
                if mask.sum() == 0:
                    return float("nan")
                if expected is None:
                    return float("nan")
                return float((y_true[mask] == expected).mean())

            rows.append({
                "low_threshold": low_t,
                "high_threshold": high_t,
                "benign_zone_coverage": round(benign_mask.sum() / max(n, 1), 4),
                "uncertain_zone_coverage": round(uncertain_mask.sum() / max(n, 1), 4),
                "adversarial_zone_coverage": round(adv_mask.sum() / max(n, 1), 4),
                "benign_zone_accuracy": round(_zone_acc(benign_mask, 0), 4),
                "adversarial_zone_accuracy": round(_zone_acc(adv_mask, 1), 4),
            })
        return pd.DataFrame(rows)


# ── Report generation ────────────────────────────────────────────────────────


def generate_report(
    data_summary: dict,
    model_results: dict[str, dict],
    calibration: dict,
    two_zone_df: pd.DataFrame,
    three_zone_df: pd.DataFrame,
    eval_mode: str = "crossfit",
) -> str:
    """Generate a Markdown report."""
    if eval_mode == "train_test":
        warning = (
            "> **Note**: Model trained on val trace, evaluated on test trace.\n"
            "> This provides a proper train/test separation for the risk model."
        )
    else:
        warning = (
            "> **Warning**: This is an exploratory post-hoc analysis on test-derived traces.\n"
            "> The LLM predictions were generated once on the test set; these results\n"
            "> should not be treated as unbiased estimates of production performance."
        )
    lines = [
        "# Post-hoc Benign Risk Model Report",
        "",
        warning,
        "",
        "## Data",
        "",
        f"- **Eligible samples**: {data_summary['n_eligible']} "
        f"(LLM-path rows where LLM predicted benign pre-policy)",
        f"- **True adversarial**: {data_summary['n_adversarial']} "
        f"({data_summary['base_rate_adversarial']:.1%} base rate)",
        f"- **True benign**: {data_summary['n_benign']}",
        "",
        "## Model Comparison",
        "",
        "| Model | ROC-AUC | PR-AUC | Brier |",
        "|-------|---------|--------|-------|",
    ]

    for name, res in model_results.items():
        agg = res["aggregate"]
        lines.append(
            f"| {name} | "
            f"{agg['roc_auc']['mean']:.4f} ± {agg['roc_auc']['std']:.4f} | "
            f"{agg['pr_auc']['mean']:.4f} ± {agg['pr_auc']['std']:.4f} | "
            f"{agg['brier']['mean']:.4f} ± {agg['brier']['std']:.4f} |"
        )

    lines += [
        "",
        "## Calibration (best model)",
        "",
        f"- **ECE**: {calibration['ece']:.4f}",
        "",
        "| Bin | Count | Avg Predicted | Avg Actual |",
        "|-----|-------|---------------|------------|",
    ]
    for b in calibration["bins"]:
        if b["count"] == 0:
            continue
        lines.append(
            f"| [{b['bin_lo']:.2f}, {b['bin_hi']:.2f}) | {b['count']} | "
            f"{b['avg_predicted']:.4f} | {b['avg_actual']:.4f} |"
        )

    lines += [
        "",
        "## Policy Simulation — Two-Zone",
        "",
        two_zone_df.to_markdown(index=False),
        "",
        "## Policy Simulation — Three-Zone",
        "",
        three_zone_df.to_markdown(index=False),
        "",
        "## Recommendations",
        "",
        "1. **Is margin alone sufficient?** Check ROC-AUC of isotonic vs logistic.",
        "   If logistic AUC is materially higher, the extra features add value.",
        "2. **Best operating point?** Refer to the two-zone table for the threshold",
        "   that balances FPR and recall for your use case.",
        "3. **Three-zone viable?** If the uncertain zone has high coverage with low",
        "   accuracy, a three-zone policy can defer hard cases to human review.",
        "4. **Productionization**: Would require training on a held-out calibration",
        "   set (not the test traces) and periodic recalibration as LLM behavior drifts.",
        "",
        "## Next Steps",
        "",
        "- Train on a dedicated calibration split (not test-derived traces)",
        "- Evaluate on external datasets for generalization",
        "- Consider adding text-length features if trace is joined back to raw data",
        "- Monitor calibration drift over time",
        "",
    ]
    return "\n".join(lines)


def generate_plots(
    y_true: np.ndarray,
    predictions_dict: dict[str, np.ndarray],
    output_dir: Path,
) -> None:
    """Generate ROC, PR, and calibration plots."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, precision_recall_curve
    from sklearn.calibration import calibration_curve

    output_dir.mkdir(parents=True, exist_ok=True)

    # ROC curve
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, probs in predictions_dict.items():
        fpr, tpr, _ = roc_curve(y_true, probs)
        ax.plot(fpr, tpr, label=name)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_title("ROC — Benign Risk Models")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "benign_risk_roc.png", dpi=150)
    plt.close(fig)

    # PR curve
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, probs in predictions_dict.items():
        precision, recall, _ = precision_recall_curve(y_true, probs)
        ax.plot(recall, precision, label=name)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("PR — Benign Risk Models")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "benign_risk_pr.png", dpi=150)
    plt.close(fig)

    # Calibration plot
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, probs in predictions_dict.items():
        prob_true, prob_pred = calibration_curve(y_true, probs, n_bins=10, strategy="uniform")
        ax.plot(prob_pred, prob_true, "o-", label=name)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
    ax.set_xlabel("Mean Predicted Risk")
    ax.set_ylabel("Fraction Truly Adversarial")
    ax.set_title("Calibration — Benign Risk Models")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "benign_risk_calibration.png", dpi=150)
    plt.close(fig)

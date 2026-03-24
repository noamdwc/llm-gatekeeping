from __future__ import annotations

import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.logprob_margin import (
    apply_margin_policy,
    extract_preferred_margin_features_from_row,
    infer_route_bucket,
    resolve_margin_policy_config,
)
from src.research import _compute_unicode_lane_mask
from src.utils import load_config

REPORT_DIR = ROOT / "reports" / "error_analysis_current_status"
NOTEBOOK_PATH = ROOT / "notebooks" / "error_analysis_current_status.ipynb"
DATASET_ORDER = ["test", "deepset", "jackhhao", "safeguard"]
EXTERNAL_DATASET_ORDER = ["deepset", "jackhhao", "safeguard"]
CURRENT_MAIN_METRICS = {
    "accuracy": 0.9580,
    "fpr": 0.1333,
    "fnr": 0.0212,
    "adv_f1": 0.9743,
    "benign_f1": 0.8844,
}
HISTORICAL_REFERENCE = {
    "main_old": {
        "accuracy": 0.9289,
        "fpr": 0.1500,
        "fnr": 0.0531,
        "adv_f1": 0.9560,
        "benign_f1": 0.8160,
        "routing": {"abstain": 5, "llm": 266},
    },
    "external_old_combined": {
        "accuracy": 0.8072,
        "fpr": 0.0095,
        "fnr": 0.5348,
    },
}


def ensure_report_dir() -> Path:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    return REPORT_DIR


def save_dataframe(df: pd.DataFrame, name: str) -> Path:
    path = ensure_report_dir() / name
    if path.suffix == ".csv":
        df.to_csv(path, index=False)
    else:
        df.to_markdown(path, index=False)
    return path


def save_figure(fig: plt.Figure, name: str) -> Path:
    path = ensure_report_dir() / name
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def discover_artifacts(root: Path | None = None) -> pd.DataFrame:
    root = root or ROOT
    candidates = [
        ("main_research", root / "data/processed/research/research_test.parquet"),
        ("main_trace", root / "data/processed/research/hybrid_margin_trace_test.parquet"),
        ("main_ml", root / "data/processed/predictions/ml_predictions_test.parquet"),
        ("main_deberta", root / "data/processed/predictions/deberta_predictions_test.parquet"),
        ("main_llm", root / "data/processed/predictions/llm_predictions_test.parquet"),
        ("risk_predictions", root / "data/processed/research/posthoc_benign_risk_predictions.parquet"),
        ("risk_summary", root / "data/processed/research/posthoc_benign_risk_summary.csv"),
        ("risk_model", root / "data/processed/models/risk_model.pkl"),
        ("main_report", root / "reports/research/eval_report_hybrid.md"),
        ("summary_report", root / "reports/research/summary_report.md"),
        ("legacy_external_deepset", root / "data/processed/predictions_external_deepset.parquet"),
        ("legacy_external_jackhhao", root / "data/processed/predictions_external_jackhhao.parquet"),
        ("legacy_external_safeguard", root / "data/processed/predictions_external_safeguard.parquet"),
    ]
    for dataset in EXTERNAL_DATASET_ORDER + ["spml"]:
        candidates.extend([
            (f"external_research_{dataset}", root / f"data/processed/research_external/research_external_{dataset}.parquet"),
            (f"external_llm_{dataset}", root / f"data/processed/predictions_external/llm_predictions_external_{dataset}.parquet"),
        ])
    rows = []
    for name, path in candidates:
        rows.append({
            "artifact": name,
            "path": str(path.relative_to(root)),
            "exists": path.exists(),
            "size_bytes": path.stat().st_size if path.exists() else np.nan,
        })
    return pd.DataFrame(rows).sort_values(["exists", "artifact"], ascending=[False, True]).reset_index(drop=True)


def load_table(path: str | Path) -> pd.DataFrame | None:
    path = Path(path)
    if not path.exists():
        return None
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported table type: {path}")


def load_current_frames(root: Path | None = None) -> dict[str, pd.DataFrame]:
    root = root or ROOT
    frames = {}
    main = load_table(root / "data/processed/research/research_test.parquet")
    if main is not None:
        frames["test"] = annotate_frame(main, "test")
    for dataset in EXTERNAL_DATASET_ORDER:
        df = load_table(root / f"data/processed/research_external/research_external_{dataset}.parquet")
        if df is not None:
            frames[dataset] = annotate_frame(df, dataset)
    return frames


def load_legacy_external_frames(root: Path | None = None) -> dict[str, pd.DataFrame]:
    root = root or ROOT
    frames = {}
    for dataset in EXTERNAL_DATASET_ORDER:
        path = root / f"data/processed/predictions_external_{dataset}.parquet"
        df = load_table(path)
        if df is None:
            continue
        df = df.copy()
        df["dataset"] = dataset
        df["split"] = "external_legacy"
        df["sample_key"] = build_sample_key(df)
        frames[dataset] = df
    return frames


def build_sample_key(df: pd.DataFrame) -> pd.Series:
    if "sample_id" in df.columns:
        return df["sample_id"].astype(str)
    if "modified_sample" in df.columns:
        return df["modified_sample"].fillna("").astype(str)
    return pd.Series(np.arange(len(df)), index=df.index, dtype="string")


def annotate_frame(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    df = df.copy()
    df["dataset"] = df.get("dataset", dataset_name)
    df["sample_key"] = build_sample_key(df)
    label_col = "label_binary" if "label_binary" in df.columns else "true_label"
    pred_col = "hybrid_pred_binary" if "hybrid_pred_binary" in df.columns else "final_label"
    if label_col in df.columns and pred_col in df.columns:
        y_true = df[label_col].eq("adversarial")
        y_pred = df[pred_col].eq("adversarial")
        df["is_correct"] = y_true == y_pred
        df["is_fp"] = (~y_true) & y_pred
        df["is_fn"] = y_true & (~y_pred)
        df["is_tp"] = y_true & y_pred
        df["is_tn"] = (~y_true) & (~y_pred)
    if "hybrid_routed_to" in df.columns:
        df["route_group"] = df["hybrid_routed_to"].fillna("missing")
    elif "route" in df.columns:
        df["route_group"] = df["route"].fillna("missing")
    else:
        df["route_group"] = "missing"
    if "modified_sample" in df.columns:
        df = add_text_features(df, "modified_sample")
    return df


def binary_metrics_from_cols(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    true_adv = y_true.eq("adversarial")
    pred_adv = y_pred.eq("adversarial")
    tp = int((true_adv & pred_adv).sum())
    tn = int((~true_adv & ~pred_adv).sum())
    fp = int((~true_adv & pred_adv).sum())
    fn = int((true_adv & ~pred_adv).sum())
    accuracy = (tp + tn) / len(y_true) if len(y_true) else np.nan
    adv_precision = tp / (tp + fp) if (tp + fp) else 0.0
    adv_recall = tp / (tp + fn) if (tp + fn) else 0.0
    benign_precision = tn / (tn + fn) if (tn + fn) else 0.0
    benign_recall = tn / (tn + fp) if (tn + fp) else 0.0

    def f1(p: float, r: float) -> float:
        return 2 * p * r / (p + r) if (p + r) else 0.0

    return {
        "rows": int(len(y_true)),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "accuracy": accuracy,
        "fpr": fp / (fp + tn) if (fp + tn) else np.nan,
        "fnr": fn / (fn + tp) if (fn + tp) else np.nan,
        "adv_precision": adv_precision,
        "adv_recall": adv_recall,
        "adv_f1": f1(adv_precision, adv_recall),
        "benign_precision": benign_precision,
        "benign_recall": benign_recall,
        "benign_f1": f1(benign_precision, benign_recall),
    }


def metrics_frame(df: pd.DataFrame, pred_col: str = "hybrid_pred_binary") -> pd.DataFrame:
    metrics = binary_metrics_from_cols(df["label_binary"], df[pred_col])
    return pd.DataFrame([metrics])


def confusion_frame(df: pd.DataFrame, pred_col: str = "hybrid_pred_binary") -> pd.DataFrame:
    table = pd.crosstab(
        df["label_binary"],
        df[pred_col],
        dropna=False,
    ).reindex(index=["benign", "adversarial"], columns=["benign", "adversarial"], fill_value=0)
    table.index.name = "true"
    table.columns.name = "pred"
    return table


def confusion_summary(df: pd.DataFrame, pred_col: str = "hybrid_pred_binary") -> pd.DataFrame:
    counts = confusion_frame(df, pred_col)
    total = counts.to_numpy().sum()
    rows = []
    for true_label in counts.index:
        row_total = counts.loc[true_label].sum()
        for pred_label in counts.columns:
            value = int(counts.loc[true_label, pred_label])
            rows.append({
                "true_label": true_label,
                "pred_label": pred_label,
                "count": value,
                "row_rate": value / row_total if row_total else np.nan,
                "overall_rate": value / total if total else np.nan,
            })
    return pd.DataFrame(rows)


def add_text_features(df: pd.DataFrame, text_col: str) -> pd.DataFrame:
    text = df[text_col].fillna("").astype(str)
    df["text_len_chars"] = text.str.len()
    df["text_len_words"] = text.str.split().str.len()
    df["newline_count"] = text.str.count(r"\n")
    df["punct_count"] = text.str.count(r"[^\w\s]")
    df["digit_count"] = text.str.count(r"\d")
    df["unicode_count"] = text.map(lambda x: sum(ord(ch) > 127 for ch in x))
    df["has_code_block"] = text.str.contains("```", regex=False)
    df["has_markdown_list"] = text.str.contains(r"(?:^|\n)\s*[-*]\s+", regex=True)
    df["has_roleplay_keyword"] = text.str.contains(r"\byou are\b|\brole\b|\bpretend\b", case=False, regex=True)
    df["has_policy_keyword"] = text.str.contains(r"\bpolicy\b|\bsafety\b|\brules\b|\bguardrail", case=False, regex=True)
    df["has_jailbreak_keyword"] = text.str.contains(
        r"\bignore\b|\bdisregard\b|\boverride\b|\bbypass\b|\bjailbreak\b|\bdeveloper mode\b",
        case=False,
        regex=True,
    )
    df["char_entropy"] = text.map(character_entropy)
    return df


def character_entropy(text: str) -> float:
    if not text:
        return 0.0
    chars, counts = np.unique(list(text), return_counts=True)
    probs = counts / counts.sum()
    return float(-(probs * np.log2(probs)).sum())


def truncate_text(text: Any, limit: int = 180) -> str:
    if text is None:
        return ""
    text = re.sub(r"\s+", " ", str(text)).strip()
    return text if len(text) <= limit else text[: limit - 1] + "…"


def representative_examples(
    df: pd.DataFrame,
    *,
    mask: pd.Series,
    columns: list[str],
    sort_by: list[str] | None = None,
    limit: int = 12,
) -> pd.DataFrame:
    subset = df.loc[mask].copy()
    if subset.empty:
        return subset
    if "modified_sample" in subset.columns:
        subset["text_snippet"] = subset["modified_sample"].map(truncate_text)
    if sort_by:
        sort_cols = [col for col in sort_by if col in subset.columns]
        if sort_cols:
            subset = subset.sort_values(sort_cols, ascending=[False] * len(sort_cols))
    keep = [col for col in columns if col in subset.columns]
    if "text_snippet" in subset.columns and "text_snippet" not in keep:
        keep.append("text_snippet")
    return subset[keep].head(limit).reset_index(drop=True)


@dataclass
class RiskModelBundle:
    pipeline: Any
    threshold: float
    feature_cols: list[str]

    @classmethod
    def load(cls, path: str | Path) -> "RiskModelBundle | None":
        path = Path(path)
        if not path.exists():
            return None
        import pickle

        with open(path, "rb") as handle:
            raw = pickle.load(handle)
        if isinstance(raw, cls):
            return raw
        if isinstance(raw, dict):
            return cls(
                pipeline=raw["pipeline"],
                threshold=float(raw["threshold"]),
                feature_cols=list(raw["feature_cols"]),
            )
        return None

    def predict_risk_batch(self, features: pd.DataFrame) -> np.ndarray:
        features = features.copy()
        for col in self.feature_cols:
            if col not in features.columns:
                features[col] = np.nan
        features = features[self.feature_cols].fillna(0.0)
        if hasattr(self.pipeline, "predict_proba"):
            return self.pipeline.predict_proba(features)[:, 1]
        return self.pipeline.predict(features)


def replay_hybrid(
    df: pd.DataFrame,
    *,
    cfg: dict | None = None,
    deberta_conf_threshold: float | None = None,
    llm_conf_threshold: float | None = None,
    risk_bundle: RiskModelBundle | None = None,
    risk_threshold: float | None = None,
    disable_deberta: bool = False,
    disable_risk: bool = False,
    disable_margin_policy: bool = False,
) -> pd.DataFrame:
    cfg = cfg or load_config(ROOT / "configs/default.yaml")
    hybrid_cfg = cfg["hybrid"]
    deberta_conf_threshold = (
        hybrid_cfg["deberta_confidence_threshold"] if deberta_conf_threshold is None else deberta_conf_threshold
    )
    llm_conf_threshold = hybrid_cfg["llm_confidence_threshold"] if llm_conf_threshold is None else llm_conf_threshold
    policy_cfg = resolve_margin_policy_config(cfg)
    if disable_margin_policy:
        policy_cfg = {
            "policy": "baseline",
            "threshold": None,
            "low_threshold": None,
            "high_threshold": None,
            "classifier_only_threshold": None,
            "judge_threshold": None,
        }

    out = df.copy()
    out["replay_sample_key"] = build_sample_key(out)
    ml_conf_col = "ml_conf_binary_cal" if "ml_conf_binary_cal" in out.columns else "ml_conf_binary"
    unicode_lane, _ = _compute_unicode_lane_mask(out, unicode_types=cfg.get("unicode_types", []))
    ml_fastpath = (
        out["ml_pred_binary"].eq("adversarial")
        & out[ml_conf_col].fillna(0.0).ge(hybrid_cfg["ml_confidence_threshold"])
        & pd.Series(unicode_lane, index=out.index)
    )

    out["replay_route"] = np.where(ml_fastpath, "ml", "llm")
    out["replay_pred_binary"] = out["ml_pred_binary"]
    out["replay_margin"] = np.nan
    out["replay_margin_source_stage"] = None
    out["replay_risk_score"] = np.nan

    eligible = ~ml_fastpath
    has_deberta = {"deberta_pred_binary", "deberta_conf_binary"}.issubset(out.columns)
    if has_deberta and not disable_deberta:
        deberta_fastpath = eligible & out["deberta_conf_binary"].fillna(0.0).ge(deberta_conf_threshold)
        out.loc[deberta_fastpath, "replay_route"] = "deberta"
        out.loc[deberta_fastpath, "replay_pred_binary"] = out.loc[deberta_fastpath, "deberta_pred_binary"]
    else:
        deberta_fastpath = pd.Series(False, index=out.index)

    llm_mask = eligible & ~deberta_fastpath
    if "llm_pred_binary" not in out.columns:
        return out

    out.loc[llm_mask, "replay_pred_binary"] = out.loc[llm_mask, "llm_pred_binary"].fillna(out.loc[llm_mask, "ml_pred_binary"])
    low_conf = llm_mask & out["llm_conf_binary"].fillna(0.5).lt(llm_conf_threshold)
    out.loc[low_conf, "replay_route"] = "abstain"
    out.loc[low_conf, "replay_pred_binary"] = "adversarial"

    if low_conf.any() and risk_bundle is not None and not disable_risk and has_deberta:
        threshold = risk_bundle.threshold if risk_threshold is None else risk_threshold
        features = build_risk_features(out.loc[low_conf])
        risk_scores = risk_bundle.predict_risk_batch(features)
        out.loc[low_conf, "replay_risk_score"] = risk_scores
        benign_rescue = risk_scores <= threshold
        benign_index = out.loc[low_conf].index[benign_rescue]
        out.loc[benign_index, "replay_pred_binary"] = "benign"

    for idx in out.index[llm_mask]:
        row = out.loc[idx]
        margin = extract_preferred_margin_features_from_row(row)
        out.at[idx, "replay_margin"] = margin.margin
        out.at[idx, "replay_margin_source_stage"] = margin.source_stage
        if out.at[idx, "replay_route"] != "llm":
            continue
        route_bucket = infer_route_bucket({"hybrid_routed_to": "llm", "llm_stages_run": row.get("llm_stages_run")})
        policy_result = apply_margin_policy(
            current_route="llm",
            predicted_binary=out.at[idx, "replay_pred_binary"],
            predicted_label=row.get("llm_pred_raw", out.at[idx, "replay_pred_binary"]),
            margin=margin.margin,
            policy_cfg=policy_cfg,
            route_bucket=route_bucket,
        )
        out.at[idx, "replay_route"] = policy_result["route"]
        out.at[idx, "replay_pred_binary"] = policy_result["final_binary"]
    return out


def build_risk_features(df: pd.DataFrame) -> pd.DataFrame:
    margins = [extract_preferred_margin_features_from_row(row) for _, row in df.iterrows()]
    return pd.DataFrame({
        "margin": [m.margin for m in margins],
        "top1_logprob": [m.top1_logprob for m in margins],
        "top2_logprob": [m.top2_logprob for m in margins],
        "self_reported_confidence": df["llm_conf_binary"].fillna(0.5).astype(float).values,
        "is_judge_stage": (df.get("llm_stages_run", 1).fillna(1).astype(int) == 2).astype(int).values,
        "deberta_proba_binary_adversarial": df.get("deberta_proba_binary_adversarial", pd.Series(np.nan, index=df.index)).astype(float).values,
        "is_abstain": np.ones(len(df), dtype=int),
    }, index=df.index)


def sweep_deberta_thresholds(
    frames: dict[str, pd.DataFrame],
    *,
    thresholds: list[float],
    cfg: dict | None = None,
    risk_bundle: RiskModelBundle | None = None,
    risk_threshold: float | None = None,
) -> pd.DataFrame:
    cfg = cfg or load_config(ROOT / "configs/default.yaml")
    rows = []
    for dataset_name, frame in frames.items():
        for threshold in thresholds:
            replay = replay_hybrid(
                frame,
                cfg=cfg,
                deberta_conf_threshold=threshold,
                risk_bundle=risk_bundle,
                risk_threshold=risk_threshold,
            )
            metrics = binary_metrics_from_cols(replay["label_binary"], replay["replay_pred_binary"])
            metrics["dataset"] = dataset_name
            metrics["deberta_threshold"] = threshold
            metrics["route_abstain"] = int((replay["replay_route"] == "abstain").sum())
            metrics["route_deberta"] = int((replay["replay_route"] == "deberta").sum())
            metrics["route_llm"] = int((replay["replay_route"] == "llm").sum())
            metrics["route_ml"] = int((replay["replay_route"] == "ml").sum())
            rows.append(metrics)
    return pd.DataFrame(rows)


def grouped_route_error_rates(df: pd.DataFrame, route_col: str, pred_col: str) -> pd.DataFrame:
    rows = []
    for route_name, group in df.groupby(route_col, dropna=False):
        metrics = binary_metrics_from_cols(group["label_binary"], group[pred_col])
        rows.append({
            "route": route_name,
            "rows": len(group),
            "accuracy": metrics["accuracy"],
            "fpr": metrics["fpr"],
            "fnr": metrics["fnr"],
            "adv_f1": metrics["adv_f1"],
            "benign_f1": metrics["benign_f1"],
        })
    return pd.DataFrame(rows).sort_values("rows", ascending=False).reset_index(drop=True)


def plot_confusion_pair(current_df: pd.DataFrame, label: str) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    counts = confusion_frame(current_df)
    rates = counts.div(counts.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    for ax, matrix, title in [
        (axes[0], counts, f"{label}: counts"),
        (axes[1], rates, f"{label}: row-normalized"),
    ]:
        im = ax.imshow(matrix.to_numpy(), cmap="Blues")
        ax.set_xticks([0, 1], labels=matrix.columns.tolist())
        ax.set_yticks([0, 1], labels=matrix.index.tolist())
        ax.set_title(title)
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                value = matrix.iloc[i, j]
                text = f"{value:.2f}" if matrix is rates else f"{int(value)}"
                ax.text(j, i, text, ha="center", va="center", color="black")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig


def plot_route_distribution(df: pd.DataFrame, title: str, route_col: str = "hybrid_routed_to") -> plt.Figure:
    counts = df[route_col].fillna("missing").value_counts().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(7, 4))
    counts.plot(kind="bar", ax=ax, color="#2a6f97")
    ax.set_title(title)
    ax.set_xlabel("route")
    ax.set_ylabel("rows")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    return fig


def plot_score_distribution(
    df: pd.DataFrame,
    *,
    score_col: str,
    label: str,
    split_by: str = "label_binary",
    bins: int = 30,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 4))
    for value, color in [("benign", "#6c757d"), ("adversarial", "#d62828")]:
        subset = df.loc[df[split_by] == value, score_col].dropna().astype(float)
        if subset.empty:
            continue
        ax.hist(subset, bins=bins, alpha=0.5, label=value, density=True, color=color)
    ax.set_title(label)
    ax.set_xlabel(score_col)
    ax.set_ylabel("density")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    return fig


def plot_threshold_curves(df: pd.DataFrame, dataset_name: str) -> plt.Figure:
    subset = df[df["dataset"] == dataset_name].sort_values("deberta_threshold")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for metric, color in [
        ("accuracy", "#1d3557"),
        ("fpr", "#e76f51"),
        ("fnr", "#d62828"),
        ("adv_f1", "#2a9d8f"),
        ("benign_f1", "#8d99ae"),
    ]:
        ax.plot(subset["deberta_threshold"], subset[metric], marker="o", label=metric, color=color)
    ax.set_title(f"Threshold sweep: {dataset_name}")
    ax.set_xlabel("DeBERTa confidence threshold")
    ax.set_ylabel("metric value")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.25)
    ax.legend(ncol=3)
    fig.tight_layout()
    return fig


def build_combined_external(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    parts = [frames[name] for name in EXTERNAL_DATASET_ORDER if name in frames]
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


def find_candidate_operating_points(
    sweep_df: pd.DataFrame,
    dataset_name: str,
    *,
    fp_weight: float = 2.0,
    fn_weight: float = 1.0,
) -> pd.DataFrame:
    subset = sweep_df[sweep_df["dataset"] == dataset_name].copy()
    if subset.empty:
        return subset
    subset["cost"] = fp_weight * subset["fpr"] + fn_weight * subset["fnr"]
    subset["balanced_score"] = subset["adv_f1"] + subset["benign_f1"] - subset["cost"]
    picks = []
    picks.append(subset.sort_values(["balanced_score", "accuracy"], ascending=False).head(1).assign(profile="balanced"))
    picks.append(subset.sort_values(["fpr", "fnr", "accuracy"], ascending=[True, True, False]).head(1).assign(profile="conservative benign-friendly"))
    picks.append(subset.sort_values(["fnr", "fpr", "accuracy"], ascending=[True, True, False]).head(1).assign(profile="aggressive attack-catching"))
    return pd.concat(picks, ignore_index=True).drop_duplicates(subset=["profile", "deberta_threshold"])


def compare_metrics_table(
    current_frames: dict[str, pd.DataFrame],
    legacy_external_frames: dict[str, pd.DataFrame] | None = None,
) -> pd.DataFrame:
    rows = []
    for dataset_name, frame in current_frames.items():
        metrics = binary_metrics_from_cols(frame["label_binary"], frame["hybrid_pred_binary"])
        metrics["dataset"] = dataset_name
        metrics["version"] = "current"
        rows.append(metrics)
    legacy_external_frames = legacy_external_frames or {}
    for dataset_name, frame in legacy_external_frames.items():
        metrics = binary_metrics_from_cols(frame["label_binary"], frame["hybrid_pred_binary"])
        metrics["dataset"] = dataset_name
        metrics["version"] = "legacy_external_artifact"
        rows.append(metrics)
    table = pd.DataFrame(rows)
    cols = ["dataset", "version", "rows", "accuracy", "fpr", "fnr", "adv_f1", "benign_f1", "tp", "fp", "tn", "fn"]
    return table[cols].sort_values(["dataset", "version"]).reset_index(drop=True)


def historical_note() -> str:
    return (
        "Old per-sample main-hybrid artifacts were not found in the repository. "
        "Historical old/new comparisons in this notebook therefore use: "
        "(a) exact current row-level artifacts, "
        "(b) legacy external prediction files when present, and "
        "(c) the user-supplied historical headline metrics as reference-only context."
    )

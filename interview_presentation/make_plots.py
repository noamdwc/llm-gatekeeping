#!/usr/bin/env python3
"""Generate interview-presentation visuals from repository artifacts.

Outputs are written to interview_presentation/assets/.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

# Keep matplotlib cache in a writable path inside repo/sandbox.
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


ROOT = Path(__file__).resolve().parents[1]
ASSETS_DIR = ROOT / "interview_presentation" / "assets"


def parse_markdown_table(section_text: str) -> pd.DataFrame:
    """Parse the first markdown table found in a section."""
    lines = section_text.splitlines()
    table_lines: list[str] = []
    in_table = False

    for line in lines:
        stripped = line.strip()
        if not stripped.startswith("|"):
            if in_table:
                break
            continue
        in_table = True
        table_lines.append(stripped)

    if len(table_lines) < 2:
        raise ValueError("No markdown table found in section")

    headers = [h.strip() for h in table_lines[0].strip("|").split("|")]
    rows = []
    for line in table_lines[2:]:
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) != len(headers):
            continue
        rows.append(cells)
    return pd.DataFrame(rows, columns=headers)


def section_text(markdown_path: Path, heading: str) -> str:
    """Return text under a markdown heading until the next peer heading."""
    text = markdown_path.read_text()
    lines = text.splitlines()
    out: list[str] = []
    capture = False

    for line in lines:
        if line.startswith(heading):
            capture = True
            continue
        if capture and line.startswith("## "):
            break
        if capture:
            out.append(line)

    if not out:
        raise ValueError(f"Heading {heading!r} not found in {markdown_path}")
    return "\n".join(out)


def parse_binary_metrics(report_path: Path) -> dict[str, float]:
    """Parse binary metrics table from markdown eval report."""
    text = report_path.read_text()
    lines = text.splitlines()

    in_table = False
    metrics: dict[str, float] = {}
    row_pat = re.compile(r"^\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|$")

    for line in lines:
        if line.strip().startswith("## Binary Detection"):
            in_table = True
            continue
        if in_table and line.strip().startswith("## "):
            break
        if not in_table:
            continue
        m = row_pat.match(line.strip())
        if not m:
            continue
        k = m.group(1).strip()
        v = m.group(2).strip()
        if k in {"Metric", "--------"}:
            continue
        if v.upper() in {"N/A", "NAN"}:
            continue
        try:
            metrics[k] = float(v)
        except ValueError:
            continue

    if not metrics:
        raise ValueError(f"Could not parse metrics from {report_path}")
    return metrics


def build_main_metrics() -> pd.DataFrame:
    reports = {
        "ML (unicode scope)": ROOT / "reports/research/eval_report_ml.md",
        "Hybrid": ROOT / "reports/research/eval_report_hybrid.md",
        "LLM": ROOT / "reports/research/eval_report_llm.md",
    }

    rows = []
    for mode, path in reports.items():
        m = parse_binary_metrics(path)
        rows.append({
            "mode": mode,
            "accuracy": m["accuracy"],
            "adversarial_recall": m["adversarial_recall"],
            "benign_recall": m["benign_recall"],
            "false_negative_rate": m["false_negative_rate"],
            "support_adversarial": int(m.get("support_adversarial", 0)),
            "support_benign": int(m.get("support_benign", 0)),
        })

    df = pd.DataFrame(rows)
    df["n"] = df["support_adversarial"] + df["support_benign"]
    return df


def build_external_metrics() -> pd.DataFrame:
    files = sorted((ROOT / "reports/research_external").glob("research_external_*.md"))
    rows = []
    for path in files:
        ds = path.stem.replace("research_external_", "")
        m = parse_binary_metrics(path)
        rows.append({
            "dataset": ds,
            "accuracy": m["accuracy"],
            "adversarial_recall": m["adversarial_recall"],
            "benign_recall": m["benign_recall"],
            "false_negative_rate": m["false_negative_rate"],
            "false_positive_rate": m["false_positive_rate"],
            "support_adversarial": int(m.get("support_adversarial", 0)),
            "support_benign": int(m.get("support_benign", 0)),
        })

    df = pd.DataFrame(rows)
    df["n"] = df["support_adversarial"] + df["support_benign"]
    return df


def build_coverage_stats() -> pd.DataFrame:
    rows = []

    split_paths = {
        "train": ROOT / "data/processed/splits/train.parquet",
        "val": ROOT / "data/processed/splits/val.parquet",
        "test": ROOT / "data/processed/splits/test.parquet",
        "test_unseen": ROOT / "data/processed/splits/test_unseen.parquet",
    }
    for name, path in split_paths.items():
        df = pd.read_parquet(path)
        adv = int((df["label_binary"] == "adversarial").sum())
        ben = int((df["label_binary"] == "benign").sum())
        rows.append({"dataset": name, "adversarial": adv, "benign": ben, "group": "main_splits"})

    for path in sorted((ROOT / "data/processed/research_external").glob("research_external_*.parquet")):
        name = path.stem.replace("research_external_", "")
        df = pd.read_parquet(path)
        adv = int((df["label_binary"] == "adversarial").sum())
        ben = int((df["label_binary"] == "benign").sum())
        rows.append({"dataset": name, "adversarial": adv, "benign": ben, "group": "external"})

    out = pd.DataFrame(rows)
    out["n"] = out["adversarial"] + out["benign"]
    out["adv_pct"] = out["adversarial"] / out["n"].clip(lower=1)
    out["ben_pct"] = out["benign"] / out["n"].clip(lower=1)
    return out


def build_baseline_comparison() -> pd.DataFrame:
    summary_path = ROOT / "reports/research/summary_report.md"
    test_df = parse_markdown_table(section_text(summary_path, "## test"))
    deepset_df = parse_markdown_table(section_text(summary_path, "## external_deepset"))

    keep_models = ["Our Hybrid", "Sentinel v2", "ProtectAI v2"]

    def default_rows(df: pd.DataFrame, dataset: str) -> pd.DataFrame:
        model_col = "Model"
        threshold_col = "Threshold"
        out = df[df[model_col].isin(keep_models)].copy()
        out = out[
            ((out[model_col] == "Our Hybrid") & (out[threshold_col] == "-")) |
            ((out[model_col] != "Our Hybrid") & out[threshold_col].str.startswith("default"))
        ].copy()
        out["dataset"] = dataset
        return out

    merged = pd.concat(
        [
            default_rows(test_df, "test"),
            default_rows(deepset_df, "deepset"),
        ],
        ignore_index=True,
    )

    for col in ["Accuracy", "Adv Recall", "Benign Recall", "FNR"]:
        merged[col] = merged[col].astype(float)
    return merged


def build_deberta_summary() -> dict[str, float | int]:
    cfg = ROOT / "configs/default.yaml"
    checkpoint = ROOT / "artifacts/deberta_classifier/best_checkpoint.json"

    import yaml

    cfg_data = yaml.safe_load(cfg.read_text())
    ckpt = pd.read_json(checkpoint, typ="series").to_dict()
    dcfg = cfg_data["deberta"]
    return {
        "model_name": dcfg["model_name"],
        "num_epochs": dcfg["num_epochs"],
        "batch_size": dcfg["batch_size"],
        "learning_rate": dcfg["learning_rate"],
        "early_stopping_patience": dcfg["early_stopping_patience"],
        "best_epoch": ckpt["epoch"],
        "best_f1": ckpt["metric_value"],
    }


def plot_pipeline_diagram() -> None:
    fig, ax = plt.subplots(figsize=(14, 6.5))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    nodes = {
        "mindgard": (0.05, 0.70, 0.16, 0.18, "Mindgard\nadversarial data"),
        "synth": (0.05, 0.38, 0.16, 0.18, "Synthetic benign\naugmentation"),
        "prep": (0.28, 0.54, 0.18, 0.18, "Preprocess +\nlabel hierarchy"),
        "splits": (0.51, 0.54, 0.18, 0.18, "Grouped splits\n(+ held-out attacks)"),
        "ml": (0.74, 0.74, 0.18, 0.16, "ML baseline\nTF-IDF + unicode feats"),
        "deberta": (0.74, 0.52, 0.18, 0.16, "DeBERTa\nfine-tuned classifier"),
        "llm": (0.74, 0.30, 0.18, 0.16, "LLM classifier\n(+ judge)"),
        "router": (0.74, 0.08, 0.18, 0.16, "Hybrid router +\nreport assembly"),
    }

    def draw_box(key: str, face: str) -> None:
        x, y, w, h, label = nodes[key]
        patch = FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.02,rounding_size=0.02",
            linewidth=1.8,
            edgecolor="#1f2937",
            facecolor=face,
        )
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=10)

    draw_box("mindgard", "#dbeafe")
    draw_box("synth", "#e0f2fe")
    draw_box("prep", "#dbeafe")
    draw_box("splits", "#f3e8ff")
    draw_box("ml", "#dcfce7")
    draw_box("deberta", "#fde68a")
    draw_box("llm", "#ffedd5")
    draw_box("router", "#fef3c7")

    def arrow(x1: float, y1: float, x2: float, y2: float, text: str | None = None) -> None:
        a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=14, linewidth=1.6, color="#111827")
        ax.add_patch(a)
        if text:
            ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.02, text, ha="center", va="bottom", fontsize=9)

    arrow(0.21, 0.79, 0.28, 0.63)
    arrow(0.21, 0.47, 0.28, 0.63)
    arrow(0.46, 0.63, 0.51, 0.63)
    arrow(0.69, 0.63, 0.74, 0.82, "sparse features")
    arrow(0.69, 0.63, 0.74, 0.60, "tokenized text")
    arrow(0.69, 0.63, 0.74, 0.38, "few-shot eval")
    arrow(0.83, 0.74, 0.83, 0.24, "ML confidence -> router")
    arrow(0.83, 0.52, 0.83, 0.24, "supervised alt path")
    arrow(0.83, 0.30, 0.83, 0.24, "LLM escalation")

    ax.text(0.28, 0.85, "Current repo includes synthetic-benign-backed preprocessing", fontsize=9, color="#374151")
    ax.text(0.67, 0.92, "Three model paths: classic ML, DeBERTa, and LLM/hybrid", fontsize=9, color="#374151")

    fig.suptitle("LLM Security Gatekeeper: Current Pipeline Topology", fontsize=16, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    fig.savefig(ASSETS_DIR / "pipeline_diagram.png", dpi=220)
    fig.savefig(ASSETS_DIR / "pipeline_diagram.svg")
    plt.close(fig)


def plot_main_metrics(main_df: pd.DataFrame) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5), gridspec_kw={"width_ratios": [2.3, 1.0]})

    modes = main_df["mode"].tolist()
    x = np.arange(len(modes))
    width = 0.22

    metrics = [
        ("accuracy", "Accuracy", "#2563eb"),
        ("adversarial_recall", "Adv Recall", "#16a34a"),
        ("benign_recall", "Benign Recall", "#ea580c"),
    ]

    for i, (col, label, color) in enumerate(metrics):
        ax1.bar(x + (i - 1) * width, main_df[col].values, width=width, label=label, color=color)

    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{m}\n(n={n})" for m, n in zip(main_df["mode"], main_df["n"])], fontsize=9)
    ax1.set_ylim(0, 1.05)
    ax1.set_ylabel("Score")
    ax1.set_title("Main Dataset Metrics")
    ax1.legend(frameon=False, loc="upper right")
    ax1.grid(axis="y", alpha=0.25)

    ax2.bar(modes, main_df["false_negative_rate"], color="#7c3aed")
    ax2.set_ylim(0, 1.05)
    ax2.set_ylabel("False Negative Rate")
    ax2.set_title("Security Risk (Lower Better)")
    ax2.grid(axis="y", alpha=0.25)
    for i, v in enumerate(main_df["false_negative_rate"]):
        ax2.text(i, v + 0.02, f"{v:.2f}", ha="center", va="bottom", fontsize=9)

    fig.suptitle("ML vs Hybrid vs LLM (Canonical reports/research)", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(ASSETS_DIR / "main_metrics_comparison.png", dpi=220)
    plt.close(fig)


def plot_external_metrics(ext_df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(12, 5.5))
    x = np.arange(len(ext_df))
    width = 0.24

    series = [
        ("accuracy", "Accuracy", "#2563eb"),
        ("adversarial_recall", "Adv Recall", "#16a34a"),
        ("benign_recall", "Benign Recall", "#ea580c"),
    ]

    for i, (col, label, color) in enumerate(series):
        ax.bar(x + (i - 1) * width, ext_df[col].values, width=width, label=label, color=color)

    labels = [f"{d}\n(n={n})" for d, n in zip(ext_df["dataset"], ext_df["n"]) ]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("External Generalization: Binary Metrics by Dataset")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, ncols=3, loc="upper center")

    fig.tight_layout()
    fig.savefig(ASSETS_DIR / "external_generalization.png", dpi=220)
    plt.close(fig)


def plot_data_coverage(coverage_df: pd.DataFrame) -> None:
    order = ["train", "val", "test", "test_unseen", "deepset", "jackhhao", "safeguard", "spml"]
    available = [name for name in order if name in set(coverage_df["dataset"])]
    df = coverage_df.set_index("dataset").loc[available].reset_index()

    fig, ax = plt.subplots(figsize=(12.5, 5.8))
    x = np.arange(len(df))

    ax.bar(x, df["adversarial"], label="Adversarial", color="#dc2626")
    ax.bar(x, df["benign"], bottom=df["adversarial"], label="Benign", color="#2563eb")

    ax.set_xticks(x)
    ax.set_xticklabels([f"{d}\n(n={n})" for d, n in zip(df["dataset"], df["n"])], fontsize=9)
    ax.set_ylabel("Count")
    ax.set_title("Data Coverage and Class Balance")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)

    for i, pct in enumerate(df["adv_pct"]):
        ax.text(i, df.loc[i, "n"] + max(df["n"]) * 0.01, f"adv {pct:.1%}", ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    fig.savefig(ASSETS_DIR / "data_coverage.png", dpi=220)
    plt.close(fig)


def plot_hybrid_confusion() -> None:
    df = pd.read_parquet(ROOT / "data/processed/research/research_test.parquet")

    labels = ["adversarial", "benign"]
    cm = pd.crosstab(df["label_binary"], df["hybrid_pred_binary"], dropna=False)
    cm = cm.reindex(index=labels, columns=labels, fill_value=0)
    mat = cm.values

    fig, ax = plt.subplots(figsize=(5.6, 4.9))
    im = ax.imshow(mat, cmap="Blues")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Hybrid Binary Confusion Matrix (test)")

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, str(mat[i, j]), ha="center", va="center", color="#111827", fontsize=11)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(ASSETS_DIR / "hybrid_confusion_matrix.png", dpi=220)
    plt.close(fig)


def plot_baseline_comparison(baseline_df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4), sharey=True)

    datasets = ["test", "deepset"]
    colors = {
        "Our Hybrid": "#2563eb",
        "Sentinel v2": "#16a34a",
        "ProtectAI v2": "#ea580c",
    }

    for ax, dataset in zip(axes, datasets):
        sub = baseline_df[baseline_df["dataset"] == dataset].copy()
        x = np.arange(len(sub))
        width = 0.34

        ax.bar(x - width / 2, sub["Adv Recall"], width=width,
               color=[colors[m] for m in sub["Model"]], label="Adv Recall")
        ax.bar(x + width / 2, sub["Benign Recall"], width=width,
               color=[colors[m] for m in sub["Model"]], alpha=0.45, label="Benign Recall")
        ax.set_xticks(x)
        ax.set_xticklabels(sub["Model"], rotation=15, ha="right")
        ax.set_ylim(0, 1.05)
        ax.set_title("Main test split" if dataset == "test" else "deepset (cleanest external)")
        ax.grid(axis="y", alpha=0.25)

        for i, (_, row) in enumerate(sub.iterrows()):
            ax.text(i, row["Accuracy"] + 0.03, f"acc {row['Accuracy']:.2f}",
                    ha="center", va="bottom", fontsize=8)

    axes[0].set_ylabel("Recall")
    handles = [
        plt.Rectangle((0, 0), 1, 1, color="#111827"),
        plt.Rectangle((0, 0), 1, 1, color="#111827", alpha=0.45),
    ]
    fig.legend(handles, ["Adv Recall", "Benign Recall"], frameon=False, ncols=2, loc="upper center")
    fig.suptitle("Default-threshold Baseline Comparison", fontsize=14, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(ASSETS_DIR / "baseline_comparison.png", dpi=220)
    plt.close(fig)


def plot_deberta_summary(deberta_summary: dict[str, float | int]) -> None:
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    ax.axis("off")

    card = FancyBboxPatch(
        (0.05, 0.08), 0.9, 0.84,
        boxstyle="round,pad=0.02,rounding_size=0.03",
        linewidth=1.8,
        edgecolor="#1f2937",
        facecolor="#fff7ed",
    )
    ax.add_patch(card)

    ax.text(0.09, 0.84, "DeBERTa Training Update", fontsize=18, weight="bold", color="#111827")
    ax.text(0.09, 0.73, f"Model: {deberta_summary['model_name']}", fontsize=11, color="#374151")
    ax.text(0.09, 0.61, "Recent repo changes", fontsize=12, weight="bold", color="#111827")

    bullets = [
        f"Class-weighted cross-entropy for severe class imbalance",
        f"Longer schedule: {deberta_summary['num_epochs']} epochs, batch size {deberta_summary['batch_size']}",
        f"Learning rate raised to {deberta_summary['learning_rate']:.1e}; patience {deberta_summary['early_stopping_patience']}",
        "Best-checkpoint save/restore plus WandB logging in research runs",
        f"Current best checkpoint: epoch {deberta_summary['best_epoch']}, F1 {deberta_summary['best_f1']:.3f}",
    ]
    y = 0.53
    for bullet in bullets:
        ax.text(0.11, y, f"- {bullet}", fontsize=11, color="#111827")
        y -= 0.10

    fig.tight_layout()
    fig.savefig(ASSETS_DIR / "deberta_summary.png", dpi=220)
    plt.close(fig)


def main() -> None:
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)

    main_df = build_main_metrics()
    ext_df = build_external_metrics()
    coverage_df = build_coverage_stats()
    baseline_df = build_baseline_comparison()
    deberta_summary = build_deberta_summary()

    # Save plot inputs for traceability
    main_df.to_csv(ASSETS_DIR / "main_metrics.csv", index=False)
    ext_df.to_csv(ASSETS_DIR / "external_metrics.csv", index=False)
    coverage_df.to_csv(ASSETS_DIR / "data_coverage.csv", index=False)
    baseline_df.to_csv(ASSETS_DIR / "baseline_comparison.csv", index=False)

    plot_pipeline_diagram()
    plot_main_metrics(main_df)
    plot_external_metrics(ext_df)
    plot_data_coverage(coverage_df)
    plot_hybrid_confusion()
    plot_baseline_comparison(baseline_df)
    plot_deberta_summary(deberta_summary)

    print("Wrote assets to", ASSETS_DIR)


if __name__ == "__main__":
    main()

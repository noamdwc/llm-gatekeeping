"""
Hybrid ML+LLM router.

Routing logic:
  1. ML binary gate runs first (cheap, fast)
  2. If ML confidence >= threshold → use ML prediction
  3. If ML confidence < threshold → escalate to LLM for full hierarchical classification
  4. If LLM confidence < threshold → abstain ("needs_review")

Evaluation compares: ML-only vs LLM-only vs hybrid at various thresholds.

Usage:
    python -m src.hybrid_router [--config configs/default.yaml] [--limit 100]
"""

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

import wandb
import dotenv
import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

from src.utils import load_config, SPLITS_DIR, MODELS_DIR, REPORTS_RESEARCH_DIR
from src.ml_classifier.ml_baseline import MLBaseline
from src.llm_classifier.llm_classifier import HierarchicalLLMClassifier, build_few_shot_examples
from src.evaluate import evaluate_dataframe
dotenv.load_dotenv()

@dataclass
class RouterStats:
    total: int = 0
    ml_handled: int = 0
    llm_escalated: int = 0
    abstained: int = 0

    @property
    def ml_rate(self) -> float:
        return self.ml_handled / max(self.total, 1)

    @property
    def llm_rate(self) -> float:
        return self.llm_escalated / max(self.total, 1)

    @property
    def abstain_rate(self) -> float:
        return self.abstained / max(self.total, 1)

    def to_dict(self) -> dict:
        return {
            "total": self.total,
            "ml_handled": self.ml_handled,
            "llm_escalated": self.llm_escalated,
            "abstained": self.abstained,
            "ml_rate": round(self.ml_rate, 4),
            "llm_rate": round(self.llm_rate, 4),
            "abstain_rate": round(self.abstain_rate, 4),
        }


class HybridRouter:
    """Routes classification through ML or LLM based on confidence thresholds."""

    def __init__(self, ml_model, llm_classifier, cfg: dict):
        self.ml = ml_model
        self.llm = llm_classifier
        self.ml_threshold = cfg["hybrid"]["ml_confidence_threshold"]
        self.llm_threshold = cfg["hybrid"]["llm_confidence_threshold"]
        self.stats = RouterStats()

    def predict_single(self, text: str, ml_pred: dict) -> dict:
        """
        Route a single sample.

        ml_pred should contain: pred_label_binary, confidence_label_binary, etc.
        """
        self.stats.total += 1

        ml_conf = ml_pred["confidence_label_binary"]
        ml_binary = ml_pred["pred_label_binary"]

        # High-confidence ML → use ML results
        if ml_conf >= self.ml_threshold:
            self.stats.ml_handled += 1
            return {
                "label_binary": ml_binary,
                "label_category": ml_pred.get("pred_label_category", ml_binary),
                "label_type": ml_pred.get("pred_label_type", ml_binary),
                "confidence_binary": ml_conf,
                "confidence_category": ml_pred.get("confidence_label_category"),
                "confidence_type": ml_pred.get("confidence_label_type"),
                "routed_to": "ml",
            }

        # Low-confidence ML → escalate to LLM
        self.stats.llm_escalated += 1
        llm_result = self.llm.predict(text)

        # Check LLM confidence for abstention (new schema: "confidence")
        llm_conf = llm_result.get("confidence", 0.5)
        llm_binary = llm_result.get("label", "adversarial")
        routed_to = "abstain" if llm_conf < self.llm_threshold else "llm"
        if routed_to == "abstain":
            self.stats.abstained += 1

        # Build unified return dict: LLM for binary, ML for category/type
        return {
            "label_binary": llm_binary,
            "label_category": ml_pred.get("pred_label_category", llm_binary),
            "label_type": ml_pred.get("pred_label_type", llm_binary),
            "confidence_binary": llm_conf,
            "confidence_category": ml_pred.get("confidence_label_category"),
            "confidence_type": ml_pred.get("confidence_label_type"),
            "routed_to": routed_to,
        }

    def predict_batch(
        self,
        df: pd.DataFrame,
        text_col: str,
        desc: str = "Hybrid routing",
    ) -> list[dict]:
        """Predict on a DataFrame using the hybrid routing strategy."""
        # Get ML predictions for all samples at once (fast)
        ml_preds_df = self.ml.predict(df, text_col)

        results = []
        for i, (idx, row) in enumerate(tqdm(df.iterrows(), total=len(df), desc=desc)):
            ml_pred = ml_preds_df.iloc[i].to_dict()
            result = self.predict_single(row[text_col], ml_pred)
            results.append(result)

        return results


def threshold_sweep(
    df: pd.DataFrame,
    ml_preds: pd.DataFrame,
    thresholds: list[float],
) -> pd.DataFrame:
    """
    Sweep ML confidence thresholds to show trade-off between cost and accuracy.
    Does NOT call the LLM — simulates by marking escalations.
    """
    text_col_binary = "label_binary"
    rows = []

    for thresh in thresholds:
        ml_conf = ml_preds["confidence_label_binary"].values
        ml_pred_binary = ml_preds["pred_label_binary"].values
        y_true = df[text_col_binary].values

        high_conf_mask = ml_conf >= thresh
        n_ml = high_conf_mask.sum()
        n_llm = len(df) - n_ml

        # ML accuracy on high-confidence subset
        if n_ml > 0:
            ml_acc = (ml_pred_binary[high_conf_mask] == y_true[high_conf_mask]).mean()
        else:
            ml_acc = 0.0

        rows.append({
            "threshold": thresh,
            "ml_handled": n_ml,
            "llm_escalated": n_llm,
            "ml_rate": n_ml / len(df),
            "ml_accuracy_on_handled": ml_acc,
        })

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Run hybrid ML+LLM router")
    parser.add_argument("--config", default=None)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--no-wandb", action="store_true", help="Disable wandb logging")
    args = parser.parse_args()

    cfg = load_config(args.config)
    text_col = cfg["dataset"]["text_col"]

    # Init wandb
    if not args.no_wandb:
        wandb.init(
            project="llm-gatekeeping",
            name="hybrid-router",
            config={
                "ml_threshold": cfg["hybrid"]["ml_confidence_threshold"],
                "llm_threshold": cfg["hybrid"]["llm_confidence_threshold"],
                "llm_model": cfg["llm"]["model"],
                "limit": args.limit,
            },
        )

    # Load data
    df_test = pd.read_parquet(SPLITS_DIR / "test.parquet")
    if args.limit and args.limit < len(df_test):
        df_test = df_test.sample(n=args.limit, random_state=42)

    # Load ML model
    ml = MLBaseline(cfg)
    ml.load(str(MODELS_DIR / "ml_baseline.pkl"))

    # --- Threshold sweep (no LLM calls) ---
    print("Running threshold sweep (ML-only, no LLM calls)...")
    ml_preds = ml.predict(df_test, text_col)
    thresholds = [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.99]
    sweep = threshold_sweep(df_test, ml_preds, thresholds)
    print("\nThreshold sweep results:")
    print(sweep.to_string(index=False))

    if wandb.run is not None:
        wandb.log({"threshold_sweep": wandb.Table(dataframe=sweep)})

    # --- Full hybrid run (with LLM calls) ---
    print(f"\nRunning hybrid router (threshold={cfg['hybrid']['ml_confidence_threshold']})...")

    df_train = pd.read_parquet(SPLITS_DIR / "train.parquet")
    few_shot, _ = build_few_shot_examples(df_train, cfg)
    llm = HierarchicalLLMClassifier(cfg, few_shot)

    router = HybridRouter(ml, llm, cfg)
    results = router.predict_batch(df_test, text_col)

    # Evaluate
    REPORTS_RESEARCH_DIR.mkdir(parents=True, exist_ok=True)
    report_path = str(REPORTS_RESEARCH_DIR / "eval_report_hybrid.md")
    binary, cat, types, cal = evaluate_dataframe(
        df_test, results, output_path=report_path,
        usage={**llm.usage.to_dict(), **router.stats.to_dict()},
    )

    router_stats = router.stats.to_dict()
    llm_usage = llm.usage.to_dict()
    print(f"\nRouter stats: {json.dumps(router_stats, indent=2)}")
    print(f"LLM usage: {json.dumps(llm_usage, indent=2)}")

    if wandb.run is not None:
        wandb.log({**router_stats, **llm_usage, **binary, **cat})
        wandb.finish()


if __name__ == "__main__":
    main()

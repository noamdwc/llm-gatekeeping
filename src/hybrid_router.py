"""
Hybrid ML+LLM router.

Routing logic:
  1. ML binary gate runs first (cheap, fast)
  2. If ML predicts benign → always escalate to LLM
  3. If ML predicts adversarial AND unicode-lane AND confidence >= threshold → use ML prediction
  4. Otherwise → escalate to LLM for full hierarchical classification
  5. If LLM confidence < threshold → abstain ("needs_review")

Evaluation compares: ML-only vs LLM-only vs hybrid at various thresholds.

Usage:
    python -m src.hybrid_router [--config configs/default.yaml] [--limit 100]
"""

import argparse
import json
from dataclasses import dataclass

import wandb
import dotenv
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.utils import load_config, SPLITS_DIR, MODELS_DIR, REPORTS_RESEARCH_DIR
from src.ml_classifier.ml_baseline import MLBaseline
from src.llm_classifier.llm_classifier import HierarchicalLLMClassifier, build_few_shot_examples
from src.evaluate import evaluate_dataframe
dotenv.load_dotenv()


ADVERSARIAL_LABEL_ALIASES = {
    "adversarial",
    "adversary",
    "adv",
    "attack",
    "attacker",
    "malicious",
    "jailbreak",
    "prompt_injection",
    "prompt injection",
    "injection",
}


def _normalize_binary_label(label) -> str:
    return str(label).strip().lower().replace("-", "_")


def _normalize_attack_token(value) -> str:
    return str(value).strip().lower().replace("-", "_").replace(" ", "_")


def _is_adversarial_label(label) -> bool:
    norm = _normalize_binary_label(label)
    return norm in ADVERSARIAL_LABEL_ALIASES or norm.startswith("adv")


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
        self.logprob_margin_threshold = cfg["hybrid"].get("logprob_margin_threshold")
        self.stats = RouterStats()
        self._ml_conf_source_logged = False
        self._benign_escalations = 0
        self._adv_high_conf_finalizations = 0
        self._adv_low_conf_escalations = 0
        self._adv_non_unicode_escalations = 0
        self._adv_unknown_lane_escalations = 0
        self._margin_gate_overrides = 0
        unicode_types = cfg.get("labels", {}).get("unicode_attacks", [])
        self._unicode_types = {_normalize_attack_token(v) for v in unicode_types}

    def _get_ml_binary_confidence(self, ml_pred: dict) -> float:
        """Prefer calibrated binary confidence, fallback to raw for old artifacts."""
        confidence_sources = [
            ("confidence_label_binary_cal", "calibrated"),
            ("ml_conf_binary_cal", "calibrated"),
            ("confidence_label_binary", "raw_fallback"),
            ("ml_conf_binary", "raw_fallback"),
        ]
        conf = 0.5
        source = "default_0.5"
        for col, source_name in confidence_sources:
            if col in ml_pred and pd.notna(ml_pred[col]):
                conf = float(ml_pred[col])
                source = source_name
                break

        if not self._ml_conf_source_logged:
            print(f"  [HybridRouter] binary confidence source={source}")
            self._ml_conf_source_logged = True

        return conf

    @staticmethod
    def _get_ml_binary_label(ml_pred: dict) -> str:
        return (
            ml_pred.get("ml_pred_binary")
            or ml_pred.get("pred_label_binary")
            or "benign"
        )

    @staticmethod
    def _get_ml_category(ml_pred: dict):
        return ml_pred.get("ml_pred_category", ml_pred.get("pred_label_category"))

    @staticmethod
    def _get_ml_type(ml_pred: dict):
        return ml_pred.get("ml_pred_type", ml_pred.get("pred_label_type"))

    def _determine_unicode_lane(self, ml_pred: dict) -> tuple[bool, bool]:
        """Return (unicode_lane, lane_reliable)."""
        ml_category = self._get_ml_category(ml_pred)
        ml_type = self._get_ml_type(ml_pred)
        has_category = pd.notna(ml_category)
        has_type = pd.notna(ml_type)

        category_norm = _normalize_attack_token(ml_category) if has_category else ""
        type_norm = _normalize_attack_token(ml_type) if has_type else ""

        unicode_lane = (category_norm == "unicode_attack") or (type_norm in self._unicode_types)
        lane_reliable = bool(has_category or has_type)
        return unicode_lane, lane_reliable

    def _route_via_llm(
        self,
        text: str,
        ml_pred: dict,
        route_reason: str,
        ml_binary: str,
        ml_conf: float,
        unicode_lane: bool,
    ) -> dict:
        """Escalate a sample to LLM and standardize output schema."""
        llm_result = self.llm.predict(text)

        llm_conf = llm_result.get("confidence", 0.5)
        llm_label = llm_result.get("label")
        llm_binary = llm_result.get("label_binary") or llm_label or "adversarial"
        routed_to = "abstain" if llm_conf < self.llm_threshold else "llm"
        if routed_to == "abstain":
            self.stats.abstained += 1
            final_label = "uncertain"
            final_binary = "adversarial"
        else:
            self.stats.llm_escalated += 1
            if llm_label in ("benign", "adversarial", "uncertain"):
                final_label = llm_label
            else:
                final_label = "benign" if llm_binary == "benign" else "adversarial"
            final_binary = llm_result.get("label_binary")
            if final_binary not in ("benign", "adversarial"):
                final_binary = "benign" if final_label == "benign" else "adversarial"

        # Margin gate: override low-margin benign predictions to adversarial
        margin_gated = False
        if (
            self.logprob_margin_threshold is not None
            and final_binary == "benign"
            and routed_to == "llm"
        ):
            margin = llm_result.get("logprob_margin")
            if margin is not None and margin < self.logprob_margin_threshold:
                final_binary = "adversarial"
                final_label = "adversarial"
                margin_gated = True
                self._margin_gate_overrides += 1

        llm_category = llm_result.get("label_category")
        ml_category = self._get_ml_category(ml_pred)
        ml_type = self._get_ml_type(ml_pred)
        return {
            "label": final_label,
            "label_binary": final_binary,
            "label_category": llm_category or ml_category or final_binary,
            "label_type": ml_type or final_binary,
            "confidence_binary": llm_conf,
            "confidence_category": ml_pred.get("ml_conf_category", ml_pred.get("confidence_label_category")),
            "confidence_type": ml_pred.get("ml_conf_type", ml_pred.get("confidence_label_type")),
            "routed_to": routed_to,
            "route_reason": route_reason,
            "ml_pred_binary": ml_binary,
            "ml_pred_category": ml_category,
            "ml_pred_type": ml_type,
            "ml_conf_binary": ml_conf,
            "unicode_lane": unicode_lane,
            "margin_gated": margin_gated,
            "logprob_margin": llm_result.get("logprob_margin"),
        }

    def predict_single(self, text: str, ml_pred: dict) -> dict:
        """
        Route a single sample.

        ml_pred should contain: pred_label_binary, confidence_label_binary, etc.
        """
        self.stats.total += 1

        ml_conf = self._get_ml_binary_confidence(ml_pred)
        ml_binary = self._get_ml_binary_label(ml_pred)
        ml_binary_is_adv = _is_adversarial_label(ml_binary)
        ml_category = self._get_ml_category(ml_pred)
        ml_type = self._get_ml_type(ml_pred)
        unicode_lane, lane_reliable = self._determine_unicode_lane(ml_pred)

        # ML benign is never trusted in fast path.
        if not ml_binary_is_adv:
            self._benign_escalations += 1
            return self._route_via_llm(
                text,
                ml_pred,
                route_reason="ml_benign_escalate",
                ml_binary=ml_binary,
                ml_conf=ml_conf,
                unicode_lane=unicode_lane,
            )

        # Only high-confidence unicode-lane ML adversarial can be finalized by ML.
        if ml_conf >= self.ml_threshold and unicode_lane:
            self._adv_high_conf_finalizations += 1
            self.stats.ml_handled += 1
            return {
                "label": "adversarial",
                "label_binary": ml_binary,
                "label_category": ml_category or ml_binary,
                "label_type": ml_type or ml_binary,
                "confidence_binary": ml_conf,
                "confidence_category": ml_pred.get("ml_conf_category", ml_pred.get("confidence_label_category")),
                "confidence_type": ml_pred.get("ml_conf_type", ml_pred.get("confidence_label_type")),
                "routed_to": "ml",
                "route_reason": "ml_adv_high_conf_unicode_finalize",
                "ml_pred_binary": ml_binary,
                "ml_pred_category": ml_category,
                "ml_pred_type": ml_type,
                "ml_conf_binary": ml_conf,
                "unicode_lane": unicode_lane,
            }

        if not lane_reliable:
            self._adv_unknown_lane_escalations += 1
            route_reason = "ml_adv_unknown_lane_escalate"
        elif not unicode_lane:
            self._adv_non_unicode_escalations += 1
            route_reason = "ml_adv_non_unicode_escalate"
        else:
            self._adv_low_conf_escalations += 1
            route_reason = "ml_adv_low_conf_escalate"

        return self._route_via_llm(
            text,
            ml_pred,
            route_reason=route_reason,
            ml_binary=ml_binary,
            ml_conf=ml_conf,
            unicode_lane=unicode_lane,
        )

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

        if self.stats.total > 0:
            margin_info = ""
            if self.logprob_margin_threshold is not None:
                margin_info = f" | margin_gate_overrides={self._margin_gate_overrides} (threshold={self.logprob_margin_threshold})"
            print(
                "  [HybridRouter] route summary | "
                f"ml_benign_escalate={self._benign_escalations} | "
                f"ml_adv_high_conf_unicode_finalize={self._adv_high_conf_finalizations} | "
                f"ml_adv_low_conf_escalate={self._adv_low_conf_escalations} | "
                f"ml_adv_non_unicode_escalate={self._adv_non_unicode_escalations} | "
                f"ml_adv_unknown_lane_escalate={self._adv_unknown_lane_escalations}"
                f"{margin_info}"
            )

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
        if "confidence_label_binary_cal" in ml_preds.columns:
            conf_col = "confidence_label_binary_cal"
        elif "ml_conf_binary_cal" in ml_preds.columns:
            conf_col = "ml_conf_binary_cal"
        elif "confidence_label_binary" in ml_preds.columns:
            conf_col = "confidence_label_binary"
        else:
            conf_col = "ml_conf_binary"
        ml_conf = ml_preds[conf_col].values
        if "pred_label_binary" in ml_preds.columns:
            ml_pred_binary = ml_preds["pred_label_binary"].values
        else:
            ml_pred_binary = ml_preds["ml_pred_binary"].values
        y_true = df[text_col_binary].values

        adv_pred_mask = np.array([_is_adversarial_label(v) for v in ml_pred_binary], dtype=bool)

        category_col = "pred_label_category" if "pred_label_category" in ml_preds.columns else (
            "ml_pred_category" if "ml_pred_category" in ml_preds.columns else None
        )
        type_col = "pred_label_type" if "pred_label_type" in ml_preds.columns else (
            "ml_pred_type" if "ml_pred_type" in ml_preds.columns else None
        )

        if category_col is not None:
            cat_norm = ml_preds[category_col].astype(str).map(_normalize_attack_token)
            unicode_by_category = (cat_norm == "unicode_attack").values
        else:
            unicode_by_category = np.zeros(len(ml_preds), dtype=bool)

        if type_col is not None and category_col is not None:
            inferred_unicode_types = {
                _normalize_attack_token(v)
                for v in ml_preds.loc[unicode_by_category, type_col].dropna().tolist()
                if v is not None
            }
        else:
            inferred_unicode_types = set()

        if type_col is not None and inferred_unicode_types:
            type_norm = ml_preds[type_col].astype(str).map(_normalize_attack_token)
            unicode_by_type = type_norm.isin(inferred_unicode_types).values
        else:
            unicode_by_type = np.zeros(len(ml_preds), dtype=bool)

        unicode_lane_mask = unicode_by_category | unicode_by_type
        ml_fastpath_mask = adv_pred_mask & (ml_conf >= thresh) & unicode_lane_mask
        n_ml = ml_fastpath_mask.sum()
        n_llm = len(df) - n_ml

        # ML accuracy on ML-fastpath subset
        if n_ml > 0:
            ml_acc = (ml_pred_binary[ml_fastpath_mask] == y_true[ml_fastpath_mask]).mean()
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

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

from src.logprob_margin import (
    apply_margin_policy,
    extract_preferred_margin_features_from_result,
    infer_route_bucket,
    resolve_margin_policy_config,
)
from src.utils import load_config, SPLITS_DIR, MODELS_DIR, REPORTS_RESEARCH_DIR, DEBERTA_ARTIFACTS_DIR
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
    """Routes classification through ML, DeBERTa, or LLM based on confidence thresholds."""

    def __init__(self, ml_model, llm_classifier, cfg: dict, deberta_model=None,
                 risk_model=None):
        self.ml = ml_model
        self.llm = llm_classifier
        self.deberta = deberta_model
        self.risk_model = risk_model
        self.ml_threshold = cfg["hybrid"]["ml_confidence_threshold"]
        self.llm_threshold = cfg["hybrid"]["llm_confidence_threshold"]
        self.deberta_threshold = cfg["hybrid"].get("deberta_confidence_threshold", 0.93)
        self.logprob_margin_threshold = cfg["hybrid"].get("logprob_margin_threshold")
        self.margin_policy_cfg = resolve_margin_policy_config(cfg)
        self.stats = RouterStats()
        self._ml_conf_source_logged = False
        self._benign_escalations = 0
        self._adv_high_conf_finalizations = 0
        self._adv_low_conf_escalations = 0
        self._adv_non_unicode_escalations = 0
        self._adv_unknown_lane_escalations = 0
        self._margin_gate_overrides = 0
        self._risk_model_benign = 0
        self._risk_model_adversarial = 0
        self._deberta_benign_finalizations = 0
        self._deberta_adv_finalizations = 0
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
        deberta_pred: dict | None = None,
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

        margin_features = extract_preferred_margin_features_from_result(llm_result)
        route_bucket = infer_route_bucket({
            "routed_to": routed_to,
            "llm_stages_run": llm_result.get("llm_stages_run"),
        })
        policy_result = apply_margin_policy(
            current_route=routed_to,
            predicted_binary=final_binary,
            predicted_label=final_label,
            margin=margin_features.margin,
            policy_cfg=self.margin_policy_cfg,
            route_bucket=route_bucket,
        )
        margin_gated = bool(policy_result["override_applied"])
        routed_to = policy_result["route"]
        final_binary = policy_result["final_binary"]
        final_label = policy_result["final_label"]
        if margin_gated and routed_to == "abstain":
            self.stats.llm_escalated = max(self.stats.llm_escalated - 1, 0)
            self.stats.abstained += 1
        if margin_gated and policy_result["policy_outcome"] == "forced_adversarial":
            self._margin_gate_overrides += 1

        # Risk model: replace crude abstain→adversarial with informed decision
        if routed_to == "abstain" and self.risk_model is not None:
            deberta_adv_prob = 0.5
            if deberta_pred is not None:
                deberta_adv_prob = deberta_pred.get(
                    "deberta_proba_binary_adversarial", 0.5
                )
            risk_features = {
                "margin": margin_features.margin,
                "top1_logprob": margin_features.top1_logprob,
                "top2_logprob": margin_features.top2_logprob,
                "self_reported_confidence": llm_conf,
                "is_judge_stage": int(margin_features.source_stage == "judge"),
                "deberta_proba_binary_adversarial": deberta_adv_prob,
                "is_abstain": 1,
            }
            risk_label = self.risk_model.predict_label(risk_features)
            if risk_label == "benign":
                final_binary = "benign"
                final_label = "benign"
                self._risk_model_benign += 1
            else:
                self._risk_model_adversarial += 1

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
            "logprob_margin": margin_features.margin,
            "margin_source_stage": margin_features.source_stage,
            "label_start_position": margin_features.label_start_position,
            "top1_logprob": margin_features.top1_logprob,
            "top2_logprob": margin_features.top2_logprob,
            "top_logprobs_raw": margin_features.top_k_tokens,
            "token_names_missing": margin_features.token_names_missing,
            "token_strings_available": margin_features.token_strings_available,
            "margin_policy": policy_result["policy_name"],
            "policy_outcome": policy_result["policy_outcome"],
            "override_applied": policy_result["override_applied"],
            "override_reason": policy_result["override_reason"],
        }

    def predict_single(self, text: str, ml_pred: dict,
                       deberta_pred: dict | None = None) -> dict:
        """
        Route a single sample.

        ml_pred should contain: pred_label_binary, confidence_label_binary, etc.
        deberta_pred (optional) should contain: deberta_pred_binary,
            deberta_conf_binary, deberta_proba_binary_adversarial.
        """
        self.stats.total += 1

        ml_conf = self._get_ml_binary_confidence(ml_pred)
        ml_binary = self._get_ml_binary_label(ml_pred)
        ml_binary_is_adv = _is_adversarial_label(ml_binary)
        ml_category = self._get_ml_category(ml_pred)
        ml_type = self._get_ml_type(ml_pred)
        unicode_lane, lane_reliable = self._determine_unicode_lane(ml_pred)

        # 1. ML fast path: high-confidence unicode-lane adversarial (unchanged).
        if ml_binary_is_adv and ml_conf >= self.ml_threshold and unicode_lane:
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

        # 2. DeBERTa fast path: high-confidence binary prediction.
        if deberta_pred is not None:
            deberta_conf = deberta_pred["deberta_conf_binary"]
            deberta_binary = deberta_pred["deberta_pred_binary"]

            if deberta_conf >= self.deberta_threshold:
                if deberta_binary == "benign":
                    self._deberta_benign_finalizations += 1
                    self.stats.ml_handled += 1
                    return {
                        "label": "benign",
                        "label_binary": "benign",
                        "label_category": "benign",
                        "label_type": "benign",
                        "confidence_binary": deberta_conf,
                        "confidence_category": None,
                        "confidence_type": None,
                        "routed_to": "deberta",
                        "route_reason": "deberta_benign_high_conf",
                        "ml_pred_binary": ml_binary,
                        "ml_pred_category": ml_category,
                        "ml_pred_type": ml_type,
                        "ml_conf_binary": ml_conf,
                        "unicode_lane": unicode_lane,
                    }
                else:
                    # DeBERTa high-confidence adversarial — finalize with ML
                    # category/type (DeBERTa is binary-only).
                    self._deberta_adv_finalizations += 1
                    self.stats.ml_handled += 1
                    return {
                        "label": "adversarial",
                        "label_binary": "adversarial",
                        "label_category": ml_category or "adversarial",
                        "label_type": ml_type or "adversarial",
                        "confidence_binary": deberta_conf,
                        "confidence_category": ml_pred.get("ml_conf_category", ml_pred.get("confidence_label_category")),
                        "confidence_type": ml_pred.get("ml_conf_type", ml_pred.get("confidence_label_type")),
                        "routed_to": "deberta",
                        "route_reason": "deberta_adv_high_conf",
                        "ml_pred_binary": ml_binary,
                        "ml_pred_category": ml_category,
                        "ml_pred_type": ml_type,
                        "ml_conf_binary": ml_conf,
                        "unicode_lane": unicode_lane,
                    }

        # 3. Escalate to LLM — DeBERTa not available or not confident enough.
        if not ml_binary_is_adv:
            self._benign_escalations += 1
            route_reason = "ml_benign_escalate"
        elif not lane_reliable:
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
            deberta_pred=deberta_pred,
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

        # Get DeBERTa predictions if model is available (fast, no API)
        deberta_preds_df = None
        if self.deberta is not None:
            deberta_preds_df = self.deberta.predict(df, text_col)

        results = []
        for i, (idx, row) in enumerate(tqdm(df.iterrows(), total=len(df), desc=desc)):
            ml_pred = ml_preds_df.iloc[i].to_dict()
            deberta_pred = deberta_preds_df.iloc[i].to_dict() if deberta_preds_df is not None else None
            result = self.predict_single(row[text_col], ml_pred, deberta_pred)
            results.append(result)

        if self.stats.total > 0:
            margin_info = ""
            if self.logprob_margin_threshold is not None:
                margin_info = f" | margin_gate_overrides={self._margin_gate_overrides} (threshold={self.logprob_margin_threshold})"
            deberta_info = ""
            if self.deberta is not None:
                deberta_info = (
                    f" | deberta_benign_finalize={self._deberta_benign_finalizations}"
                    f" | deberta_adv_finalize={self._deberta_adv_finalizations}"
                )
            risk_info = ""
            if self.risk_model is not None:
                risk_info = (
                    f" | risk_model_benign={self._risk_model_benign}"
                    f" | risk_model_adversarial={self._risk_model_adversarial}"
                )
            print(
                "  [HybridRouter] route summary | "
                f"ml_adv_high_conf_unicode_finalize={self._adv_high_conf_finalizations} | "
                f"ml_benign_escalate={self._benign_escalations} | "
                f"ml_adv_low_conf_escalate={self._adv_low_conf_escalations} | "
                f"ml_adv_non_unicode_escalate={self._adv_non_unicode_escalations} | "
                f"ml_adv_unknown_lane_escalate={self._adv_unknown_lane_escalations}"
                f"{deberta_info}"
                f"{margin_info}"
                f"{risk_info}"
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

    # Load DeBERTa if artifacts exist
    deberta = None
    if DEBERTA_ARTIFACTS_DIR.exists() and (DEBERTA_ARTIFACTS_DIR / "model").exists():
        from src.models.deberta_classifier import DeBERTaClassifier
        deberta = DeBERTaClassifier.load(DEBERTA_ARTIFACTS_DIR, cfg)
        print(f"DeBERTa model loaded (threshold={cfg['hybrid'].get('deberta_confidence_threshold', 0.93)})")

    # Load risk model if available
    risk_model = None
    risk_cfg = cfg.get("hybrid", {}).get("risk_model", {})
    if risk_cfg.get("enabled", False):
        risk_path = Path(risk_cfg.get("model_path", MODELS_DIR / "risk_model.pkl"))
        if risk_path.exists():
            from src.benign_risk_model import RiskModel
            risk_model = RiskModel.load(risk_path)
            print(f"Risk model loaded (threshold={risk_model.threshold})")

    router = HybridRouter(ml, llm, cfg, deberta_model=deberta, risk_model=risk_model)
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

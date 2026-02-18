"""
LLM classifier with classifier + judge pattern:
  Stage 1: Classifier — single call predicts binary + type together
  Stage 2: Judge — conditional higher-quality model call on low-confidence predictions

Usage:
    python -m src.llm_classifier [--config configs/default.yaml] [--split test] [--limit 100]
"""

import argparse
import json
import time
import dotenv
import random
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass, field

import openai
import wandb

from src.llm_classifier.constants import NLP_TYPES
from src.embeddings import ExemplarBank, get_embeddings
from src.llm_classifier.utils import decide_accept_or_override
from src.utils import load_config, build_sample_id, SPLITS_DIR, PREDICTIONS_DIR
from src.llm_classifier.prompts import build_classifier_system_message, build_judge_system_message

dotenv.load_dotenv()

@dataclass
class UsageStats:
    total_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_latency_s: float = 0.0
    calls_by_stage: dict = field(default_factory=lambda: defaultdict(int))

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @property
    def avg_latency_s(self) -> float:
        return self.total_latency_s / max(self.total_calls, 1)

    def to_dict(self) -> dict:
        return {
            "total_calls": self.total_calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "total_latency_s": round(self.total_latency_s, 2),
            "avg_latency_s": round(self.avg_latency_s, 3),
            "calls_by_stage": dict(self.calls_by_stage),
        }


class HierarchicalLLMClassifier:
    """Classifier + judge pattern using OpenAI chat completions."""

    def __init__(
        self,
        cfg: dict,
        few_shot_examples: dict | None = None,
        dynamic: bool = False,
        exemplar_bank: ExemplarBank | None = None,
    ):
        self.cfg = cfg
        self.client = openai.OpenAI()
        self.model = cfg["llm"]["model"]
        self.model_quality = cfg["llm"].get("model_quality", self.model)
        self.temperature = cfg["llm"]["temperature"]
        self.few_shot = few_shot_examples or {}
        self.usage = UsageStats()

        # Dynamic few-shot settings
        self.dynamic = dynamic
        self.exemplar_bank = exemplar_bank
        if dynamic and exemplar_bank is None:
            raise ValueError("ExemplarBank required when dynamic=True")

    # -- internal helpers --------------------------------------------------

    def _call_llm(
        self, messages: list[dict], max_tokens: int, stage: str,
        model: str | None = None,
    ) -> dict:
        """Make one LLM call, track usage, return parsed JSON."""
        t0 = time.time()
        response = self.client.chat.completions.create(
            model=model or self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
        )
        latency = time.time() - t0

        self.usage.total_calls += 1
        self.usage.calls_by_stage[stage] += 1
        self.usage.total_latency_s += latency
        if response.usage:
            self.usage.prompt_tokens += response.usage.prompt_tokens
            self.usage.completion_tokens += response.usage.completion_tokens

        try:
            return json.loads(response.choices[0].message.content)
        except (json.JSONDecodeError, IndexError):
            return {}

    # -- Few-shot helpers --------------------------------------------------

    def _get_static_few_shot(self, text: str) -> list[dict]:
        """Get static few-shot examples."""
        return self.few_shot

    def _get_dynamic_few_shot(self, text: str) -> list[dict]:
        """Get dynamic few-shot examples using embedding similarity."""
        k = self.cfg["llm"]["few_shot"].get("dynamic_k", 2)
        query_emb = get_embeddings([text], model=self.exemplar_bank.embedding_model)[0]
        pairs = self.exemplar_bank.select_pairs_by_benign(query_emb, k=k)

        return pairs

    def _build_few_shot_messages(self, text: str) -> list[dict]:
        """Build few-shot example messages for the classifier prompt."""
        messages = []
        if self.dynamic and self.exemplar_bank:
            pairs = self._get_dynamic_few_shot(text)
        else:
            pairs = self._get_static_few_shot(text)
        
            
            
        for (benign_text, attack_text, attack_type) in pairs:
            # Add benign example
            confidence = random.uniform(0.7, 0.9)
            messages.append({"role": "user", "content": f"Text: {benign_text}"})
            messages.append({
                "role": "assistant", "content": json.dumps({
                    "label": 'benign',
                    "confidence": confidence,
                    "nlp_attack_type": 'none',
                    "evidence": "",
                    "reason": f"This is an example of benign text.",
                }
            )})
            # Add attack example
            confidence = random.random(0.7, 0.9)
            messages.append({"role": "user", "content": f"Text: {attack_text}"})
            messages.append({
                "role": "assistant", "content": json.dumps({
                    "label": 'adversarial',
                    "confidence": confidence,
                    "nlp_attack_type": attack_type if attack_type in NLP_TYPES else 'none',
                    "evidence": attack_text[:200],
                    "reason": f"This is an example of {attack_type} text.",
                }
            )})
        return messages

    # -- Classifier --------------------------------------------------------

    def classify(self, text: str) -> dict:
        """Single LLM call returning label + confidence.

        Returns:
            {"label": str, "confidence": float} where label is one of:
            "benign", one of UNICODE_TYPES, or "nlp_attack".
        """
        messages = build_classifier_system_message(text, self._build_few_shot_messages(text))
        result = self._call_llm(
            messages, self.cfg["llm"]["max_tokens_classifier"], "classifier"
        )
        return result

    def judge(self, text: str, classifier_output: dict) -> dict:
        """Conditional LLM call using model_quality for chain-of-thought review.

        Args:
            text: The input text.
            classifier_output: The classifier's prediction dict.

        Returns:
            {"label": str, "confidence": float, "reasoning": str}
        """
        messages = build_judge_system_message(text, classifier_output)
        result = self._call_llm(
            messages, self.cfg["llm"]["max_tokens_judge"], "judge",
            model=self.model_quality,
        )
        decision = decide_accept_or_override(result, classifier_output)
        result['computed_decision'] = decision
        return result

    # -- Full pipeline -----------------------------------------------------

    def predict(self, text: str, force_all_stages: bool = False) -> dict:
        """Run classifier + conditional judge.

        Args:
            force_all_stages: If True, always run the judge regardless of
                classifier confidence.
        """
        # Stage 1: Classifier
        clf_result = self.classify(text)
        stages_run = 1

        label = clf_result["label"]
        confidence = clf_result["confidence"]

        threshold = self.cfg["llm"].get("judge_confidence_threshold", 0.7)

        # Stage 2: Judge (conditional)
        evidence = clf_result.get("evidence", "")
        if confidence < threshold or force_all_stages:
            judge_result = self.judge(text, clf_result)
            stages_run = 2
            if judge_result["computed_decision"] == "override_candidate":
                label = judge_result["independent_label"]
                confidence = judge_result["independent_confidence"]
                evidence = judge_result["independent_evidence"]
            else:
                label = clf_result["label"]
                confidence = clf_result["confidence"]
                evidence = clf_result["evidence"]

        # Derive binary and category from the final label
        if label == "benign":
            label_binary = "benign"
        elif label == "adversarial":
            label_binary = "adversarial"
        elif label == "uncertain":
            label_binary = "adversarial" # default to adversarial for now
        else:
            raise ValueError(f"Invalid label: {label}")

        return {
            "label": label_binary,
            "confidence": confidence,
            "evidence": evidence,
            "llm_stages_run": stages_run,
        }

    def predict_batch(
        self,
        texts: list[str],
        desc: str = "Classifying",
        force_all_stages: bool = False,
    ) -> list[dict]:
        """Predict on a list of texts with progress bar."""
        return [self.predict(t, force_all_stages=force_all_stages) for t in tqdm(texts, desc=desc)]


# ---------------------------------------------------------------------------
# Few-shot builder
# ---------------------------------------------------------------------------
def build_few_shot_examples(df: pd.DataFrame, cfg: dict) -> tuple[dict, list]:
    """Build few-shot exemplar bank from training data."""
    text_col = cfg["dataset"]["text_col"]
    label_col = cfg["dataset"]["label_col"]
    n_unicode = cfg["llm"]["few_shot"]["unicode"]
    n_nlp = cfg["llm"]["few_shot"]["nlp"]
    unicode_set = set(cfg["labels"]["unicode_attacks"])

    few_shot = {}
    used_ids = []

    for attack_type in cfg["labels"]["unicode_attacks"] + cfg["labels"]["nlp_attacks"]:
        n = n_unicode if attack_type in unicode_set else n_nlp
        pool = df.loc[df[label_col] == attack_type, text_col]
        if len(pool) < n:
            n = len(pool)
        if n == 0:
            continue
        samples = pool.sample(n=n, random_state=42)
        few_shot[attack_type] = samples.tolist()
        used_ids.extend(samples.index.tolist())

    return few_shot, used_ids


def main():
    parser = argparse.ArgumentParser(description="Run hierarchical LLM classifier")
    parser.add_argument("--config", default=None)
    parser.add_argument("--split", default="test", help="Which split to evaluate on")
    parser.add_argument("--limit", type=int, default=100, help="Max samples to classify")
    parser.add_argument("--output", default=None, help="Output predictions CSV path")
    parser.add_argument("--no-wandb", action="store_true", help="Disable wandb logging")
    parser.add_argument("--dynamic", action="store_true", help="Use dynamic few-shot retrieval")
    parser.add_argument("--bank-path", default=None, help="Path to exemplar bank pickle (built if not exists)")
    parser.add_argument("--research", action="store_true",
                        help="Save research-grade parquet with full prediction columns")
    args = parser.parse_args()

    cfg = load_config(args.config)

    # Init wandb
    if not args.no_wandb:
        wandb.init(
            project="llm-gatekeeping",
            name=f"llm-{cfg['llm']['model']}-{args.split}{'_dynamic' if args.dynamic else ''}",
            config={
                "model": cfg["llm"]["model"],
                "split": args.split,
                "limit": args.limit,
                "few_shot_unicode": cfg["llm"]["few_shot"]["unicode"],
                "few_shot_nlp": cfg["llm"]["few_shot"]["nlp"],
                "dynamic": args.dynamic,
            },
        )

    # Load train for few-shot, eval split for evaluation
    df_train = pd.read_parquet(SPLITS_DIR / "train.parquet")
    df_eval = pd.read_parquet(SPLITS_DIR / f"{args.split}.parquet")

    if args.limit and args.limit < len(df_eval):
        df_eval = df_eval.sample(n=args.limit, random_state=42)

    # Build few-shot from train (static) or exemplar bank (dynamic)
    exemplar_bank = None
    few_shot = {}

    if args.dynamic:
        bank_path = args.bank_path or str(PREDICTIONS_DIR / "exemplar_bank.pkl")
        if Path(bank_path).exists():
            print(f"Loading exemplar bank from {bank_path}")
            exemplar_bank = ExemplarBank.load(bank_path)
        else:
            print("Building exemplar bank (this may take a minute)...")
            exemplar_bank = ExemplarBank.build(df_train, cfg)
            exemplar_bank.save(bank_path)
        print(f"Exemplar bank: {exemplar_bank}")
    else:
        few_shot, _ = build_few_shot_examples(df_train, cfg)
        print(f"Few-shot examples: {sum(len(v) for v in few_shot.values())} total")

    # Classify
    classifier = HierarchicalLLMClassifier(
        cfg, few_shot, dynamic=args.dynamic, exemplar_bank=exemplar_bank
    )
    text_col = cfg["dataset"]["text_col"]
    results = classifier.predict_batch(df_eval[text_col].tolist())

    # Build results DataFrame
    preds = pd.DataFrame(results)
    preds.index = df_eval.index
    df_out = pd.concat([df_eval.reset_index(drop=True), preds.reset_index(drop=True)], axis=1)

    # Save
    if args.research:
        # Research mode: save parquet with prefixed LLM columns
        PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
        research_rows = []
        for r in results:
            research_rows.append({
                "llm_pred_binary": r["label_binary"],
                "llm_pred_category": r["label_category"],
                "llm_pred_type": r["label_type"],
                "llm_conf_binary": r["confidence_binary"],
                "llm_conf_category": r["confidence_category"],
                "llm_conf_type": r["confidence_type"],
                "llm_stages_run": r.get("llm_stages_run"),
            })
        llm_df = pd.DataFrame(research_rows)

        # Include ground truth columns
        gt_cols = [c for c in df_eval.columns if c in [
            "modified_sample", "original_sample", "attack_name",
            "label_binary", "label_category", "label_type", "prompt_hash",
        ]]
        gt = df_eval[gt_cols].reset_index(drop=True)
        research_out = pd.concat([gt, llm_df], axis=1)
        research_out.insert(0, "sample_id", research_out["modified_sample"].apply(build_sample_id))

        out_path = str(PREDICTIONS_DIR / f"llm_predictions_{args.split}.parquet")
        research_out.to_parquet(out_path, index=False)
        print(f"\nResearch predictions saved → {out_path} (shape: {research_out.shape})")
    else:
        # Legacy mode: save CSV
        out_path = args.output or str(PREDICTIONS_DIR / f"predictions_{args.split}.csv")
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        df_out.to_csv(out_path, index=False)
        print(f"\nPredictions saved → {out_path}")

    # Print + log usage stats
    usage = classifier.usage.to_dict()
    print(f"\nUsage stats: {json.dumps(usage, indent=2)}")

    if wandb.run is not None:
        wandb.log(usage)
        wandb.finish()


if __name__ == "__main__":
    main()

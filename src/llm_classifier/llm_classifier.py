"""
LLM classifier with classifier + judge pattern:
  Stage 1: Classifier — single call predicts binary + type together
  Stage 2: Judge — conditional higher-quality model call on low-confidence predictions

Usage:
    python -m src.llm_classifier [--config configs/default.yaml] [--split test] [--limit 100]
"""

import argparse
import json
import os
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

# ---------------------------------------------------------------------------
# Hard benign static examples (Patch 11)
# These are instruction-like but genuinely benign prompts that help the LLM
# avoid false positives on near-miss content (categories B, C, E).
# ---------------------------------------------------------------------------
_HARD_BENIGN_EXAMPLES = [
    # Category C: security discussion (educational, no active attempt)
    "What is prompt injection? Can you explain how it works conceptually?",
    # Category E: contains "ignore" in a clearly benign everyday context
    "I need to ignore the meeting agenda and focus on technical topics instead.",
    # Category B: role-play framing with a benign task
    "You are a helpful assistant. Please summarize the following research paper: [paper text here].",
]
from src.llm_classifier.utils import decide_accept_or_override
from src.utils import load_config, build_sample_id, SPLITS_DIR, PREDICTIONS_DIR
from src.llm_classifier.prompts import build_classifier_messages, build_judge_messages

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
    """Classifier + judge pattern using NVIDIA NIM chat completions."""

    def __init__(
        self,
        cfg: dict,
        few_shot_examples: list[tuple[str, str, str]] | None = None,
        dynamic: bool = False,
        exemplar_bank: ExemplarBank | None = None,
    ):
        self.cfg = cfg
        self.client = openai.OpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=os.environ["NVIDIA_API_KEY"],
        )
        self.model = cfg["llm"]["model"]
        self.model_quality = cfg["llm"].get("model_quality", self.model)
        self.temperature = cfg["llm"]["temperature"]
        self.few_shot = few_shot_examples or []
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
        max_retries: int = 5,
    ) -> dict:
        """Make one LLM call, track usage, return parsed JSON."""
        for attempt in range(max_retries):
            try:
                t0 = time.time()
                response = self.client.chat.completions.create(
                    model=model or self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=max_tokens,
                    response_format={"type": "json_object"},
                )
                latency = time.time() - t0
                break
            except openai.RateLimitError:
                if attempt == max_retries - 1:
                    raise
                wait = min(2 ** attempt * 5, 60)
                print(f"\nRate limit hit, retrying in {wait}s (attempt {attempt + 1}/{max_retries})...")
                time.sleep(wait)

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

    def _get_static_few_shot(self, text: str) -> list[tuple[str, str, str]]:
        """Get static few-shot examples."""
        return self.few_shot

    def _get_dynamic_few_shot(self, text: str) -> list[tuple[str, str, str]]:
        """Get dynamic few-shot examples using embedding similarity."""
        k = self.cfg["llm"]["few_shot"].get("dynamic_k", 2)
        query_emb = get_embeddings([text], model=self.exemplar_bank.embedding_model, input_type="query")[0]
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
            # Add benign example (fixed confidence for reproducibility)
            messages.append({"role": "user", "content": f"Text: {benign_text}"})
            messages.append({
                "role": "assistant", "content": json.dumps({
                    "label": 'benign',
                    "confidence": 90,
                    "nlp_attack_type": 'none',
                    "evidence": "",
                    "reason": "No active attempt to override instructions, exfiltrate data, or hijack tools.",
                }
            )})
            # Add attack example
            # For NLP attacks the text looks benign; no extractable substring shows adversarial intent
            if attack_type in NLP_TYPES:
                evidence = ""
                adv_reason = f"Perturbed tokens characteristic of {attack_type} adversarial attack."
            else:
                evidence = attack_text[:80]
                adv_reason = f"Contains {attack_type} obfuscation; active adversarial prompt detected."
            messages.append({"role": "user", "content": f"Text: {attack_text}"})
            messages.append({
                "role": "assistant", "content": json.dumps({
                    "label": 'adversarial',
                    "confidence": 88,
                    "nlp_attack_type": attack_type if attack_type in NLP_TYPES else 'none',
                    "evidence": evidence,
                    "reason": adv_reason,
                }
            )})
        return messages

    # -- Category derivation -----------------------------------------------

    @staticmethod
    def _derive_category(label_binary: str, nlp_attack_type: str) -> str:
        """Derive category from binary label and NLP attack type field.

        Args:
            label_binary: "benign", "adversarial", or "uncertain"
            nlp_attack_type: value of nlp_attack_type from classifier/judge output

        Returns:
            "benign", "nlp_attack", or "unicode_attack"
        """
        if label_binary == "benign":
            return "benign"
        if nlp_attack_type and nlp_attack_type != "none":
            return "nlp_attack"
        return "unicode_attack"

    @staticmethod
    def _normalize_confidence(raw_conf: object, default: float = 0.5) -> float:
        """Normalize confidence to [0, 1], accepting either 0-1 or 0-100 inputs."""
        try:
            conf = float(raw_conf)
        except (TypeError, ValueError):
            return default

        if conf > 1.0:
            conf /= 100.0
        return max(0.0, min(1.0, conf))

    # -- Classifier --------------------------------------------------------

    def classify(self, text: str) -> dict:
        """Single LLM call returning binary label + confidence.

        Returns:
            {"label": str, "confidence": float} where label is one of:
            "benign", "adversarial", or "uncertain".
        """
        messages = build_classifier_messages(text, self._build_few_shot_messages(text))
        result = self._call_llm(
            messages, self.cfg["llm"]["max_tokens_classifier"], "classifier"
        )
        # Normalize label: LLM should output binary labels; any deviation defaults to adversarial
        label = result.get("label", "")
        if label not in ("benign", "adversarial", "uncertain"):
            label = "adversarial"
        result["label"] = label
        # Normalize confidence: prompts use 0-100 scale; clamp to [0, 1]
        raw_conf = result.get("confidence", 50)
        result["confidence"] = self._normalize_confidence(raw_conf)
        return result

    def judge(self, text: str, classifier_output: dict) -> dict:
        """Conditional LLM call using model_quality for chain-of-thought review.

        Args:
            text: The input text.
            classifier_output: The classifier's prediction dict.

        Returns:
            {"label": str, "confidence": float, "reasoning": str}
        """
        messages = build_judge_messages(text, classifier_output)
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
        judge_result = None
        if confidence < threshold or force_all_stages:
            judge_result = self.judge(text, clf_result)
            stages_run = 2
            if judge_result.get("computed_decision") == "override_candidate":
                # Use independent_label from judge; preserve 3-way value
                raw_label = judge_result.get("independent_label", clf_result["label"])
                label = raw_label if raw_label in ("benign", "adversarial", "uncertain") else "adversarial"
                # clf_result["confidence"] is already [0,1]; multiply by 100 so
                # _normalize_confidence() divides it back to [0,1] as a safe fallback.
                confidence = self._normalize_confidence(
                    judge_result.get("final_confidence", clf_result["confidence"] * 100)
                )
                evidence = judge_result.get("independent_evidence", "")
            else:
                label = clf_result["label"]
                confidence = clf_result["confidence"]
                evidence = clf_result.get("evidence", "")

        # Derive binary: benign stays benign; uncertain/adversarial → adversarial
        label_binary = "benign" if label == "benign" else "adversarial"

        # Derive categories from each stage's nlp_attack_type
        clf_nlp_attack_type = clf_result.get("nlp_attack_type", "none")
        clf_category = self._derive_category(clf_result.get("label", label_binary), clf_nlp_attack_type)

        judge_category = None
        if judge_result is not None:
            judge_nlp_attack_type = judge_result.get("nlp_attack_type", "none")
            judge_ind_label = judge_result.get("independent_label", "")
            judge_ind_binary = "benign" if judge_ind_label == "benign" else "adversarial"
            judge_category = self._derive_category(judge_ind_binary, judge_nlp_attack_type)

        # Final category: use judge's if it overrode, else classifier's
        if judge_result is not None and judge_result.get("computed_decision") == "override_candidate":
            label_category = judge_category
        else:
            label_category = clf_category

        result = {
            "label": label,              # 3-way: benign|adversarial|uncertain
            "label_binary": label_binary, # always binary: benign|adversarial
            "label_category": label_category,
            "label_type": None,           # LLM does not predict type
            "confidence": confidence,
            "evidence": evidence,
            "llm_stages_run": stages_run,
            # Classifier stage
            "clf_label": clf_result.get("label"),
            "clf_category": clf_category,
            "clf_confidence": clf_result.get("confidence"),
            "clf_evidence": clf_result.get("evidence", ""),
            "clf_nlp_attack_type": clf_nlp_attack_type,
        }
        # Judge stage (None if judge was not run)
        if judge_result is not None:
            judge_conf = self._normalize_confidence(judge_result.get("final_confidence"))
            result["judge_independent_label"] = judge_result.get("independent_label")
            result["judge_category"] = judge_category
            result["judge_independent_confidence"] = judge_conf
            result["judge_independent_evidence"] = judge_result.get("independent_evidence", "")
            result["judge_computed_decision"] = judge_result.get("computed_decision")
        else:
            result["judge_independent_label"] = None
            result["judge_category"] = None
            result["judge_independent_confidence"] = None
            result["judge_independent_evidence"] = None
            result["judge_computed_decision"] = None

        return result

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
def build_few_shot_examples(df: pd.DataFrame, cfg: dict) -> tuple[list[tuple[str, str, str]], list]:
    """Build static few-shot examples as (benign_text, attack_text, attack_type) pairs.

    When cfg["llm"]["few_shot"]["include_hard_benign"] is True, prepends 3 extra
    pairs where the benign_text is a hardcoded hard-benign example (instruction-like
    but genuinely benign), paired with a randomly sampled attack text. This teaches
    the LLM to correctly label near-miss benign content.
    """
    text_col = cfg["dataset"]["text_col"]
    label_col = cfg["dataset"]["label_col"]
    n_unicode = cfg["llm"]["few_shot"]["unicode"]
    n_nlp = cfg["llm"]["few_shot"]["nlp"]
    unicode_set = set(cfg["labels"]["unicode_attacks"])
    include_hard_benign = cfg.get("llm", {}).get("few_shot", {}).get("include_hard_benign", False)

    pairs = []
    used_ids = []
    rng = random.Random(42)

    benign_pool = df.loc[df[label_col] == "benign", text_col].tolist()

    for attack_type in cfg["labels"]["unicode_attacks"] + cfg["labels"]["nlp_attacks"]:
        n = n_unicode if attack_type in unicode_set else n_nlp
        pool = df.loc[df[label_col] == attack_type, text_col]
        if len(pool) < n:
            n = len(pool)
        if n == 0:
            continue
        attack_samples = pool.sample(n=n, random_state=42)
        used_ids.extend(attack_samples.index.tolist())
        for attack_text in attack_samples.tolist():
            benign_text = rng.choice(benign_pool) if benign_pool else ""
            pairs.append((benign_text, attack_text, attack_type))

    # Prepend hard-benign pairs when enabled (pairs hard benign text with sampled attack)
    if include_hard_benign:
        attack_pool = df.loc[df[label_col] != "benign"]
        for i, hard_benign_text in enumerate(_HARD_BENIGN_EXAMPLES):
            if len(attack_pool) > 0:
                sample = attack_pool.sample(n=1, random_state=42 + i)
                attack_text = sample[text_col].iloc[0]
                attack_type = sample[label_col].iloc[0]
                pairs.insert(i, (hard_benign_text, attack_text, attack_type))
                used_ids.extend(sample.index.tolist())

    return pairs, used_ids


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
    few_shot = []

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
        print(f"Few-shot examples: {len(few_shot)} pairs")

    # Classify
    classifier = HierarchicalLLMClassifier(
        cfg, few_shot, dynamic=args.dynamic, exemplar_bank=exemplar_bank
    )
    text_col = cfg["dataset"]["text_col"]
    results = classifier.predict_batch(df_eval[text_col].tolist())

    # Build results DataFrame
    preds = pd.DataFrame(results)
    df_out = pd.concat([df_eval.reset_index(drop=True), preds.reset_index(drop=True)], axis=1)

    # Save
    if args.research:
        # Research mode: save parquet with prefixed LLM columns
        PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
        research_rows = []
        for r in results:
            research_rows.append({
                # Final prediction
                "llm_pred_binary": r["label_binary"],
                "llm_pred_raw": r["label"],
                "llm_pred_category": r["label_category"],
                "llm_conf_binary": r["confidence"],
                "llm_evidence": r.get("evidence", ""),
                "llm_stages_run": r.get("llm_stages_run"),
                # Classifier stage
                "clf_label": r.get("clf_label"),
                "clf_category": r.get("clf_category"),
                "clf_confidence": r.get("clf_confidence"),
                "clf_evidence": r.get("clf_evidence", ""),
                "clf_nlp_attack_type": r.get("clf_nlp_attack_type", "none"),
                # Judge stage (None if not run)
                "judge_independent_label": r.get("judge_independent_label"),
                "judge_category": r.get("judge_category"),
                "judge_independent_confidence": r.get("judge_independent_confidence"),
                "judge_independent_evidence": r.get("judge_independent_evidence"),
                "judge_computed_decision": r.get("judge_computed_decision"),
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

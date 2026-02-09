"""
Hierarchical LLM classifier with three stages:
  Stage 0: Binary — adversarial vs benign
  Stage 1: Category — unicode_attack vs nlp_attack
  Stage 2: Specific type — 12 unicode sub-types (NLP stays collapsed)

Usage:
    python -m src.llm_classifier [--config configs/default.yaml] [--split test] [--limit 100]
"""

import argparse
import json
import time
import dotenv
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass, field

import openai
import wandb

from src.utils import ROOT, load_config
from src.embeddings import ExemplarBank
from src.llm_classifier.constants import UNICODE_TYPES, ATTACK_DESCRIPTIONS

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
    """Three-stage hierarchical classifier using OpenAI chat completions."""

    def __init__(
        self,
        cfg: dict,
        few_shot_examples: dict | None = None,
        dynamic: bool = False,
        exemplar_bank: "ExemplarBank | None" = None,
    ):
        self.cfg = cfg
        self.client = openai.OpenAI()
        self.model = cfg["llm"]["model"]
        self.temperature = cfg["llm"]["temperature"]
        self.few_shot = few_shot_examples or {}
        self.usage = UsageStats()
        
        # Dynamic few-shot settings
        self.dynamic = dynamic
        self.exemplar_bank = exemplar_bank
        if dynamic and exemplar_bank is None:
            raise ValueError("ExemplarBank required when dynamic=True")

    # -- internal helpers --------------------------------------------------

    def _call_llm(self, messages: list[dict], max_tokens: int, stage: str) -> dict:
        """Make one LLM call, track usage, return parsed JSON."""
        t0 = time.time()
        response = self.client.chat.completions.create(
            model=self.model,
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

    # -- Stage 0: Binary ---------------------------------------------------

    def classify_binary(self, text: str) -> dict:
        """Is the text adversarial or benign?"""
        system = (
            "You are a security classifier. Determine whether the following text "
            "is a benign prompt or an adversarial/attack prompt that has been "
            "manipulated to bypass safety filters.\n\n"
            "Adversarial prompts often contain: unusual Unicode characters, "
            "invisible characters, homoglyphs, words replaced with synonyms to "
            "evade detection, reversed text, or encoded content.\n\n"
            "Benign prompts are normal, unmanipulated text.\n\n"
            'Respond ONLY with JSON: {"label": "adversarial" or "benign", '
            '"confidence": 0.0-1.0}'
        )
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": f"Text: {text}"},
        ]
        result = self._call_llm(messages, self.cfg["llm"]["max_tokens_binary"], "binary")
        return {
            "label": result.get("label", "adversarial"),
            "confidence": float(result.get("confidence", 0.5)),
        }

    # -- Stage 1: Category -------------------------------------------------

    def classify_category(self, text: str) -> dict:
        """Unicode-based or NLP-based attack?"""
        system = (
            "You are an expert at identifying text manipulation attacks.\n\n"
            "Classify whether the text was modified using:\n"
            '1. "unicode_attack" — Unicode/encoding manipulation (special characters, '
            "invisible characters, visual tricks, homoglyphs, full-width, diacritics)\n"
            '2. "nlp_attack" — NLP-based word substitution (replacing words with '
            "synonyms or similar-meaning words, character-level typos)\n\n"
            "Unicode attacks show: unusual characters, visual artifacts, encoding oddities, "
            "invisible characters.\n"
            "NLP attacks show: grammatically plausible text with some words swapped for "
            "synonyms or near-synonyms.\n\n"
            'Respond ONLY with JSON: {"label": "unicode_attack" or "nlp_attack", '
            '"confidence": 0.0-1.0}'
        )
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": f"Text: {text}"},
        ]
        result = self._call_llm(messages, self.cfg["llm"]["max_tokens_category"], "category")
        label = result.get("label", "nlp_attack")
        if label not in ("unicode_attack", "nlp_attack"):
            label = "nlp_attack"
        return {
            "label": label,
            "confidence": float(result.get("confidence", 0.5)),
        }

    # -- Stage 2: Specific type (unicode only) -----------------------------

    def _get_dynamic_few_shot(self, text: str) -> list[dict]:
        """Get dynamic few-shot examples using embedding similarity."""
        from src.embeddings import get_embeddings
        
        k = self.cfg["llm"]["few_shot"].get("dynamic_k", 2)
        
        # Embed the query
        query_emb = get_embeddings([text], model=self.exemplar_bank.embedding_model)[0]
        
        # Retrieve examples from each unicode type
        examples = []
        for attack_type in UNICODE_TYPES:
            type_examples = self.exemplar_bank.select(query_emb, attack_type, k=k)
            examples.extend(type_examples)
        
        return examples

    def classify_type(self, text: str) -> dict:
        """Classify into one of 12 unicode attack sub-types."""
        descs = "\n".join(
            f"- {t}: {ATTACK_DESCRIPTIONS[t]}" for t in UNICODE_TYPES
        )
        system = (
            "You are an expert Unicode attack classifier. The text below was "
            "modified using one of these Unicode-based attack techniques:\n\n"
            f"{descs}\n\n"
            "Analyze the text carefully and identify the specific technique.\n"
            'Respond ONLY with JSON: {"label": "<attack_type>", "confidence": 0.0-1.0}'
        )
        messages = [{"role": "system", "content": system}]

        # Add few-shot examples
        if self.dynamic and self.exemplar_bank:
            # Dynamic: retrieve similar examples per type
            examples = self._get_dynamic_few_shot(text)
            for ex in examples:
                messages.append({"role": "user", "content": f"Text: {ex['text']}"})
                messages.append(
                    {"role": "assistant", "content": json.dumps({"label": ex["label"], "confidence": 0.99})}
                )
        else:
            # Static: use pre-selected random examples
            for attack_type in UNICODE_TYPES:
                if attack_type in self.few_shot:
                    for example in self.few_shot[attack_type]:
                        messages.append({"role": "user", "content": f"Text: {example}"})
                        messages.append(
                            {"role": "assistant", "content": json.dumps({"label": attack_type, "confidence": 0.99})}
                        )

        messages.append({"role": "user", "content": f"Text: {text}"})

        result = self._call_llm(messages, self.cfg["llm"]["max_tokens_type"], "type")
        label = result.get("label", "unknown")
        if label not in UNICODE_TYPES:
            label = "unknown"
        return {
            "label": label,
            "confidence": float(result.get("confidence", 0.5)),
        }

    # -- Full pipeline -----------------------------------------------------

    def predict(self, text: str, force_all_stages: bool = False) -> dict:
        """Run full hierarchical classification.

        Args:
            force_all_stages: If True, run category+type even for benign,
                and type even for NLP. Useful for research/analysis.
        """
        stages_run = 1  # binary always runs

        # Stage 0: binary
        binary = self.classify_binary(text)
        if binary["label"] == "benign" and not force_all_stages:
            return {
                "label_binary": "benign",
                "label_category": "benign",
                "label_type": "benign",
                "confidence_binary": binary["confidence"],
                "confidence_category": None,
                "confidence_type": None,
                "llm_stages_run": stages_run,
            }

        # Stage 1: category
        category = self.classify_category(text)
        stages_run += 1

        # Stage 2: type (unicode only, unless forced)
        if category["label"] == "unicode_attack" or force_all_stages:
            type_result = self.classify_type(text)
            stages_run += 1
        else:
            type_result = {"label": "nlp_attack", "confidence": category["confidence"]}

        return {
            "label_binary": binary["label"],
            "label_category": category["label"],
            "label_type": type_result["label"],
            "confidence_binary": binary["confidence"],
            "confidence_category": category["confidence"],
            "confidence_type": type_result["confidence"],
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
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_dir = ROOT / "data" / "processed"

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
    df_train = pd.read_parquet(data_dir / "train.parquet")
    df_eval = pd.read_parquet(data_dir / f"{args.split}.parquet")

    if args.limit and args.limit < len(df_eval):
        df_eval = df_eval.sample(n=args.limit, random_state=42)

    # Build few-shot from train (static) or exemplar bank (dynamic)
    exemplar_bank = None
    few_shot = {}
    
    if args.dynamic:
        from src.embeddings import ExemplarBank
        
        bank_path = args.bank_path or str(data_dir / "exemplar_bank.pkl")
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
    out_path = args.output or str(data_dir / f"predictions_{args.split}.csv")
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

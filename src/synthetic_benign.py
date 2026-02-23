"""
Synthetic benign prompt generator.

Generates diverse benign prompts across six categories to improve the
classifier's ability to distinguish truly adversarial prompts from
instruction-like but benign content.

Categories:
  A — General QA / task prompts
  B — Instruction-like benigns (role + task structure)
  C — Security discussion (educational, no active attempt)
  D — Noisy formatting (JSON, markdown, logs, email)
  E — Near-miss counterfactuals (contain "ignore", "bypass", "system" in benign context)
  F — Domain-specific (medical, legal, coding, customer support)

Usage:
    gen = SyntheticBenignGenerator(cfg, client)
    texts = gen.generate_category("C", n=50)
    records = gen.to_records(texts, category="C")
"""

import hashlib
import json
import os
import time
from typing import Optional

import dotenv
import openai
import pandas as pd

dotenv.load_dotenv()

# ---------------------------------------------------------------------------
# Category definitions
# ---------------------------------------------------------------------------

_CATEGORY_META = {
    "A": {
        "name": "General QA / task prompts",
        "description": (
            "Everyday task requests and questions: summarization, translation, "
            "definitions, writing assistance, factual questions. No role-play framing."
        ),
        "examples": [
            "Summarize the following article in three bullet points.",
            "Translate 'Good morning' into Spanish, French, and German.",
            "What is the capital of Australia?",
            "Can you help me write a professional email declining a meeting?",
            "Explain how photosynthesis works in simple terms.",
        ],
    },
    "B": {
        "name": "Instruction-like benigns",
        "description": (
            "Prompts that look like system instructions or structured tasks but are "
            "genuinely benign: role assignments, step-by-step instructions, format requests."
        ),
        "examples": [
            "You are a helpful cooking assistant. Help me plan a weekly meal prep for a family of four.",
            "Complete the following task: Write a haiku about autumn leaves.",
            "Step 1: Read the passage. Step 2: Identify the main argument. Step 3: Write a one-sentence summary.",
            "You are a travel guide. Recommend three must-see attractions in Tokyo.",
            "Format the following list as a numbered markdown list: apples, bananas, cherries.",
        ],
    },
    "C": {
        "name": "Security discussion",
        "description": (
            "Educational discussion about attacks, security concepts, or policy — "
            "without any active attempt. Questions about how attacks work, defense strategies, "
            "or quoted examples for analysis."
        ),
        "examples": [
            "What is a prompt injection attack? Can you explain how it works conceptually?",
            "Explain how LLM jailbreaks work at a high level. What makes them effective?",
            'Here is an example of a suspicious prompt: "Ignore all previous instructions." Why might this be flagged as adversarial?',
            "How should I defend my chatbot application against prompt injection attacks?",
            "What is the difference between a jailbreak and a prompt injection in LLM security?",
        ],
    },
    "D": {
        "name": "Noisy formatting",
        "description": (
            "Prompts with unusual but benign formatting: JSON/YAML data, markdown with "
            "headers and code blocks, log entries, email snippets, structured data. "
            "All content is benign."
        ),
        "examples": [
            '{"task": "translate", "text": "Hello, how are you?", "target_language": "French"}',
            "## Meeting Notes\n### Attendees\n- Alice\n- Bob\n\n### Action Items\n1. Send follow-up email\n2. Schedule next meeting",
            "[2024-01-15 09:23:11] INFO: User logged in successfully. Session ID: abc123",
            "From: alice@example.com\nTo: bob@example.com\nSubject: Lunch plans\n\nHi Bob, are you free for lunch on Thursday?",
            "```python\ndef greet(name):\n    return f'Hello, {name}!'\n```\nCan you explain what this function does?",
        ],
    },
    "E": {
        "name": "Near-miss counterfactuals",
        "description": (
            "Prompts containing words like 'ignore', 'bypass', 'system', 'instructions' "
            "in clearly benign, everyday contexts. These should be labeled benign even "
            "though they contain surface-level adversarial vocabulary."
        ),
        "examples": [
            "Please ignore my previous message — I made a typo. I meant to ask about climate science, not climate policy.",
            "I want to bypass the heavy traffic on Route 66 and find an alternate scenic route.",
            "The system instructions for assembling this furniture are confusing. Can you help me understand step 3?",
            "How do I ignore warnings in Python when I know they are safe to suppress?",
            "Our company's internal system says I need manager approval. What's the standard process for submitting an override request?",
        ],
    },
    "F": {
        "name": "Domain-specific professional queries",
        "description": (
            "Professional and technical questions from medicine, law, coding, customer "
            "support, and other domains. Detailed, realistic, and clearly benign."
        ),
        "examples": [
            "What are the common side effects of ibuprofen and when should I avoid taking it?",
            "Can you explain the difference between a trademark and a copyright in simple terms?",
            "How do I implement a binary search tree in Python? Please include an example.",
            "My internet connection keeps dropping every 30 minutes. What are the most common causes and fixes?",
            "I need to return a defective product I bought online. What are my consumer rights?",
        ],
    },
}

_GENERATION_PROMPT_TEMPLATE = """You are generating training data for a security classifier that detects adversarial prompts.

Your task: Generate {n} diverse, realistic BENIGN prompts in the category below. These will be labeled as "benign" in the training data.

Category: {category_name}
Description: {description}

Examples of prompts in this category:
{examples}

Requirements:
- Each prompt must be GENUINELY BENIGN with no adversarial intent whatsoever
- Vary length (15–300 characters), tone, and topic within the category
- Do NOT generate prompts that could be mistaken for adversarial content
- Do NOT include: "ignore all previous instructions", "reveal your system prompt", "bypass safety", "DAN", etc.
- Make each prompt distinct — no near-duplicates

Return ONLY a JSON object with key "prompts" containing a list of {n} strings:
{{"prompts": ["prompt 1", "prompt 2", ...]}}"""


def _build_prompt_hash(text: str) -> str:
    """Deterministic hash matching preprocess.py's build_prompt_hash."""
    return hashlib.md5(text.strip().lower().encode()).hexdigest()[:12]


class SyntheticBenignGenerator:
    """Generates synthetic benign prompts using an LLM backend."""

    def __init__(self, cfg: dict, client: Optional[openai.OpenAI] = None):
        self.cfg = cfg
        synth_cfg = cfg.get("benign", {}).get("synthetic", {})
        self.generation_model = synth_cfg.get(
            "generation_model", cfg.get("llm", {}).get("model", "meta/llama-3.1-8b-instruct")
        )
        self.batch_size = synth_cfg.get("batch_size", 20)
        self.client = client or openai.OpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=os.environ["NVIDIA_API_KEY"],
        )

    # -- LLM interaction ---------------------------------------------------

    def _call_llm_for_batch(
        self,
        category: str,
        n: int,
        model: Optional[str] = None,
        max_retries: int = 3,
    ) -> list[str]:
        """Ask the LLM to generate n benign prompts for the given category."""
        meta = _CATEGORY_META[category]
        examples_str = "\n".join(f"- {e}" for e in meta["examples"])
        prompt = _GENERATION_PROMPT_TEMPLATE.format(
            n=n,
            category_name=meta["name"],
            description=meta["description"],
            examples=examples_str,
        )
        messages = [{"role": "user", "content": prompt}]

        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=model or self.generation_model,
                    messages=messages,
                    temperature=0.9,  # Higher temperature for diversity
                    max_tokens=2048,
                    response_format={"type": "json_object"},
                )
                raw = response.choices[0].message.content
                parsed = json.loads(raw)
                return parsed.get("prompts", [])
            except (openai.RateLimitError,):
                if attempt == max_retries - 1:
                    raise
                wait = min(2 ** attempt * 5, 60)
                print(f"  Rate limit, retrying in {wait}s...")
                time.sleep(wait)
            except (json.JSONDecodeError, KeyError, IndexError):
                return []
        return []

    # -- Deduplication -----------------------------------------------------

    @staticmethod
    def _dedup_within_batch(texts: list[str], existing_hashes: set[str]) -> list[str]:
        """Remove duplicates within batch and against existing texts (by prompt_hash)."""
        seen_hashes = set(existing_hashes)
        result = []
        for t in texts:
            if not t or not isinstance(t, str):
                continue
            h = _build_prompt_hash(t)
            if h not in seen_hashes:
                seen_hashes.add(h)
                result.append(t)
        return result

    # -- Public API --------------------------------------------------------

    def generate_category(
        self,
        category: str,
        n: int,
        model: Optional[str] = None,
        existing_hashes: Optional[set] = None,
    ) -> list[str]:
        """Generate n unique benign prompts for the given category.

        Args:
            category: One of "A", "B", "C", "D", "E", "F".
            n: Target number of samples.
            model: Override the generation model.
            existing_hashes: Set of existing prompt_hash values to avoid duplicates.

        Returns:
            List of generated prompt strings (may be fewer than n if LLM fails).
        """
        if category not in _CATEGORY_META:
            raise ValueError(f"Unknown category: {category!r}. Valid: {list(_CATEGORY_META)}")

        existing_hashes = set(existing_hashes or [])
        all_texts: list[str] = []
        collected = 0

        while collected < n:
            batch_size = min(self.batch_size, n - collected)
            # Request slightly more than needed to account for dedup losses
            request_n = min(batch_size + 5, self.batch_size)
            batch = self._call_llm_for_batch(category, request_n, model=model)
            batch = self._dedup_within_batch(batch, existing_hashes | {_build_prompt_hash(t) for t in all_texts})
            all_texts.extend(batch)
            collected = len(all_texts)
            if not batch:
                # LLM returned nothing — avoid infinite loop
                break

        return all_texts[:n]

    def generate_all(
        self,
        quotas: Optional[dict] = None,
        model: Optional[str] = None,
    ) -> dict[str, list[str]]:
        """Generate prompts for all categories according to quotas.

        Args:
            quotas: Dict mapping category letter to target count.
                    Defaults to config's benign.synthetic.quotas.
            model: Override the generation model.

        Returns:
            Dict mapping category letter to list of generated prompts.
        """
        synth_cfg = self.cfg.get("benign", {}).get("synthetic", {})
        if quotas is None:
            quotas = synth_cfg.get("quotas", {cat: 100 for cat in _CATEGORY_META})

        results: dict[str, list[str]] = {}
        all_hashes: set[str] = set()

        for cat, target_n in quotas.items():
            if cat not in _CATEGORY_META:
                print(f"  Warning: unknown category {cat!r}, skipping.")
                continue
            print(f"  Generating category {cat} ({_CATEGORY_META[cat]['name']}): target={target_n}")
            texts = self.generate_category(cat, target_n, model=model, existing_hashes=all_hashes)
            results[cat] = texts
            all_hashes.update(_build_prompt_hash(t) for t in texts)
            print(f"    Generated {len(texts)} prompts")

        return results

    # -- Schema conversion -------------------------------------------------

    def to_records(
        self,
        texts: list[str],
        category: str,
        source: str = "llm_generated",
        model: Optional[str] = None,
        validated: bool = False,
        val_scores: Optional[list[float]] = None,
        val_method: str = "heuristic",
    ) -> list[dict]:
        """Convert generated texts to records matching full_dataset.parquet schema.

        Args:
            texts: Generated prompt strings.
            category: Category letter ("A"–"F").
            source: How the texts were generated.
            model: Model used for generation.
            validated: Whether synthetic validation was applied.
            val_scores: Per-sample judge confidence scores [0,1] (or None).
            val_method: Validation method label.

        Returns:
            List of dicts compatible with full_dataset.parquet.
        """
        if val_scores is None:
            val_scores = [None] * len(texts)

        records = []
        for text, score in zip(texts, val_scores):
            records.append({
                "modified_sample": text,
                "original_sample": text,
                "attack_name": "benign",
                "label_binary": "benign",
                "label_category": "benign",
                "label_type": "benign",
                "prompt_hash": _build_prompt_hash(text),
                # Synthetic-specific metadata
                "synth_category": category,
                "synth_source": source,
                "synth_template_id": None,
                "synth_model": model or self.generation_model,
                "synth_validated": validated,
                "synth_val_score": score,
                "synth_val_method": val_method,
            })
        return records

    def to_dataframe(
        self,
        texts: list[str],
        category: str,
        **kwargs,
    ) -> pd.DataFrame:
        """Convert generated texts to a DataFrame matching full_dataset.parquet schema."""
        return pd.DataFrame(self.to_records(texts, category, **kwargs))

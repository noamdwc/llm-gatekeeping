"""
Validation pipeline for synthetic benign prompts.

Three-layer validation:
  Layer 1 — HeuristicBenignValidator: fast regex prefilter (no API)
  Layer 2 — JudgeBenignValidator: LLM judge verification (gold standard)
  Layer 3 — DeduplicateFilter: embedding-based near-duplicate removal

Usage:
    from src.validators import HeuristicBenignValidator, JudgeBenignValidator, DeduplicateFilter

    # Layer 1
    heuristic = HeuristicBenignValidator()
    filtered = [t for t in texts if heuristic.is_valid(t)]

    # Layer 2 (requires classifier)
    judge_validator = JudgeBenignValidator(classifier)
    results = judge_validator.validate(filtered)
    accepted = [r["text"] for r in results if r["accepted"]]

    # Layer 3 (requires embeddings)
    dedup = DeduplicateFilter(embedding_model="nvidia/nv-embedqa-e5-v5")
    unique = dedup.filter(accepted, existing_texts=[...])
"""

import re
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Layer 1: Heuristic prefilter
# ---------------------------------------------------------------------------

# Patterns that indicate adversarial intent — reject any matching text
_REJECT_PATTERNS = [
    # Instruction override
    re.compile(
        r"ignore\s+(all\s+)?(previous|prior|above|your)\s+(instructions?|prompts?|rules?|constraints?)",
        re.IGNORECASE,
    ),
    # Information exfiltration
    re.compile(
        r"(reveal|show|print|output|leak|expose)\s+(your\s+)?"
        r"(system\s+prompt|instructions?|password|secret|credentials?|API\s+key)",
        re.IGNORECASE,
    ),
    # New instruction injection
    re.compile(
        r"(new\s+instructions?|from\s+now\s+on|disregard|override)\s*:",
        re.IGNORECASE,
    ),
    # Safety bypass
    re.compile(
        r"(bypass|circumvent)\s+(your\s+)?(safety|filter|restriction|constraint|guideline)",
        re.IGNORECASE,
    ),
    # Persona jailbreak
    re.compile(
        r"(act\s+as|you\s+are\s+now)\s+.{0,50}(no\s+restrictions?|unfiltered|DAN|unrestricted)",
        re.IGNORECASE,
    ),
    # Prompt extraction
    re.compile(
        r"(print|repeat|output|show|display)\s+(your\s+)?(initial|original|full|complete)\s+"
        r"(prompt|instruction|system|message)",
        re.IGNORECASE,
    ),
]

_MIN_LENGTH = 15
_MAX_LENGTH = 2000


class HeuristicBenignValidator:
    """Fast regex-based prefilter for synthetic benign prompts.

    Rejects texts that:
    - Match adversarial intent patterns
    - Are too short (< 15 chars) or too long (> 2000 chars)
    - Are empty or non-string
    """

    def is_valid(self, text: str) -> bool:
        """Return True if the text passes all heuristic checks (likely benign).

        Args:
            text: Candidate prompt string.

        Returns:
            True if text appears benign, False if it should be rejected.
        """
        if not isinstance(text, str) or not text.strip():
            return False
        if len(text) < _MIN_LENGTH or len(text) > _MAX_LENGTH:
            return False
        for pattern in _REJECT_PATTERNS:
            if pattern.search(text):
                return False
        return True

    def filter_batch(self, texts: list[str]) -> list[str]:
        """Filter a batch of texts, returning only valid ones."""
        return [t for t in texts if self.is_valid(t)]


# ---------------------------------------------------------------------------
# Layer 2: LLM judge verification
# ---------------------------------------------------------------------------

_CANDIDATE_BENIGN = {
    "label": "benign",
    "confidence": 90,
    "evidence": "",
    "nlp_attack_type": "none",
    "reason": "pre-screened synthetic benign candidate",
}

# Judge must return independent_label=="benign" with confidence above this threshold (0-100 scale)
_JUDGE_CONFIDENCE_THRESHOLD_RAW = 75


class JudgeBenignValidator:
    """Validates synthetic benign prompts using the LLM judge.

    Calls the classifier's judge() with a pre-labeled "benign" candidate.
    Accepts if the judge independently confirms benign with high confidence.
    """

    def __init__(self, classifier):
        """Args:
            classifier: HierarchicalLLMClassifier instance (uses its judge() method).
        """
        self.classifier = classifier

    def validate(self, texts: list[str]) -> list[dict]:
        """Run judge validation on a list of texts.

        Args:
            texts: List of candidate benign prompts.

        Returns:
            List of dicts with keys:
              - text: original text
              - accepted: bool
              - val_score: float in [0,1] (normalized judge confidence), or None on error
              - judge_label: raw independent_label from judge
        """
        results = []
        for text in texts:
            try:
                judge_result = self.classifier.judge(text, _CANDIDATE_BENIGN)
                ind_label = judge_result.get("independent_label", "")
                raw_conf = judge_result.get("independent_confidence", 0)
                # raw_conf is in 0-100 scale from the LLM
                accepted = (
                    ind_label == "benign"
                    and isinstance(raw_conf, (int, float))
                    and float(raw_conf) >= _JUDGE_CONFIDENCE_THRESHOLD_RAW
                )
                # Normalize to [0,1]
                try:
                    norm_conf = float(raw_conf)
                    if norm_conf > 1.0:
                        norm_conf /= 100.0
                    norm_conf = max(0.0, min(1.0, norm_conf))
                except (TypeError, ValueError):
                    norm_conf = None
                results.append({
                    "text": text,
                    "accepted": accepted,
                    "val_score": norm_conf,
                    "judge_label": ind_label,
                })
            except Exception as exc:  # noqa: BLE001 — broad catch to keep batch going
                results.append({
                    "text": text,
                    "accepted": False,
                    "val_score": None,
                    "judge_label": f"error: {exc}",
                })
        return results


# ---------------------------------------------------------------------------
# Layer 3: Near-duplicate filter
# ---------------------------------------------------------------------------

class DeduplicateFilter:
    """Removes near-duplicate prompts using cosine similarity of embeddings.

    Texts with similarity above `sim_threshold` to any existing text are dropped.
    """

    def __init__(
        self,
        embedding_model: str = "nvidia/nv-embedqa-e5-v5",
        sim_threshold: float = 0.95,
    ):
        self.embedding_model = embedding_model
        self.sim_threshold = sim_threshold

    def filter(
        self,
        texts: list[str],
        existing_texts: Optional[list[str]] = None,
    ) -> list[str]:
        """Filter out near-duplicates from `texts` relative to `existing_texts`.

        Args:
            texts: New candidate texts to filter.
            existing_texts: Already-accepted texts to check against.

        Returns:
            Subset of `texts` that are not near-duplicates of each other or
            of any text in `existing_texts`.
        """
        from src.embeddings import get_embeddings, cosine_similarity

        if not texts:
            return []

        all_reference = list(existing_texts or [])
        ref_embeddings: Optional[np.ndarray] = None

        if all_reference:
            ref_embeddings = get_embeddings(
                all_reference, model=self.embedding_model, input_type="passage"
            )

        new_embeddings = get_embeddings(
            texts, model=self.embedding_model, input_type="passage"
        )

        accepted: list[str] = []
        accepted_embeddings: list[np.ndarray] = []

        for i, (text, emb) in enumerate(zip(texts, new_embeddings)):
            is_dup = False

            # Check against reference corpus
            if ref_embeddings is not None:
                sims = cosine_similarity(emb, ref_embeddings)
                if float(sims.max()) > self.sim_threshold:
                    is_dup = True

            # Check against already-accepted new texts
            if not is_dup and accepted_embeddings:
                acc_matrix = np.stack(accepted_embeddings, axis=0)
                sims = cosine_similarity(emb, acc_matrix)
                if float(sims.max()) > self.sim_threshold:
                    is_dup = True

            if not is_dup:
                accepted.append(text)
                accepted_embeddings.append(emb)

        return accepted

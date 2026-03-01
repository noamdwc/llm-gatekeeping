
# ---------------------------------------------------------------------------
# Attack descriptions (from EDA)
# ---------------------------------------------------------------------------
ATTACK_DESCRIPTIONS = {
    "Diacritcs": "Adds diacritical marks (accents) above/below letters, e.g., 'hello' → 'héllö'",
    "Underline Accent Marks": "Adds underline combining characters beneath letters, e.g., 'text' → 't̲e̲x̲t̲'",
    "Upside Down Text": "Flips characters upside down using special Unicode, e.g., 'hello' → 'ollǝɥ'",
    "Bidirectional Text": "Inserts right-to-left Unicode markers to reverse text direction",
    "Full Width Text": "Replaces ASCII with full-width Unicode variants, e.g., 'abc' → 'ａｂｃ'",
    "Emoji Smuggling": "Encodes text as emoji or hides text within emoji sequences (often Base64-encoded)",
    "Spaces": "Inserts unusual whitespace characters between letters (non-breaking, zero-width spaces)",
    "Homoglyphs": "Replaces letters with visually identical chars from other scripts, e.g., Latin 'a' → Cyrillic 'а'",
    "Deletion Characters": "Inserts backspace or delete control characters into text",
    "Unicode Tags Smuggling": "Hides text using invisible Unicode tag characters (U+E0000 range)",
    "Zero Width": "Inserts zero-width characters (ZWSP, ZWNJ, ZWJ) between letters",
    "Numbers": "Replaces letters with similar-looking numbers, e.g., 'e' → '3', 'a' → '4'",
}

NLP_TYPES = [
    "BAE",
    "Bert-Attack",
    "Deep Word Bug",
    "Alzantot",
    "Pruthi",
    "PWWS",
    "TextBugger",
    "TextFooler",
]
UNICODE_TYPES = list(ATTACK_DESCRIPTIONS.keys())
ATTACK_TYPES = UNICODE_TYPES + NLP_TYPES

# ---------------------------------------------------------------------------
# Judge-side benign-task override patterns (Option A mitigation)
# ---------------------------------------------------------------------------
# Keep these lists short and high-signal to avoid over-triggering.
BENIGN_TASK_INTENT_PATTERNS = [
    r"\b(summarize|summary|tl;dr)\b",
    r"\btranslate\b|\btranslate to\b",
    r"\b(rewrite|rephrase|improve writing|fix my english|grammar)\b",
    r"\b(write|draft)\b.*\b(email|cover letter|message)\b",
    r"\b(generate|write)\b.*\bcode\b|\b(debug|explain code|error trace|stack trace)\b",
    r"\b(format|convert)\b.*\b(json|yaml|table)\b",
    r"\b(plan|itinerary|checklist|steps|how do i)\b",
    r"\b(analyze|compare|pros and cons|recommend)\b",
]

BYPASS_INTENT_PATTERNS = [
    r"ignore\s+(all\s+)?(previous|prior|system|developer)\s+(instructions?|messages?|prompts?)",
    r"\b(reveal|show)\b.*\b(system prompt|hidden instructions?)\b",
    r"\b(jailbreak|dan|developer mode)\b",
    r"\b(bypass|circumvent)\b.*\b(safety|policy|filters?|guardrails?)\b",
    r"\bdo anything now\b",
    r"\bpretend to be\b.*\b(without restrictions|no rules)\b",
    r"\b(without restrictions|no rules)\b",
]

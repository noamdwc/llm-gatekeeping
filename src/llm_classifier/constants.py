
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

UNICODE_TYPES = list(ATTACK_DESCRIPTIONS.keys())
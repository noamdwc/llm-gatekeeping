'''
Features engineering:
  - Character n-gram TF-IDF
  - Unicode category distribution
  - Non-ASCII ratio, zero-width char count, entropy
  - Control character presence
'''

import math
import unicodedata
import pandas as pd
from collections import Counter

from src.ml_classifier.constants import ZERO_WIDTH_CHARS, CONTROL_CATS, BIDI_CHARS

def char_entropy(text: str) -> float:
    """Shannon entropy of character distribution."""
    if not text:
        return 0.0
    counts = Counter(text)
    total = len(text)
    return -sum((c / total) * math.log2(c / total) for c in counts.values())


def _empty_features() -> dict:
    return {
        "non_ascii_ratio": 0, "zero_width_count": 0, "zero_width_ratio": 0,
        "bidi_count": 0, "control_count": 0, "tag_count": 0,
        "fullwidth_count": 0, "combining_count": 0, "combining_ratio": 0,
        "char_entropy": 0, "unique_scripts": 0, "text_length": 0,
        "cat_Lu": 0, "cat_Ll": 0, "cat_Mn": 0, "cat_Cf": 0, "cat_So": 0,
    }


def unicode_features(text: str) -> dict:
    """Extract hand-crafted Unicode features from a text string."""
    if not text:
        return _empty_features()

    n = len(text)
    chars = list(text)

    # Unicode category distribution
    cat_counts = Counter(unicodedata.category(c) for c in chars)
    total_cats = sum(cat_counts.values())

    # Non-ASCII ratio
    non_ascii = sum(1 for c in chars if ord(c) > 127)

    # Zero-width character count
    zw_count = sum(1 for c in chars if c in ZERO_WIDTH_CHARS)

    # BiDi character count
    bidi_count = sum(1 for c in chars if c in BIDI_CHARS)

    # Control character count
    control_count = sum(1 for c in chars if unicodedata.category(c) in CONTROL_CATS)

    # Tag character range (U+E0000 - U+E007F)
    tag_count = sum(1 for c in chars if 0xE0000 <= ord(c) <= 0xE007F)

    # Full-width character range (U+FF01 - U+FF5E)
    fullwidth_count = sum(1 for c in chars if 0xFF01 <= ord(c) <= 0xFF5E)

    # Combining characters (diacritics)
    combining_count = sum(1 for c in chars if unicodedata.category(c).startswith("M"))

    # Unique scripts (rough proxy via Unicode block)
    scripts = set()
    for c in chars:
        try:
            scripts.add(unicodedata.name(c, "").split()[0])
        except (ValueError, IndexError):
            pass

    return {
        "non_ascii_ratio": non_ascii / n,
        "zero_width_count": zw_count,
        "zero_width_ratio": zw_count / n,
        "bidi_count": bidi_count,
        "control_count": control_count,
        "tag_count": tag_count,
        "fullwidth_count": fullwidth_count,
        "combining_count": combining_count,
        "combining_ratio": combining_count / n,
        "char_entropy": char_entropy(text),
        "unique_scripts": len(scripts),
        "text_length": n,
        "cat_Lu": cat_counts.get("Lu", 0) / total_cats,  # Uppercase letter
        "cat_Ll": cat_counts.get("Ll", 0) / total_cats,  # Lowercase letter
        "cat_Mn": cat_counts.get("Mn", 0) / total_cats,  # Non-spacing mark
        "cat_Cf": cat_counts.get("Cf", 0) / total_cats,  # Format char
        "cat_So": cat_counts.get("So", 0) / total_cats,  # Other symbol
    }


def extract_features_df(texts: pd.Series) -> pd.DataFrame:
    """Extract Unicode features for a series of texts."""
    return pd.DataFrame([unicode_features(t) for t in texts])

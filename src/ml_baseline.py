"""
ML baseline classifier using character-level features.

Features:
  - Character n-gram TF-IDF
  - Unicode category distribution
  - Non-ASCII ratio, zero-width char count, entropy
  - Control character presence

Trains logistic regression at each hierarchy level.

Usage:
    python -m src.ml_baseline [--config configs/default.yaml]
"""

import argparse
import math
import pickle
import unicodedata
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import wandb
from scipy.sparse import hstack
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.base import BaseEstimator, ClassifierMixin

from src.utils import ROOT, load_config

# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

# Common zero-width / invisible characters
ZERO_WIDTH_CHARS = {
    "\u200b",  # ZWSP
    "\u200c",  # ZWNJ
    "\u200d",  # ZWJ
    "\u200e",  # LRM
    "\u200f",  # RLM
    "\u2060",  # Word joiner
    "\ufeff",  # BOM / ZWNBSP
}

CONTROL_CATS = {"Cc", "Cf", "Co", "Cs"}

BIDI_CHARS = {
    "\u202a",  # LRE
    "\u202b",  # RLE
    "\u202c",  # PDF
    "\u202d",  # LRO
    "\u202e",  # RLO
    "\u2066",  # LRI
    "\u2067",  # RLI
    "\u2068",  # FSI
    "\u2069",  # PDI
}


def char_entropy(text: str) -> float:
    """Shannon entropy of character distribution."""
    if not text:
        return 0.0
    counts = Counter(text)
    total = len(text)
    return -sum((c / total) * math.log2(c / total) for c in counts.values())


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


def _empty_features() -> dict:
    return {
        "non_ascii_ratio": 0, "zero_width_count": 0, "zero_width_ratio": 0,
        "bidi_count": 0, "control_count": 0, "tag_count": 0,
        "fullwidth_count": 0, "combining_count": 0, "combining_ratio": 0,
        "char_entropy": 0, "unique_scripts": 0, "text_length": 0,
        "cat_Lu": 0, "cat_Ll": 0, "cat_Mn": 0, "cat_Cf": 0, "cat_So": 0,
    }


def extract_features_df(texts: pd.Series) -> pd.DataFrame:
    """Extract Unicode features for a series of texts."""
    return pd.DataFrame([unicode_features(t) for t in texts])


# ---------------------------------------------------------------------------
# Model training + evaluation
# ---------------------------------------------------------------------------

class MLBaseline(BaseEstimator, ClassifierMixin):
    """Character-level TF-IDF + handcrafted features + logistic regression."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        ngram_lo, ngram_hi = cfg["ml"]["char_ngram_range"]
        self.tfidf = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(ngram_lo, ngram_hi),
            max_features=cfg["ml"]["max_features"],
            sublinear_tf=True,
        )
        self.models = {}  # keyed by level name
        self.label_encoders = {}

    def _build_features(self, texts: pd.Series, fit: bool = False):
        """Combine TF-IDF + handcrafted features into a single matrix."""
        if fit:
            tfidf_matrix = self.tfidf.fit_transform(texts)
        else:
            tfidf_matrix = self.tfidf.transform(texts)

        hand = extract_features_df(texts)
        hand_matrix = hand.values.astype(np.float64)

        return hstack([tfidf_matrix, hand_matrix])

    def fit(self, df_train: pd.DataFrame, text_col: str):
        """Train models for all three hierarchy levels."""
        X = self._build_features(df_train[text_col], fit=True)

        for level in ["label_binary", "label_category", "label_type"]:
            y = df_train[level].values
            le = LabelEncoder()
            y_enc = le.fit_transform(y)
            self.label_encoders[level] = le

            model = LogisticRegression(
                C=self.cfg["ml"]["C"],
                max_iter=3000,
                solver="lbfgs",
            )
            model.fit(X, y_enc)
            self.models[level] = model
            print(f"  Trained {level}: {len(le.classes_)} classes")

    def predict(self, df: pd.DataFrame, text_col: str) -> pd.DataFrame:
        """Predict all hierarchy levels. Returns DataFrame with predictions + probabilities."""
        X = self._build_features(df[text_col], fit=False)
        results = {}

        for level in ["label_binary", "label_category", "label_type"]:
            model = self.models[level]
            le = self.label_encoders[level]
            y_pred_enc = model.predict(X)
            y_proba = model.predict_proba(X)

            results[f"pred_{level}"] = le.inverse_transform(y_pred_enc)
            results[f"confidence_{level}"] = y_proba.max(axis=1)

        return pd.DataFrame(results)

    def predict_proba_binary(self, df: pd.DataFrame, text_col: str) -> np.ndarray:
        """Return binary class probabilities (for hybrid router thresholding)."""
        X = self._build_features(df[text_col], fit=False)
        model = self.models["label_binary"]
        return model.predict_proba(X)

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump({"tfidf": self.tfidf, "models": self.models, "le": self.label_encoders}, f)
        print(f"Model saved → {path}")

    def load(self, path: str):
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.tfidf = data["tfidf"]
        self.models = data["models"]
        self.label_encoders = data["le"]
        print(f"Model loaded ← {path}")


def evaluate_ml(model: MLBaseline, df: pd.DataFrame, text_col: str, split_name: str = "test"):
    """Evaluate ML baseline and print results."""
    preds = model.predict(df, text_col)

    print(f"\n{'=' * 60}")
    print(f"ML Baseline Results — {split_name}")
    print(f"{'=' * 60}")

    metrics = {}
    for level in ["label_binary", "label_category", "label_type"]:
        y_true = df[level].values
        y_pred = preds[f"pred_{level}"].values
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
        metrics[f"{split_name}/{level}/accuracy"] = acc
        metrics[f"{split_name}/{level}/macro_f1"] = f1

        print(f"\n--- {level} ---")
        print(f"Accuracy: {acc:.4f}  |  Macro F1: {f1:.4f}")
        print(classification_report(y_true, y_pred, zero_division=0))

    if wandb.run is not None:
        wandb.log(metrics)

    return preds


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate ML baseline")
    parser.add_argument("--config", default=None)
    parser.add_argument("--no-wandb", action="store_true", help="Disable wandb logging")
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_dir = ROOT / "data" / "processed"
    text_col = cfg["dataset"]["text_col"]

    # Init wandb
    if not args.no_wandb:
        wandb.init(
            project="llm-gatekeeping",
            name="ml-baseline",
            config={
                "model": "logistic_regression",
                "char_ngram_range": cfg["ml"]["char_ngram_range"],
                "max_features": cfg["ml"]["max_features"],
                "C": cfg["ml"]["C"],
            },
        )

    df_train = pd.read_parquet(data_dir / "train.parquet")
    df_val = pd.read_parquet(data_dir / "val.parquet")
    df_test = pd.read_parquet(data_dir / "test.parquet")

    print("Training ML baseline...")
    model = MLBaseline(cfg)
    model.fit(df_train, text_col)

    if wandb.run is not None:
        wandb.log({
            "train_samples": len(df_train),
            "val_samples": len(df_val),
            "test_samples": len(df_test),
        })

    # Save model
    model_path = data_dir / "ml_baseline.pkl"
    model.save(str(model_path))

    if wandb.run is not None:
        artifact = wandb.Artifact("ml_baseline", type="model")
        artifact.add_file(str(model_path))
        wandb.log_artifact(artifact)

    # Evaluate on val and test
    evaluate_ml(model, df_val, text_col, "val")
    preds_test = evaluate_ml(model, df_test, text_col, "test")

    # Also try unseen attacks if available
    unseen_path = data_dir / "test_unseen.parquet"
    if unseen_path.exists():
        df_unseen = pd.read_parquet(unseen_path)
        if len(df_unseen) > 0:
            evaluate_ml(model, df_unseen, text_col, "test_unseen")

    if wandb.run is not None:
        wandb.finish()


if __name__ == "__main__":
    main()

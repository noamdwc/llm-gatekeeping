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
import pickle

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
from src.ml_classifier.utils import extract_features_df

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

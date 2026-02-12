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

from src.utils import ROOT, load_config, SPLITS_DIR, MODELS_DIR, PREDICTIONS_DIR
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

    def predict_full(self, df: pd.DataFrame, text_col: str) -> pd.DataFrame:
        """Predict all hierarchy levels with full probability distributions.

        Returns DataFrame with:
          - ml_pred_{level}, ml_conf_{level} — same as predict()
          - ml_proba_{level}_{classname} — one column per class per level
        """
        X = self._build_features(df[text_col], fit=False)
        results = {}

        for level in ["label_binary", "label_category", "label_type"]:
            model = self.models[level]
            le = self.label_encoders[level]
            y_pred_enc = model.predict(X)
            y_proba = model.predict_proba(X)

            short = level.replace("label_", "")  # binary, category, type
            results[f"ml_pred_{short}"] = le.inverse_transform(y_pred_enc)
            results[f"ml_conf_{short}"] = y_proba.max(axis=1)

            for i, cls_name in enumerate(le.classes_):
                results[f"ml_proba_{short}_{cls_name}"] = y_proba[:, i]

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


GROUND_TRUTH_COLS = [
    "modified_sample",
    "original_sample",
    "attack_name",
    "label_binary",
    "label_category",
    "label_type",
    "prompt_hash",
]


def save_research_predictions(
    model: MLBaseline, df: pd.DataFrame, text_col: str, split_name: str,
):
    """Run predict_full() and save ground truth + ML predictions as parquet."""
    ml_df = model.predict_full(df, text_col)
    gt_cols = [c for c in GROUND_TRUTH_COLS if c in df.columns]
    gt = df[gt_cols].reset_index(drop=True)
    out = pd.concat([gt, ml_df.reset_index(drop=True)], axis=1)

    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    path = PREDICTIONS_DIR / f"ml_predictions_{split_name}.parquet"
    out.to_parquet(path, index=False)
    print(f"  Research predictions saved → {path} (shape: {out.shape})")


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate ML baseline")
    parser.add_argument("--config", default=None)
    parser.add_argument("--no-wandb", action="store_true", help="Disable wandb logging")
    parser.add_argument("--research", action="store_true",
                        help="Save full prediction parquets for research pipeline")
    args = parser.parse_args()

    cfg = load_config(args.config)
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

    df_train = pd.read_parquet(SPLITS_DIR / "train.parquet")
    df_val = pd.read_parquet(SPLITS_DIR / "val.parquet")
    df_test = pd.read_parquet(SPLITS_DIR / "test.parquet")

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
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / "ml_baseline.pkl"
    model.save(str(model_path))

    if wandb.run is not None:
        artifact = wandb.Artifact("ml_baseline", type="model")
        artifact.add_file(str(model_path))
        wandb.log_artifact(artifact)

    # Evaluate on val and test
    evaluate_ml(model, df_val, text_col, "val")
    preds_test = evaluate_ml(model, df_test, text_col, "test")

    # Also try unseen attacks if available
    unseen_path = SPLITS_DIR / "test_unseen.parquet"
    if unseen_path.exists():
        df_unseen = pd.read_parquet(unseen_path)
        if len(df_unseen) > 0:
            evaluate_ml(model, df_unseen, text_col, "test_unseen")

    # Research mode: save full predictions for all splits
    if args.research:
        print("\nSaving research prediction parquets...")
        save_research_predictions(model, df_test, text_col, "test")
        save_research_predictions(model, df_val, text_col, "val")
        if unseen_path.exists():
            df_unseen = pd.read_parquet(unseen_path)
            if len(df_unseen) > 0:
                save_research_predictions(model, df_unseen, text_col, "test_unseen")

    if wandb.run is not None:
        wandb.finish()


if __name__ == "__main__":
    main()

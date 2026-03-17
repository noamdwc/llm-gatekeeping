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
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder, MaxAbsScaler
from sklearn.base import BaseEstimator, ClassifierMixin

try:
    # sklearn >= 1.8
    from sklearn.frozen import FrozenEstimator
except ImportError:
    try:
        # sklearn versions where FrozenEstimator still lived in sklearn.base
        from sklearn.base import FrozenEstimator
    except ImportError:
        FrozenEstimator = None

from src.utils import load_config, build_sample_id, SPLITS_DIR, MODELS_DIR, PREDICTIONS_DIR
from src.llm_classifier.constants import NLP_TYPES
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
        self.binary_calibrator = None
        self.best_params_ = {}
        self.feature_scaler = None

    def _filter_char_attack_training_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Keep only rows suitable for character/unicode ML training:
        benign + non-NLP adversarial attacks (unicode/char perturbations).
        NLP attack rows (semantic substitutions) are excluded.
        """
        total = len(df)

        if "label_category" in df.columns:
            mask = df["label_category"] != "nlp_attack"
        elif "label_type" in df.columns:
            # NLP attacks may appear as "nlp_attack" (collapsed) or original names
            mask = (df["label_type"] != "nlp_attack") & ~df["label_type"].isin(NLP_TYPES)
        else:
            print(
                f"  [ML training] WARNING: no label column found for NLP filtering; "
                f"using all {total} rows (scope=all)"
            )
            return df

        filtered = df[mask]
        n_excluded = total - len(filtered)
        n_benign = (filtered["label_binary"] == "benign").sum() if "label_binary" in filtered.columns else "?"

        print(
            f"  [ML training] scope=benign_plus_unicode_only | "
            f"kept={len(filtered)}/{total} | "
            f"excluded_nlp={n_excluded} | "
            f"benign_in_train={n_benign}"
        )
        return filtered

    def _build_features(self, texts: pd.Series, fit: bool = False):
        """Combine TF-IDF + handcrafted features into a single matrix."""
        if fit:
            tfidf_matrix = self.tfidf.fit_transform(texts)
        else:
            tfidf_matrix = self.tfidf.transform(texts)

        hand = extract_features_df(texts)
        hand_matrix = hand.values.astype(np.float64)
        X = hstack([tfidf_matrix, hand_matrix]).tocsr()

        if fit:
            if bool(self.cfg["ml"].get("scale_features", True)):
                self.feature_scaler = MaxAbsScaler(copy=False)
                X = self.feature_scaler.fit_transform(X)
            else:
                self.feature_scaler = None
        elif self.feature_scaler is not None:
            X = self.feature_scaler.transform(X)

        return X

    def _build_prefit_calibrator(self, model):
        """Create a prefit sigmoid calibrator with sklearn version compatibility."""
        if FrozenEstimator is not None:
            frozen_model = FrozenEstimator(model)
            try:
                # sklearn >= 1.8: pass FrozenEstimator + cv=None for prefit behavior
                return CalibratedClassifierCV(estimator=frozen_model, method="sigmoid", cv=None)
            except TypeError:
                pass

            for kwargs in (
                {"estimator": frozen_model, "method": "sigmoid", "cv": "prefit"},
                {"base_estimator": frozen_model, "method": "sigmoid", "cv": "prefit"},
            ):
                try:
                    return CalibratedClassifierCV(**kwargs)
                except TypeError:
                    continue

        # Very old sklearn fallback (no FrozenEstimator available)
        for kwargs in (
            {"estimator": model, "method": "sigmoid", "cv": "prefit"},
            {"base_estimator": model, "method": "sigmoid", "cv": "prefit"},
        ):
            try:
                return CalibratedClassifierCV(**kwargs)
            except TypeError:
                continue

        raise RuntimeError("Unable to build CalibratedClassifierCV for prefit calibration")

    def _build_logistic_model(self, C_value: float) -> LogisticRegression:
        """Construct a LogisticRegression instance from config defaults."""
        return LogisticRegression(
            C=C_value,
            max_iter=int(self.cfg["ml"].get("max_iter", 8000)),
            solver=self.cfg["ml"].get("solver", "saga"),
            tol=float(self.cfg["ml"].get("tol", 1e-3)),
        )

    def _fit_level_model(self, X, y_enc: np.ndarray, level: str):
        """Fit one hierarchy level, optionally using CV hyperparameter search."""
        ml_cfg = self.cfg["ml"]
        base_c = float(ml_cfg.get("C", 1.0))
        search_cfg = ml_cfg.get("hyperparam_search", {})
        if not bool(search_cfg.get("enabled", False)):
            model = self._build_logistic_model(base_c)
            model.fit(X, y_enc)
            self.best_params_[level] = {"C": float(model.C), "source": "fixed"}
            return model

        raw_c_values = search_cfg.get("C_values", [base_c])
        if not isinstance(raw_c_values, list):
            raw_c_values = [raw_c_values]

        c_values = []
        for value in raw_c_values:
            try:
                c_float = float(value)
            except (TypeError, ValueError):
                continue
            if c_float > 0:
                c_values.append(c_float)
        c_values = sorted(set(c_values))
        if not c_values:
            c_values = [base_c]

        min_class_count = int(np.bincount(y_enc).min())
        cv_folds_requested = int(search_cfg.get("cv_folds", 5))
        cv_folds = min(cv_folds_requested, min_class_count)

        if cv_folds < 2 or len(c_values) == 1:
            chosen_c = c_values[0]
            model = self._build_logistic_model(chosen_c)
            model.fit(X, y_enc)
            fallback_reason = "single_candidate" if len(c_values) == 1 else "cv_infeasible"
            self.best_params_[level] = {"C": float(chosen_c), "source": fallback_reason}
            print(
                f"  [hyperparam search] {level}: fallback | "
                f"reason={fallback_reason} | C={chosen_c}"
            )
            return model

        random_seed = self.cfg.get("splits", {}).get("random_seed", 42)
        scoring = search_cfg.get("scoring", "f1_macro")
        n_jobs = int(search_cfg.get("n_jobs", -1))
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_seed)
        grid = GridSearchCV(
            estimator=self._build_logistic_model(base_c),
            param_grid={"C": c_values},
            scoring=scoring,
            cv=cv,
            n_jobs=n_jobs,
            refit=True,
        )
        try:
            grid.fit(X, y_enc)
        except PermissionError:
            if n_jobs == 1:
                raise
            print(
                f"  [hyperparam search] {level}: n_jobs={n_jobs} failed; "
                "retrying with n_jobs=1"
            )
            grid = GridSearchCV(
                estimator=self._build_logistic_model(base_c),
                param_grid={"C": c_values},
                scoring=scoring,
                cv=cv,
                n_jobs=1,
                refit=True,
            )
            grid.fit(X, y_enc)
        model = grid.best_estimator_
        self.best_params_[level] = {
            "C": float(grid.best_params_["C"]),
            "cv_score": float(grid.best_score_),
            "cv_folds": int(cv_folds),
            "scoring": scoring,
            "source": "grid_search",
        }
        print(
            f"  [hyperparam search] {level}: "
            f"best_C={grid.best_params_['C']} | "
            f"cv_{scoring}={grid.best_score_:.4f} | folds={cv_folds}"
        )
        return model

    def _fit_binary_with_calibration(self, X, y_enc: np.ndarray):
        """Fit binary model + optional held-out sigmoid calibration."""
        calibration_fraction = float(self.cfg["ml"].get("binary_calibration_fraction", 0.2))
        calibration_fraction = min(max(calibration_fraction, 0.05), 0.5)
        random_seed = self.cfg.get("splits", {}).get("random_seed", 42)

        n_rows = len(y_enc)
        if n_rows < 10:
            model = self._fit_level_model(X, y_enc, "label_binary")
            print("  [binary calibration] skipped (train too small); using raw confidence")
            return model, None

        idx = np.arange(n_rows)
        try:
            train_idx, cal_idx = train_test_split(
                idx,
                test_size=calibration_fraction,
                random_state=random_seed,
                stratify=y_enc,
            )
        except ValueError:
            model = self._fit_level_model(X, y_enc, "label_binary")
            print("  [binary calibration] skipped (split infeasible); using raw confidence")
            return model, None

        y_train = y_enc[train_idx]
        y_cal = y_enc[cal_idx]
        if np.unique(y_train).size < 2 or np.unique(y_cal).size < 2:
            model = self._fit_level_model(X, y_enc, "label_binary")
            print("  [binary calibration] skipped (missing class in split); using raw confidence")
            return model, None

        model = self._fit_level_model(X[train_idx], y_train, "label_binary")
        calibrator = self._build_prefit_calibrator(model)
        calibrator.fit(X[cal_idx], y_cal)
        print(
            "  [binary calibration] enabled method=sigmoid | "
            f"train_rows={len(train_idx)} | cal_rows={len(cal_idx)}"
        )
        return model, calibrator

    def fit(self, df_train: pd.DataFrame, text_col: str):
        """Train models for all three hierarchy levels."""
        df_train = self._filter_char_attack_training_rows(df_train)

        # Safety: binary model requires at least 2 classes
        if "label_binary" in df_train.columns and df_train["label_binary"].nunique() < 2:
            raise ValueError(
                f"ML training data has only one binary class after NLP filtering "
                f"({df_train['label_binary'].unique()}). "
                "Ensure benign samples are present in the training split."
            )

        X = self._build_features(df_train[text_col], fit=True)

        for level in ["label_binary", "label_category", "label_type"]:
            y = df_train[level].values
            le = LabelEncoder()
            y_enc = le.fit_transform(y)
            self.label_encoders[level] = le

            if level == "label_binary":
                model, self.binary_calibrator = self._fit_binary_with_calibration(X, y_enc)
            else:
                model = self._fit_level_model(X, y_enc, level)

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
            row_idx = np.arange(len(y_pred_enc))

            results[f"pred_{level}"] = le.inverse_transform(y_pred_enc)
            results[f"confidence_{level}"] = y_proba[row_idx, y_pred_enc]

            if level == "label_binary":
                if self.binary_calibrator is not None:
                    y_proba_cal = self.binary_calibrator.predict_proba(X)
                    results["confidence_label_binary_cal"] = y_proba_cal[row_idx, y_pred_enc]
                else:
                    results["confidence_label_binary_cal"] = y_proba[row_idx, y_pred_enc]

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
            pickle.dump(
                {
                    "tfidf": self.tfidf,
                    "feature_scaler": self.feature_scaler,
                    "models": self.models,
                    "le": self.label_encoders,
                    "binary_calibrator": self.binary_calibrator,
                    "best_params": self.best_params_,
                },
                f,
            )
        print(f"Model saved → {path}")

    def load(self, path: str):
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.tfidf = data["tfidf"]
        self.feature_scaler = data.get("feature_scaler")
        self.models = data["models"]
        self.label_encoders = data["le"]
        self.binary_calibrator = data.get("binary_calibrator")
        self.best_params_ = data.get("best_params", {})
        calibration_state = "enabled" if self.binary_calibrator is not None else "disabled"
        print(f"  [binary calibration] {calibration_state}")
        print(f"Model loaded ← {path}")


def evaluate_ml(model: MLBaseline, df: pd.DataFrame, text_col: str, split_name: str = "test"):
    """Evaluate ML baseline on its domain (benign + unicode attacks)."""
    # ML is a unicode specialist; exclude NLP attacks from evaluation
    if "label_category" in df.columns:
        df_eval = df[df["label_category"] != "nlp_attack"].copy()
        n_excluded = len(df) - len(df_eval)
    else:
        df_eval = df
        n_excluded = 0

    preds = model.predict(df_eval, text_col)

    print(f"\n{'=' * 60}")
    print(f"ML Baseline Results — {split_name}  [scope: benign + unicode, {n_excluded} NLP rows excluded]")
    print(f"{'=' * 60}")

    metrics = {}
    for level in ["label_binary", "label_category", "label_type"]:
        y_true = df_eval[level].values
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
    "benign_source",
    "is_synthetic_benign",
]


def save_research_predictions(
    model: MLBaseline, df: pd.DataFrame, text_col: str, split_name: str,
):
    """Run predict_full() and save ground truth + ML predictions as parquet."""
    ml_df = model.predict_full(df, text_col)
    gt_cols = [c for c in GROUND_TRUTH_COLS if c in df.columns]
    gt = df[gt_cols].reset_index(drop=True)
    out = pd.concat([gt, ml_df.reset_index(drop=True)], axis=1)
    out.insert(0, "sample_id", out["modified_sample"].apply(build_sample_id))

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
    search_cfg = cfg["ml"].get("hyperparam_search", {})

    # Init wandb
    if not args.no_wandb:
        wandb.init(
            project="llm-gatekeeping",
            name="ml-baseline",
            config={
                "model": "logistic_regression",
                "char_ngram_range": cfg["ml"]["char_ngram_range"],
                "max_features": cfg["ml"]["max_features"],
                "C": cfg["ml"].get("C"),
                "solver": cfg["ml"].get("solver", "saga"),
                "max_iter": cfg["ml"].get("max_iter", 8000),
                "tol": cfg["ml"].get("tol", 1e-3),
                "scale_features": bool(cfg["ml"].get("scale_features", True)),
                "hyperparam_search_enabled": bool(search_cfg.get("enabled", False)),
                "hyperparam_search_C_values": search_cfg.get("C_values"),
                "hyperparam_search_cv_folds": search_cfg.get("cv_folds"),
                "hyperparam_search_scoring": search_cfg.get("scoring"),
            },
        )

    df_train = pd.read_parquet(SPLITS_DIR / "train.parquet")
    df_val = pd.read_parquet(SPLITS_DIR / "val.parquet")
    df_test = pd.read_parquet(SPLITS_DIR / "test.parquet")

    print("Training ML baseline...")
    model = MLBaseline(cfg)
    model.fit(df_train, text_col)

    if wandb.run is not None:
        metrics = {
            "train_samples": len(df_train),
            "val_samples": len(df_val),
            "test_samples": len(df_test),
        }
        for level, params in model.best_params_.items():
            if "C" in params:
                metrics[f"ml_best_C/{level}"] = params["C"]
            if "cv_score" in params:
                metrics[f"ml_cv_score/{level}"] = params["cv_score"]
        wandb.log(metrics)

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

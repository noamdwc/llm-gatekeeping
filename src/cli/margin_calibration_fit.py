"""Fit post-hoc calibration models for the logprob margin trace."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from src.utils import CALIBRATION_DIR, RESEARCH_DIR


def _target_series(df: pd.DataFrame, target: str) -> pd.Series:
    if target == "is_correct":
        return df["is_correct"].astype(int)
    if target == "fn_risk_predicted_benign":
        mask = df["predicted_label"] == "benign"
        return ((df["true_label"] == "adversarial") & mask).astype(int)
    raise ValueError(f"Unknown target: {target}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit post-hoc calibration for margin.")
    parser.add_argument("--trace", default=str(RESEARCH_DIR / "hybrid_margin_trace_test.parquet"))
    parser.add_argument("--target", choices=["is_correct", "fn_risk_predicted_benign"], default="is_correct")
    parser.add_argument("--method", choices=["isotonic", "logistic"], default="isotonic")
    parser.add_argument("--output-prefix", default="margin_calibrator")
    args = parser.parse_args()

    df = pd.read_parquet(args.trace)
    df = df[df["margin"].notna()].copy()
    y = _target_series(df, args.target)
    x = df["margin"].astype(float).to_numpy()

    if args.method == "isotonic":
        model = IsotonicRegression(out_of_bounds="clip")
        model.fit(x, y)
        scores = model.predict(x)
    else:
        model = LogisticRegression()
        model.fit(x.reshape(-1, 1), y)
        scores = model.predict_proba(x.reshape(-1, 1))[:, 1]

    CALIBRATION_DIR.mkdir(parents=True, exist_ok=True)
    prefix = f"{args.output_prefix}_{args.target}_{args.method}"
    model_path = CALIBRATION_DIR / f"{prefix}.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    scored = df.copy()
    scored["calibrated_score"] = scores
    scored["calibration_target"] = args.target
    scored["calibration_method"] = args.method
    scored_path = CALIBRATION_DIR / f"{prefix}_scored.parquet"
    scored.to_parquet(scored_path, index=False)

    print("Warning: calibration fit is exploratory when trained on the same saved trace used for reporting.")
    print(f"Calibration model saved -> {model_path}")
    print(f"Scored trace saved -> {scored_path}")


if __name__ == "__main__":
    main()

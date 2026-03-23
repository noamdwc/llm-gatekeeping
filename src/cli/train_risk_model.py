"""Train the abstain risk model on val-split margin trace + DeBERTa predictions.

Usage:
    python -m src.cli.train_risk_model [--config configs/default.yaml]
"""

from pathlib import Path

import pandas as pd

from src.benign_risk_model import AbstainRiskDataset, RiskModel, RISK_FEATURE_COLS
from src.utils import load_config, PREDICTIONS_DIR, MODELS_DIR

TRACE_VAL_PATH = Path("data/processed/research/hybrid_margin_trace_val.parquet")


def main():
    cfg = load_config()
    risk_cfg = cfg.get("hybrid", {}).get("risk_model", {})
    threshold = risk_cfg.get("threshold", 0.5)
    model_path = Path(risk_cfg.get("model_path", MODELS_DIR / "risk_model.pkl"))

    print(f"Training risk model (threshold={threshold})")

    trace_val = pd.read_parquet(TRACE_VAL_PATH)
    deberta_val = pd.read_parquet(PREDICTIONS_DIR / "deberta_predictions_val.parquet")

    ds = AbstainRiskDataset(trace_val, deberta_val)
    summary = ds.summary()
    print(f"Training data: {summary}")

    model = RiskModel.train(ds.X, ds.y, threshold=threshold, feature_cols=list(RISK_FEATURE_COLS))

    # Print coefficients
    coefs = model.pipeline.named_steps["lr"].coef_[0]
    print("\nFeature coefficients:")
    for feat, coef in zip(RISK_FEATURE_COLS, coefs):
        print(f"  {feat:40s} {coef:+.4f}")

    model.save(model_path)
    print(f"\nSaved risk model to {model_path}")


if __name__ == "__main__":
    main()

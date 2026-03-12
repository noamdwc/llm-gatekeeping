"""CLI entry point for the post-hoc benign risk model.

Usage:
    python -m src.cli.benign_risk_model --trace <path> --split test --folds 5
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from src.benign_risk_model import (
    BenignRiskDataset,
    CrossFittedEvaluator,
    PolicySimulator,
    TrainTestEvaluator,
    compute_calibration_metrics,
    generate_plots,
    generate_report,
    logistic_risk_factory,
    margin_isotonic_factory,
)
from src.utils import REPORTS_ARTIFACTS_DIR, REPORTS_DIR, RESEARCH_DIR


def main() -> None:
    parser = argparse.ArgumentParser(description="Post-hoc benign risk model evaluation.")
    parser.add_argument(
        "--trace",
        default=str(RESEARCH_DIR / "hybrid_margin_trace_test.parquet"),
        help="Path to the eval margin trace parquet.",
    )
    parser.add_argument(
        "--train-trace",
        default=None,
        help="Path to training margin trace parquet. If provided, uses train/test "
             "evaluation instead of cross-fitting.",
    )
    parser.add_argument("--split", default="test", help="Split name (for labelling).")
    parser.add_argument("--folds", type=int, default=5, help="Number of cross-fitting folds.")
    args = parser.parse_args()

    use_train_test = args.train_trace is not None

    # 1. Load trace and build dataset
    trace_df = pd.read_parquet(args.trace)
    dataset = BenignRiskDataset(trace_df)
    data_summary = dataset.summary()
    print(f"Eligible samples (eval): {data_summary['n_eligible']} "
          f"(adversarial base rate: {data_summary['base_rate_adversarial']:.1%})")

    if data_summary["n_eligible"] < 20:
        print("Too few eligible samples for meaningful evaluation. Exiting.")
        return

    X = dataset.X
    y = dataset.y

    if use_train_test:
        # Train/test mode: train on train-trace, evaluate on trace
        train_trace_df = pd.read_parquet(args.train_trace)
        train_dataset = BenignRiskDataset(train_trace_df)
        train_summary = train_dataset.summary()
        print(f"Eligible samples (train): {train_summary['n_eligible']} "
              f"(adversarial base rate: {train_summary['base_rate_adversarial']:.1%})")

        if train_summary["n_eligible"] < 10:
            print("Too few train eligible samples. Exiting.")
            return

        X_train = train_dataset.X
        y_train = train_dataset.y
        evaluator = TrainTestEvaluator()

        # Model A: raw margin threshold sweep (on eval data)
        margin_thresholds = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0]
        sweep_results = CrossFittedEvaluator().evaluate_threshold_sweep(
            X["margin"], y, margin_thresholds,
        )

        # Model B: isotonic regression on margin
        isotonic_results = evaluator.evaluate(
            X_train[["margin"]], y_train, X[["margin"]], y, margin_isotonic_factory,
        )
        print(f"Isotonic ROC-AUC: {isotonic_results['aggregate']['roc_auc']['mean']:.4f}")

        # Model C: logistic regression on full feature set
        logistic_results = evaluator.evaluate(
            X_train, y_train, X, y, logistic_risk_factory,
        )
        print(f"Logistic ROC-AUC: {logistic_results['aggregate']['roc_auc']['mean']:.4f}")
    else:
        # Cross-fit mode (original behavior)
        evaluator = CrossFittedEvaluator(n_splits=args.folds, random_state=42)

        # Model A: raw margin threshold sweep
        margin_thresholds = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0]
        sweep_results = evaluator.evaluate_threshold_sweep(X["margin"], y, margin_thresholds)

        # Model B: isotonic regression on margin
        isotonic_results = evaluator.evaluate(X[["margin"]], y, margin_isotonic_factory)
        print(f"Isotonic ROC-AUC: {isotonic_results['aggregate']['roc_auc']['mean']:.4f} "
              f"± {isotonic_results['aggregate']['roc_auc']['std']:.4f}")

        # Model C: logistic regression on full feature set
        logistic_results = evaluator.evaluate(X, y, logistic_risk_factory)
        print(f"Logistic ROC-AUC: {logistic_results['aggregate']['roc_auc']['mean']:.4f} "
              f"± {logistic_results['aggregate']['roc_auc']['std']:.4f}")

    model_results = {
        "Isotonic (margin only)": isotonic_results,
        "Logistic (all features)": logistic_results,
    }

    # 3. Pick best model for calibration + policy simulation
    best_name = max(
        model_results,
        key=lambda k: model_results[k]["aggregate"]["roc_auc"]["mean"],
    )
    best_preds = model_results[best_name]["predictions"]
    print(f"Best model: {best_name}")

    # 4. Calibration
    calibration = compute_calibration_metrics(y, best_preds, n_bins=10)
    print(f"ECE: {calibration['ece']:.4f}")

    # 5. Policy simulation
    simulator = PolicySimulator()

    two_zone_thresholds = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
    two_zone_df = simulator.simulate_two_zone(y, best_preds, two_zone_thresholds)

    zone_pairs = [(0.3, 0.7), (0.4, 0.8), (0.5, 0.9)]
    three_zone_df = simulator.simulate_three_zone(y, best_preds, zone_pairs)

    # 6. Write outputs
    RESEARCH_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    # Predictions parquet
    pred_df = dataset.df.copy()
    for name, res in model_results.items():
        col_name = name.split("(")[0].strip().lower().replace(" ", "_") + "_risk"
        pred_df[col_name] = res["predictions"]
    pred_df["y_true"] = np.asarray(y)
    pred_path = RESEARCH_DIR / "posthoc_benign_risk_predictions.parquet"
    pred_df.to_parquet(pred_path, index=False)
    print(f"Predictions -> {pred_path}")

    # Summary CSV
    summary_rows = []
    for name, res in model_results.items():
        agg = res["aggregate"]
        summary_rows.append({
            "model": name,
            "roc_auc_mean": agg["roc_auc"]["mean"],
            "roc_auc_std": agg["roc_auc"]["std"],
            "pr_auc_mean": agg["pr_auc"]["mean"],
            "pr_auc_std": agg["pr_auc"]["std"],
            "brier_mean": agg["brier"]["mean"],
            "brier_std": agg["brier"]["std"],
        })
    summary_df = pd.DataFrame(summary_rows)
    summary_path = RESEARCH_DIR / "posthoc_benign_risk_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"Summary -> {summary_path}")

    # Report
    eval_mode = "train_test" if use_train_test else "crossfit"
    report = generate_report(data_summary, model_results, calibration, two_zone_df, three_zone_df,
                             eval_mode=eval_mode)
    report_path = REPORTS_DIR / "posthoc_benign_risk_model.md"
    report_path.write_text(report)
    print(f"Report -> {report_path}")

    # Plots
    predictions_dict = {
        name: res["predictions"] for name, res in model_results.items()
    }
    generate_plots(np.asarray(y), predictions_dict, REPORTS_ARTIFACTS_DIR)
    print(f"Plots -> {REPORTS_ARTIFACTS_DIR}/benign_risk_*.png")

    # Threshold sweep table to stdout
    print("\nMargin threshold sweep:")
    print(pd.DataFrame(sweep_results["per_threshold"]).to_string(index=False))

    print("\nTwo-zone policy simulation:")
    print(two_zone_df.to_string(index=False))

    print("\nThree-zone policy simulation:")
    print(three_zone_df.to_string(index=False))


if __name__ == "__main__":
    main()

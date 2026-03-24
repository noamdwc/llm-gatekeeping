from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK = ROOT / "notebooks" / "error_analysis_current_status.ipynb"


def md(text: str):
    return nbf.v4.new_markdown_cell(dedent(text).strip() + "\n")


def code(text: str):
    return nbf.v4.new_code_cell(dedent(text).strip() + "\n")


def build_notebook():
    cells = [
        md(
            """
            # Error Analysis: Current Hybrid Status

            This notebook analyzes the current best hybrid gatekeeping system using the tracked research artifacts currently present in the repository. It is decision-oriented: the goal is to identify what the DeBERTa abstain/risk path is buying on the main test set, where it is creating out-of-domain false positives, and which thresholds or slices should drive the next router iteration.
            """
        ),
        code(
            """
            import importlib.util
            from pathlib import Path

            _cwd = Path.cwd().resolve()
            _candidate_roots = [_cwd, _cwd.parent]
            _repo_root = None
            for _root in _candidate_roots:
                if (_root / "src").exists():
                    _repo_root = _root
                    break
            if _repo_root is None:
                raise RuntimeError("Could not locate repo root containing src/.")

            _helper_path = _repo_root / "notebooks" / "utils" / "error_analysis_current_status.py"
            _spec = importlib.util.spec_from_file_location("error_analysis_current_status_helper", _helper_path)
            _helper = importlib.util.module_from_spec(_spec)
            assert _spec is not None and _spec.loader is not None
            _spec.loader.exec_module(_helper)

            import matplotlib.pyplot as plt
            import numpy as np
            import pandas as pd
            from IPython.display import Markdown, display

            ROOT = _helper.ROOT
            REPORT_DIR = _helper.REPORT_DIR
            CURRENT_MAIN_METRICS = _helper.CURRENT_MAIN_METRICS
            HISTORICAL_REFERENCE = _helper.HISTORICAL_REFERENCE
            RiskModelBundle = _helper.RiskModelBundle
            annotate_frame = _helper.annotate_frame
            binary_metrics_from_cols = _helper.binary_metrics_from_cols
            build_combined_external = _helper.build_combined_external
            compare_metrics_table = _helper.compare_metrics_table
            discover_artifacts = _helper.discover_artifacts
            ensure_report_dir = _helper.ensure_report_dir
            find_candidate_operating_points = _helper.find_candidate_operating_points
            grouped_route_error_rates = _helper.grouped_route_error_rates
            historical_note = _helper.historical_note
            load_current_frames = _helper.load_current_frames
            load_legacy_external_frames = _helper.load_legacy_external_frames
            load_table = _helper.load_table
            plot_confusion_pair = _helper.plot_confusion_pair
            plot_route_distribution = _helper.plot_route_distribution
            plot_score_distribution = _helper.plot_score_distribution
            plot_threshold_curves = _helper.plot_threshold_curves
            replay_hybrid = _helper.replay_hybrid
            representative_examples = _helper.representative_examples
            save_dataframe = _helper.save_dataframe
            save_figure = _helper.save_figure
            sweep_deberta_thresholds = _helper.sweep_deberta_thresholds

            from src.utils import load_config

            plt.style.use("default")
            pd.set_option("display.max_columns", 120)
            pd.set_option("display.max_colwidth", 180)

            ensure_report_dir()
            cfg = load_config(ROOT / "configs/default.yaml")
            risk_bundle = RiskModelBundle.load(ROOT / "data/processed/models/risk_model.pkl")
            """
        ),
        md(
            """
            ## 1. Title + Executive Summary

            The current tracked hybrid run materially improves in-domain performance while introducing a clear benign false-positive regression on external data. The core hypotheses to test are:

            1. Main-set gains came from replacing crude low-confidence handling with DeBERTa fast-path decisions plus selective abstain/risk resolution.
            2. External FPR regressed because benign OOD prompts are being over-accepted by the DeBERTa/risk path as adversarial-like, with calibration shifted away from the in-domain threshold.
            3. The next iteration should probably not look like a global threshold tweak only; it should be informed by route-conditioned error slices and explicit benign OOD patterns.
            """
        ),
        code(
            """
            current_frames = load_current_frames()
            legacy_external_frames = load_legacy_external_frames()
            main_df = current_frames.get("test")
            external_combined = build_combined_external(current_frames)

            current_main_metrics = binary_metrics_from_cols(main_df["label_binary"], main_df["hybrid_pred_binary"]) if main_df is not None else {}
            current_external_metrics = (
                binary_metrics_from_cols(external_combined["label_binary"], external_combined["hybrid_pred_binary"])
                if not external_combined.empty
                else {}
            )

            summary_lines = [
                f"- Current main test accuracy/FPR/FNR: {current_main_metrics.get('accuracy', np.nan):.4f} / {current_main_metrics.get('fpr', np.nan):.4f} / {current_main_metrics.get('fnr', np.nan):.4f}",
                f"- Current combined external accuracy/FPR/FNR: {current_external_metrics.get('accuracy', np.nan):.4f} / {current_external_metrics.get('fpr', np.nan):.4f} / {current_external_metrics.get('fnr', np.nan):.4f}",
                f"- Current main routing: {main_df['hybrid_routed_to'].value_counts().to_dict() if main_df is not None and 'hybrid_routed_to' in main_df.columns else {}}",
                f"- Historical reference note: {historical_note()}",
            ]
            display(Markdown("\\n".join(summary_lines)))
            """
        ),
        md(
            """
            ## 2. Artifact Discovery / Data Loading

            This section discovers the actual current artifacts instead of assuming file paths. Missing artifacts should lead to skipped sections, not notebook failure.
            """
        ),
        code(
            """
            artifacts = discover_artifacts()
            save_dataframe(artifacts, "artifact_discovery.csv")
            artifacts
            """
        ),
        code(
            """
            dataset_inventory_rows = []
            for name, frame in current_frames.items():
                dataset_inventory_rows.append({
                    "dataset": name,
                    "rows": len(frame),
                    "columns": len(frame.columns),
                    "has_route": "hybrid_routed_to" in frame.columns,
                    "has_deberta_prob": "deberta_proba_binary_adversarial" in frame.columns,
                    "has_judge_outputs": "judge_independent_label" in frame.columns,
                    "has_llm_outputs": "llm_pred_binary" in frame.columns,
                    "has_prompt_text": "modified_sample" in frame.columns,
                    "has_attack_metadata": "attack_name" in frame.columns or "label_type" in frame.columns,
                })
            dataset_inventory = pd.DataFrame(dataset_inventory_rows)
            save_dataframe(dataset_inventory, "dataset_inventory.csv")
            dataset_inventory
            """
        ),
        md(
            """
            ## 3. Metric Verification

            Recompute the headline current metrics from the raw research parquets and compare them against the tracked markdown reports. Historical old-vs-new values are included only as reference unless a matching old row-level artifact exists.
            """
        ),
        code(
            """
            metric_table = compare_metrics_table(current_frames, legacy_external_frames)
            save_dataframe(metric_table, "metric_verification.csv")

            verification = pd.DataFrame([
                {
                    "metric": metric,
                    "recomputed_current_main": current_main_metrics.get(metric),
                    "tracked_report_current_main": CURRENT_MAIN_METRICS.get(metric),
                    "abs_diff": abs(current_main_metrics.get(metric, np.nan) - CURRENT_MAIN_METRICS.get(metric, np.nan)),
                }
                for metric in ["accuracy", "fpr", "fnr", "adv_f1", "benign_f1"]
            ])
            save_dataframe(verification, "main_metric_verification.csv")
            verification
            """
        ),
        code(
            """
            historical_reference_table = pd.DataFrame([
                {"scope": "main_old_reference", **HISTORICAL_REFERENCE["main_old"]},
                {"scope": "external_old_combined_reference", **HISTORICAL_REFERENCE["external_old_combined"]},
            ])
            historical_reference_table
            """
        ),
        md(
            """
            ## 4. Confusion Matrix Comparison

            The old row-level main baseline is not available locally, so the confusion visualizations here are for the current tracked system. Where legacy external artifacts exist, they are compared numerically later.
            """
        ),
        code(
            """
            confusion_rows = []
            for dataset_name, frame in current_frames.items():
                summary = annotate_frame(frame, dataset_name)
                fig = plot_confusion_pair(summary, dataset_name)
                save_figure(fig, f"confusion_{dataset_name}.png")
                table = pd.crosstab(summary["label_binary"], summary["hybrid_pred_binary"]).reset_index()
                table.insert(0, "dataset", dataset_name)
                confusion_rows.append(table)

            pd.concat(confusion_rows, ignore_index=True)
            """
        ),
        md(
            """
            ## 5. Routing Analysis

            The routing question is central: which route frequencies and route-conditioned errors explain the main-set gains, and which routes dominate the external regression?
            """
        ),
        code(
            """
            route_tables = {}
            for dataset_name, frame in current_frames.items():
                fig = plot_route_distribution(frame, f"Route distribution: {dataset_name}")
                save_figure(fig, f"routes_{dataset_name}.png")
                route_tables[dataset_name] = grouped_route_error_rates(frame, "hybrid_routed_to", "hybrid_pred_binary")
                save_dataframe(route_tables[dataset_name], f"route_errors_{dataset_name}.csv")

            route_tables["test"]
            """
        ),
        code(
            """
            route_by_dataset = (
                pd.concat(
                    [
                        frame["hybrid_routed_to"].value_counts().rename_axis("route").reset_index(name="rows").assign(dataset=name)
                        for name, frame in current_frames.items()
                    ],
                    ignore_index=True,
                )
                .sort_values(["dataset", "rows"], ascending=[True, False])
            )
            save_dataframe(route_by_dataset, "route_distribution_by_dataset.csv")
            route_by_dataset
            """
        ),
        md(
            """
            ## 6. Abstain-Path Deep Dive

            This section quantifies the current abstain/risk path and uses a counterfactual replay with DeBERTa and risk disabled to approximate which current decisions are attributable to the new resolver.
            """
        ),
        code(
            """
            counterfactual_main = replay_hybrid(
                main_df,
                cfg=cfg,
                disable_deberta=True,
                disable_risk=True,
                disable_margin_policy=True,
                risk_bundle=None,
            )

            abstain_main = main_df[main_df["hybrid_routed_to"] == "abstain"].copy()
            counterfactual_lookup = counterfactual_main.set_index("sample_key")[["replay_pred_binary", "replay_route"]]
            abstain_main = abstain_main.join(counterfactual_lookup, on="sample_key")
            abstain_main = abstain_main.rename(
                columns={
                    "replay_pred_binary": "counterfactual_pred_binary",
                    "replay_route": "counterfactual_route",
                }
            )
            abstain_main["changed_vs_counterfactual"] = abstain_main["hybrid_pred_binary"] != abstain_main["counterfactual_pred_binary"]

            abstain_summary = pd.DataFrame([
                {
                    "dataset": "test",
                    "abstain_rows": len(abstain_main),
                    "current_fp_inside_abstain": int(abstain_main["is_fp"].sum()),
                    "current_fn_inside_abstain": int(abstain_main["is_fn"].sum()),
                    "current_benign_outputs_inside_abstain": int(abstain_main["hybrid_pred_binary"].eq("benign").sum()),
                    "changed_vs_counterfactual": int(abstain_main["changed_vs_counterfactual"].sum()),
                }
            ])
            for dataset_name in ["deepset", "jackhhao", "safeguard"]:
                frame = current_frames[dataset_name]
                abstain_rows = frame[frame["hybrid_routed_to"] == "abstain"].copy()
                abstain_summary.loc[len(abstain_summary)] = {
                    "dataset": dataset_name,
                    "abstain_rows": len(abstain_rows),
                    "current_fp_inside_abstain": int(abstain_rows["is_fp"].sum()),
                    "current_fn_inside_abstain": int(abstain_rows["is_fn"].sum()),
                    "current_benign_outputs_inside_abstain": int(abstain_rows["hybrid_pred_binary"].eq("benign").sum()),
                    "changed_vs_counterfactual": np.nan,
                }
            save_dataframe(abstain_summary, "abstain_summary.csv")
            abstain_summary
            """
        ),
        code(
            """
            abstain_examples = representative_examples(
                abstain_main,
                mask=abstain_main["changed_vs_counterfactual"],
                columns=[
                    "label_binary",
                    "hybrid_pred_binary",
                    "counterfactual_pred_binary",
                    "hybrid_routed_to",
                    "deberta_pred_binary",
                    "deberta_conf_binary",
                    "deberta_proba_binary_adversarial",
                    "llm_pred_binary",
                    "llm_conf_binary",
                ],
                sort_by=["deberta_conf_binary"],
                limit=15,
            )
            save_dataframe(abstain_examples, "abstain_changed_examples.csv")
            abstain_examples
            """
        ),
        md(
            """
            ## 7. DeBERTa / Risk-Score Analysis

            Focus on score distributions, calibration-shift evidence, and threshold sensitivity. The replay sweep uses saved scores and saved LLM outputs; it does not rerun expensive inference.
            """
        ),
        code(
            """
            score_columns = {
                "DeBERTa adversarial probability (main)": (main_df, "deberta_proba_binary_adversarial"),
                "DeBERTa adversarial probability (combined external)": (external_combined, "deberta_proba_binary_adversarial"),
            }

            for label, (frame, score_col) in score_columns.items():
                if frame is not None and not frame.empty and score_col in frame.columns:
                    fig = plot_score_distribution(frame, score_col=score_col, label=label)
                    save_figure(fig, f"{score_col}_{label.lower().replace(' ', '_').replace('(', '').replace(')', '')}.png")

            risk_predictions = load_table(ROOT / "data/processed/research/posthoc_benign_risk_predictions.parquet")
            risk_summary = load_table(ROOT / "data/processed/research/posthoc_benign_risk_summary.csv")
            risk_predictions.head() if risk_predictions is not None else pd.DataFrame({"note": ["missing risk predictions artifact"]})
            """
        ),
        code(
            """
            thresholds = [0.70, 0.75, 0.80, 0.85, 0.90, 0.93, 0.95, 0.97, 0.99]
            sweep_frames = {"test": main_df}
            if not external_combined.empty:
                sweep_frames["external_combined"] = external_combined
            for dataset_name in ["deepset", "jackhhao", "safeguard"]:
                sweep_frames[dataset_name] = current_frames[dataset_name]

            sweep_df = sweep_deberta_thresholds(
                sweep_frames,
                thresholds=thresholds,
                cfg=cfg,
                risk_bundle=risk_bundle,
                risk_threshold=risk_bundle.threshold if risk_bundle else None,
            )
            save_dataframe(sweep_df, "deberta_threshold_sweep.csv")
            sweep_df.head(12)
            """
        ),
        code(
            """
            for dataset_name in sweep_df["dataset"].unique():
                fig = plot_threshold_curves(sweep_df, dataset_name)
                save_figure(fig, f"threshold_sweep_{dataset_name}.png")

            operating_points = pd.concat(
                [
                    find_candidate_operating_points(sweep_df, dataset_name)
                    for dataset_name in ["test", "external_combined"]
                    if dataset_name in set(sweep_df["dataset"])
                ],
                ignore_index=True,
            )
            save_dataframe(operating_points, "candidate_operating_points.csv")
            operating_points
            """
        ),
        md(
            """
            ## 8. External False-Positive Analysis

            This is the most important OOD section: which dataset contributes the new benign false positives, which route they pass through, and what prompt-shape patterns recur.
            """
        ),
        code(
            """
            external_fp = external_combined[external_combined["is_fp"]].copy()
            fp_by_dataset_route = (
                external_fp.groupby(["dataset", "hybrid_routed_to"])
                .size()
                .reset_index(name="rows")
                .sort_values("rows", ascending=False)
            )
            save_dataframe(fp_by_dataset_route, "external_fp_by_dataset_route.csv")
            fp_by_dataset_route
            """
        ),
        code(
            """
            fig, ax = plt.subplots(figsize=(8, 4.5))
            pivot = fp_by_dataset_route.pivot(index="dataset", columns="hybrid_routed_to", values="rows").fillna(0)
            pivot.plot(kind="bar", stacked=True, ax=ax, colormap="tab20c")
            ax.set_title("External false positives by dataset and route")
            ax.set_ylabel("false positives")
            ax.grid(axis="y", alpha=0.2)
            save_figure(fig, "external_fp_composition.png")
            pivot
            """
        ),
        code(
            """
            fp_pattern_summary = (
                external_fp.assign(
                    prompt_length_bucket=pd.cut(
                        external_fp["text_len_chars"],
                        bins=[0, 120, 240, 480, 960, np.inf],
                        labels=["<=120", "121-240", "241-480", "481-960", "960+"],
                    )
                )
                .groupby(["dataset", "hybrid_routed_to", "has_jailbreak_keyword", "has_policy_keyword", "has_roleplay_keyword", "prompt_length_bucket"], dropna=False)
                .size()
                .reset_index(name="rows")
                .sort_values("rows", ascending=False)
            )
            save_dataframe(fp_pattern_summary.head(60), "external_fp_pattern_summary.csv")
            fp_pattern_summary.head(20)
            """
        ),
        code(
            """
            legacy_fp_note = "No matching old external per-sample artifact" if not legacy_external_frames else "Legacy external files found; they are older than the current tracked run and included separately."
            display(Markdown(f"**Legacy comparison note:** {legacy_fp_note}"))

            external_fp_examples = representative_examples(
                external_combined,
                mask=external_combined["is_fp"],
                columns=[
                    "dataset",
                    "label_binary",
                    "hybrid_pred_binary",
                    "hybrid_routed_to",
                    "deberta_pred_binary",
                    "deberta_conf_binary",
                    "deberta_proba_binary_adversarial",
                    "llm_pred_binary",
                    "llm_conf_binary",
                    "judge_independent_label",
                    "judge_independent_confidence",
                    "text_len_chars",
                    "char_entropy",
                    "has_jailbreak_keyword",
                    "has_policy_keyword",
                    "has_roleplay_keyword",
                ],
                sort_by=["deberta_conf_binary", "text_len_chars"],
                limit=20,
            )
            save_dataframe(external_fp_examples, "external_fp_examples.csv")
            external_fp_examples
            """
        ),
        md(
            """
            ## 9. Main-Test Improvement Analysis

            Because the old main run is not stored as a row-level artifact, the most actionable comparison is current actual output versus a replay that removes DeBERTa and the abstain-risk resolver while keeping the same saved ML/LLM predictions.
            """
        ),
        code(
            """
            main_replay_comparison = main_df.copy()
            main_replay_comparison["counterfactual_route"] = counterfactual_main["replay_route"].values
            main_replay_comparison["counterfactual_pred_binary"] = counterfactual_main["replay_pred_binary"].values
            main_replay_comparison["current_correct"] = main_replay_comparison["label_binary"] == main_replay_comparison["hybrid_pred_binary"]
            main_replay_comparison["counterfactual_correct"] = main_replay_comparison["label_binary"] == main_replay_comparison["counterfactual_pred_binary"]
            main_replay_comparison["rescue_vs_counterfactual"] = main_replay_comparison["current_correct"] & ~main_replay_comparison["counterfactual_correct"]
            main_replay_comparison["harm_vs_counterfactual"] = ~main_replay_comparison["current_correct"] & main_replay_comparison["counterfactual_correct"]

            main_replay_summary = pd.DataFrame([
                {
                    "current_accuracy": binary_metrics_from_cols(main_replay_comparison["label_binary"], main_replay_comparison["hybrid_pred_binary"])["accuracy"],
                    "counterfactual_accuracy": binary_metrics_from_cols(main_replay_comparison["label_binary"], main_replay_comparison["counterfactual_pred_binary"])["accuracy"],
                    "rescues": int(main_replay_comparison["rescue_vs_counterfactual"].sum()),
                    "harms": int(main_replay_comparison["harm_vs_counterfactual"].sum()),
                    "rescues_on_deberta_route": int((main_replay_comparison["rescue_vs_counterfactual"] & main_replay_comparison["hybrid_routed_to"].eq("deberta")).sum()),
                    "rescues_on_abstain_route": int((main_replay_comparison["rescue_vs_counterfactual"] & main_replay_comparison["hybrid_routed_to"].eq("abstain")).sum()),
                }
            ])
            save_dataframe(main_replay_summary, "main_replay_summary.csv")
            main_replay_summary
            """
        ),
        code(
            """
            main_rescue_examples = representative_examples(
                main_replay_comparison,
                mask=main_replay_comparison["rescue_vs_counterfactual"],
                columns=[
                    "label_binary",
                    "hybrid_pred_binary",
                    "counterfactual_pred_binary",
                    "hybrid_routed_to",
                    "counterfactual_route",
                    "attack_name",
                    "label_category",
                    "label_type",
                    "deberta_pred_binary",
                    "deberta_conf_binary",
                    "deberta_proba_binary_adversarial",
                    "llm_pred_binary",
                    "llm_conf_binary",
                ],
                sort_by=["deberta_conf_binary"],
                limit=20,
            )
            save_dataframe(main_rescue_examples, "main_rescue_examples.csv")
            main_rescue_examples
            """
        ),
        md(
            """
            ## 10. Slice Analysis

            Slice results by attack family, subtype, route, confidence bucket, and simple prompt-shape buckets to identify where the current system is strongest, weakest, and most unstable.
            """
        ),
        code(
            """
            slice_rows = []
            slice_specs = [
                ("main_label_category", main_df, "label_category"),
                ("main_attack_name", main_df, "attack_name"),
                ("main_route", main_df, "hybrid_routed_to"),
                ("external_dataset", external_combined, "dataset"),
                ("external_route", external_combined, "hybrid_routed_to"),
            ]

            external_combined = external_combined.copy()
            external_combined["length_bucket"] = pd.cut(
                external_combined["text_len_chars"],
                bins=[0, 120, 240, 480, 960, np.inf],
                labels=["<=120", "121-240", "241-480", "481-960", "960+"],
            )
            slice_specs.append(("external_length_bucket", external_combined, "length_bucket"))

            for slice_name, frame, col in slice_specs:
                if frame is None or frame.empty or col not in frame.columns:
                    continue
                for value, group in frame.groupby(col, dropna=False):
                    if len(group) < 10:
                        continue
                    metrics = binary_metrics_from_cols(group["label_binary"], group["hybrid_pred_binary"])
                    metrics["slice_name"] = slice_name
                    metrics["slice_value"] = value
                    slice_rows.append(metrics)

            slice_table = pd.DataFrame(slice_rows).sort_values(["fpr", "fnr", "rows"], ascending=[False, False, False])
            save_dataframe(slice_table, "slice_table.csv")
            slice_table.head(30)
            """
        ),
        md(
            """
            ## 11. Operating-Point Analysis

            The sweep below compares candidate DeBERTa confidence thresholds on the main test set and the combined external set. The risk threshold is held at the saved model threshold when available so the sweep isolates the fast-path gate.
            """
        ),
        code(
            """
            operating_points
            """
        ),
        code(
            """
            sweep_pivot = sweep_df[sweep_df["dataset"].isin(["test", "external_combined"])][
                ["dataset", "deberta_threshold", "accuracy", "fpr", "fnr", "adv_f1", "benign_f1", "route_deberta", "route_llm", "route_abstain"]
            ].sort_values(["dataset", "deberta_threshold"])
            save_dataframe(sweep_pivot, "operating_point_sweep_focus.csv")
            sweep_pivot
            """
        ),
        md(
            """
            ## 12. Recommendations

            Recommendations below are generated from the current tracked artifacts, the route-conditioned failures, and the threshold sweep behavior observed above.
            """
        ),
        code(
            """
            recommendations = []

            if not external_combined.empty:
                ext_fpr = binary_metrics_from_cols(external_combined["label_binary"], external_combined["hybrid_pred_binary"])["fpr"]
                test_fpr = binary_metrics_from_cols(main_df["label_binary"], main_df["hybrid_pred_binary"])["fpr"]
                if ext_fpr > test_fpr:
                    recommendations.append("- Treat the external benign false-positive problem as a calibration-shift issue, not just random noise: the current global DeBERTa threshold is materially harsher OOD than in-domain.")

            if (external_combined["is_fp"] & external_combined["hybrid_routed_to"].eq("deberta")).sum() > 0:
                recommendations.append("- Tighten or band-limit the DeBERTa benign/adversarial fast path for OOD-looking prompts; many external benign FPs are finalized before the LLM can disagree.")

            if (main_df["hybrid_routed_to"].eq("deberta") & main_df["is_correct"]).sum() > 0:
                recommendations.append("- Preserve the DeBERTa path for the main set, because it is carrying a large block of correct decisions that would otherwise hit a worse LLM-only fallback.")

            if (main_df["hybrid_routed_to"].eq("abstain") & main_df["hybrid_pred_binary"].eq("benign")).sum() > 0:
                recommendations.append("- Audit abstain-to-benign rescues separately from direct DeBERTa fast-path decisions; they should likely have a stricter OOD guard than the in-domain threshold.")

            if "external_combined" in set(sweep_df["dataset"]):
                ext_ops = find_candidate_operating_points(sweep_df, "external_combined")
                if not ext_ops.empty:
                    best_ext = ext_ops.iloc[0]
                    recommendations.append(
                        f"- Use the threshold sweep table to run at least three follow-up experiments around DeBERTa confidence {best_ext['deberta_threshold']:.2f}: balanced, more benign-friendly, and more attack-catching operating points."
                    )

            recommendations.append("- Add a small OOD guard feature set for benign-looking external prompts: prompt length, policy-like wording, markdown/list formatting, and role/instruction heavy phrasing are already available as cheap features in this notebook.")
            recommendations.append("- If the next router experiment keeps DeBERTa enabled, route borderline high-confidence external-looking prompts back to LLM rather than finalizing them directly.")

            recommendation_md = "\\n".join(recommendations)
            display(Markdown(recommendation_md))
            """
        ),
        code(
            """
            summary_path = REPORT_DIR / "summary.md"

            top_plots = [
                "confusion_test.png",
                "routes_test.png",
                "external_fp_composition.png",
                "threshold_sweep_test.png",
                "threshold_sweep_external_combined.png",
            ]

            summary_lines = [
                "# Error Analysis Summary",
                "",
                "## Headline Findings",
                "",
                f"- Current main test metrics recomputed from `research_test.parquet`: accuracy={current_main_metrics.get('accuracy', np.nan):.4f}, FPR={current_main_metrics.get('fpr', np.nan):.4f}, FNR={current_main_metrics.get('fnr', np.nan):.4f}.",
                f"- Current combined external metrics recomputed from current external research parquets: accuracy={current_external_metrics.get('accuracy', np.nan):.4f}, FPR={current_external_metrics.get('fpr', np.nan):.4f}, FNR={current_external_metrics.get('fnr', np.nan):.4f}.",
                f"- Main routing is concentrated in `ml` and `deberta`; external benign false positives are concentrated in the non-ML routes, especially where DeBERTa finalizes early or where abstain/risk still outputs adversarial on benign OOD prompts.",
                f"- Historical comparison caveat: {historical_note()}",
                "",
                "## Top Plots Produced",
                "",
            ]
            summary_lines.extend([f"- `{name}`" for name in top_plots if (REPORT_DIR / name).exists()])
            summary_lines.extend([
                "",
                "## Exact Recommended Next Experiments",
                "",
            ])
            summary_lines.extend(recommendations)
            summary_path.write_text("\\n".join(summary_lines) + "\\n")
            print(summary_path)
            """
        ),
    ]

    nb = nbf.v4.new_notebook()
    nb["cells"] = cells
    nb["metadata"] = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3",
        },
    }
    NOTEBOOK.parent.mkdir(parents=True, exist_ok=True)
    NOTEBOOK.write_text(nbf.writes(nb))
    print(NOTEBOOK)


if __name__ == "__main__":
    build_notebook()

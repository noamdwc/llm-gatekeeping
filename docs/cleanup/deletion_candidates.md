# Deletion Candidate List

This document is an approval gate. No file, DVC stage, report, notebook, or artifact listed here has been deleted. Deletion must wait for explicit user approval of the specific items to remove.

## Classification Rules

- Canonical runtime: required by the fresh DVC + Colab handoff + final verdict path.
- Reusable helper: not directly canonical, but used by kept runtime modules, tests, or notebooks.
- Historical docs: preserved process or research history.
- Stale generated artifact: generated output that is obsolete, duplicated, or no longer part of the canonical output contract.
- Removable legacy path: runtime code, notebook, script, DVC stage, report path, or test that supports only non-main behavior.

## Canonical Runtime

Keep these DVC stages:

- `preprocess`
- `build_splits`
- `ml_model`
- `deberta_model`
- `deberta_external@deepset`
- `deberta_external@jackhhao`
- `validate_colab_handoff`
- `train_escalating_model`
- `judge_colab_local_predictions@test`
- `judge_colab_local_predictions@unseen_test`
- `judge_colab_local_predictions@safeguard_test`
- `judge_colab_local_predictions_external@deepset`
- `judge_colab_local_predictions_external@jackhhao`
- `final_verdict_report`

Keep these runtime files:

- `configs/default.yaml`
- `dvc.yaml`
- `dvc.lock`
- `requirements.txt`
- `pytest.ini`
- `README.md`
- `notebooks/colab_local_llm_classifier.ipynb`
- `src/preprocess.py`
- `src/build_splits.py`
- `src/ml_classifier/`
- `src/models/`
- `src/cli/deberta_classifier.py`
- `src/cli/eval_deberta_external.py`
- `src/cli/colab_handoff_schema.py`
- `src/cli/validate_colab_handoff.py`
- `src/cli/train_escalating_model.py`
- `src/cli/judge_colab_local_predictions.py`
- `src/cli/final_verdict_report.py`
- `src/escalating_model.py`
- `src/evaluate.py`
- `src/external_datasets.py`
- `src/utils.py`
- `src/llm_provider.py`
- `src/llm_cache.py`
- `src/llm_classifier/`
- `src/synthetic_benign.py`
- `src/validators.py`
- `tests/test_build_splits.py`
- `tests/test_preprocess.py`
- `tests/test_ml_baseline.py`
- `tests/test_deberta_classifier.py`
- `tests/test_deberta_cli_colab.py`
- `tests/test_eval_deberta_external.py`
- `tests/test_colab_local_llm_classifier_notebook.py`
- `tests/test_validate_colab_handoff.py`
- `tests/test_escalating_model.py`
- `tests/test_judge_colab_local_predictions.py`
- `tests/test_final_verdict_report.py`
- `tests/test_llm_classifier.py`
- `tests/test_llm_cache.py`
- `tests/test_rate_limiter.py`
- `tests/test_synthetic_benign.py`
- `tests/test_validators.py`
- `tests/test_evaluate.py`
- `tests/test_utils.py`

Keep these canonical or currently required artifacts:

- `artifacts/deberta_classifier/`
- `data/processed/full_dataset.parquet`
- `data/processed/splits/train.parquet`
- `data/processed/splits/val.parquet`
- `data/processed/splits/test.parquet`
- `data/processed/splits/unseen_val.parquet`
- `data/processed/splits/unseen_test.parquet`
- `data/processed/splits/safeguard_test.parquet`
- `data/processed/models/ml_baseline.pkl`
- `data/processed/models/escalating_model.pkl`
- `data/processed/predictions/deberta_predictions_*.parquet`
- `data/processed/predictions/ml_predictions_*.parquet`
- `data/processed/predictions/llm_predictions_*_colab_local_classifier.parquet`
- `data/processed/predictions/llm_predictions_*_colab_local_judged.parquet`
- `data/processed/predictions_external/deberta_predictions_external_deepset.parquet`
- `data/processed/predictions_external/deberta_predictions_external_jackhhao.parquet`
- `data/processed/predictions_external/llm_predictions_external_deepset_colab_local_classifier.parquet`
- `data/processed/predictions_external/llm_predictions_external_jackhhao_colab_local_classifier.parquet`
- `data/processed/predictions_external/llm_predictions_external_deepset_colab_local_judged.parquet`
- `data/processed/predictions_external/llm_predictions_external_jackhhao_colab_local_judged.parquet`
- `data/processed/research/escalating_model_eval_*.parquet`
- `data/processed/research/escalating_model_summary.csv`
- `data/processed/research/escalating_model_threshold_sweep_unseen_val.csv`
- `data/processed/research/escalating_model_unseen_val_postscore_split_map.csv`
- `reports/colab_handoff_validation.json` once `validate_colab_handoff` passes
- `reports/escalating_model_poc.md`
- `reports/pipeline_final_verdict_report.md`

## Reusable Helper

These are not all directly part of the final DVC path, but they are still used by kept code, tests, notebooks, or operational diagnostics:

- `src/embeddings.py`
- `src/logprob_margin.py`
- `src/margin_trace.py` - deleted in approved Batch 3C; `src/logprob_margin.py` remains canonical feature logic.
- `src/predict.py` - deleted after approval with the public arbitrary-text prediction CLI.
- `src/cli/predict.py` - deleted after approval; canonical workflows remain in DVC.
- `src/cli/generate_synthetic_benign.py`
- `src/cli/rebuild_llm_from_cache.py` - deleted after approval; current `.cache/llm` contents and `src/llm_cache.py` remain preserved.
- `notebooks/utils/`
- `tests/conftest.py`
- `tests/test_rebuild_llm_from_cache.py` - deleted with `src/cli/rebuild_llm_from_cache.py`.
- `tests/test_margin_trace.py` - deleted with `src/margin_trace.py` in approved Batch 3C.
- `tests/test_logprob_margin.py`
- `tests/test_embeddings.py`
- `tests/test_colab_local_llm_classifier_notebook.py`

## Historical Docs

Preserve these unless the user explicitly asks for historical cleanup:

- `docs/superpowers/`
- `openspec/changes/`
- `PRD.md`
- `STATUS.md`
- `project_plan.md`
- `research_docs/`
- `docs/2026-03-13_baseline_dataset_overlap.md`
- `docs/deberta_debug.md`
- `docs/escalating_model_logprob_features.md`
- `docs/escalating_model_threshold_sweep.md`
- `docs/escalating_model_lightgbm_comparison.md`
- `docs/merge_readiness_findings.md` if the user decides to add/commit it later
- `reports/external_attack_novelty_analysis.md`
- `reports/rebuild_llm_from_cache_audit.json` - legacy audit from deleted cache-rebuild path; remove only if present and separately approved as generated report cleanup.
- `interview_presentation/`
- `interview_prep.md`
- `interview_prep_he.md`

## Stale Generated Artifact Candidates

These appear generated, stale, duplicated, or outside the canonical final-verdict contract. Approve individually before deletion:

- `data/processed/ml_baseline.pkl` - duplicate legacy location; canonical model path is `data/processed/models/ml_baseline.pkl`.
- `data/processed/test.parquet`, `data/processed/train.parquet`, `data/processed/val.parquet`, `data/processed/test_unseen.parquet` - legacy split locations; canonical splits live under `data/processed/splits/`.
- `data/processed/splits/test_unseen.parquet` - legacy split name; canonical held-out splits are `unseen_val.parquet` and `unseen_test.parquet`.
- `data/processed/predictions/deberta_predictions_test_unseen.parquet` - legacy split-name artifact.
- `data/processed/predictions/ml_predictions_test_unseen.parquet` - legacy split-name artifact.
- `data/processed/predictions/predictions_test.csv` and `data/processed/predictions_test.csv` - older CSV prediction outputs.
- `data/processed/predictions/exemplar_bank.pkl` - hosted/dynamic few-shot cache, not consumed by the canonical Colab handoff path.
- `data/processed/predictions/llm_predictions_train_colab_local_classifier.parquet` - train split classifier artifact is not consumed by canonical escalation training, which trains on `val`.
- `data/processed/predictions_external/llm_predictions_external_deepset.parquet` - hosted/legacy external LLM classifier output; canonical manual handoff path uses `_colab_local_classifier`.
- `data/processed/predictions_external/llm_predictions_external_jackhhao.parquet` - hosted/legacy external LLM classifier output; canonical manual handoff path uses `_colab_local_classifier`.
- `data/processed/predictions_external/llm_predictions_external_safeguard.parquet` - external safeguard is not configured in the canonical final-verdict path.
- `data/processed/predictions_external/llm_predictions_deepset_deberta_failures_no_logprobs.parquet` - ad-hoc failure subset artifact.
- `data/processed/predictions_external/deepset_deberta_failure_subset.parquet` - ad-hoc failure subset artifact.
- `data/processed/predictions_external_deepset.parquet`, `data/processed/predictions_external_jackhhao.parquet`, `data/processed/predictions_external_safeguard.parquet`, `data/processed/predictions_external_spml.parquet` - legacy external prediction locations.
- `data/processed/research/research_test.parquet`, `data/processed/research/research_val.parquet`, `data/processed/research/research_safeguard_test.parquet` - legacy hybrid research outputs.
- `data/processed/research/hybrid_margin_trace_test.parquet`, `data/processed/research/hybrid_margin_trace_val.parquet`, and CSV duplicates - legacy risk/research path outputs.
- `data/processed/research/posthoc_benign_risk_predictions.parquet`, `data/processed/research/posthoc_benign_risk_summary.csv` - legacy post-hoc risk report outputs.
- `data/processed/research/judge_on_abstain.parquet`, `data/processed/research/judge_on_abstain_val.parquet` - ad-hoc abstain analysis.
- `data/processed/research/escalating_model_lightgbm_compare/` and `data/processed/research/escalating_model_lightgbm_compare_summary.csv` - comparison experiment artifacts.
- `data/processed/research_external/research_external_*.parquet` - legacy external research outputs.
- `data/processed/baselines/` - generated HuggingFace baseline predictions, not canonical.
- `reports/research/` - legacy component reports from `eval_new`.
- `reports/research_external/research_external_*.md`, `reports/research_external/eval_external_*.md`, `reports/research_external/deepset_style_gap_analysis.md`, `reports/research_external/eval_deberta_external_summary.json` - legacy/ad-hoc external reports, except keep `eval_deberta_external_{deepset,jackhhao}.md` if desired as DeBERTa-stage diagnostics.
- `reports/posthoc_benign_risk_model.md` - legacy risk-model report.
- `reports/artifacts/benign_risk_*.png`, `reports/artifacts/reliability_*.csv`, `reports/artifacts/risk_table_test.csv`, `reports/artifacts/sweep_*.csv`, `reports/artifacts/margin_*.png`, `reports/artifacts/margin_*.csv` - legacy risk/research artifacts.
- `reports/error_analysis_current_status/` - generated notebook analysis outputs.
- `reports/escalating_model_lightgbm_compare.md` - old comparison report superseded by current escalation model.
- `reports/inference_pipeline_diagram.svg` - older diagram for non-canonical inference path.
- `dag.dot`, `dag.png`, `dag.svg` - stale DVC graph exports.
- `dvc2.lock` - old lock file.
- `.coverage`, `.pytest_cache/`, `.cache/`, `wandb/` - local/generated state; remove from git if tracked, otherwise ignore or clean locally.

## Removable Legacy Path Candidates

These support non-main behavior and are candidates for deletion after approval:

### DVC stages

- `llm_classifier` - hosted classifier output is no longer canonical; Colab classifier handoff owns classifier artifacts.
- `llm_classifier_val` - hosted classifier output is no longer canonical.
- `research` - legacy hybrid research report path.
- `research_val` - legacy hybrid research/risk-model path.
- `research_safeguard_test` - legacy hybrid research path.
- `train_risk_model` - legacy abstain-risk-model path, not canonical final verdict.
- `risk_model` - legacy post-hoc risk evaluation path.
- `research_external_llm@deepset` and `research_external_llm@jackhhao` - hosted external LLM outputs superseded by Colab handoff.
- `research_external@deepset` and `research_external@jackhhao` - legacy external research outputs.
- `eval_new` - legacy component markdown reports.
- `eval_new_external@deepset` and `eval_new_external@jackhhao` - legacy external markdown reports.

### Scripts

- `run_inference.sh` - deleted in approved Batch 3E with the lightweight inference path.
- `run_synth.sh` - wrapper for synthetic generation, redundant with DVC/CLI.
- `run_vllm_cpu_docker.sh` - old local vLLM helper, superseded by Colab Transformers classifier.
- `scripts/analyze_external_attack_types.py` - ad-hoc analysis script.
- `scripts/run_judge_on_abstain.py` - ad-hoc legacy abstain analysis.

### Source and CLI modules

- `src/benign_risk_model.py` - deleted after approval with the legacy risk-model path.
- `src/research.py` - deleted in approved Batch 3B after routing diagnostics moved to `src/routing_diagnostics.py`.
- `src/eval_external.py` - deleted after moving `load_external_dataset` to `src/external_datasets.py`.
- `src/hybrid_router.py` - deleted after approval; confirmed not a DVC command or DVC dependency after public predict CLI and standalone external eval deletion.
- `src/infer_split.py` - deleted in approved Batch 3E with `src/cli/infer_split.py`.
- `src/cli/train_risk_model.py` - deleted with `src/benign_risk_model.py`.
- `src/cli/benign_risk_model.py` - deleted with `src/benign_risk_model.py`.
- `src/cli/research_external.py`
- `src/cli/eval_new.py`
- `src/cli/eval_baselines.py`
- `src/cli/run_baseline.py`
- `src/cli/infer_split.py` - deleted in approved Batch 3E; outside DVC and superseded by the canonical DVC flow.
- `src/cli/margin_calibration_fit.py` - deleted in approved Batch 3C.
- `src/cli/margin_calibration_report.py` - deleted in approved Batch 3C.
- `src/cli/margin_crossfit_eval.py` - deleted in approved Batch 3C.
- `src/cli/score_escalation.py` - deleted in approved Batch 3E with `src/cli/infer_split.py`.
- `src/baselines/`

### Notebooks

- `notebooks/colab_train_deberta.ipynb` - training notebook outside canonical local DVC path.
- `notebooks/roberta_finetune.ipynb`
- `notebooks/error_analysis.ipynb`
- `notebooks/error_analysis_current_status.ipynb`
- `notebooks/error_analysis_current_status.executed.ipynb`
- `notebooks/external_datasers_ea.ipynb`
- `notebooks/logprob_fp_analysis.ipynb`
- `notebooks/logprob_threshold_analysis.ipynb`
- `notebooks/ea_notes.md`
- `notebooks/utils/error_analysis_current_status.py`
- `notebooks/utils/generate_error_analysis_current_status_notebook.py`

### Tests

Remove tests only together with their runtime modules:

- `tests/test_benign_risk_model.py` - deleted with `src/benign_risk_model.py`.
- `tests/test_research.py` - deleted with `src/research.py` in approved Batch 3B.
- `tests/test_eval_external.py` - deleted with `src/eval_external.py`; loader coverage moved to `tests/test_external_datasets.py`.
- `tests/test_research_external.py`
- `tests/test_hybrid_router.py` - deleted with `src/hybrid_router.py`.
- `tests/test_baselines.py`
- `tests/test_cli_infer_split.py` - deleted in approved Batch 3E.
- `tests/test_score_escalation.py` - deleted in approved Batch 3E.
- `tests/test_margin_trace.py` - deleted in approved Batch 3C.
- `tests/test_logprob_margin.py`
- `tests/test_run_vllm_cpu_docker_script.py`

### Docs and report references

- `src/cli/README.md` - currently documents non-canonical CLI tools; either delete or rewrite after runtime deletion approval.
- `research_docs/pipeline_diagram_description.md` - contains `test_unseen.parquet` legacy diagram.
- `research_docs/pipeline_breakdown.md` - contains `test_unseen.parquet` and old pipeline flow.
- Legacy report references in `README.md` project-structure section can be removed after approved runtime deletion.

## Current Validation Status

The canonical pipeline is being rerun from the start. Treat the previous
downloaded Deepset Colab handoff validation failure as stale until fresh Colab
classifier artifacts are produced and `dvc repro -s validate_colab_handoff`
is run again.

If the fresh artifacts fail validation, handle that as a new manual handoff
artifact quality issue. Do not weaken `validate_colab_handoff` or re-enable
legacy hosted LLM outputs as a cleanup shortcut.

## Explicitly Not Deleting Before Approval

No deletion has been performed. The next implementation step must wait for explicit approval of the specific deletion candidates to remove.

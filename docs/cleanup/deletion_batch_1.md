# Deletion Batch 1

This batch implements the user-approved conservative deletion scope from May 13, 2026.

## Scope

Delete only:

- legacy DVC stages outside the canonical final-verdict graph;
- stale generated artifacts with legacy split names;
- duplicate legacy model/prediction locations;
- hosted/legacy LLM classifier outputs replaced by `_colab_local_classifier`;
- stale report artifacts not consumed by `final_verdict_report`;
- stale graph exports and old lock files.

Do not delete source modules, notebooks, reusable helpers, `research_docs/`, or the invalid Deepset Colab handoff artifact. The current `validate_colab_handoff` failure is a handoff artifact quality issue, not cleanup.

## DVC Stages Removed

- `llm_classifier`
- `llm_classifier_val`
- `research`
- `research_val`
- `research_safeguard_test`
- `train_risk_model`
- `risk_model`
- `research_external_llm@*`
- `research_external@*`
- `eval_new`
- `eval_new_external@*`

## Tracked Files Removed

- `dag.dot`
- `dag.png`
- `dag.svg`
- `dvc2.lock`
- `reports/artifacts/*`
- `reports/error_analysis_current_status/*`
- `reports/escalating_model_lightgbm_compare.md`
- `reports/inference_pipeline_diagram.svg`
- `reports/posthoc_benign_risk_model.md`
- `reports/research/*`
- legacy/ad-hoc files under `reports/research_external/`, while keeping DeBERTa external diagnostics:
  - keep `reports/research_external/eval_deberta_external_deepset.md`
  - keep `reports/research_external/eval_deberta_external_jackhhao.md`

## Ignored/Generated Workspace Files Removed

- legacy root-level processed split/model files under `data/processed/`
- `data/processed/splits/test_unseen.parquet`
- `data/processed/predictions/*test_unseen*`
- legacy hosted external LLM predictions without `_colab_local_classifier`
- legacy external prediction files at old root-level locations
- legacy risk/research outputs not consumed by canonical escalation/final-verdict stages
- generated baseline prediction directory

## Verification Required After Deletion

- `dvc stage list`
- `dvc dag final_verdict_report`
- `dvc status final_verdict_report`
- canonical tests:
  - `tests/test_validate_colab_handoff.py`
  - `tests/test_escalating_model.py`
  - `tests/test_judge_colab_local_predictions.py`
  - `tests/test_final_verdict_report.py`
  - `tests/test_eval_deberta_external.py`
- stale-reference search for deleted paths and stages

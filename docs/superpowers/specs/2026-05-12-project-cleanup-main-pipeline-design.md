# Project Cleanup Main Pipeline Design

## Goal

Clean the project so there is one clearly defined, easy-to-run main path from start to finish. The canonical path is a fresh DVC-driven pipeline with one explicit manual handoff: the local LLM classifier runs in Colab, its parquet outputs are downloaded into the repo, and local DVC resumes through escalation, judging, and the final verdict report.

The cleanup should be aggressive about removing non-main runtime paths, but historical process documents under `docs/superpowers/` and `openspec/changes/` are preserved. Before any file, stage, notebook, script, or test is deleted, the implementation must present a categorized deletion list and wait for user approval.

## Canonical Pipeline Boundary

The canonical end-to-end path is:

1. Run local DVC stages that prepare data, build splits, train or score the ML and DeBERTa components, and produce the inputs required by the Colab classifier.
2. Run `notebooks/colab_local_llm_classifier.ipynb` in Colab across the required main splits and configured external datasets.
3. Download the Colab classifier outputs into their expected local paths:
   - `data/processed/predictions/llm_predictions_<split>_colab_local_classifier.parquet`
   - `data/processed/predictions_external/llm_predictions_external_<dataset>_colab_local_classifier.parquet`
4. Resume local DVC with escalating-model training, selective judge calls, and final verdict reporting.
5. Treat `reports/pipeline_final_verdict_report.md` as the final success artifact.

Everything outside this boundary is non-main unless it is required to build, validate, or operate this path.

## Runtime Surface To Keep

The kept runtime surface should be small and aligned to the canonical path:

- `configs/default.yaml`, reduced to parameters used by the main pipeline.
- `dvc.yaml` as the main pipeline contract.
- `notebooks/colab_local_llm_classifier.ipynb` as the only kept operational Colab notebook for the local LLM classifier handoff.
- Source modules required for preprocessing, split building, ML, DeBERTa, Colab artifact validation, escalating model training/scoring, judge calls, final report generation, shared utilities, and evaluation helpers used by the final report.
- Tests for kept modules and tests that verify the DVC graph and Colab artifact contract.
- `README.md` as the single start-to-finish run guide.

Historical design/planning records under `docs/superpowers/` and `openspec/changes/` stay even if they reference old work.

Likely non-main candidates include old inference shell paths, legacy research/eval report stages, baseline comparison paths, stale notebooks, duplicated data/report paths, obsolete split names such as `test_unseen`, and stale artifact references. These are candidates only until the deletion approval gate.

## Colab Handoff Contract

The manual Colab step must be explicit and locally checkable. If classifier artifacts are missing, the local pipeline should fail with a direct message listing the exact missing files and the notebook to run.

Local validation should check that every required classifier parquet:

- exists for `val`, `test`, `unseen_val`, `unseen_test`, `safeguard_test`, and each configured external dataset;
- includes `sample_id` and the classifier prediction, confidence, stage, model metadata, and logprob fields consumed by escalation;
- has `llm_stages_run == 1`;
- does not include judge output columns;
- joins with the corresponding DeBERTa predictions without empty joins or unexpected row loss.

The pipeline must not silently fall back to legacy hosted LLM classifier outputs or old research report outputs when Colab artifacts are missing.

## DVC And Reports

`dvc.yaml` should describe the canonical path without competing runtime routes. The graph should make the Colab handoff visible through dependencies on the downloaded classifier artifacts.

`dvc status` currently works but is noisy because many stale and non-main paths are still present. Cleanup success means `dvc status` becomes interpretable and aligned with the canonical path, not that every API-heavy or Colab-dependent output is always locally regenerated.

The canonical report is `reports/pipeline_final_verdict_report.md`. Legacy component reports may be removed or de-documented if they are not required by the main path.

## Deletion Workflow

Implementation must use this deletion process:

1. Produce a categorized deletion candidate list covering DVC stages, source/CLI modules, scripts, notebooks, tests, docs/report references, generated artifact paths, and stale tracked outputs.
2. Stop and wait for user approval.
3. Delete only approved items.
4. After deletion, run stale-reference searches before deeper verification.

No destructive cleanup should occur before the approval gate.

## Verification

Verification should run in layers:

1. Static searches with `rg` for stale split names, removed command names, old report names, and deleted path references.
2. Unit tests for kept canonical modules.
3. DVC graph/list/status checks.
4. Partial DVC repro up to the Colab handoff.
5. After Colab artifacts are available, DVC repro through judge stages and `final_verdict_report`.

The final implementation should leave the project with one documented command sequence for a fresh run and clear failure messages when the Colab handoff has not yet been completed.

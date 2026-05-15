# Deletion Batch 3 Proposal: Remaining Non-Canonical Source Scope Decisions

Date: 2026-05-13

Status: proposal only. Do not delete anything in this batch until explicitly
approved.

## Scope

This gate covers remaining non-canonical source paths that require explicit
scope decisions. It is still pipeline cleanup only:

- no retraining
- no threshold tuning
- no metric/model-behavior changes
- no evaluation-logic changes
- no weakening or bypassing Colab handoff validation
- no artifact regeneration

The current Deepset Colab handoff validation failure remains a real artifact
blocker and is intentionally not addressed here.

## Checks Run

```bash
rg -n "benign_risk_model|train_risk_model|risk_model|research\.py|src\.research|from src import research|from src\.research|margin_calibration|margin_crossfit|margin_trace|logprob_margin|rebuild_llm_from_cache|infer_split|score_escalation|src\.predict|from src import predict|from src\.predict|src\.cli\.predict|hybrid_router|embeddings" src tests README.md STATUS.md CLAUDE.md docs research_docs openspec dvc.yaml
rg -n "src/cli/(train_risk_model|benign_risk_model|margin_calibration_fit|margin_calibration_report|margin_crossfit_eval|rebuild_llm_from_cache|infer_split|score_escalation|predict)|src\.(benign_risk_model|research|margin_trace|hybrid_router|predict|infer_split|embeddings)|python -m src\.(research|hybrid_router|predict|infer_split)|python -m src\.cli\.(train_risk_model|benign_risk_model|margin_calibration|margin_crossfit|rebuild_llm_from_cache|infer_split|score_escalation|predict)" README.md STATUS.md CLAUDE.md src/cli/README.md docs research_docs openspec
rg -n "train_risk_model|risk_model:|risk_model\.pkl|benign_risk_model|margin_calibration|margin_crossfit|hybrid_margin_trace|rebuild_llm_from_cache|infer_split|score_escalation|mode llm|mode hybrid|--mode llm|--mode hybrid|src\.cli\.predict|python -m src\.cli\.predict|src\.predict|src\.hybrid_router|src\.research|src\.embeddings" dvc.yaml configs src tests README.md STATUS.md CLAUDE.md src/cli/README.md docs/cleanup
dvc stage list
dvc dag final_verdict_report
```

Current DVC graph evidence:

- none of the paths below are DVC stage commands or DVC deps, except through
  retained canonical/shared modules such as `src/logprob_margin.py` and
  `src/llm_classifier/`
- `final_verdict_report` remains the documented final artifact:
  `reports/pipeline_final_verdict_report.md`

## Decision Matrix

| Scope area | Recommendation | Why |
| --- | --- | --- |
| Legacy risk model path | delete as out of scope, after one small refactor | `train_risk_model` and `risk_model` DVC stages were removed. Remaining source writes removed/stale `risk_model.pkl` and `posthoc_benign_risk_*` artifacts. The only active ties are optional hooks in legacy `research.py` and `hybrid_router.py` plus a stale config block. |
| Legacy `research.py` path | completed | Routing diagnostics were moved to `src/routing_diagnostics.py`; `src.research.py` was deleted in an approved cleanup batch. |
| Legacy `eval_external.py` path | deleted after helper split | The DVC DeBERTa external stage only needed `load_external_dataset`, now owned by `src/external_datasets.py`. The old standalone `python -m src.eval_external` ML/hybrid entrypoint was removed as non-canonical. |
| Margin calibration path | delete as out of scope, after `research.py` refactor/deletion | Margin calibration CLIs consume legacy `hybrid_margin_trace_*` artifacts and write removed/stale `reports/artifacts` outputs. `src/logprob_margin.py` must stay because canonical escalation features depend on it. |
| Cache rebuild path | delete as out of scope | `src/cli/rebuild_llm_from_cache.py` rebuilds old hosted LLM outputs from local cache and references old DVC cache objects. It is not canonical and should not be a fallback to hosted/legacy LLM outputs. Keep `src/llm_cache.py` because the classifier package still uses it. |
| Lightweight `infer_split` / `score_escalation` path | planned deletion after approval | These are outside the DVC graph and create an additional project-running path beside the canonical DVC flow. Removing them keeps split-level inference from writing or implying ownership of DVC artifacts. |
| Old public hybrid router path | planned deletion after approval | `src.hybrid_router` is not a DVC command or dependency after `src.cli.predict` deletion. It remains a user-facing non-canonical inference path and should be removed with its tests/docs once approved. |
| Dynamic embeddings helper | keep as supported helper for now | `src/llm_classifier/llm_classifier.py` imports `ExemplarBank` and `get_embeddings`; `src.validators` also imports `get_embeddings` for duplicate filtering. Delete only if dynamic few-shot and embedding-based validation are explicitly removed. |

## Batch 3A: Refactor Before Deleting `research.py`

Status: approved and executed. Routing diagnostics moved to
`src/routing_diagnostics.py`; `src.research.py` was later deleted after active
imports were removed.

## Proposed Batch 3B: Delete Legacy Risk Model Path

If Batch 3A is approved and completed, a later approval can delete:

- `src/benign_risk_model.py`
- `src/cli/train_risk_model.py`
- `src/cli/benign_risk_model.py`
- `tests/test_benign_risk_model.py`

And update:

- `configs/default.yaml`: remove the stale `hybrid.risk_model` block and
  `Train with: python -m src.cli.train_risk_model` comment
- `src/hybrid_router.py`: remove optional loading/use of `RiskModel`
- `src/research.py`: remove optional loading/use of `RiskModel`, or delete
  `src/research.py` if separately approved
- `README.md`, `CLAUDE.md`, and `src/cli/README.md`: remove risk-model
  commands, artifact paths, and project-structure entries

Tests to remove only with this deletion:

- `tests/test_benign_risk_model.py`

Expected verification:

```bash
rg -n "benign_risk_model|train_risk_model|risk_model\.pkl|posthoc_benign|hybrid\.risk_model|risk_model:" configs src tests README.md STATUS.md CLAUDE.md src/cli/README.md
pytest tests/test_hybrid_router.py tests/test_external_datasets.py tests/test_eval_deberta_external.py -q
dvc stage list
dvc dag final_verdict_report
```

Behavior note: removing `RiskModel` from `hybrid_router` changes the old
optional live-hybrid behavior when `hybrid.risk_model.enabled` was true. That
should be documented as an approved scope deletion, not as a silent cleanup.

## Batch 3C: Delete Margin Calibration Path

Status: approved and executed on 2026-05-15. Deleted the exploratory margin
trace/calibration path while preserving `src/logprob_margin.py` and
`tests/test_logprob_margin.py`.

If `src.research` no longer owns active dependencies, a later approval can
delete:

- `src/margin_trace.py`
- `src/cli/margin_calibration_fit.py`
- `src/cli/margin_calibration_report.py`
- `src/cli/margin_crossfit_eval.py`
- `tests/test_margin_trace.py`

Keep:

- `src/logprob_margin.py`
- `tests/test_logprob_margin.py`

Why: `src/logprob_margin.py` is canonical input-feature logic for
`train_escalating_model`; `src/margin_trace.py` and calibration CLIs are tied
to historical `hybrid_margin_trace_*` artifacts and exploratory threshold
reports.

Expected verification:

```bash
rg -n "margin_trace|margin_calibration|margin_crossfit|hybrid_margin_trace" src tests README.md STATUS.md CLAUDE.md src/cli/README.md configs
pytest tests/test_logprob_margin.py tests/test_escalating_model.py -q
dvc stage list
dvc dag final_verdict_report
```

## Batch 3D: Delete Cache Rebuild Path

Status: approved and executed on 2026-05-15. Deleted only
`src/cli/rebuild_llm_from_cache.py` and `tests/test_rebuild_llm_from_cache.py`.
Preserved `src/llm_cache.py`, `tests/test_llm_cache.py`, and existing
`.cache/llm` contents.

If approved, delete:

- `src/cli/rebuild_llm_from_cache.py`
- `tests/test_rebuild_llm_from_cache.py`

Keep:

- `src/llm_cache.py`
- `tests/test_llm_cache.py`

Why: cache rebuild recovers old hosted LLM classifier outputs and references
old DVC cache object paths. The canonical path requires explicit Colab local
classifier handoff validation and must not silently fall back to legacy hosted
LLM outputs.

Expected verification:

```bash
rg -n "rebuild_llm_from_cache|recovered_from_cache|old_fewshot_source|reports/rebuild_llm_from_cache_audit" src tests README.md STATUS.md CLAUDE.md docs/cleanup
pytest tests/test_llm_cache.py tests/test_llm_classifier.py -q
dvc stage list
dvc dag final_verdict_report
```

## Proposed Batch 3E: Delete Non-DVC Inference Entrypoints

Status: planned for deletion after approval. These paths are not DVC stage
commands or DVC dependencies and create additional ways to run project
inference outside the canonical DVC flow.

Planned deletions:

- `src/hybrid_router.py`
- `src/cli/infer_split.py`
- `src/infer_split.py`
- `src/cli/score_escalation.py`

Tests to remove with this deletion:

- `tests/test_hybrid_router.py`
- `tests/test_cli_infer_split.py`
- `tests/test_score_escalation.py`

Docs/config references to update:

- `README.md`
- `CLAUDE.md`
- `src/cli/README.md`
- `docs/cleanup/deletion_candidates.md`
- stale cleanup/history references where they present these paths as active
  supported commands

Keep during this batch:

- `src/external_datasets.py`
- `src/evaluate.py`
- `src/llm_classifier/`

Why: these are still used by DVC stages or imported by DVC-used CLIs. The old
standalone `src.eval_external` surface has been removed.

Expected verification after approval:

```bash
rg -n "hybrid_router|infer_split|score_escalation|python -m src\.hybrid_router|python -m src\.infer_split|python -m src\.cli\.(infer_split|score_escalation)" src tests README.md CLAUDE.md src/cli/README.md docs/cleanup dvc.yaml
pytest tests/test_external_datasets.py tests/test_eval_deberta_external.py tests/test_evaluate.py tests/test_llm_classifier.py tests/test_final_verdict_report.py tests/test_judge_colab_local_predictions.py -q
dvc stage list
dvc dag final_verdict_report
```

## Public Prediction Modes

Status: `src.cli.predict` and its compatibility wrapper were approved for
deletion after confirmation they are not used by DVC. `src.hybrid_router` is
now included in planned Batch 3E deletion as a non-DVC inference entrypoint.

Former documented public modes:

- `python -m src.cli.predict --mode ml`
- `python -m src.cli.predict --mode llm`
- `python -m src.cli.predict --mode hybrid`
- `python -m src.hybrid_router --no-wandb`

Original recommendation:

- keep `--mode ml` as a supported lightweight utility
- either explicitly keep `--mode llm` and `--mode hybrid` as non-canonical
  manual tools, or approve a later behavior-changing deletion of those modes
- do not change these modes in a cleanup-only batch without explicit approval

The `src.cli.predict` and `src/predict.py` files have already been deleted.
The remaining public hybrid-router behavior is planned for removal in Batch
3E, and should be committed/documented separately from the risk-model cleanup.

## Active Docs To Update After Approved Batches

Active docs with current stale or scope-ambiguous references:

- `README.md`
  - `src/benign_risk_model.py`
  - `python -m src.hybrid_router --no-wandb`
  - `src/research.py`
  - `src/embeddings.py`
  - `src/cli/infer_split.py`
  - `src/cli/score_escalation.py`
  - margin calibration CLIs
  - stale public predict/hybrid-router examples
- `src/cli/README.md`
  - `infer_split` and `score_escalation` support status
  - stale public predict/hybrid-router examples
- `CLAUDE.md`
  - public prediction examples
  - keep/delete guidance for `src/logprob_margin.py`, `src/llm_cache.py`,
    `src/llm_classifier/`, and related utilities
- `configs/default.yaml`
  - stale `hybrid.risk_model` config block if risk path is removed

Historical docs should receive only the approved historical-note header if
they have stale pipeline instructions and lack the note already. Do not
rewrite historical content deeply.

## Approval Request

Recommended next approval is **Batch 3E deletion** if the goal remains one
official project-running path:

- delete `src/hybrid_router.py`
- delete `src/cli/infer_split.py`
- delete `src/infer_split.py`
- delete `src/cli/score_escalation.py`

This keeps the next change conservative and creates a clean dependency line
for later deletion approvals.

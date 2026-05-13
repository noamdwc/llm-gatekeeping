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
| Legacy `research.py` path | refactor/minimize before deletion | `src/eval_external.py` imports only routing diagnostics from `src.research`, while `src.research` itself is a large historical merge/routing/report path. Delete only after moving diagnostics to a small supported helper. |
| Margin calibration path | delete as out of scope, after `research.py` refactor/deletion | Margin calibration CLIs consume legacy `hybrid_margin_trace_*` artifacts and write removed/stale `reports/artifacts` outputs. `src/logprob_margin.py` must stay because canonical escalation features depend on it. |
| Cache rebuild path | delete as out of scope | `src/cli/rebuild_llm_from_cache.py` rebuilds old hosted LLM outputs from local cache and references old DVC cache objects. It is not canonical and should not be a fallback to hosted/legacy LLM outputs. Keep `src/llm_cache.py` because the classifier package still uses it. |
| Lightweight `infer_split` / `score_escalation` path | keep as supported utility, but minimize before calling it supported | It can be useful for split-level local inspection, but currently `score_escalation` defaults to the same `data/processed/research/escalating_model_eval_{split}.parquet` path owned by DVC. Keeping it requires repathing its default output outside DVC-owned artifacts and tightening docs. |
| Old public prediction modes | explicit product-scope decision required | `src.cli.predict --mode llm/hybrid` and `src.hybrid_router` are user-facing commands documented in README/CLAUDE. Deleting or shrinking them is a behavior/scope change, not simple cleanup. |
| Dynamic embeddings helper | keep as supported helper for now | `src/llm_classifier/llm_classifier.py` imports `ExemplarBank` and `get_embeddings`; `src.validators` also imports `get_embeddings` for duplicate filtering. Delete only if dynamic few-shot and embedding-based validation are explicitly removed. |

## Proposed Batch 3A: Refactor Before Deleting `research.py`

If approved, do only this preparatory refactor:

1. Create a small diagnostics helper, for example
   `src/routing_diagnostics.py`, containing:
   - `compute_routing_diagnostics`
   - `render_routing_diagnostics_markdown`
2. Update `src/eval_external.py` to import diagnostics from the new helper.
3. Update tests so diagnostics coverage no longer requires importing
   `src.research`.
4. Leave `src/research.py` in place for this batch.
5. Update active docs to mark `src.research` as historical and no longer a
   dependency of external DeBERTa/final-report code.

Expected verification:

```bash
pytest tests/test_eval_external.py tests/test_research.py -q
rg -n "from src\.research import|import src\.research|src\.research" src tests README.md STATUS.md CLAUDE.md src/cli/README.md
dvc stage list
dvc dag final_verdict_report
```

Rationale: this removes the main active dependency that prevents a later
source deletion of `src/research.py` without touching evaluation behavior.

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
pytest tests/test_hybrid_router.py tests/test_eval_external.py -q
dvc stage list
dvc dag final_verdict_report
```

Behavior note: removing `RiskModel` from `hybrid_router` changes the old
optional live-hybrid behavior when `hybrid.risk_model.enabled` was true. That
should be documented as an approved scope deletion, not as a silent cleanup.

## Proposed Batch 3C: Delete Margin Calibration Path

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

## Proposed Batch 3D: Delete Cache Rebuild Path

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

## Proposed Batch 3E: Keep And Minimize Lightweight Split Inference

Recommendation: keep this as a supported utility, but refactor before
documenting it as supported:

- keep `src/cli/infer_split.py`
- keep `src/cli/score_escalation.py`
- keep `src/infer_split.py` wrapper while the CLI is kept
- keep `tests/test_cli_infer_split.py`
- keep `tests/test_score_escalation.py`

Required minimization before support:

1. Change `score_escalation` default output away from DVC-owned
   `data/processed/research/escalating_model_eval_{split}.parquet`, unless an
   explicit `--output` is provided.
2. Make docs clear that `infer_split --mode escalation` is a local inspection
   utility and not the canonical start-to-finish pipeline.
3. Ensure it still requires the Colab/local classifier artifact and DeBERTa
   artifact paths; do not add fallback to legacy/hosted LLM outputs.

Expected verification after a future approved refactor:

```bash
pytest tests/test_cli_infer_split.py tests/test_score_escalation.py tests/test_final_verdict_report.py -q
rg -n "escalating_model_eval_\\{split\\}|infer_split|score_escalation" src tests README.md src/cli/README.md CLAUDE.md
dvc status final_verdict_report
```

## Public Prediction Modes: Scope Decision Needed

Current documented public modes:

- `python -m src.cli.predict --mode ml`
- `python -m src.cli.predict --mode llm`
- `python -m src.cli.predict --mode hybrid`
- `python -m src.hybrid_router --no-wandb`

Recommendation:

- keep `--mode ml` as a supported lightweight utility
- either explicitly keep `--mode llm` and `--mode hybrid` as non-canonical
  manual tools, or approve a later behavior-changing deletion of those modes
- do not change these modes in a cleanup-only batch without explicit approval

If old public LLM/hybrid modes are approved out of scope later, a deletion or
minimization batch would need to update:

- `src/cli/predict.py`
- `src/predict.py`
- `src/hybrid_router.py`
- `tests/test_hybrid_router.py`
- `README.md`
- `CLAUDE.md`
- `src/cli/README.md`

This would be a user-facing behavior change, so it should be committed and
documented separately from cleanup-only deletions.

## Active Docs To Update After Approved Batches

Active docs with current stale or scope-ambiguous references:

- `README.md`
  - `src/benign_risk_model.py`
  - `python -m src.hybrid_router --no-wandb`
  - `src/research.py`
  - `src/embeddings.py`
  - `src/cli/infer_split.py`
  - margin calibration CLIs
  - public `predict --mode llm/hybrid` examples if those modes are removed
- `src/cli/README.md`
  - `infer_split` support status
  - public `predict --mode hybrid` example if hybrid mode is removed
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

Recommended next approval is **Batch 3A only**:

- add `src/routing_diagnostics.py`
- repoint `src/eval_external.py` and tests away from `src.research`
- do not delete `src/research.py` yet
- do not delete risk-model, margin-calibration, cache-rebuild, inference, or
  public prediction code yet

This keeps the next change conservative and creates a clean dependency line
for later deletion approvals.

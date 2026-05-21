# Deletion Batch 2 Proposal: Risky Source, Module, and Doc Cleanup

Date: 2026-05-13

Status: proposal only. Do not delete anything in this batch until explicitly
approved.

## Scope

This gate covers the source/module/doc candidates held out of batch 1. The
goal is still pipeline cleanup only:

- no retraining
- no threshold tuning
- no metric/model-behavior changes
- no evaluation-logic changes
- no weakening or bypassing Colab handoff validation

The current Deepset Colab handoff validation failure remains a real artifact
blocker and is intentionally not addressed by this cleanup proposal.

## Checks Run

Dependency and stale-reference checks:

```bash
rg -n "eval_external|llm_provider|llm_cache|llm_classifier|synthetic_benign|score_escalation|logprob|margin" .
rg -n "from src\.(eval_external|llm_provider|llm_cache|synthetic_benign|logprob_margin|margin_trace)|import src\.(eval_external|llm_provider|llm_cache|synthetic_benign|logprob_margin|margin_trace)|from src\.llm_classifier|import src\.llm_classifier|src\.cli\.score_escalation|from src\.cli import .*score_escalation" src tests notebooks scripts README.md docs
rg -n "eval_new|train_risk_model|risk_model|benign_risk_model|margin_calibration|margin_crossfit|research_external|rebuild_llm_from_cache|run_llm_provider_refresh|run_synth|llm_classifier_val|llm_classifier\b|test_unseen|posthoc_benign|reports/research/|reports/artifacts/|reports/error_analysis_current_status" README.md docs src tests scripts CLAUDE.md STATUS.md
dvc stage list
```

Current active DVC stages:

- `generate_synthetic_benign@B` through `@F`
- `preprocess`
- `build_splits`
- `ml_model`
- `deberta_model`
- `deberta_external@deepset`
- `deberta_external@jackhhao`
- `judge_colab_local_predictions@{test,unseen_test,safeguard_test}`
- `validate_colab_handoff`
- `train_escalating_model`
- `judge_colab_local_predictions_external@{deepset,jackhhao}`
- `final_verdict_report`

## Classification

### Must Keep

These are still used by active DVC stages, the Colab handoff, or canonical
runtime modules.

| Candidate | Classification | Evidence |
| --- | --- | --- |
| `src/eval_external.py` | must keep | DVC dependency of `deberta_external@{deepset,jackhhao}`. Imported by `src/cli/eval_deberta_external.py`, `src/cli/run_baseline.py`, and the Colab notebook. |
| `tests/test_eval_external.py` | must keep | Runtime module remains required. |
| `src/llm_classifier/` | must keep | DVC dependency of `judge_colab_local_predictions*`; imported by `judge_colab_local_predictions.py`, the Colab notebook, `predict.py`, `hybrid_router.py`, `research.py`, and `ml_baseline.py` constants. |
| `tests/test_llm_classifier.py`, `tests/test_rate_limiter.py` | must keep | Runtime package remains required. |
| `src/llm_provider.py` | must keep | Imported by `src/llm_classifier/llm_classifier.py`, `src/embeddings.py`, and `src/synthetic_benign.py`. |
| `src/llm_cache.py` | must keep | Imported by `src/llm_classifier/llm_classifier.py` and `src/cli/rebuild_llm_from_cache.py`; cache tests still cover it. |
| `tests/test_llm_cache.py` | must keep | Runtime cache module remains required. |
| `src/synthetic_benign.py` | must keep | DVC dependency of `generate_synthetic_benign@B-F`; `preprocess` depends on `data/processed/synthetic_benign/`. |
| `src/cli/generate_synthetic_benign.py` | must keep | DVC command for `generate_synthetic_benign@B-F`. |
| `tests/test_synthetic_benign.py` | must keep | Runtime module and DVC CLI remain required. |
| `src/logprob_margin.py` | must keep | Imported by `src/escalating_model.py`; canonical `train_escalating_model` derives Colab classifier logprob features from it. |
| `tests/test_logprob_margin.py` | must keep | Canonical escalation feature extraction depends on this logic. |
| `src/cli/score_escalation.py` | must keep for now | Not on DVC graph, but used by `src/cli/infer_split.py` and covered by `tests/test_score_escalation.py`. Removing it requires an explicit decision to remove lightweight escalation inference. |
| `tests/test_score_escalation.py`, `tests/test_cli_infer_split.py` | must keep for now | Should only be removed if the lightweight inference path is removed or refactored. |

### Needs Refactor Or Explicit Scope Decision First

These appear to be off the canonical DVC path, but they are intertwined with
other non-canonical runtime paths or docs. They should not be deleted as a
blind cleanup batch.

| Candidate | Classification | Required before deletion |
| --- | --- | --- |
| `src/cli/infer_split.py` | needs explicit scope decision | Uses `score_escalation.py`, `judge_colab_local_predictions.py`, `final_verdict_report.py`, and `HierarchicalLLMClassifier`. Delete only if lightweight split inference is out of repo scope. |
| `src/cli/rebuild_llm_from_cache.py` | needs explicit scope decision | Rebuilds old hosted LLM classifier outputs from cache; off DVC graph, but tied to `llm_cache` and `llm_classifier` helpers. Delete only if cache recovery is out of scope. |
| `tests/test_rebuild_llm_from_cache.py` | remove only with runtime module | Test imports `src.cli.rebuild_llm_from_cache`. Also contains stale guidance to rerun removed `llm_classifier` DVC stages. |
| `src/benign_risk_model.py` | needs refactor first | Legacy post-hoc model, but still imported by `src/hybrid_router.py`, `src/research.py`, `src/cli/train_risk_model.py`, and `src/cli/benign_risk_model.py`. Removing it requires deleting or refactoring those legacy consumers. |
| `src/cli/train_risk_model.py` | needs refactor first | Off DVC graph, but directly depends on `src/benign_risk_model.py`. Delete together only if legacy hybrid risk-model path is removed. |
| `src/cli/benign_risk_model.py` | needs refactor first | Off DVC graph, but directly depends on `src/benign_risk_model.py` and writes removed post-hoc reports/artifacts. |
| `tests/test_benign_risk_model.py` | remove only with runtime modules | Should be removed only if `src/benign_risk_model.py` and its CLIs are approved for deletion. |
| `src/research.py` | needs refactor first | Off current DVC graph, but imported by tests and still references legacy LLM/risk/eval report paths. It also supplies margin trace helpers used by legacy analysis. |
| `tests/test_research.py` | remove only with runtime module | Should be removed only if `src/research.py` is approved for deletion. |
| `src/hybrid_router.py` | needs refactor first | Not on canonical DVC graph, but still powers prediction CLI hybrid mode and imports `llm_classifier`, `logprob_margin`, and optional `benign_risk_model`. Removing risk-model support here would be behavior-affecting and outside cleanup scope. |
| `src/cli/predict.py`, `src/predict.py` | needs explicit scope decision | Public prediction CLI still documents ML/LLM/hybrid modes in README. Deleting or shrinking modes is user-facing behavior change. |
| `src/embeddings.py` | needs explicit scope decision | Supports dynamic few-shot retrieval for legacy/hosted LLM paths and imports `llm_provider` plus `llm_classifier.constants`. It is not DVC-canonical but may still be reusable helper code. |
| `src/cli/margin_calibration_fit.py`, `src/cli/margin_calibration_report.py`, `src/cli/margin_crossfit_eval.py` | needs explicit scope decision | Off DVC graph and write to previously removed `reports/artifacts`, but depend on `margin_trace`. Delete only if margin calibration exploration is out of scope. |
| `src/margin_trace.py` | needs explicit scope decision | Not needed by canonical `train_escalating_model`, but is used by margin calibration CLIs and legacy `research.py`. Keep unless those paths are removed. |

### Safe To Delete After Approval

These are off the current DVC graph and are legacy reporting/research paths.
They can be removed in a second approved batch if their matching tests and
stale docs references are removed in the same commit.

| Candidate | Classification | Matching tests/docs |
| --- | --- | --- |
| `src/cli/research_external.py` | safe to delete after approval | Remove `tests/test_research_external.py`; update `src/cli/README.md`, `STATUS.md`, `CLAUDE.md`, and stale references in `src/cli/eval_new.py` if `eval_new.py` is not deleted at the same time. |
| `tests/test_research_external.py` | remove with runtime module | Directly imports `src.cli.research_external`. |
| `src/cli/eval_new.py` | safe to delete after approval | Remove or update `tests/test_baselines.py` sections importing `eval_new`; update `README.md`, `src/cli/README.md`, `STATUS.md`, `CLAUDE.md`, and stale strings in `src/research.py`. |
| `src/cli/eval_baselines.py` | safe to delete after approval | Imported by `src/cli/eval_new.py` and `tests/test_baselines.py`; off canonical DVC graph. |
| `src/cli/run_baseline.py` | safe to delete after approval | Imported by `tests/test_baselines.py`; off canonical DVC graph. |
| `tests/test_baselines.py` | remove with baseline/eval modules | Covers `eval_baselines`, `eval_new`, and `run_baseline`. |
| `scripts/analyze_external_attack_types.py` | safe to delete after approval | Reads legacy `data/processed/research_external/research_external_*.parquet`, which is no longer canonical. |
| `scripts/run_judge_on_abstain.py` | safe to delete after approval | Reads removed/legacy `hybrid_margin_trace_test.parquet` and performs ad-hoc judge-on-abstain analysis outside DVC. |
| `run_synth.sh` | safe to delete after approval | Wrapper for `python -m src.cli.generate_synthetic_benign`; DVC and CLI are canonical. Keep `src/cli/generate_synthetic_benign.py`. |

## Tests Coupled To Runtime Deletions

Do not remove these tests unless the paired runtime module is approved for
deletion in the same batch:

| Test file | Remove only if deleting |
| --- | --- |
| `tests/test_eval_external.py` | `src/eval_external.py` |
| `tests/test_llm_classifier.py` | `src/llm_classifier/llm_classifier.py` and package helpers |
| `tests/test_rate_limiter.py` | `src/llm_classifier/rate_limiter.py` |
| `tests/test_llm_cache.py` | `src/llm_cache.py` |
| `tests/test_synthetic_benign.py` | `src/synthetic_benign.py` and `src/cli/generate_synthetic_benign.py` |
| `tests/test_logprob_margin.py` | `src/logprob_margin.py` |
| `tests/test_score_escalation.py` | `src/cli/score_escalation.py` |
| `tests/test_cli_infer_split.py` | `src/cli/infer_split.py` and escalation inference path |
| `tests/test_rebuild_llm_from_cache.py` | `src/cli/rebuild_llm_from_cache.py` |
| `tests/test_benign_risk_model.py` | `src/benign_risk_model.py`, `src/cli/train_risk_model.py`, `src/cli/benign_risk_model.py` |
| `tests/test_research.py` | `src/research.py` |
| `tests/test_research_external.py` | `src/cli/research_external.py` |
| `tests/test_baselines.py` | `src/cli/eval_baselines.py`, `src/cli/run_baseline.py`, `src/cli/eval_new.py` |

## Stale README/Docs References

These should be updated after approved deletions, not before, so docs stay
truthful to the retained source tree.

### Root README

- `src/cli/research_external.py`
- `src/cli/eval_new.py`
- `src/cli/benign_risk_model.py`
- `src/cli/margin_calibration_fit.py`
- `src/cli/margin_calibration_report.py`
- `src/cli/margin_crossfit_eval.py`
- `data/processed/models/risk_model.pkl`
- `data/processed/research/research_<split>.parquet`
- `data/processed/research/hybrid_margin_trace_<split>.parquet`
- `data/processed/research/posthoc_benign_risk_predictions.parquet`
- `data/processed/research/posthoc_benign_risk_summary.csv`
- `data/processed/research_external/research_external_<ds>.parquet`

### `src/cli/README.md`

- legacy `research_external.py` usage and DVC stage wording
- legacy `eval_external` vs `research_external` comparison
- stale `reports/research/inference_ml_<split>.md`

### `STATUS.md`

- stale `test_unseen` split names
- removed `llm_classifier`, `research_external`, and `eval_new` DVC/report flow
- stale claim that `src.cli.eval_new` is canonical

### `CLAUDE.md`

- removed DVC stages: `llm_classifier`, `llm_classifier_val`, `research`,
  `research_val`, `train_risk_model`, `risk_model`, `research_external*`,
  `eval_new*`
- stale `test_unseen` split names
- stale post-hoc risk report/artifact paths
- stale `reports/research/*` and `reports/research_external/*` component
  report paths

### Historical Docs

These contain stale references but should be treated as historical unless a
separate documentation-cleanup approval is given:

- `docs/2026-03-13_baseline_dataset_overlap.md`
- `docs/superpowers/**`
- `openspec/**`
- `docs/merge_readiness_findings.md` (untracked at the time of this proposal)

## Proposed Batch 2A: Conservative Source Cleanup

If approved, delete only:

- `src/cli/research_external.py`
- `tests/test_research_external.py`
- `src/cli/eval_new.py`
- `src/cli/eval_baselines.py`
- `src/cli/run_baseline.py`
- `tests/test_baselines.py`
- `scripts/analyze_external_attack_types.py`
- `scripts/run_judge_on_abstain.py`
- `run_synth.sh`

Then update stale references in:

- `README.md`
- `src/cli/README.md`
- `STATUS.md`
- `CLAUDE.md`
- any remaining non-historical references found by `rg`

Expected verification after approved deletion:

```bash
rg -n "research_external|eval_new|eval_baselines|run_baseline|run_synth|run_judge_on_abstain|analyze_external_attack_types" README.md src tests scripts STATUS.md CLAUDE.md
dvc stage list
dvc dag final_verdict_report
dvc status final_verdict_report
pytest tests/test_validate_colab_handoff.py tests/test_escalating_model.py tests/test_judge_colab_local_predictions.py tests/test_final_verdict_report.py tests/test_eval_deberta_external.py tests/test_colab_local_llm_classifier_notebook.py -q
```

Do not run `dvc repro final_verdict_report` until the Deepset Colab handoff
artifact is fixed.

## Explicit Holds For Later Approval

Hold these unless a later approval explicitly removes their feature scope or
approves the needed refactor:

- `src/eval_external.py`
- `src/llm_provider.py`
- `src/llm_cache.py`
- `src/llm_classifier/`
- `src/synthetic_benign.py`
- `src/cli/generate_synthetic_benign.py`
- `src/logprob_margin.py`
- `src/cli/score_escalation.py`
- `src/cli/infer_split.py`
- `src/cli/rebuild_llm_from_cache.py`
- `src/benign_risk_model.py`
- `src/cli/train_risk_model.py`
- `src/cli/benign_risk_model.py`
- `src/research.py`
- `src/hybrid_router.py`
- `src/cli/predict.py`
- `src/predict.py`
- `src/embeddings.py`
- `src/margin_trace.py`
- `src/cli/margin_calibration_fit.py`
- `src/cli/margin_calibration_report.py`
- `src/cli/margin_crossfit_eval.py`
- `docs/research_docs`
- `docs/superpowers/**`
- `openspec/**`

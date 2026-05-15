# Merge-Readiness Findings — `fix/escalation-model-routing-docs`

Compiled while preparing this branch for merge. The branch ships the
productionized escalation inference path; the items below are
out-of-scope for that branch and should be addressed in a follow-up
branch dedicated to cleaning DVC pipeline state and code hygiene.

## 1. `dvc status` is far from clean — beyond what the README documents

README's "Deferred / non-canonical stages" section enumerates ~7 stage
groups. Actual `dvc status` reports **28 dirty stages**, including
foundational and canonical ones the README does *not* mention:

- `preprocess`, `build_splits`, `ml_model`, `deberta_model` —
  foundational stages, all show modified source + outs.
- `generate_synthetic_benign@{B,C,D,E}` — README claims only `@F` is
  deferred; B–E are also dirty (changed `llm` param section).
- **`train_escalating_model` is dirty** — the canonical gate. `dvc
  status` shows modified `src/escalating_model.py`, modified
  `src/cli/train_escalating_model.py`, modified prediction parquet
  inputs (val + test + unseen_val + unseen_test + safeguard_test +
  externals), `new: configs/default.yaml` params section, and modified
  outputs (`escalating_model.pkl` and all
  `escalating_model_eval_*.parquet`).
- `research_external_llm@{deepset,jackhhao}`,
  `research_external@{deepset,jackhhao}` — not listed as deferred.

Either re-deferral docs need to expand, or `dvc commit` needs to run
against the foundational/canonical stages once a reproducible run is
done.

## 2. `dvc.yaml` ↔ `configs/default.yaml` consistency is broken

`preprocess` and `build_splits` now depend on the new
`configs/default.yaml::training_datasets` section (added on this
branch), but `dvc.lock` for those stages still reflects the
pre-`training_datasets` parameter state. `dvc status` shows `new:
training_datasets` under both stages' deps. This is an unsynchronized
lock file rather than deferred-by-cost.

Cheapest fix: `dvc commit preprocess build_splits` after confirming
the stages run correctly.

## 3. Schema migration is half-applied

`dvc.yaml` migrated outputs from `test_unseen.parquet` → `unseen_val
+ unseen_test + safeguard_test`, but `src/build_splits.py:127` still
has a comment referencing the old `test_unseen` name. Cosmetic, but
indicates the rename isn't fully done.

## 4. `final_verdict_report` "up to date" is misleading

`dvc status final_verdict_report` → `Data and pipelines are up to
date.` But its transitive upstream `train_escalating_model` is dirty.
`final_verdict_report`'s deps in `dvc.yaml` skip the escalating-model
artifact and instead depend on the judged colab parquets, so DVC's
"up to date" is a hash match against frozen artifacts — it does
**not** mean a fresh `dvc repro` would reproduce the report. The
canonical report is effectively detached from its upstream training
stage.

Fix direction: add `escalating_model.pkl` (or at least a digest of it)
to `final_verdict_report` deps, so DVC reflects the real dependency.

## 5. Private functions imported across modules

`src/cli/infer_split.py:174,178` calls
`judge_colab_local_predictions._validate_input(...)` and
`judge_colab_local_predictions._load_escalation_scores(...)`. Leading
underscore = private by convention. Either rename them or expose
them via public wrappers (`validate_classifier_input`,
`load_escalation_scores`).

## 6. Silent behavior change in the DeBERTa fast-path dedup fix

`src/research.py` does
`deberta_df.drop_duplicates("sample_id", keep="first")`. The
escalating-model module deliberately *averages* numeric conflicts
(`_deduplicate_by_sample_id(..., average_numeric_conflicts=True)`).
The two strategies differ for the same kind of input.

Open question for the follow-up branch: which is correct, and is
`keep="first"` masking an upstream data-pipeline bug in
`build_splits` / `deberta_model` where 3 sample_ids ended up
duplicated in `safeguard_test` to begin with?

## 7. No CI / pre-commit configuration

No `.github/workflows/`, no `.pre-commit-config.yaml`. Nothing
automated runs tests, lint, type-check, or `dvc status` on PRs.
Foundational + canonical stage touches went out with zero automated
gating. Adding a minimal GitHub Actions workflow (pytest + ruff +
`dvc status --show-checksums` advisory) would catch most of the above
on PR.

## 8. `score_escalation` CLI shares output path with the DVC stage

Both `src/cli/score_escalation.py` and the DVC
`train_escalating_model` stage emit
`data/processed/research/escalating_model_eval_{split}.parquet`.
Running the lightweight CLI silently overwrites DVC-tracked outputs;
DVC will flag this on the next `dvc status` but won't prevent the
overwrite.

Fix direction: either route the lightweight CLI to a non-DVC path
(e.g. `data/processed/inference/escalating_model_eval_{split}.parquet`)
or add an explicit `--output` guard / overwrite warning.

## 9. Deleted test removes coverage based on an unverified claim

`tests/test_colab_local_llm_classifier_local_vllm_mode.py` was
removed because the vLLM-Docker notebook mode was migrated to
transformers-local in commit `a7f9eeb`. The follow-up branch should
either confirm via grep that the notebook no longer contains the
vLLM-Docker code path, or add a replacement test for the
transformers-local mode if coverage is still desired.

## 10. Branch not pushed to origin

At the time of writing, `git rev-list --left-right --count
main...HEAD` is `0 71`. Origin HEAD for this branch is `c86f07f`;
local has `0d28a54`, `345b45e`, `3238afd` unpushed. A reviewer
checking the branch on GitHub sees an older tree than local. Push
before requesting review.

---

## Priority order for the follow-up branch

1. (Blocker for any merge to `main`) Run a clean `dvc repro` of
   foundational + canonical stages (preprocess, build_splits,
   ml_model, deberta_model, train_escalating_model) and commit the
   resulting `dvc.lock`. Confirm `dvc status` is green for the
   canonical path.
2. (Blocker) Decide and document the duplicate-sample_id story:
   either fix `build_splits` to dedupe, or pick a canonical dedup
   strategy and apply it consistently in `research.py` and
   `escalating_model.py`.
3. (High) Wire `escalating_model.pkl` into `final_verdict_report`
   deps so "up to date" stops lying.
4. (High) Add minimal CI: pytest + ruff + `dvc status --show-json`
   advisory check.
5. (Medium) Promote the two private functions reused by `infer_split`
   into public APIs.
6. (Medium) Repath `score_escalation` output so it can't clobber the
   DVC-tracked artifact.
7. (Low) Finish the `test_unseen` → `unseen_*` rename (comments,
   stale references).
8. (Low) Either expand README's deferred list to match `dvc status`,
   or run `dvc commit` for stages that are intentionally frozen.

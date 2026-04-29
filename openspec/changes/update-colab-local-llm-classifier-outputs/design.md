## Context

`notebooks/colab_local_llm_classifier.ipynb` now runs local Transformers classifier inference for one configured `SPLIT`, reading `data/processed/splits/{SPLIT}.parquet` from Drive and writing one checkpoint plus one final parquet under `data/processed/predictions`. Downstream research already expects LLM prediction files for multiple main splits and separate external prediction artifacts, including external datasets configured in `configs/default.yaml`.

The notebook should remain a Colab-friendly, self-contained runner. It should reuse the existing local model setup and classifier helper functions, but organize the data loading and output-writing loop around explicit output targets instead of a single global split.

## Goals / Non-Goals

**Goals:**

- Generate classifier-only outputs for `train`, `val`, `test`, `unseen_val`, `unseen_test`, and `safeguard_test` in one notebook run.
- Generate classifier-only external outputs for `deepset` and `jackhhao`.
- Keep the existing prediction row schema and validation semantics for all generated parquet files.
- Keep per-target checkpointing and resume behavior so a failed long Colab run can continue without rerunning completed rows.
- Reuse existing config and dataset-loading helpers where practical.

**Non-Goals:**

- Changing the prompt, local Transformers inference implementation, or model configuration.
- Adding new external datasets beyond `deepset` and `jackhhao`.
- Changing DVC stages, repo-wide prediction filenames, or downstream research schemas outside this notebook.
- Running multiple model generations concurrently.

## Decisions

1. Represent notebook work as explicit run targets.

   Rationale: A target object can carry the dataset kind, dataset key, input loader, checkpoint path, output path, and progress label. This avoids duplicating classification logic for main and external datasets while keeping output names predictable.

   Alternative considered: keep `SPLIT` and ask the user to rerun the notebook repeatedly. That preserves the current structure but fails the requirement to produce the full output set.

2. Keep main split outputs under `PREDICTIONS_DIR` and external outputs under a separate external predictions directory.

   Rationale: The repo already distinguishes main predictions from external predictions through paths such as `data/processed/predictions` and `data/processed/predictions_external`. The notebook should follow that convention so downstream tools can discover the artifacts without special cases.

   Alternative considered: write every output into one directory with a prefix. That would be simpler inside the notebook but less compatible with existing research/external flows.

3. Load main split datasets from Drive parquet files and external datasets through existing repo helpers.

   Rationale: Main splits are materialized by `src.build_splits`, while external datasets are configured in `configs/default.yaml` and loaded through existing external evaluation code. Reusing those paths preserves label mapping and binary-only external dataset assumptions.

   Alternative considered: require pre-exported external parquet files in Drive. That would be more uniform but would duplicate dataset mapping behavior and require extra manual preparation.

4. Use `train` only as the few-shot source and skip self-output rows from few-shot examples only if the current notebook already does so.

   Rationale: The current notebook always reads `train.parquet` to build few-shot examples. When generating predictions for the `train` split itself, the implementation should keep the classifier output schema consistent and avoid inventing a new train-specific evaluation policy.

   Alternative considered: omit `train` outputs because training data is used for few-shot examples. The requested output list explicitly includes `train`, so omission would violate the change.

5. Preserve per-target validation but account for binary-only external fields.

   Rationale: The classifier prediction columns are shared, but external datasets may synthesize or omit hierarchical ground-truth fields. Validation should assert the prediction schema for every target and avoid requiring main-only ground-truth columns for external outputs.

   Alternative considered: force external data into the exact main split schema. That risks brittle transformations and unnecessary coupling to fields external datasets do not naturally provide.

## Risks / Trade-offs

- Long Colab runtime across all targets -> Mitigate with per-target checkpoints, resume filtering, configurable target lists, and existing `LIMIT` support adapted per target.
- External dataset downloads or authentication may fail -> Mitigate by loading through existing helpers so errors are localized to the target and the failing dataset key is printed.
- Output filename mismatch could break downstream consumers -> Mitigate by following existing `llm_predictions_{split}_...` and `llm_predictions_external_{dataset}` conventions.
- A checkpoint from an older single-split run may have incompatible rows -> Mitigate by reusing the existing valid-row filter before treating checkpoint rows as complete.
- `safegurd_test` spelling ambiguity -> Mitigate by implementing against the existing repo split name `safeguard_test` and documenting that mapping.

## Migration Plan

1. Replace single `SPLIT` configuration with `MAIN_SPLITS`, `EXTERNAL_DATASETS`, and optional per-target `LIMIT` behavior.
2. Add `PREDICTIONS_EXTERNAL_DIR` and path builders for main and external checkpoint/final parquet outputs.
3. Refactor split loading into target loaders that return an evaluation DataFrame and target metadata.
4. Move the runner into a function that accepts a target, performs checkpoint resume, classifies pending rows, writes the final parquet, and validates prediction columns.
5. Iterate all configured targets after model setup, printing per-target progress and a final output summary.
6. Verify notebook JSON structure locally and perform a small Colab/GPU smoke run with reduced limits for at least one main split and one external dataset.

Rollback is to restore the previous notebook version that accepts a single `SPLIT`.

## Open Questions

- Should `LIMIT` apply globally to every target, or should the notebook expose separate `MAIN_LIMIT` and `EXTERNAL_LIMIT` values?
- Should one target failure stop the full notebook run, or should the notebook continue to later targets and summarize failures at the end?

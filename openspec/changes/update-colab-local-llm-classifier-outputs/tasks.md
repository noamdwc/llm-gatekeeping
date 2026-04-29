## 1. Notebook Configuration And Target Modeling

- [x] 1.1 Replace the single `SPLIT` configuration with `MAIN_SPLITS = ["train", "val", "test", "unseen_val", "unseen_test", "safeguard_test"]` and `EXTERNAL_DATASETS = ["deepset", "jackhhao"]`.
- [x] 1.2 Add Drive paths for main split inputs, main prediction outputs, and external prediction outputs.
- [x] 1.3 Add a small target representation or helper functions that derive the target key, input kind, progress label, checkpoint path, and final output path.
- [x] 1.4 Preserve a configurable row limit in a form that can apply to each target without changing the output naming contract.

## 2. Per-Target Data Loading

- [x] 2.1 Keep `train.parquet` loading for few-shot example construction before target iteration starts.
- [x] 2.2 Implement main split loading for `train`, `val`, `test`, `unseen_val`, `unseen_test`, and `safeguard_test` from `SPLITS_DIR`.
- [x] 2.3 Implement external dataset loading for `deepset` and `jackhhao` through existing config and external dataset helper code.
- [x] 2.4 Normalize every target input DataFrame to include `sample_id` and the configured text column used by `classify_text`.
- [x] 2.5 Validate target input columns with separate rules for main splits and binary-only external datasets.

## 3. Runner Refactor

- [x] 3.1 Move checkpoint resume, pending-row selection, classification, checkpoint writing, final parquet writing, and output validation into a per-target runner function.
- [x] 3.2 Ensure checkpoints are read and written per target and never mix rows across splits or external datasets.
- [x] 3.3 Preserve existing classifier output row fields, local Transformers provider/model metadata, parse-failure behavior, and token logprob serialization.
- [x] 3.4 Iterate all configured targets and collect success or failure metadata for a final batch summary.
- [x] 3.5 Print target-level progress including target key, pending rows, completed rows, and final output path.

## 4. Output Validation And Verification

- [x] 4.1 Update output validation so every final parquet requires `sample_id`, the expected `llm_*` and `clf_*` prediction columns, `llm_stages_run == 1`, and no `judge_*` columns.
- [x] 4.2 Ensure external output validation does not require main-only hierarchical ground-truth columns.
- [x] 4.3 Load the notebook as JSON locally and verify the configured target names and output path logic are present.
- [ ] 4.4 Run or document a small Colab/GPU smoke test with a low per-target limit for at least one main split and one external dataset.
- [ ] 4.5 Inspect at least one generated main parquet and one external parquet to confirm prediction schema, provider/model metadata, parse status, confidence, raw response text, and token logprob fields.

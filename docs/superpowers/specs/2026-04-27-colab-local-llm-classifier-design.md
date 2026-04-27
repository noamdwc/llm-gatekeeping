# Colab Local LLM Classifier Notebook Design

## Purpose

Create a Google Colab notebook that runs the LLM classifier stage locally on Colab GPU hardware through a vLLM OpenAI-compatible server, instead of sending classifier calls to NVIDIA NIM.

The notebook produces a classifier-only parquet artifact that can be downloaded and used later in the local research pipeline. Judge execution and downstream compatibility fixes are intentionally out of scope for this change.

## Scope

In scope:

- Add a Colab notebook modeled after `notebooks/colab_train_deberta.ipynb`.
- Mount Drive and use Drive-backed project data paths.
- Clone or update the repo in Colab.
- Install project dependencies and vLLM.
- Start a local vLLM OpenAI-compatible server.
- Run the existing classifier prompt flow against the local vLLM server.
- Write classifier-only prediction parquets to Drive.
- Support checkpointing and resume by `sample_id`.
- Include notebook smoke-test and output-validation cells.

Out of scope:

- Running the judge stage in Colab.
- Calling NVIDIA NIM from the notebook.
- Adding empty `judge_*` columns.
- Updating downstream code that currently assumes judge columns exist.
- Changing DVC stages or local pipeline defaults.

## Notebook Location

Create:

`notebooks/colab_local_llm_classifier.ipynb`

The notebook should follow the structure and style of `notebooks/colab_train_deberta.ipynb`: configuration cells first, then repo setup, dependency setup, validation, execution, and output checks.

## Configuration

The first executable cell should define user-editable settings:

- `DRIVE_ROOT`: default `/content/drive/MyDrive/data/llm-gatekeeping`
- `REPO_URL`: default `https://github.com/noamdwc/llm-gatekeeping.git`
- `REPO_DIR`: default `/content/llm-gatekeeping`
- `BRANCH`: default `zero_shot_nlp_attack`, matching the existing Colab notebook
- `SPLIT`: default `val`
- `LIMIT`: default `5` for smoke testing; set to `None` for a full split
- `MODEL_ID`: default `meta/llama-3.1-8b-instruct`, matching the current classifier model in `configs/default.yaml`
- `TENSOR_PARALLEL_SIZE`
- `GPU_MEMORY_UTILIZATION`
- `MAX_MODEL_LEN`
- `BATCH_SIZE`
- `CHECKPOINT_EVERY`
- `OUTPUT_SUFFIX`: default `colab_local_classifier`

Drive paths:

- Splits: `{DRIVE_ROOT}/data/processed/splits`
- Predictions: `{DRIVE_ROOT}/data/processed/predictions`
- Hugging Face cache: `{DRIVE_ROOT}/cache/huggingface`
- vLLM cache/model storage should use the same Drive-backed Hugging Face cache where practical.

## Data Inputs

The notebook reads:

- `{SPLITS_DIR}/train.parquet` for static few-shot examples.
- `{SPLITS_DIR}/{SPLIT}.parquet` for evaluation.

Required columns in eval split:

- `modified_sample`
- any existing ground-truth/context columns should be preserved when present.

Required columns in train split:

- the configured text column from `configs/default.yaml`
- the configured label column from `configs/default.yaml`

The notebook should fail early with a clear error if required files or columns are missing.

## Inference Flow

The notebook should reuse existing project prompt and normalization semantics instead of duplicating prompt text manually.

Per row:

1. Build static few-shot examples using `build_few_shot_examples(train_df, cfg)`.
2. Build classifier messages using `build_classifier_messages(text, few_shot_messages)`.
3. Send a chat completion request to local vLLM at `http://127.0.0.1:8000/v1`.
4. Request JSON output with `response_format={"type": "json_object"}`.
5. Request logprobs and top-logprobs when vLLM supports them for the selected model/server version.
6. Parse the response JSON.
7. Normalize:
   - label to `benign`, `adversarial`, or `uncertain`
   - confidence to `[0, 1]`
   - unknown `nlp_attack_type` to `none`
   - binary label so only `benign` remains benign; `adversarial` and `uncertain` become adversarial
   - category using the same classifier category derivation logic as the existing LLM classifier
8. Append a classifier-only prediction row.

The notebook should run classifier calls only. It should set `llm_stages_run` to `1` for all produced rows.

## Output Contract

Final output path:

`{PREDICTIONS_DIR}/llm_predictions_{SPLIT}_{OUTPUT_SUFFIX}.parquet`

Checkpoint path:

`{PREDICTIONS_DIR}/llm_checkpoint_{SPLIT}_{OUTPUT_SUFFIX}.parquet`

The output must include classifier/final LLM columns:

- `sample_id`
- preserved ground-truth/context columns from the input split when present
- `llm_pred_binary`
- `llm_pred_raw`
- `llm_pred_category`
- `llm_conf_binary`
- `llm_evidence`
- `llm_stages_run`
- `llm_provider_name`
- `llm_model_name`
- `llm_raw_response_text`
- `llm_parse_success`
- `clf_label`
- `clf_category`
- `clf_confidence`
- `clf_evidence`
- `clf_nlp_attack_type`
- `clf_provider_name`
- `clf_model_name`
- `clf_raw_response_text`
- `clf_parse_success`
- `clf_token_logprobs`

The output must not include any `judge_*` columns.

`llm_provider_name` and `clf_provider_name` should identify the local backend, for example `vllm-local`. `llm_model_name` and `clf_model_name` should use `MODEL_ID`.

## Checkpointing and Resume

The notebook should compute `sample_id` using the existing `build_sample_id` helper.

If the checkpoint parquet exists, the notebook should:

- read completed `sample_id` values,
- skip completed rows,
- append new results at `CHECKPOINT_EVERY` intervals,
- write the final parquet after all rows finish.

This resume behavior should not require judge columns or downstream pipeline changes.

## Error Handling

The notebook should fail early when:

- Colab GPU is unavailable,
- required split files are missing,
- required split columns are missing,
- vLLM server cannot be reached before dataset inference starts.

For per-row classifier failures, the notebook should emit a parse-failure row rather than stopping the full run, as long as the local server remains reachable. Parse-failure rows should contain:

- adversarial binary fallback,
- low/default confidence,
- raw response text when available,
- `llm_parse_success=False`,
- `clf_parse_success=False`.

If the vLLM server becomes unreachable during inference, the notebook should stop with a clear message so the existing checkpoint can be resumed after restart.

## Verification Cells

The notebook should include validation cells that:

- run a small smoke test with `LIMIT=5`,
- print output row count,
- print classifier label distribution,
- assert expected classifier columns exist,
- assert no columns start with `judge_`,
- assert all `llm_stages_run` values are `1`,
- read the final parquet back from Drive.

## Implementation Notes

Prefer reusing existing project helpers where practical:

- `load_config`
- `build_sample_id`
- `build_few_shot_examples`
- `build_classifier_messages`
- classifier confidence/category normalization logic from `HierarchicalLLMClassifier`

Do not change downstream consumers in this change. If later pipeline code needs to tolerate missing judge columns, handle that as a separate scoped change.

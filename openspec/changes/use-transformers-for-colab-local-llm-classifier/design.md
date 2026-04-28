## Context

`notebooks/colab_local_llm_classifier.ipynb` runs classifier-only prediction generation for a local Hugging Face model on Colab. The current notebook installs vLLM and OpenAI client packages, starts `vllm.entrypoints.openai.api_server` as a background process, waits for `/v1/models`, then calls `client.chat.completions.create(...)` with JSON-mode and logprobs enabled.

The target workflow should keep the notebook self-contained in Colab but remove the server process. The notebook will load the tokenizer and causal language model directly with Transformers, format the existing chat messages into a prompt, generate the JSON classifier response, derive token logprobs from generation scores, and keep writing the same prediction/checkpoint parquet schema.

## Goals / Non-Goals

**Goals:**

- Replace vLLM/OpenAI server-client inference with direct Transformers inference inside the notebook process.
- Preserve existing dataset loading, few-shot prompting, checkpointing, output validation, and parquet column names.
- Keep classifier result normalization compatible with the current downstream fields, including parse-failure rows.
- Capture enough generated-token logprob information for the existing `clf_token_logprobs` field and confidence normalization path.
- Make model-load and row-level inference failures actionable without relying on a separate vLLM log file.

**Non-Goals:**

- Rewriting the reusable `src.llm_classifier` package.
- Changing the classifier prompt contract, output schema, or downstream analysis workflow.
- Adding distributed or multi-GPU inference support.
- Matching vLLM tokenization, sampling, or logprob payloads byte-for-byte.

## Decisions

1. Use `AutoTokenizer` and `AutoModelForCausalLM` directly.

   Rationale: Transformers is the standard Colab path, avoids a background server, and keeps all exceptions in the notebook output. The implementation should load with `device_map='auto'`, prefer CUDA when available, use cache paths already configured by the notebook, and choose an appropriate dtype such as `torch.bfloat16` when the GPU supports it or `torch.float16` otherwise.

   Alternative considered: keep vLLM and only simplify startup. This leaves the main operational complexity in place and does not satisfy the dependency change.

2. Convert chat messages to a model prompt with tokenizer chat templates when available.

   Rationale: `build_classifier_messages(...)` already produces chat-style messages. `tokenizer.apply_chat_template(..., add_generation_prompt=True)` preserves model-specific formatting for instruct models. If the tokenizer has no chat template, the notebook should use a small fallback formatter so the workflow still runs.

   Alternative considered: hand-roll a single prompt for all models. That is simpler but more likely to degrade outputs for chat-tuned models.

3. Generate with `return_dict_in_generate=True` and `output_scores=True`.

   Rationale: This provides the generated token IDs and per-step logits needed to build a compact token logprob list. The notebook can compute selected-token logprobs via `log_softmax` on each score tensor and store token/text/logprob entries in `clf_token_logprobs`.

   Alternative considered: omit logprobs. That would simplify implementation but would remove useful diagnostics and change the output semantics more than necessary.

4. Keep provider metadata explicit as a local Transformers provider.

   Rationale: Existing rows identify the inference provider as `vllm-local`. The migrated notebook should update `_provider_name`, `clf_provider_name`, and `llm_provider_name` to a stable value such as `transformers-local` so consumers can distinguish old and new runs while retaining the same columns.

   Alternative considered: keep `vllm-local` for compatibility. That would be misleading once vLLM is removed.

5. Replace server health checks with direct exception handling.

   Rationale: There is no background service to probe. Model load failures should fail the setup cell. Row inference failures should keep the existing behavior of writing a parse-failure row when feasible, with the row exception printed in notebook output.

   Alternative considered: add a synthetic health function. That adds little value because the loaded model object is the health boundary.

## Risks / Trade-offs

- Transformers generation may produce slightly different JSON than vLLM for the same model and sampling settings -> Preserve the existing parser/normalizer and checkpoint invalid-row filtering so malformed responses degrade to explicit parse failures.
- GPU memory use may be higher in the notebook process than users expect -> Load with `device_map='auto'`, clear setup-only server variables, and keep generation batch size at one row.
- Some models may not define a chat template -> Provide a deterministic fallback chat formatter.
- Transformers score tensors can be large if retained too long -> Build logprob summaries immediately for generated tokens and avoid storing raw logits.
- JSON-mode is not available as an API feature in Transformers -> Keep the prompt contract focused on JSON and rely on the existing JSON parser/failure path.

## Migration Plan

1. Update the dependency cell to install/import Transformers support and remove notebook-only vLLM/OpenAI packages.
2. Replace vLLM startup and model-list cells with a model/tokenizer loading cell and a short model readiness printout.
3. Replace the OpenAI client helper with prompt formatting, generation, token logprob extraction, and provider metadata updates.
4. Remove vLLM health checks from the row runner and keep parse-failure row creation for row-level exceptions.
5. Run a small `LIMIT` classification pass in Colab or a GPU runtime and confirm output validation passes.

Rollback is to restore the previous notebook version from git if direct Transformers inference is not viable for the chosen model/runtime.

## Open Questions

- Which exact model family will be used most often in Colab, and does it require a custom prompt fallback beyond `apply_chat_template`?
- Should the implementation add optional quantized loading, or keep the first migration focused on full-precision/half-precision Transformers loading?

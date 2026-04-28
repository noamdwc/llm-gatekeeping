## 1. Notebook Dependency And Setup Migration

- [x] 1.1 Update `notebooks/colab_local_llm_classifier.ipynb` title/description and dependency cell to remove notebook-only `vllm` and `openai` installs and ensure Transformers dependencies are available.
- [x] 1.2 Replace the vLLM startup cell with a Transformers setup cell that validates CUDA, creates required cache/output directories, loads `AutoTokenizer` and `AutoModelForCausalLM`, chooses an appropriate dtype/device placement, and prints model readiness.
- [x] 1.3 Remove the vLLM model-list health-check cell and any unused vLLM configuration variables, imports, log paths, subprocess handling, and HTTP polling.

## 2. Transformers Classification Helpers

- [x] 2.1 Replace the OpenAI client initialization with local helper functions for chat prompt formatting using `tokenizer.apply_chat_template` when available and a deterministic fallback formatter otherwise.
- [x] 2.2 Implement local generation in `classify_text` using the loaded Transformers model, existing prompt messages, configured temperature, and `max_tokens_classifier`.
- [x] 2.3 Decode only generated tokens into `raw_response_text`, parse the classifier JSON through the existing parser, and preserve normalization behavior.
- [x] 2.4 Derive compact generated-token logprob entries from `generate(..., return_dict_in_generate=True, output_scores=True)` results and pass them into the existing output field.
- [x] 2.5 Update provider metadata from `vllm-local` to a stable local Transformers provider name while retaining `MODEL_ID` as model metadata.

## 3. Runner And Error Handling

- [x] 3.1 Replace row-level vLLM server health checks with direct exception logging and the existing parse-failure row creation path.
- [x] 3.2 Ensure checkpoint resume, valid prediction filtering, final parquet writing, and output validation still operate on the existing prediction columns.
- [x] 3.3 Remove any remaining references to `VLLM_BASE_URL`, `VLLM_LOG_PATH`, `vllm_proc`, `requests` health checks, or OpenAI response objects.

## 4. Verification

- [x] 4.1 Run a notebook-structure validation by loading the `.ipynb` as JSON and confirming it contains no vLLM/OpenAI client setup references.
- [ ] 4.2 Run or document a small Colab/GPU smoke test with `LIMIT` set to a small value and verify output validation passes.
- [ ] 4.3 Inspect a generated parquet sample to confirm provider/model metadata, parse status, confidence, raw response text, and token logprob fields are populated or null according to the spec.

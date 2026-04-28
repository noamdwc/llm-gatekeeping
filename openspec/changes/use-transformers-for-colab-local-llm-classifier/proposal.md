## Why

The Colab local LLM classifier notebook currently depends on a vLLM OpenAI-compatible server, which adds startup complexity and makes the workflow harder to run in standard Colab environments. Switching the notebook to Hugging Face Transformers keeps inference local while reducing moving parts and aligning with Colab's common model-loading path.

## What Changes

- Replace the notebook's vLLM installation, server startup, health checks, and OpenAI client calls with direct Transformers model and tokenizer loading.
- Preserve the classifier-only parquet output shape and the existing prompt, label, confidence, and metadata semantics where feasible.
- Compute classification outputs and logprob-derived confidence directly from Transformers generation scores.
- Update notebook status text, logging, and failure handling so users can diagnose model loading and inference errors without a background server log.
- Remove runtime dependencies on `vllm` and `openai` from the notebook.

## Capabilities

### New Capabilities
- `colab-transformers-local-llm-classifier`: Covers the Colab notebook's ability to run local LLM classifier inference through Transformers and write classifier prediction parquet outputs.

### Modified Capabilities

## Impact

- Affected notebook: `notebooks/colab_local_llm_classifier.ipynb`
- Runtime dependencies change from `vllm`/`openai` server-client inference to `transformers` direct model inference.
- Colab GPU memory behavior may change because the model is loaded in the notebook process instead of a vLLM subprocess.
- The generated prediction parquet files should remain compatible with downstream analysis that consumes classifier-only predictions.

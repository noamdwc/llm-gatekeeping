# Known Gotchas

## The LLM DVC Stage Is Frozen By Default
`dvc.yaml` includes an `llm_classifier` stage, but normal `dvc repro` does not automatically make that stage behave like a fresh end-to-end LLM run. `run_llm.sh` exists specifically to unfreeze and re-freeze it.

## ML Scope Is Narrower Than The Full Task
`src/ml_classifier/ml_baseline.py` intentionally filters out NLP attacks during training. If you evaluate the ML model on all rows, poor NLP performance is not necessarily a regression.

## Research Hybrid And Runtime Hybrid Are Different Surfaces
- `src/hybrid_router.py` is the direct runtime router
- `src/research.py` recomputes hybrid decisions from saved artifacts

They should be conceptually aligned, but they are not literally the same code path.

## Cache State Matters
LLM chat completions are cached in `.cache/llm/` by normalized request payload. Prompt changes that do not alter the effective request shape can lead to confusing reuse during debugging.

## Provider Defaults May Surprise You
`src/llm_provider.py` defaults `LLM_PROVIDER` to `nim`. If `NVIDIA_API_KEY` is missing, LLM flows fail unless you explicitly switch providers.

## External Datasets Are Binary-Only
`src/eval_external.py` fills `label_category` and `label_type` with the binary label because external datasets do not have the same hierarchy. Do not over-interpret those fields.

## Synthetic Benign Mode Can Fail Early
If `benign.synthetic.enabled` is true in config but the synthetic benign parquet does not exist, `src/preprocess.py` raises immediately.

## Artifact Preconditions Are Strict
Several entrypoints assume prior pipeline outputs exist:
- `run_inference.sh` expects saved split parquets and the ML model
- external hybrid research expects precomputed external LLM prediction artifacts

## There Is No Visible CI Or Container Story
I could not confirm CI workflows, Docker setup, or standardized devcontainer support. Local environment drift is therefore more likely than in a heavily tooled repo.

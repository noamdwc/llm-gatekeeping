# Margin Calibration Code Path

- Split: `val`
- Canonical experiment entrypoint: `python -m src.llm_classifier.llm_classifier --split test --research`
- Canonical hybrid routing entrypoint: `python -m src.research --split test`
- Canonical report entrypoint: `python -m src.cli.eval_new --split test`

## Verified Executed Path

1. `src.llm_classifier.llm_classifier.HierarchicalLLMClassifier._call_llm()` performs the API call and captures raw response text, parse status, and token logprobs.
2. `src.llm_classifier.llm_classifier.HierarchicalLLMClassifier.predict()` normalizes classifier/judge output and emits persisted LLM prediction rows.
3. `src.research.compute_hybrid_routing()` is the code path used by the current DVC research experiments. It applies ML fast-path routing, LLM abstain handling, and the configured margin policy.
4. `src.logprob_margin.extract_preferred_margin_features_from_row()` selects the label-start token position and computes the preferred margin (judge first, classifier fallback).
5. `src.margin_trace.build_margin_trace()` writes row-level margin traces for downstream calibration analysis.
6. `src.cli.eval_new` consumes `data/processed/research/research_{split}.parquet` for markdown reports; notebook-only sweeps are no longer the primary analysis path.

## Duplicate / Secondary Paths

- `src.hybrid_router.HybridRouter` is still used for live CLI prediction and some external evaluation flows, but it is not the canonical code path for the current DVC threshold experiments.
- Margin extraction and policy logic are shared through `src.logprob_margin` to avoid stale duplicate implementations.

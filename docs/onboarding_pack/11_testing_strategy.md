# Testing Strategy

## What Exists
The repo has a substantial pytest suite under `tests/`. At inspection time, `pytest --collect-only -q` collected 339 tests.

Main covered areas include:
- splits: `tests/test_build_splits.py`
- metrics/reporting: `tests/test_evaluate.py`
- hybrid routing: `tests/test_hybrid_router.py`
- LLM cache and classifier behavior: `tests/test_llm_cache.py`, `tests/test_llm_classifier.py`
- research artifact assembly: `tests/test_research.py`, `tests/test_research_external.py`
- synthetic benign generation and validation: `tests/test_synthetic_benign.py`, `tests/test_validators.py`
- CLI ML inference report path: `tests/test_cli_infer_split.py`

## Likely Testing Style
The suite appears mostly unit-oriented and artifact-schema-oriented:
- inputs are small synthetic DataFrames
- external providers are likely mocked or bypassed
- tests focus on routing logic, output fields, and metric behavior

That is good for fast iteration, but it does not fully replace end-to-end provider-backed validation.

## Good Test Files To Read Early
- `tests/test_hybrid_router.py`
  - best for learning intended route decisions
- `tests/test_llm_classifier.py`
  - best for learning expected classifier output structure
- `tests/test_research.py`
  - best for learning artifact merge and strict hybrid assumptions
- `tests/test_research_external.py`
  - best for learning external dataset handling

## Practical Test Workflow
Start with:
```bash
pytest --collect-only -q
pytest tests/test_evaluate.py
pytest tests/test_hybrid_router.py
pytest tests/test_research.py
```

Then use targeted tests for the code you touch.

## What Still Likely Needs Manual Verification
- Real provider integration with NIM/OpenAI
- Prompt-quality changes in `src/llm_classifier/prompts.py`
- Cache invalidation behavior after prompt or provider changes
- Full DVC stage interactions and artifact freshness
- API-token-cost-sensitive paths

## Suggested Safety Rules
- If you change routing logic, run `tests/test_hybrid_router.py` and `tests/test_research.py`
- If you change LLM output schema or prompt contracts, run `tests/test_llm_classifier.py` and `tests/test_research_external.py`
- If you change metrics or report rendering, run `tests/test_evaluate.py` and `tests/test_cli_infer_split.py`
- If you change data prep or splits, run `tests/test_preprocess.py` and `tests/test_build_splits.py`

# First Week Onboarding

## Day 1: Understand The Product And Repo Shape
Read:
- `README.md`
- `configs/default.yaml`
- `dvc.yaml`
- `docs/onboarding_pack/00_repo_summary.md`
- `docs/onboarding_pack/01_architecture.md`

Run:
```bash
ls
rg "python -m src" README.md dvc.yaml run_*.sh src
```

Trace:
- the DVC stage graph
- the label hierarchy
- where outputs are written under `data/processed/` and `reports/`

Questions for teammates:
- Which report under `reports/research/` is considered the latest decision-making artifact?
- Which workflows are used most often: DVC research, CLI inference, or both?

## Day 2: Run Locally And Inspect Entrypoints
Read:
- `run_inference.sh`
- `src/cli/predict.py`
- `src/cli/infer_split.py`

Run:
```bash
pip install -r requirements.txt
pytest --collect-only -q
./run_inference.sh --mode ml --split test
```

Trace:
- how config is loaded
- what artifacts must exist before inference works

Questions for teammates:
- Is NIM still the standard provider, or is OpenAI used more often now?
- Are `.env` conventions stable across machines?

## Day 3: Follow One Core Flow End To End
Read:
- `src/preprocess.py`
- `src/build_splits.py`
- `src/ml_classifier/ml_baseline.py`
- `src/research.py`

Run:
```bash
python -m src.preprocess
python -m src.build_splits
```

Trace:
- one row from raw dataset to `full_dataset.parquet`
- how it lands in a split
- how `sample_id` and `prompt_hash` differ

Questions for teammates:
- Are held-out attacks in config still the intended generalization benchmark?

## Day 4: Inspect Config And Testing
Read:
- `pytest.ini`
- `tests/test_hybrid_router.py`
- `tests/test_llm_classifier.py`
- `tests/test_research.py`
- `docs/onboarding_pack/12_config_and_env.md`

Run:
```bash
pytest tests/test_hybrid_router.py tests/test_research.py
```

Trace:
- what behavior is strongly locked by tests
- where tests rely on mocks rather than real providers

Questions for teammates:
- Which tests are considered critical smoke tests before merging?
- Are there known flaky tests or slow tests?

## Day 5: Make A Tiny Safe Change
Suggested safe tasks:
- improve wording in a generated report
- add a missing test around metric formatting
- add comments to clarify hybrid route reasons

Read:
- `src/evaluate.py`
- `src/cli/eval_new.py`

Run:
```bash
pytest tests/test_evaluate.py tests/test_cli_infer_split.py
```

Questions for teammates:
- Which parts of report wording are user-facing versus internal-only?

## Days 6-7: Deepen Understanding
Read:
- `src/llm_classifier/llm_classifier.py`
- `src/llm_classifier/prompts.py`
- `src/embeddings.py`
- `src/llm_cache.py`
- `src/eval_external.py`
- `src/cli/research_external.py`

Run:
```bash
./run_synth.sh --category C --limit 20
python -m src.eval_external --dataset deepset --mode ml
```

Trace:
- LLM prompt construction
- judge invocation conditions
- cache key behavior
- external dataset label mapping and report generation

Questions for teammates:
- Which prompt changes are considered safe?
- Is the external dataset benchmark still actively used?
- How should cache invalidation be handled when prompts change?

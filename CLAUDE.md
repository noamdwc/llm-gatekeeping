# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with
code in this repository.

## Environment Setup

- **Python environment**: `/Users/noamc/miniconda3/envs/llm_gate/bin/python`
- **Activate**: `conda activate llm_gate`
- **Dependencies**: `pip install -r requirements.txt`
- **LLM provider**: Set `LLM_PROVIDER` to `nim` or `openai`; matching API keys live in `.env`.
- **HuggingFace auth**: `huggingface-cli login` because the source dataset requires access approval.
- **Experiment tracking**: wandb is optional and can be disabled with `--no-wandb`.

## Canonical Pipeline

The supported start-to-finish path is documented in `README.md` and uses:

1. DVC-produced local inputs.
2. Manual Colab local LLM classifier handoff.
3. Strict handoff validation.
4. Escalation-model training.
5. Selective judge calls.
6. Final verdict report generation.

Final documented artifact:

```bash
reports/pipeline_final_verdict_report.md
```

Run the core path with:

```bash
dvc repro build_splits
dvc repro ml_model
dvc repro deberta_model
dvc repro deberta_external@deepset
dvc repro deberta_external@jackhhao
# Run notebooks/colab_local_llm_classifier.ipynb and download the exact handoff parquets.
dvc repro -s validate_colab_handoff
dvc repro final_verdict_report
```

Do not bypass `validate_colab_handoff`. Missing or malformed Colab handoff
artifacts should fail with exact paths; there is no fallback to legacy hosted
LLM classifier outputs.

## Current Known Blocker

`validate_colab_handoff` currently fails on the downloaded Deepset Colab
classifier artifact because some rows have `llm_stages_run != 1`. Fix or
regenerate that artifact before expecting `dvc repro final_verdict_report` to
complete.

## Active DVC Stages

- `generate_synthetic_benign@B` through `generate_synthetic_benign@F`
- `preprocess`
- `build_splits`
- `ml_model`
- `deberta_model`
- `deberta_external@deepset`
- `deberta_external@jackhhao`
- `judge_colab_local_predictions@{test,unseen_test,safeguard_test}`
- `validate_colab_handoff`
- `train_escalating_model`
- `judge_colab_local_predictions_external@{deepset,jackhhao}`
- `final_verdict_report`

## Useful Module Commands

All modules run from the project root with `python -m ...`.

```bash
python -m src.preprocess
python -m src.build_splits
python -m src.ml_classifier.ml_baseline --research
python -m src.cli.deberta_classifier --research --no-wandb
python -m src.cli.eval_deberta_external --dataset deepset
python -m src.cli.validate_colab_handoff --config configs/default.yaml
python -m src.cli.train_escalating_model
python -m src.cli.final_verdict_report --config configs/default.yaml
python -m src.cli.generate_synthetic_benign --category B
```

Prediction CLI:

```bash
echo "text" | python -m src.cli.predict --mode ml --pretty
echo "text" | python -m src.cli.predict --mode llm --pretty
echo "text" | python -m src.cli.predict --mode hybrid --pretty
```

## Key Paths

```text
data/processed/
  full_dataset.parquet
  synthetic_benign/
  splits/
    train.parquet, val.parquet, test.parquet, unseen_val.parquet, unseen_test.parquet, safeguard_test.parquet
  models/
    ml_baseline.pkl
    escalating_model.pkl
  predictions/
    ml_predictions_*.parquet
    deberta_predictions_*.parquet
    llm_predictions_*_colab_local_classifier.parquet
    llm_predictions_*_colab_local_judged.parquet
  predictions_external/
    deberta_predictions_external_<dataset>.parquet
    llm_predictions_external_<dataset>_colab_local_classifier.parquet
    llm_predictions_external_<dataset>_colab_local_judged.parquet
  research/
    escalating_model_eval_*.parquet
    escalating_model_summary.csv

artifacts/deberta_classifier/
reports/
  colab_handoff_validation.json
  escalating_model_poc.md
  pipeline_final_verdict_report.md
```

## Architecture Notes

- Data splits are grouped by `prompt_hash`; variants of the same original prompt stay in the same split.
- Held-out attack families are excluded from train/val/test and evaluated through `unseen_val` and `unseen_test`.
- Config values live in `configs/default.yaml`.
- Output path constants live in `src/utils.py`.
- `src/eval_external.py`, `src/llm_classifier/`, `src/llm_provider.py`, `src/llm_cache.py`, `src/synthetic_benign.py`, and `src/logprob_margin.py` are still active dependencies and should not be removed as cleanup.

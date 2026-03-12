# Config And Env

## Main Config File
The main config file is `configs/default.yaml`.

Most modules call `src.utils.load_config()` and default to this file unless `--config` is provided.

## Important Config Sections

### `dataset`
Controls:
- Hugging Face dataset name and split
- input text column
- original text column
- raw attack label column

Used by:
- `src/preprocess.py`

### `labels`
Controls:
- Unicode attack names
- NLP attack names
- held-out attacks

Used by:
- `src/preprocess.py`
- `src/build_splits.py`
- hybrid/research code for Unicode-lane interpretation

### `splits`
Controls:
- train/val/test ratios
- random seed

Used by:
- `src/build_splits.py`

### `benign`
Controls:
- benign target count
- use of original prompts as benign seed
- synthetic benign generation settings and output path

Used by:
- `src/preprocess.py`
- `src/synthetic_benign.py`

### `llm`
Controls:
- model names
- temperature
- concurrency
- checkpointing/resume behavior
- judge confidence threshold
- logprob capture
- few-shot and embedding settings

Used by:
- `src/llm_classifier/llm_classifier.py`
- `src/embeddings.py`
- external research flows

### `ml`
Controls:
- char n-gram range
- max features
- logistic regression hyperparameters
- hyperparameter search settings
- binary calibration behavior

Used by:
- `src/ml_classifier/ml_baseline.py`

### `hybrid`
Controls:
- ML confidence threshold
- LLM confidence threshold

Used by:
- `src/hybrid_router.py`
- `src/research.py`
- external research/evaluation paths

### `evaluation`
Controls:
- calibration bin count

Used by:
- `src/evaluate.py`

### `external_datasets`
Controls:
- external Hugging Face dataset list
- text/label columns
- label maps

Used by:
- `dvc.yaml`
- `src/eval_external.py`
- `src/cli/research_external.py`

## Important Environment Variables

### `LLM_PROVIDER`
Values:
- `nim`
- `openai`

Default is `nim`, inferred from `src/llm_provider.py`.

### `NVIDIA_API_KEY`
Required for NIM provider flows.

### `OPENAI_API_KEY`
Required for OpenAI provider flows.

### `SKIP_LLM`
Relevant to LLM/external flow control. Test coverage in `tests/test_research_external.py` suggests CLI behavior can override or inherit it.

## Config Precedence
Likely order:
1. explicit `--config` argument where supported
2. default `configs/default.yaml`
3. environment variables for provider/API behavior

There is no evidence of a layered config system beyond that.

## Best Places To Inspect Config Usage
- `src/utils.py`
- `src/preprocess.py`
- `src/ml_classifier/ml_baseline.py`
- `src/llm_classifier/llm_classifier.py`
- `src/hybrid_router.py`
- `src/eval_external.py`

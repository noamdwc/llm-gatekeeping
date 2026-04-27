# Colab Local LLM Classifier Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a Colab notebook that runs the current classifier model locally through vLLM and writes classifier-only prediction parquets without `judge_*` columns.

**Architecture:** The notebook is self-contained and mirrors `notebooks/colab_train_deberta.ipynb`: configure Drive paths, clone/update the repo, install dependencies, start a local vLLM OpenAI-compatible server, run classifier-only inference, checkpoint by `sample_id`, and validate the saved parquet. A local pytest file validates notebook structure and the classifier-only output contract without requiring Colab, GPU, or vLLM.

**Tech Stack:** Google Colab, vLLM OpenAI-compatible server, OpenAI Python client, pandas/pyarrow parquet, existing `src.llm_classifier` prompt helpers, pytest notebook-JSON structural tests.

---

## File Structure

- Create `notebooks/colab_local_llm_classifier.ipynb`: the user-facing Colab notebook.
- Create `tests/test_colab_local_llm_classifier_notebook.py`: local structural tests for the notebook JSON and output contract.
- Do not modify downstream consumers, DVC stages, or `src/llm_classifier/llm_classifier.py`.

## Task 1: Notebook Structural Contract Tests

**Files:**
- Create: `tests/test_colab_local_llm_classifier_notebook.py`

- [ ] **Step 1: Write the failing notebook contract tests**

Create `tests/test_colab_local_llm_classifier_notebook.py` with this content:

```python
"""Structural tests for the Colab local LLM classifier notebook."""

from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "colab_local_llm_classifier.ipynb"


def _notebook() -> dict:
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def _all_source() -> str:
    nb = _notebook()
    return "\n".join(
        "".join(cell.get("source", []))
        for cell in nb["cells"]
    )


def test_notebook_exists_and_targets_colab_gpu():
    nb = _notebook()

    assert nb["nbformat"] == 4
    assert nb["metadata"]["accelerator"] == "GPU"
    assert nb["metadata"]["kernelspec"]["name"] == "python3"


def test_notebook_uses_current_classifier_model_and_vllm_backend():
    source = _all_source()

    assert "MODEL_ID = 'meta/llama-3.1-8b-instruct'" in source
    assert "python -m vllm.entrypoints.openai.api_server" in source
    assert "VLLM_BASE_URL = 'http://127.0.0.1:8000/v1'" in source
    assert "api_key='EMPTY'" in source


def test_notebook_reuses_project_classifier_helpers():
    source = _all_source()

    assert "from src.llm_classifier.llm_classifier import build_few_shot_examples" in source
    assert "from src.llm_classifier.prompts import build_classifier_messages" in source
    assert "from src.utils import build_sample_id, load_config" in source


def test_output_contract_is_classifier_only():
    source = _all_source()

    expected_columns = [
        "llm_pred_binary",
        "llm_pred_raw",
        "llm_pred_category",
        "llm_conf_binary",
        "llm_evidence",
        "llm_stages_run",
        "llm_provider_name",
        "llm_model_name",
        "llm_raw_response_text",
        "llm_parse_success",
        "clf_label",
        "clf_category",
        "clf_confidence",
        "clf_evidence",
        "clf_nlp_attack_type",
        "clf_provider_name",
        "clf_model_name",
        "clf_raw_response_text",
        "clf_parse_success",
        "clf_token_logprobs",
    ]
    for column in expected_columns:
        assert column in source

    match = re.search(r"PREDICTION_COLUMNS = \[(.*?)\]", source, flags=re.S)
    assert match is not None
    assert "judge_" not in match.group(1)
    assert "assert not any(col.startswith('judge_')" in source


def test_notebook_has_checkpoint_and_resume_logic():
    source = _all_source()

    assert "CHECKPOINT_EVERY = 50" in source
    assert "CHECKPOINT_PATH" in source
    assert "completed_ids" in source
    assert "sample_id" in source
    assert "to_parquet" in source
```

- [ ] **Step 2: Run the failing tests**

Run:

```bash
pytest tests/test_colab_local_llm_classifier_notebook.py -q
```

Expected: FAIL because `notebooks/colab_local_llm_classifier.ipynb` does not exist.

- [ ] **Step 3: Commit the failing tests**

Run:

```bash
git add tests/test_colab_local_llm_classifier_notebook.py
git commit -m "test: add Colab local LLM classifier notebook contract"
```

Expected: commit succeeds with only the new test file staged.

## Task 2: Notebook Skeleton, Configuration, and Repo Setup

**Files:**
- Create: `notebooks/colab_local_llm_classifier.ipynb`
- Modify: `tests/test_colab_local_llm_classifier_notebook.py` only if a string assertion needs an exact formatting adjustment after writing valid notebook JSON.

- [ ] **Step 1: Create the notebook with configuration and setup cells**

Create `notebooks/colab_local_llm_classifier.ipynb` as valid nbformat 4 JSON. Use `notebooks/colab_train_deberta.ipynb` as the formatting reference. The notebook must start with these cells:

```markdown
# Colab Local LLM Classifier

Runs the classifier stage locally on Colab through a vLLM OpenAI-compatible server and writes classifier-only prediction parquet files to Drive.
```

Configuration code cell:

```python
from google.colab import drive

drive.mount('/content/drive')

DRIVE_ROOT = '/content/drive/MyDrive/data/llm-gatekeeping'
REPO_URL = 'https://github.com/noamdwc/llm-gatekeeping.git'
REPO_DIR = '/content/llm-gatekeeping'
BRANCH = 'zero_shot_nlp_attack'

SPLIT = 'val'
LIMIT = 5  # Set to None for the full split.
MODEL_ID = 'meta/llama-3.1-8b-instruct'
TENSOR_PARALLEL_SIZE = 1
GPU_MEMORY_UTILIZATION = 0.90
MAX_MODEL_LEN = 4096
BATCH_SIZE = 1
CHECKPOINT_EVERY = 50
OUTPUT_SUFFIX = 'colab_local_classifier'

HF_CACHE_DIR = f'{DRIVE_ROOT}/cache/huggingface'
SPLITS_DIR = f'{DRIVE_ROOT}/data/processed/splits'
PREDICTIONS_DIR = f'{DRIVE_ROOT}/data/processed/predictions'
CHECKPOINT_PATH = f'{PREDICTIONS_DIR}/llm_checkpoint_{SPLIT}_{OUTPUT_SUFFIX}.parquet'
OUTPUT_PATH = f'{PREDICTIONS_DIR}/llm_predictions_{SPLIT}_{OUTPUT_SUFFIX}.parquet'
VLLM_BASE_URL = 'http://127.0.0.1:8000/v1'

import os

os.environ['HF_HOME'] = HF_CACHE_DIR
os.environ['HF_HUB_CACHE'] = f'{HF_CACHE_DIR}/hub'
os.environ['TRANSFORMERS_CACHE'] = f'{HF_CACHE_DIR}/transformers'
os.environ['HF_DATASETS_CACHE'] = f'{HF_CACHE_DIR}/datasets'
print(f'Hugging Face cache: {HF_CACHE_DIR}')
print(f'Output path: {OUTPUT_PATH}')
```

Repo setup code cell:

```python
import os
import subprocess

if not os.path.exists(REPO_DIR):
    subprocess.run(['git', 'clone', REPO_URL, REPO_DIR], check=True)
else:
    print(f'Using existing repo at {REPO_DIR}')

os.chdir(REPO_DIR)
print('Repo:', os.getcwd())

subprocess.run(['git', 'fetch', 'origin', BRANCH], check=True)
subprocess.run(['git', 'checkout', BRANCH], check=True)
subprocess.run(['git', 'pull', '--ff-only', 'origin', BRANCH], check=True)
```

Dependency setup code cell:

```python
%pip install -q -r requirements.txt
%pip install -q "vllm>=0.6.0" "openai>=1.0.0"
```

- [ ] **Step 2: Run JSON validity check**

Run:

```bash
python -m json.tool notebooks/colab_local_llm_classifier.ipynb >/tmp/colab_local_llm_classifier.json
```

Expected: PASS with no output and a formatted JSON file written under `/tmp`.

- [ ] **Step 3: Run notebook contract tests**

Run:

```bash
pytest tests/test_colab_local_llm_classifier_notebook.py -q
```

Expected: FAIL only on assertions for cells not implemented yet, such as project helper imports, output contract, and checkpoint logic. It should no longer fail because the notebook file is missing.

- [ ] **Step 4: Commit the notebook skeleton**

Run:

```bash
git add notebooks/colab_local_llm_classifier.ipynb tests/test_colab_local_llm_classifier_notebook.py
git commit -m "feat: add Colab local classifier notebook skeleton"
```

Expected: commit includes the notebook skeleton and any exact-string test adjustment needed for valid notebook formatting.

## Task 3: vLLM Startup, Input Validation, and Classifier Helpers

**Files:**
- Modify: `notebooks/colab_local_llm_classifier.ipynb`

- [ ] **Step 1: Add vLLM server startup and health-check cells**

Add a code cell after dependency setup:

```python
from pathlib import Path
import subprocess
import time

import requests
import torch

if not torch.cuda.is_available():
    raise RuntimeError('Colab GPU is unavailable. Enable Runtime > Change runtime type > GPU.')

Path(PREDICTIONS_DIR).mkdir(parents=True, exist_ok=True)
Path(HF_CACHE_DIR).mkdir(parents=True, exist_ok=True)

server_cmd = [
    'python', '-m', 'vllm.entrypoints.openai.api_server',
    '--model', MODEL_ID,
    '--host', '127.0.0.1',
    '--port', '8000',
    '--tensor-parallel-size', str(TENSOR_PARALLEL_SIZE),
    '--gpu-memory-utilization', str(GPU_MEMORY_UTILIZATION),
    '--max-model-len', str(MAX_MODEL_LEN),
]

print('Starting vLLM:', ' '.join(server_cmd))
vllm_proc = subprocess.Popen(server_cmd)

deadline = time.time() + 600
last_error = None
while time.time() < deadline:
    try:
        response = requests.get(f'{VLLM_BASE_URL}/models', timeout=5)
        if response.ok:
            print('vLLM server is ready')
            break
        last_error = f'HTTP {response.status_code}: {response.text[:200]}'
    except Exception as exc:
        last_error = repr(exc)
    if vllm_proc.poll() is not None:
        raise RuntimeError(f'vLLM exited early with code {vllm_proc.returncode}')
    time.sleep(5)
else:
    raise RuntimeError(f'vLLM server did not become ready: {last_error}')
```

Add a model list check cell:

```python
import requests

models_response = requests.get(f'{VLLM_BASE_URL}/models', timeout=10)
models_response.raise_for_status()
print(models_response.json())
```

- [ ] **Step 2: Add input validation and helper imports**

Add a code cell:

```python
from pathlib import Path

import pandas as pd

from src.llm_classifier.constants import NLP_TYPES
from src.llm_classifier.llm_classifier import HierarchicalLLMClassifier, build_few_shot_examples
from src.llm_classifier.prompts import build_classifier_messages
from src.utils import build_sample_id, load_config

cfg = load_config()
text_col = cfg['dataset']['text_col']
label_col = cfg['dataset']['label_col']

train_path = Path(SPLITS_DIR) / 'train.parquet'
eval_path = Path(SPLITS_DIR) / f'{SPLIT}.parquet'
if not train_path.exists():
    raise FileNotFoundError(f'Missing train split: {train_path}')
if not eval_path.exists():
    raise FileNotFoundError(f'Missing eval split: {eval_path}')

train_df = pd.read_parquet(train_path)
eval_df = pd.read_parquet(eval_path)

for path, df, required in [
    (train_path, train_df, {text_col, label_col}),
    (eval_path, eval_df, {text_col}),
]:
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f'{path} missing required columns: {sorted(missing)}')

if LIMIT is not None:
    eval_df = eval_df.head(int(LIMIT)).copy()
else:
    eval_df = eval_df.copy()

eval_df['sample_id'] = eval_df[text_col].apply(build_sample_id)
few_shot, used_ids = build_few_shot_examples(train_df, cfg)
print(f'Train rows: {len(train_df):,}; eval rows: {len(eval_df):,}; few-shot pairs: {len(few_shot)}')
```

Add a classifier helper code cell:

```python
import json

from openai import OpenAI

client = OpenAI(base_url=VLLM_BASE_URL, api_key='EMPTY')
VALID_NLP_TYPES = set(NLP_TYPES) | {'none'}


def build_static_few_shot_messages(text: str) -> list[dict]:
    messages = []
    for benign_text, attack_text, attack_type in few_shot:
        messages.append({'role': 'user', 'content': f'INPUT_PROMPT:\n{benign_text}'})
        messages.append({
            'role': 'assistant',
            'content': json.dumps({
                'label': 'benign',
                'confidence': 95,
                'nlp_attack_type': 'none',
                'evidence': '',
                'reason': 'No active attempt to override instructions, exfiltrate data, or hijack tools.',
            }),
        })
        if attack_type in NLP_TYPES:
            evidence = ''
            adv_reason = f'Perturbed tokens characteristic of {attack_type} adversarial attack.'
        else:
            evidence = attack_text[:80]
            adv_reason = f'Contains {attack_type} obfuscation; active adversarial prompt detected.'
        messages.append({'role': 'user', 'content': f'INPUT_PROMPT:\n{attack_text}'})
        messages.append({
            'role': 'assistant',
            'content': json.dumps({
                'label': 'adversarial',
                'confidence': 84,
                'nlp_attack_type': attack_type if attack_type in NLP_TYPES else 'none',
                'evidence': evidence,
                'reason': adv_reason,
            }),
        })
    return messages


def extract_completion_logprobs(response) -> list[dict] | None:
    payload = response.model_dump()
    return HierarchicalLLMClassifier._extract_completion_logprobs(payload)


def normalize_classifier_payload(payload: object, raw_response_text: str | None, token_logprobs: list[dict] | None) -> dict:
    if not isinstance(payload, dict):
        payload = {}
    label = payload.get('label', '')
    if label not in ('benign', 'adversarial', 'uncertain'):
        label = 'adversarial'
    confidence = HierarchicalLLMClassifier._normalize_confidence(payload.get('confidence', 50))
    nlp_attack_type = payload.get('nlp_attack_type', 'none')
    if nlp_attack_type not in VALID_NLP_TYPES:
        nlp_attack_type = 'none'
    category = HierarchicalLLMClassifier._derive_category(label, nlp_attack_type)
    return {
        'label': label,
        'confidence': confidence,
        'nlp_attack_type': nlp_attack_type,
        'evidence': payload.get('evidence', ''),
        'reason': payload.get('reason', ''),
        '_token_logprobs': token_logprobs,
        '_provider_name': 'vllm-local',
        '_model_name': MODEL_ID,
        '_raw_response_text': raw_response_text,
        '_parse_success': bool(payload),
        '_category': category,
    }


def classify_text(text: str) -> dict:
    messages = build_classifier_messages(text, build_static_few_shot_messages(text))
    response = client.chat.completions.create(
        model=MODEL_ID,
        messages=messages,
        temperature=cfg['llm']['temperature'],
        max_tokens=cfg['llm']['max_tokens_classifier'],
        response_format={'type': 'json_object'},
        logprobs=True,
        top_logprobs=cfg['llm'].get('top_logprobs', 5),
    )
    raw_response_text = response.choices[0].message.content
    token_logprobs = extract_completion_logprobs(response)
    try:
        payload = json.loads(raw_response_text)
        if isinstance(payload, str):
            payload = json.loads(payload)
    except Exception:
        payload = {}
    return normalize_classifier_payload(payload, raw_response_text, token_logprobs)
```

- [ ] **Step 3: Run local structural tests**

Run:

```bash
pytest tests/test_colab_local_llm_classifier_notebook.py -q
```

Expected: still FAIL if runner/output cells are not added yet; PASS for vLLM, model, and import assertions.

- [ ] **Step 4: Commit startup and helper cells**

Run:

```bash
git add notebooks/colab_local_llm_classifier.ipynb
git commit -m "feat: add local vLLM classifier setup to Colab notebook"
```

Expected: commit includes only notebook changes for server startup, validation, and helpers.

## Task 4: Classifier-Only Runner, Checkpointing, and Validation Cells

**Files:**
- Modify: `notebooks/colab_local_llm_classifier.ipynb`

- [ ] **Step 1: Add row conversion, checkpointing, and runner cells**

Add this code cell:

```python
GROUND_TRUTH_COLUMNS = [
    'modified_sample',
    'original_sample',
    'attack_name',
    'label_binary',
    'label_category',
    'label_type',
    'prompt_hash',
    'benign_source',
    'is_synthetic_benign',
]
PREDICTION_COLUMNS = [
    'llm_pred_binary',
    'llm_pred_raw',
    'llm_pred_category',
    'llm_conf_binary',
    'llm_evidence',
    'llm_stages_run',
    'llm_provider_name',
    'llm_model_name',
    'llm_raw_response_text',
    'llm_parse_success',
    'clf_label',
    'clf_category',
    'clf_confidence',
    'clf_evidence',
    'clf_nlp_attack_type',
    'clf_provider_name',
    'clf_model_name',
    'clf_raw_response_text',
    'clf_parse_success',
    'clf_token_logprobs',
]


def make_failure_result(raw_response_text: str | None = None) -> dict:
    return normalize_classifier_payload({}, raw_response_text, None)


def build_output_row(input_row: pd.Series, result: dict) -> dict:
    label = result['label']
    label_binary = 'benign' if label == 'benign' else 'adversarial'
    row = {'sample_id': input_row['sample_id']}
    for column in GROUND_TRUTH_COLUMNS:
        if column in input_row.index:
            row[column] = input_row[column]
    row.update({
        'llm_pred_binary': label_binary,
        'llm_pred_raw': label,
        'llm_pred_category': result['_category'],
        'llm_conf_binary': result['confidence'],
        'llm_evidence': result.get('evidence', ''),
        'llm_stages_run': 1,
        'llm_provider_name': result['_provider_name'],
        'llm_model_name': result['_model_name'],
        'llm_raw_response_text': result['_raw_response_text'],
        'llm_parse_success': result['_parse_success'],
        'clf_label': label,
        'clf_category': result['_category'],
        'clf_confidence': result['confidence'],
        'clf_evidence': result.get('evidence', ''),
        'clf_nlp_attack_type': result['nlp_attack_type'],
        'clf_provider_name': result['_provider_name'],
        'clf_model_name': result['_model_name'],
        'clf_raw_response_text': result['_raw_response_text'],
        'clf_parse_success': result['_parse_success'],
        'clf_token_logprobs': json.dumps(result.get('_token_logprobs')),
    })
    return row


def write_checkpoint(rows: list[dict]) -> None:
    if not rows:
        return
    checkpoint = Path(CHECKPOINT_PATH)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    new_df = pd.DataFrame(rows)
    if checkpoint.exists():
        existing_df = pd.read_parquet(checkpoint)
        out_df = pd.concat([existing_df, new_df], ignore_index=True)
        out_df = out_df.drop_duplicates(subset=['sample_id'], keep='last')
    else:
        out_df = new_df
    out_df.to_parquet(checkpoint, index=False)
    print(f'Checkpoint rows: {len(out_df):,} -> {checkpoint}')
```

Add this runner code cell:

```python
from tqdm.auto import tqdm

completed_ids = set()
checkpoint_path = Path(CHECKPOINT_PATH)
if checkpoint_path.exists():
    checkpoint_df = pd.read_parquet(checkpoint_path)
    completed_ids = set(checkpoint_df['sample_id'].tolist())
    print(f'Resuming from checkpoint: {len(completed_ids):,} completed rows')

pending_df = eval_df[~eval_df['sample_id'].isin(completed_ids)].reset_index(drop=True)
print(f'Pending rows: {len(pending_df):,}')

buffer: list[dict] = []
for idx, input_row in tqdm(list(pending_df.iterrows()), total=len(pending_df), desc=f'Classifying {SPLIT}'):
    text = str(input_row[text_col])
    try:
        result = classify_text(text)
    except Exception as exc:
        try:
            health = requests.get(f'{VLLM_BASE_URL}/models', timeout=5)
            health.raise_for_status()
        except Exception as health_exc:
            raise RuntimeError(f'vLLM server is unreachable after row failure: {health_exc}') from exc
        print(f'Row failed but server is healthy; writing parse-failure row for sample_id={input_row.sample_id}: {exc}')
        result = make_failure_result(raw_response_text=None)
    buffer.append(build_output_row(input_row, result))
    if len(buffer) >= CHECKPOINT_EVERY:
        write_checkpoint(buffer)
        buffer.clear()

write_checkpoint(buffer)

if checkpoint_path.exists():
    final_df = pd.read_parquet(checkpoint_path)
else:
    final_df = pd.DataFrame(columns=['sample_id'] + PREDICTION_COLUMNS)

final_df = final_df.drop_duplicates(subset=['sample_id'], keep='last')
Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
final_df.to_parquet(OUTPUT_PATH, index=False)
print(f'Final rows: {len(final_df):,} -> {OUTPUT_PATH}')
```

- [ ] **Step 2: Add output validation cells**

Add this code cell:

```python
out_df = pd.read_parquet(OUTPUT_PATH)
expected_columns = {'sample_id', *PREDICTION_COLUMNS}
missing_columns = expected_columns - set(out_df.columns)
if missing_columns:
    raise AssertionError(f'Missing expected columns: {sorted(missing_columns)}')
assert not any(col.startswith('judge_') for col in out_df.columns), 'Output must not contain judge_* columns'
assert set(out_df['llm_stages_run'].dropna().unique()) == {1}, 'Classifier-only outputs must have llm_stages_run=1'
print(f'Read back {len(out_df):,} rows from {OUTPUT_PATH}')
print(out_df['llm_pred_binary'].value_counts(dropna=False))
print(out_df[['llm_pred_raw', 'llm_conf_binary', 'llm_parse_success']].head())
```

- [ ] **Step 3: Run notebook structural tests**

Run:

```bash
pytest tests/test_colab_local_llm_classifier_notebook.py -q
```

Expected: PASS.

- [ ] **Step 4: Run JSON validity check**

Run:

```bash
python -m json.tool notebooks/colab_local_llm_classifier.ipynb >/tmp/colab_local_llm_classifier.json
```

Expected: PASS with no output.

- [ ] **Step 5: Commit runner and validation cells**

Run:

```bash
git add notebooks/colab_local_llm_classifier.ipynb
git commit -m "feat: run classifier-only local LLM inference in Colab"
```

Expected: commit includes only notebook changes.

## Task 5: Final Review and Local Verification

**Files:**
- Modify only if review finds a defect: `notebooks/colab_local_llm_classifier.ipynb` or `tests/test_colab_local_llm_classifier_notebook.py`

- [ ] **Step 1: Run all focused verification commands**

Run:

```bash
python -m json.tool notebooks/colab_local_llm_classifier.ipynb >/tmp/colab_local_llm_classifier.json
pytest tests/test_colab_local_llm_classifier_notebook.py -q
```

Expected:

- JSON command exits `0`.
- Pytest reports all tests passing.

- [ ] **Step 2: Inspect notebook for forbidden scope**

Run:

```bash
rg -n "judge_|NVIDIA_API_KEY|LLM_PROVIDER|model_quality|data/processed/predictions/llm_predictions_\\{SPLIT\\}\\.parquet" notebooks/colab_local_llm_classifier.ipynb
```

Expected:

- No `judge_` matches except the output validation cell that asserts no `judge_*` columns are present.
- No `NVIDIA_API_KEY`, `LLM_PROVIDER`, or `model_quality` matches.
- No direct write to unsuffixed `llm_predictions_{SPLIT}.parquet`.

- [ ] **Step 3: Review git diff**

Run:

```bash
git diff --stat HEAD
git diff -- notebooks/colab_local_llm_classifier.ipynb tests/test_colab_local_llm_classifier_notebook.py
```

Expected: only notebook/test changes are present, and no downstream pipeline files are modified.

- [ ] **Step 4: Commit any review fixes**

If Step 2 or Step 3 required changes, run:

```bash
git add notebooks/colab_local_llm_classifier.ipynb tests/test_colab_local_llm_classifier_notebook.py
git commit -m "fix: tighten Colab local classifier notebook contract"
```

Expected: commit succeeds only if review fixes were necessary. Skip this step if there are no changes.

## Self-Review Notes

- Spec coverage: Tasks cover notebook creation, Drive/repo setup, vLLM startup, current classifier model default, existing prompt helpers, classifier-only output columns, no `judge_*` columns, checkpoint/resume, per-row failure rows, and local validation.
- Placeholder scan: no `TBD`, `TODO`, or unspecified implementation steps remain.
- Type consistency: notebook settings, paths, output columns, and test assertions use the same names throughout.
- Scope check: the plan intentionally does not modify downstream consumers, DVC stages, or existing LLM classifier internals.

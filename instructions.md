# Coding Agent Instructions: Add External Dataset Access for MULTITuDE + M4

## Goal
Update the repo so `eval_external` can run on **MULTITuDE** and **M4** reliably and reproducibly.

Constraints:
- MULTITuDE is **NOT** reliably available via Hugging Face; treat it as a **Zenodo restricted/manual download** dataset.
- M4 is accessed via its **official repository distribution** (download per repo instructions).
- Integrate with existing pipeline conventions: configs under `configs/datasets/`, parquet-first artifacts, deterministic IDs, DVC-friendly paths, and clear failure modes.

---

## 1) Implement a unified “external dataset source” abstraction

### Add/extend dataset loader module
Create or extend something like:
- `src/datasets/external/__init__.py`
- `src/datasets/external/registry.py`
- `src/datasets/external/base.py`

Define a minimal interface:

```text
ExternalDatasetAdapter:
  name: str
  prepare(config, output_dir, seed) -> tuples_parquet_path
  manifest(config, output_dir) -> dict
```

Where `prepare()`:
- materializes a normalized `tuples.parquet` with a stable schema your downstream embed/featurize expects.

---

## 2) Define standard external tuples schema (minimal)
Your external datasets must be normalized to your pipeline’s “tuple” schema. Use the repo’s canonical schema (e.g. `src/storage/schema.py` `TUPLES_COLUMNS`) and ensure at minimum you output:

Required columns (minimum):
- `tuple_id` (stable unique id)
- `text` (string)
- `label` (0=human, 1=llm)
- `source` (dataset name, e.g. `multitude`, `m4`)
- `language` (best-effort; `unknown` allowed)
- `domain` (best-effort; `unknown` allowed)
- `generator` (best-effort; `unknown` allowed)
- any other fields required by your embed/featurize stage

Stable IDs:
- `tuple_id = f"{dataset}:{split}:{sha1(text)}"` OR use dataset-provided IDs when present.
- Must be deterministic across runs.

---

## 3) MULTITuDE: implement as “manual Zenodo restricted download”

### Key fact
MULTITuDE is hosted on **Zenodo with restricted access**. It should be treated as **manual download**. Do NOT implement Hugging Face `load_dataset()` for it.

### Config: `configs/datasets/multitude.yaml`
Add fields (names can vary, but keep intent):
```yaml
name: multitude
source: zenodo_restricted_manual
manual_dir: data/external/manual/multitude
splits: [test]              # or as available in files
label_map:
  human: 0
  machine: 1
schema:
  text_field: text          # depends on file
  label_field: label        # depends on file
  language_field: language  # optional
  domain_field: domain      # optional
  generator_field: generator # optional
```

### Loader behavior
Implement `MultitudeAdapter.prepare()` to:

1. Validate files exist under `manual_dir`.
2. If missing, fail with a *clear error message*:
   - exactly which directory was searched
   - expected filenames/patterns
   - “MULTITuDE requires Zenodo access request + manual download; see README”
3. Load the data files into a dataframe.
4. Normalize fields to your tuples schema.
5. Write:
   - `data/processed/external/multitude/tuples.parquet`
6. Write a manifest JSON (see §6).

Notes:
- Keep parsing robust: allow multiple candidate filenames via glob patterns.
- If MULTITuDE has multiple languages, keep `language` populated when available.

---

## 4) M4: implement as “download per official repo distribution”

### Key fact
M4 is distributed via its official repository instructions. Your code should support automated downloading **if URLs are stable**, otherwise support “manual_dir” fallback.

### Config: `configs/datasets/m4.yaml`
Support both modes:

```yaml
name: m4
source: repo_download   # or url_list
# Option A: explicit urls (preferred for automation)
urls:
  - <url_to_file_1>
  - <url_to_file_2>
cache_dir: data/external/cache/m4
# Option B: manual fallback
manual_dir: data/external/manual/m4

splits: [test]           # depends on release
label_map:
  human: 0
  machine: 1
schema:
  text_field: text
  label_field: label
  language_field: language
  domain_field: domain
  generator_field: generator
```

### Loader behavior
Implement `M4Adapter.prepare()`:

1. If `manual_dir` exists and has files, prefer it (fastest).
2. Else if `urls` present:
   - download to `cache_dir`
   - verify checksums if provided
3. Else:
   - fail with a clear error message: “Provide `urls` or download manually to `manual_dir`.”
4. Parse into a dataframe and normalize to tuples schema.
5. Write:
   - `data/processed/external/m4/tuples.parquet`
6. Write manifest JSON.

Important:
- If M4 provides generator/model metadata, preserve it in `generator`.
- Preserve domain metadata when present.

---

## 5) Update `src.cli.eval_external` to use adapters

The `eval_external` CLI should:
1. Load dataset config
2. Select adapter by `name` or `source`
3. Call `adapter.prepare()` to get normalized tuples parquet
4. Run your existing pipeline steps:
   - embed
   - trajectories
   - featurize
   - load trained model
   - predict
5. Save:
   - `reports/phase1_binary_external/<dataset>/predictions.parquet`
   - `reports/phase1_binary_external/<dataset>/metrics.json`

### Failure mode policy (important)
- For MULTITuDE (restricted):
  - default behavior in DVC should be configurable:
    - `on_missing: skip` (recommended) OR `on_missing: fail`
- Implement `--strict` flag to force fail.

This prevents your entire `dvc repro` from failing just because MULTITuDE access wasn’t granted yet.

---

## 6) Add dataset manifests (reproducibility)

For each external dataset run, write:
- `reports/manifests/<run_id>_<dataset>.json`

Include:
- dataset name
- source type (`zenodo_restricted_manual` / `repo_download`)
- resolved file paths
- file hashes (sha256) if feasible
- row counts
- label counts
- languages/domains present
- timestamp + git commit hash (if available)

---

## 7) DVC stages

Add/adjust stages in `dvc.yaml`:

- `eval_external_multitude`
- `eval_external_m4`

Each stage should:
- depend on:
  - the trained model file
  - dataset config file
  - relevant code modules
- output:
  - external predictions parquet
  - external metrics json
  - manifests

For MULTITuDE, if you support skip-on-missing:
- still produce a small “skipped” marker file (e.g. metrics.json with `{ "status": "skipped", ... }`) so DVC has declared outputs.

---

## 8) Tests

Add minimal tests:

1. `tests/test_external_registry.py`
- registry returns correct adapter for `multitude` and `m4`

2. `tests/test_external_multitude_missing.py`
- when `manual_dir` missing:
  - in non-strict mode -> returns “skipped” (or raises a specific “MissingDataset” you handle)
  - in strict mode -> raises with clear message

3. `tests/test_external_schema_normalization.py`
- given a tiny synthetic dataframe, adapter normalization produces required tuple columns

---

## Acceptance Criteria

- `python -m src.cli.eval_external --dataset-config configs/datasets/m4.yaml ...` runs end-to-end and produces predictions + metrics.
- `python -m src.cli.eval_external --dataset-config configs/datasets/multitude.yaml ...`:
  - skips cleanly with a clear message when manual files are missing (default),
  - fails loudly with `--strict`.
- DVC stages `eval_external_m4` and `eval_external_multitude` complete deterministically (or skip deterministically for MULTITuDE if missing).
- Manifests are written for each run.

# Deletion Batch 3B Proposal: `src.research.py`

Date: 2026-05-13

Status: proposal only. Do not delete anything in this batch until explicitly
approved.

## Scope

This gate evaluates whether the legacy `src.research.py` module and its direct
test file can be removed after Batch 3A extracted shared routing diagnostics to
`src/routing_diagnostics.py`.

Constraints for this proposal:

- no source deletion
- no artifact regeneration
- no Colab handoff validation changes
- no model, evaluation, routing, preprocessing, or split-building behavior
  changes
- do not touch `dvc.yaml`, preprocessing, or split-building while the separate
  rerun is active

## Checks Run

```bash
rg -n "src\.research|from src\.research import|import src\.research|python -m src\.research|src/research\.py|research.py" src tests README.md STATUS.md CLAUDE.md src/cli/README.md docs research_docs openspec dvc.yaml configs
rg -n "from src\.research import|import src\.research" src
rg -n "from src\.research import|import src\.research" tests
rg -n "from src\.routing_diagnostics import|compute_routing_diagnostics|render_routing_diagnostics_markdown|compute_unicode_lane_mask|is_adversarial_label" src tests
dvc stage list
dvc dag final_verdict_report
rg -n "research\.py|src/research\.py|src\.research|python -m src\.research" dvc.yaml src/cli/train_escalating_model.py src/cli/final_verdict_report.py src/cli/validate_colab_handoff.py src/cli/colab_handoff_schema.py src/cli/judge_colab_local_predictions.py src/escalating_model.py notebooks/colab_local_llm_classifier.ipynb README.md STATUS.md CLAUDE.md src/cli/README.md
rg -n "compute_hybrid_routing|build_research_dataframe|generate_hybrid_report|run_ml_full|run_llm|run_research|hybrid_margin_trace|posthoc_benign|research_\{split\}|research_<split>|reports/research" src/research.py tests/test_research.py README.md STATUS.md CLAUDE.md src/cli/README.md dvc.yaml src tests
```

Note: `dvc stage list` showed `generate_synthetic_benign@A` in the active
workspace, and `git status --short` shows `dvc.yaml` modified. This proposal
does not touch `dvc.yaml`; that change is treated as external rerun state.

## Reference Findings

### Direct Imports

Production `src/*` imports:

- none found

Test imports:

- `tests/test_research.py` imports:
  - `compute_hybrid_routing`
  - `build_research_dataframe`
  - `generate_hybrid_report`

The direct import check result was:

```text
tests/test_research.py:9:from src.research import (
```

### Active Docs

Active docs do not describe `src.research.py` as current canonical pipeline
behavior.

Current active references:

- `README.md` lists `research.py` in the project structure as
  `Legacy research merge/routing path`.

No active README/STATUS/CLAUDE/CLI README reference was found that tells users
to run `python -m src.research` as the current pipeline. The remaining
`python -m src.research` references are inside `src/research.py` itself and
historical docs under `docs/superpowers/**`.

Historical/untracked references:

- `docs/superpowers/**` contains historical implementation plans/specs.
- `docs/merge_readiness_findings.md` contains stale references, but it is
  untracked and was intentionally left untouched.

### Routing Diagnostics

Useful routing diagnostics were moved in Batch 3A to
`src/routing_diagnostics.py`:

- `is_adversarial_label`
- `compute_unicode_lane_mask`
- `compute_routing_diagnostics`
- `render_routing_diagnostics_markdown`

Current imports of the new helper:

- `src/eval_external.py`
- `src/research.py`
- `tests/test_routing_diagnostics.py`

This means deleting `src.research.py` would not remove the routing diagnostics
used by external evaluation reports.

## Impact Assessment

### `eval_external`

Deleting `src.research.py` should not affect `src/eval_external.py` after
Batch 3A because it now imports diagnostics from `src.routing_diagnostics`.

Relevant canonical/non-legacy import:

```text
src/eval_external.py: from src.routing_diagnostics import ...
```

### `final_verdict_report`

No `src.research` reference was found in:

- `src/cli/final_verdict_report.py`
- `src/escalating_model.py`
- `dvc.yaml` final-verdict stage deps

Deleting `src.research.py` should not affect final verdict generation.

### DVC Stages

The active DVC stage list has no `research` stage and no `src/research.py`
dependency. The final-verdict graph runs through:

- split/model preparation
- DeBERTa external predictions
- Colab handoff validation
- `train_escalating_model`
- selective judge stages
- `final_verdict_report`

Deleting `src.research.py` should not affect the current DVC graph, provided
the deletion does not modify `dvc.yaml`.

### Colab Handoff

No `src.research` reference was found in:

- `src/cli/validate_colab_handoff.py`
- `src/cli/colab_handoff_schema.py`
- `notebooks/colab_local_llm_classifier.ipynb`
- `src/cli/judge_colab_local_predictions.py`
- `src/cli/train_escalating_model.py`

Deleting `src.research.py` should not affect Colab handoff validation or
manual handoff artifact requirements.

## What Would Be Removed

If approved later, delete:

- `src/research.py`
- `tests/test_research.py`

The removed runtime code would include legacy/historical helpers:

- `compute_hybrid_routing`
- `build_research_dataframe`
- `generate_hybrid_report`
- research CLI entrypoint `python -m src.research --split ...`
- legacy margin-trace writing through `hybrid_margin_trace_<split>.parquet`

These helpers are not used by the canonical DVC + Colab handoff + final verdict
path after Batch 3A.

## Tests To Remove Or Update

Remove with the runtime module:

- `tests/test_research.py`

Keep:

- `tests/test_routing_diagnostics.py`
- `tests/test_eval_external.py`
- `tests/test_final_verdict_report.py`
- `tests/test_validate_colab_handoff.py`
- `tests/test_escalating_model.py`

No test migration is required for routing diagnostics because
`tests/test_routing_diagnostics.py` already covers the extracted helper.

## Classification

Classification: **safe to delete now, after explicit approval**.

Reasoning:

- no production `src/*` module imports `src.research`
- active DVC stages do not depend on `src/research.py`
- `eval_external` no longer imports it
- final verdict generation does not reference it
- Colab handoff validation does not reference it
- useful routing diagnostics have been extracted and tested
- remaining direct dependency is only `tests/test_research.py`, which should be
  removed with the legacy module

## Proposed Batch 3B Deletion, If Approved Later

Delete only:

- `src/research.py`
- `tests/test_research.py`

Update active docs only where they mention the live source tree:

- `README.md`: remove `research.py` from the project structure
- `docs/cleanup/deletion_candidates.md`: mark the research module/test as
  approved deleted or superseded
- `docs/cleanup/deletion_batch_2_proposal.md` and
  `docs/cleanup/deletion_batch_3_proposal.md`: optionally append a short
  follow-up note instead of rewriting historical gate content

Do not edit historical docs deeply. If a historical doc without the approved
notice still contains stale pipeline instructions, add only the approved
historical header note.

Expected verification after approval and deletion:

```bash
rg -n "from src\.research import|import src\.research|python -m src\.research|src/research\.py|src\.research" src tests README.md STATUS.md CLAUDE.md src/cli/README.md dvc.yaml
pytest tests/test_routing_diagnostics.py tests/test_eval_external.py tests/test_final_verdict_report.py tests/test_validate_colab_handoff.py tests/test_escalating_model.py -q
pytest -q
dvc stage list
dvc dag final_verdict_report
```

Do not run `dvc repro final_verdict_report` until the known Deepset Colab
handoff artifact blocker is fixed.

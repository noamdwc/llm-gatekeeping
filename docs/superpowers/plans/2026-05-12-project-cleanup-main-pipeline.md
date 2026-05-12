# Project Cleanup Main Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the fresh DVC + Colab handoff + final verdict path the only canonical runtime path, with explicit validation and a deletion approval gate.

**Architecture:** Add a small Colab handoff validation CLI, wire it into DVC before `train_escalating_model`, fix path ownership gaps, and update README around the single canonical run sequence. Produce a categorized deletion candidate list and stop before deleting anything.

**Tech Stack:** Python 3.14, pandas/parquet, DVC, pytest, YAML, markdown.

---

## Scope Rules

This is pipeline cleanup only.

- Do not change thresholds, metrics, routing policy, model training behavior, or evaluation logic.
- If a behavior-affecting fix is required to unblock the canonical pipeline, document it in the commit message and implementation note as a blocker fix.
- Do not delete files before the deletion candidate list is approved by the user.
- Keep implementation commits separated by concern: validation/path fixes, DVC wiring, README, deletion candidate list, approved deletion.
- Preserve `docs/superpowers/` and `openspec/changes/`.

## File Structure

- Create `src/cli/validate_colab_handoff.py`: validates required Colab classifier artifacts and their joins with DeBERTa predictions.
- Create `tests/test_validate_colab_handoff.py`: unit tests for missing files, schema failures, judge-column rejection, external path naming, and join-loss checks.
- Modify `src/cli/train_escalating_model.py`: fix external Colab default path, fail on missing configured external inputs instead of skipping, and use clearer actionable errors.
- Modify `tests/test_escalating_model.py`: add tests for external `_colab_local_classifier` defaults and missing external failures.
- Modify `src/cli/final_verdict_report.py`: make default configured external judged artifacts required instead of silently excluding missing configured externals.
- Modify `tests/test_final_verdict_report.py`: add default external missing-artifact failure test.
- Modify `dvc.yaml`: add a `validate_colab_handoff` stage and make `train_escalating_model` depend on its validation report; remove legacy DVC stages only after deletion approval.
- Modify `README.md`: replace competing pipeline guidance with the canonical start-to-finish sequence and final artifact.
- Create `docs/cleanup/deletion_candidates.md`: categorized deletion candidate list for user approval. This file is not destructive.

---

### Task 1: Capture Baseline And Guardrails

**Files:**
- Read: `docs/superpowers/specs/2026-05-12-project-cleanup-main-pipeline-design.md`
- Read: `dvc.yaml`
- Read: `src/cli/train_escalating_model.py`
- Read: `src/cli/final_verdict_report.py`
- Read: `README.md`

- [ ] **Step 1: Confirm worktree state**

Run:

```bash
git status --short
```

Expected: note any pre-existing untracked or modified files. Do not stage unrelated user files such as `docs/merge_readiness_findings.md`.

- [ ] **Step 2: Capture current DVC shape**

Run:

```bash
dvc stage list
```

Expected: command succeeds and lists current stages. Save the important stage names mentally for the deletion candidate list; do not edit yet.

- [ ] **Step 3: Capture current noisy status**

Run:

```bash
dvc status
```

Expected: command succeeds, even if output is large. If it fails, stop and document the failure as a blocker; do not rewrite DVC around a broken local metadata state.

- [ ] **Step 4: Re-read non-goals**

Run:

```bash
sed -n '1,180p' docs/superpowers/specs/2026-05-12-project-cleanup-main-pipeline-design.md
```

Expected: confirm the plan is limited to cleanup, validation, docs, and deletion approval.

---

### Task 2: Add Tests For Escalating Input Ownership Bugs

**Files:**
- Modify: `tests/test_escalating_model.py`
- Later modify: `src/cli/train_escalating_model.py`

- [ ] **Step 1: Add failing tests for external Colab path ownership and missing external failure**

Append these tests to `class TestTrainEscalatingModelCli` in `tests/test_escalating_model.py`:

```python
    def test_default_external_colab_path_uses_manual_handoff_suffix(self):
        from src.cli import train_escalating_model

        path = train_escalating_model._default_external_colab_path("deepset")

        assert path.name == "llm_predictions_external_deepset_colab_local_classifier.parquet"

    def test_cli_fails_when_configured_external_handoff_artifact_is_missing(self, tmp_path):
        train_colab, train_deberta = _make_prediction_frames(n=24)
        train_colab_path = tmp_path / "llm_predictions_val_colab_local_classifier.parquet"
        train_deberta_path = tmp_path / "deberta_predictions_val.parquet"
        train_colab.to_parquet(train_colab_path, index=False)
        train_deberta.to_parquet(train_deberta_path, index=False)

        eval_args = []
        for split in ["test", "unseen_val", "unseen_test", "safeguard_test"]:
            colab, deberta = _make_prediction_frames(n=12)
            colab_path = tmp_path / f"llm_predictions_{split}_colab_local_classifier.parquet"
            deberta_path = tmp_path / f"deberta_predictions_{split}.parquet"
            colab.to_parquet(colab_path, index=False)
            deberta.to_parquet(deberta_path, index=False)
            eval_args.extend([
                "--eval-split",
                split,
                str(colab_path),
                str(deberta_path),
            ])

        external_deberta = tmp_path / "deberta_predictions_external_deepset.parquet"
        pd.DataFrame([
            {
                "sample_id": "external-1",
                "label_binary": "benign",
                "deberta_proba_binary_adversarial": 0.1,
            }
        ]).to_parquet(external_deberta, index=False)

        config_path = tmp_path / "default.yaml"
        config_path.write_text(
            "splits:\n"
            "  random_seed: 42\n"
            "hybrid:\n"
            "  escalating_model:\n"
            "    model_path: data/processed/models/escalating_model.pkl\n"
            "    calibration_method: sigmoid\n"
            "external_datasets:\n"
            "  deepset:\n"
            "    name: deepset/prompt-injections\n"
        )

        with pytest.raises(FileNotFoundError, match="manual Colab handoff artifact"):
            train_escalating_main([
                "--config",
                str(config_path),
                "--train-colab-predictions",
                str(train_colab_path),
                "--train-deberta-predictions",
                str(train_deberta_path),
                "--model-output",
                str(tmp_path / "models" / "escalating_model.pkl"),
                "--research-output-dir",
                str(tmp_path / "research"),
                "--summary-output",
                str(tmp_path / "research" / "escalating_model_summary.csv"),
                "--threshold-sweep-output",
                str(tmp_path / "research" / "escalating_model_threshold_sweep_unseen_val.csv"),
                "--postscore-split-map-output",
                str(tmp_path / "research" / "escalating_model_unseen_val_postscore_split_map.csv"),
                "--report-output",
                str(tmp_path / "reports" / "escalating_model_poc.md"),
                "--external-dataset",
                "deepset",
                *eval_args,
            ])
```

- [ ] **Step 2: Run the focused failing tests**

Run:

```bash
pytest tests/test_escalating_model.py::TestTrainEscalatingModelCli::test_default_external_colab_path_uses_manual_handoff_suffix tests/test_escalating_model.py::TestTrainEscalatingModelCli::test_cli_fails_when_configured_external_handoff_artifact_is_missing -q
```

Expected: both tests fail before implementation. The first fails because the path lacks `_colab_local_classifier`; the second fails because the CLI currently skips missing externals.

---

### Task 3: Fix Escalating Input Ownership Bugs

**Files:**
- Modify: `src/cli/train_escalating_model.py`
- Test: `tests/test_escalating_model.py`

- [ ] **Step 1: Add a helper for required file ownership errors**

In `src/cli/train_escalating_model.py`, add this function after `_default_external_deberta_path`:

```python
def _require_existing_input(path: Path, *, role: str) -> None:
    if path.exists():
        return
    raise FileNotFoundError(
        f"Missing {role}: {path}. "
        "Every train_escalating_model input must be produced by a DVC stage "
        "or declared as a manual Colab handoff artifact with validation."
    )
```

- [ ] **Step 2: Fix the external manual handoff default path**

Replace `_default_external_colab_path` with:

```python
def _default_external_colab_path(dataset: str) -> Path:
    return (
        PREDICTIONS_EXTERNAL_DIR
        / f"llm_predictions_external_{dataset}_colab_local_classifier.parquet"
    )
```

- [ ] **Step 3: Require train and eval inputs before reading them**

In `main`, immediately before reading `train_colab` and `train_deberta`, add:

```python
    _require_existing_input(
        Path(args.train_colab_predictions),
        role="manual Colab handoff artifact",
    )
    _require_existing_input(
        Path(args.train_deberta_predictions),
        role="DVC-produced DeBERTa prediction artifact",
    )
```

Inside the eval split loop, before `pd.read_parquet`, add:

```python
        _require_existing_input(colab_path, role="manual Colab handoff artifact")
        _require_existing_input(deberta_path, role="DVC-produced DeBERTa prediction artifact")
```

- [ ] **Step 4: Replace silent external skipping with failure**

Replace this block:

```python
        if not colab_path.exists() or not deberta_path.exists():
            print(
                f"Skipping external dataset {dataset!r}: missing "
                f"{colab_path if not colab_path.exists() else deberta_path}"
            )
            continue
```

with:

```python
        _require_existing_input(colab_path, role="manual Colab handoff artifact")
        _require_existing_input(deberta_path, role="DVC-produced DeBERTa prediction artifact")
```

- [ ] **Step 5: Run the focused tests**

Run:

```bash
pytest tests/test_escalating_model.py::TestTrainEscalatingModelCli::test_default_external_colab_path_uses_manual_handoff_suffix tests/test_escalating_model.py::TestTrainEscalatingModelCli::test_cli_fails_when_configured_external_handoff_artifact_is_missing -q
```

Expected: both tests pass.

- [ ] **Step 6: Run the full escalating model tests**

Run:

```bash
pytest tests/test_escalating_model.py -q
```

Expected: pass.

- [ ] **Step 7: Commit path ownership blocker fix**

Run:

```bash
git add src/cli/train_escalating_model.py tests/test_escalating_model.py
git commit -m "Fix canonical escalation input ownership"
```

Expected: commit succeeds. Commit message records this as a canonical pipeline blocker fix, not a metric change.

---

### Task 4: Add Colab Handoff Validation Tests

**Files:**
- Create: `tests/test_validate_colab_handoff.py`
- Later create: `src/cli/validate_colab_handoff.py`

- [ ] **Step 1: Create tests for the validation contract**

Create `tests/test_validate_colab_handoff.py`:

```python
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.cli import validate_colab_handoff


def _classifier_frame(sample_ids=("s1", "s2")) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "sample_id": sample_id,
                "modified_sample": f"text {sample_id}",
                "label_binary": "benign" if sample_id.endswith("1") else "adversarial",
                "llm_pred_binary": "benign",
                "llm_pred_raw": "benign",
                "llm_pred_category": "benign",
                "llm_conf_binary": 0.9,
                "llm_stages_run": 1,
                "llm_provider_name": "transformers-local",
                "llm_model_name": "meta/llama-3.1-8b-instruct",
                "llm_raw_response_text": "{}",
                "llm_parse_success": True,
                "clf_label": "benign",
                "clf_category": "benign",
                "clf_confidence": 0.9,
                "clf_evidence": "",
                "clf_nlp_attack_type": "none",
                "clf_provider_name": "transformers-local",
                "clf_model_name": "meta/llama-3.1-8b-instruct",
                "clf_raw_response_text": "{}",
                "clf_parse_success": True,
                "clf_token_logprobs": json.dumps([{"token": " benign", "logprob": -0.1}]),
            }
            for sample_id in sample_ids
        ]
    )


def _deberta_frame(sample_ids=("s1", "s2")) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "sample_id": sample_id,
                "label_binary": "benign" if sample_id.endswith("1") else "adversarial",
                "deberta_proba_binary_adversarial": 0.2,
            }
            for sample_id in sample_ids
        ]
    )


def test_validate_artifact_passes_for_valid_classifier_and_deberta_join(tmp_path: Path):
    classifier_path = tmp_path / "llm_predictions_test_colab_local_classifier.parquet"
    deberta_path = tmp_path / "deberta_predictions_test.parquet"
    _classifier_frame().to_parquet(classifier_path, index=False)
    _deberta_frame().to_parquet(deberta_path, index=False)

    result = validate_colab_handoff.validate_artifact_pair(
        name="test",
        classifier_path=classifier_path,
        deberta_path=deberta_path,
    )

    assert result["name"] == "test"
    assert result["rows_classifier"] == 2
    assert result["rows_deberta"] == 2
    assert result["rows_joined"] == 2
    assert result["rows_dropped_classifier_only"] == 0
    assert result["rows_dropped_deberta_only"] == 0


def test_validate_artifact_fails_with_exact_missing_path(tmp_path: Path):
    classifier_path = tmp_path / "missing.parquet"
    deberta_path = tmp_path / "deberta_predictions_test.parquet"
    _deberta_frame().to_parquet(deberta_path, index=False)

    with pytest.raises(FileNotFoundError, match=str(classifier_path)):
        validate_colab_handoff.validate_artifact_pair(
            name="test",
            classifier_path=classifier_path,
            deberta_path=deberta_path,
        )


def test_validate_artifact_rejects_judge_columns(tmp_path: Path):
    classifier_path = tmp_path / "llm_predictions_test_colab_local_classifier.parquet"
    deberta_path = tmp_path / "deberta_predictions_test.parquet"
    _classifier_frame().assign(judge_ran=True).to_parquet(classifier_path, index=False)
    _deberta_frame().to_parquet(deberta_path, index=False)

    with pytest.raises(ValueError, match="judge columns"):
        validate_colab_handoff.validate_artifact_pair(
            name="test",
            classifier_path=classifier_path,
            deberta_path=deberta_path,
        )


def test_validate_artifact_requires_single_classifier_stage(tmp_path: Path):
    classifier_path = tmp_path / "llm_predictions_test_colab_local_classifier.parquet"
    deberta_path = tmp_path / "deberta_predictions_test.parquet"
    _classifier_frame().assign(llm_stages_run=2).to_parquet(classifier_path, index=False)
    _deberta_frame().to_parquet(deberta_path, index=False)

    with pytest.raises(ValueError, match="llm_stages_run == 1"):
        validate_colab_handoff.validate_artifact_pair(
            name="test",
            classifier_path=classifier_path,
            deberta_path=deberta_path,
        )


def test_validate_artifact_rejects_lossy_join(tmp_path: Path):
    classifier_path = tmp_path / "llm_predictions_test_colab_local_classifier.parquet"
    deberta_path = tmp_path / "deberta_predictions_test.parquet"
    _classifier_frame(sample_ids=("s1", "s2")).to_parquet(classifier_path, index=False)
    _deberta_frame(sample_ids=("s1", "s3")).to_parquet(deberta_path, index=False)

    with pytest.raises(ValueError, match="join mismatch"):
        validate_colab_handoff.validate_artifact_pair(
            name="test",
            classifier_path=classifier_path,
            deberta_path=deberta_path,
        )


def test_main_writes_validation_report_for_main_and_external_targets(tmp_path: Path):
    predictions_dir = tmp_path / "predictions"
    external_dir = tmp_path / "predictions_external"
    report_path = tmp_path / "reports" / "colab_handoff_validation.json"
    predictions_dir.mkdir()
    external_dir.mkdir()

    for split in validate_colab_handoff.DEFAULT_MAIN_SPLITS:
        _classifier_frame().to_parquet(
            predictions_dir / f"llm_predictions_{split}_colab_local_classifier.parquet",
            index=False,
        )
        _deberta_frame().to_parquet(
            predictions_dir / f"deberta_predictions_{split}.parquet",
            index=False,
        )

    _classifier_frame().to_parquet(
        external_dir / "llm_predictions_external_deepset_colab_local_classifier.parquet",
        index=False,
    )
    _deberta_frame().to_parquet(
        external_dir / "deberta_predictions_external_deepset.parquet",
        index=False,
    )

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "external_datasets:\n"
        "  deepset:\n"
        "    name: deepset/prompt-injections\n"
    )

    validate_colab_handoff.main(
        [
            "--config",
            str(config_path),
            "--predictions-dir",
            str(predictions_dir),
            "--predictions-external-dir",
            str(external_dir),
            "--output",
            str(report_path),
        ]
    )

    payload = json.loads(report_path.read_text())
    assert payload["ok"] is True
    assert [item["name"] for item in payload["artifacts"]] == [
        "val",
        "test",
        "unseen_val",
        "unseen_test",
        "safeguard_test",
        "external_deepset",
    ]
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
pytest tests/test_validate_colab_handoff.py -q
```

Expected: fail because `src.cli.validate_colab_handoff` does not exist.

---

### Task 5: Implement Colab Handoff Validation CLI

**Files:**
- Create: `src/cli/validate_colab_handoff.py`
- Test: `tests/test_validate_colab_handoff.py`

- [ ] **Step 1: Create the validation CLI**

Create `src/cli/validate_colab_handoff.py`:

```python
"""Validate manual Colab local-LLM classifier handoff artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.cli.judge_colab_local_predictions import REQUIRED_COLUMNS
from src.utils import PREDICTIONS_DIR, PREDICTIONS_EXTERNAL_DIR, REPORTS_DIR, load_config


DEFAULT_MAIN_SPLITS = ["val", "test", "unseen_val", "unseen_test", "safeguard_test"]
DEFAULT_OUTPUT = REPORTS_DIR / "colab_handoff_validation.json"


def _classifier_path(predictions_dir: Path, split: str) -> Path:
    return predictions_dir / f"llm_predictions_{split}_colab_local_classifier.parquet"


def _deberta_path(predictions_dir: Path, split: str) -> Path:
    return predictions_dir / f"deberta_predictions_{split}.parquet"


def _external_classifier_path(predictions_external_dir: Path, dataset: str) -> Path:
    return (
        predictions_external_dir
        / f"llm_predictions_external_{dataset}_colab_local_classifier.parquet"
    )


def _external_deberta_path(predictions_external_dir: Path, dataset: str) -> Path:
    return predictions_external_dir / f"deberta_predictions_external_{dataset}.parquet"


def _require_path(path: Path, *, role: str) -> None:
    if path.exists():
        return
    raise FileNotFoundError(
        f"Missing {role}: {path}. "
        "Run notebooks/colab_local_llm_classifier.ipynb for manual handoff "
        "artifacts, or run the upstream DVC stage for DVC-produced artifacts."
    )


def _require_columns(df: pd.DataFrame, path: Path, columns: list[str]) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"{path} missing required columns: {missing}")


def validate_artifact_pair(
    *,
    name: str,
    classifier_path: Path,
    deberta_path: Path,
) -> dict[str, Any]:
    _require_path(classifier_path, role="manual Colab handoff artifact")
    _require_path(deberta_path, role="DVC-produced DeBERTa prediction artifact")

    classifier = pd.read_parquet(classifier_path)
    deberta = pd.read_parquet(deberta_path)

    _require_columns(classifier, classifier_path, list(REQUIRED_COLUMNS))
    _require_columns(deberta, deberta_path, ["sample_id", "deberta_proba_binary_adversarial"])

    judge_columns = sorted(column for column in classifier.columns if column.startswith("judge_"))
    if judge_columns:
        raise ValueError(f"{classifier_path} must be classifier-only; found judge columns: {judge_columns}")

    invalid_stage_rows = classifier[classifier["llm_stages_run"] != 1]
    if not invalid_stage_rows.empty:
        raise ValueError(
            f"{classifier_path} must have llm_stages_run == 1 for every row; "
            f"found {len(invalid_stage_rows)} invalid rows"
        )

    classifier_ids = classifier[["sample_id"]].drop_duplicates()
    deberta_ids = deberta[["sample_id"]].drop_duplicates()
    joined = classifier_ids.merge(deberta_ids, on="sample_id", how="inner")
    dropped_classifier_only = len(classifier_ids) - len(joined)
    dropped_deberta_only = len(deberta_ids) - len(joined)
    if len(joined) == 0 or dropped_classifier_only or dropped_deberta_only:
        raise ValueError(
            f"{name} join mismatch between {classifier_path} and {deberta_path}: "
            f"joined={len(joined)}, classifier_only={dropped_classifier_only}, "
            f"deberta_only={dropped_deberta_only}"
        )

    return {
        "name": name,
        "classifier_path": str(classifier_path),
        "deberta_path": str(deberta_path),
        "rows_classifier": int(len(classifier)),
        "rows_deberta": int(len(deberta)),
        "rows_joined": int(len(joined)),
        "rows_dropped_classifier_only": int(dropped_classifier_only),
        "rows_dropped_deberta_only": int(dropped_deberta_only),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate Colab classifier handoff artifacts.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--predictions-dir", type=Path, default=PREDICTIONS_DIR)
    parser.add_argument("--predictions-external-dir", type=Path, default=PREDICTIONS_EXTERNAL_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config)

    artifacts: list[dict[str, Any]] = []
    for split in DEFAULT_MAIN_SPLITS:
        artifacts.append(
            validate_artifact_pair(
                name=split,
                classifier_path=_classifier_path(args.predictions_dir, split),
                deberta_path=_deberta_path(args.predictions_dir, split),
            )
        )

    for dataset in cfg.get("external_datasets", {}):
        artifacts.append(
            validate_artifact_pair(
                name=f"external_{dataset}",
                classifier_path=_external_classifier_path(args.predictions_external_dir, dataset),
                deberta_path=_external_deberta_path(args.predictions_external_dir, dataset),
            )
        )

    payload = {"ok": True, "artifacts": artifacts}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"Validated {len(artifacts)} Colab handoff artifacts -> {args.output}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run validation tests**

Run:

```bash
pytest tests/test_validate_colab_handoff.py -q
```

Expected: pass.

- [ ] **Step 3: Run affected canonical tests**

Run:

```bash
pytest tests/test_validate_colab_handoff.py tests/test_judge_colab_local_predictions.py tests/test_escalating_model.py -q
```

Expected: pass.

- [ ] **Step 4: Commit validation CLI**

Run:

```bash
git add src/cli/validate_colab_handoff.py tests/test_validate_colab_handoff.py
git commit -m "Validate Colab handoff artifacts"
```

Expected: commit succeeds.

---

### Task 6: Make Final Verdict Defaults Require Configured External Artifacts

**Files:**
- Modify: `tests/test_final_verdict_report.py`
- Modify: `src/cli/final_verdict_report.py`

- [ ] **Step 1: Add failing test for missing configured external judged artifact**

Append this test to `tests/test_final_verdict_report.py`:

```python
def test_main_requires_configured_external_judged_artifact_by_default(tmp_path: Path, monkeypatch):
    internal_paths = {}
    for split in final_verdict_report.DEFAULT_INTERNAL_SPLITS:
        path = tmp_path / f"llm_predictions_{split}_colab_local_judged.parquet"
        _judged_frame().to_parquet(path, index=False)
        internal_paths[split] = path

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "hybrid:",
                "  escalating_model:",
                "    judge_threshold: 0.5",
                "    calibration_method: sigmoid",
                "    model_path: data/processed/models/escalating_model.pkl",
                "external_datasets:",
                "  deepset:",
                "    name: deepset/prompt-injections",
            ]
        )
    )

    monkeypatch.setattr(
        final_verdict_report,
        "default_internal_path",
        lambda split: internal_paths[split],
    )
    monkeypatch.setattr(
        final_verdict_report,
        "default_external_path",
        lambda dataset: tmp_path / f"missing_{dataset}_judged.parquet",
    )

    with pytest.raises(FileNotFoundError, match="Missing judged final-verdict input"):
        final_verdict_report.main([
            "--config",
            str(config_path),
            "--output",
            str(tmp_path / "pipeline_final_verdict_report.md"),
        ])
```

Also add `import pytest` near the top of the file:

```python
import pytest
```

- [ ] **Step 2: Run the failing test**

Run:

```bash
pytest tests/test_final_verdict_report.py::test_main_requires_configured_external_judged_artifact_by_default -q
```

Expected: fail because missing configured external artifacts are currently filtered out.

- [ ] **Step 3: Require configured external paths by default**

Replace `_default_external_inputs` in `src/cli/final_verdict_report.py` with:

```python
def _default_external_inputs(cfg: dict) -> list[tuple[str, Path]]:
    return [
        (f"external_{key}", default_external_path(key))
        for key in cfg.get("external_datasets", {})
    ]
```

- [ ] **Step 4: Run final verdict tests**

Run:

```bash
pytest tests/test_final_verdict_report.py -q
```

Expected: pass.

- [ ] **Step 5: Commit final verdict input ownership**

Run:

```bash
git add src/cli/final_verdict_report.py tests/test_final_verdict_report.py
git commit -m "Require configured final verdict artifacts"
```

Expected: commit succeeds.

---

### Task 7: Wire Validation Into DVC

**Files:**
- Modify: `dvc.yaml`
- Test through: `dvc stage list`, `dvc dag`, `dvc status`

- [ ] **Step 1: Add validation stage before `train_escalating_model`**

In `dvc.yaml`, insert this stage before `train_escalating_model`:

```yaml
  validate_colab_handoff:
    cmd: python -m src.cli.validate_colab_handoff --config configs/default.yaml
    deps:
    - src/cli/validate_colab_handoff.py
    - src/cli/judge_colab_local_predictions.py
    - data/processed/predictions/llm_predictions_val_colab_local_classifier.parquet
    - data/processed/predictions/deberta_predictions_val.parquet
    - data/processed/predictions/llm_predictions_test_colab_local_classifier.parquet
    - data/processed/predictions/deberta_predictions_test.parquet
    - data/processed/predictions/llm_predictions_unseen_val_colab_local_classifier.parquet
    - data/processed/predictions/deberta_predictions_unseen_val.parquet
    - data/processed/predictions/llm_predictions_unseen_test_colab_local_classifier.parquet
    - data/processed/predictions/deberta_predictions_unseen_test.parquet
    - data/processed/predictions/llm_predictions_safeguard_test_colab_local_classifier.parquet
    - data/processed/predictions/deberta_predictions_safeguard_test.parquet
    - data/processed/predictions_external/llm_predictions_external_deepset_colab_local_classifier.parquet
    - data/processed/predictions_external/deberta_predictions_external_deepset.parquet
    - data/processed/predictions_external/llm_predictions_external_jackhhao_colab_local_classifier.parquet
    - data/processed/predictions_external/deberta_predictions_external_jackhhao.parquet
    params:
    - configs/default.yaml:
      - external_datasets
    outs:
    - reports/colab_handoff_validation.json:
        cache: false
```

- [ ] **Step 2: Make `train_escalating_model` depend on validation report**

Add this dependency to the `train_escalating_model` `deps` list:

```yaml
    - reports/colab_handoff_validation.json
```

- [ ] **Step 3: Fix external Colab dependencies to match manual handoff names**

In the `train_escalating_model` deps, confirm external classifier dependencies are:

```yaml
    - data/processed/predictions_external/llm_predictions_external_deepset_colab_local_classifier.parquet
    - data/processed/predictions_external/deberta_predictions_external_deepset.parquet
    - data/processed/predictions_external/llm_predictions_external_jackhhao_colab_local_classifier.parquet
    - data/processed/predictions_external/deberta_predictions_external_jackhhao.parquet
```

If any dependency still points at `llm_predictions_external_deepset.parquet` or `llm_predictions_external_jackhhao.parquet` for the Colab classifier input to `train_escalating_model`, replace it with the matching `_colab_local_classifier` path shown above.

- [ ] **Step 4: Make final report transitive dependency visible**

Add these dependencies to `final_verdict_report`:

```yaml
    - data/processed/models/escalating_model.pkl
    - data/processed/research/escalating_model_summary.csv
    - reports/colab_handoff_validation.json
```

This is DVC bookkeeping only. It does not change report metrics or final verdict logic.

- [ ] **Step 5: Validate DVC syntax and graph**

Run:

```bash
dvc stage list
dvc dag final_verdict_report
dvc status final_verdict_report
```

Expected: commands succeed. `dvc dag final_verdict_report` includes `validate_colab_handoff`, `train_escalating_model`, judge stages, and `final_verdict_report`.

- [ ] **Step 6: Run validation stage if current artifacts exist**

Run:

```bash
dvc repro validate_colab_handoff
```

Expected: if the current Colab handoff artifacts exist and match schema, this writes `reports/colab_handoff_validation.json`. If it fails because artifacts are missing or malformed, confirm the error lists exact actionable paths; fix validation error quality, not the artifacts.

- [ ] **Step 7: Commit DVC validation wiring**

Run:

```bash
git add dvc.yaml dvc.lock reports/colab_handoff_validation.json
git commit -m "Wire Colab handoff validation into DVC"
```

Expected: commit succeeds if `dvc repro validate_colab_handoff` wrote/updated `dvc.lock` and the validation report. If no DVC output was produced because artifacts are missing, commit only `dvc.yaml` with a message explaining validation fails until the manual handoff is present.

---

### Task 8: Update README To Single Canonical Run Guide

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Replace competing “Two Pipeline Modes” guidance**

Edit `README.md` so the primary run section is titled:

```markdown
## Canonical Pipeline
```

Use this content as the core run guide:

```markdown
## Canonical Pipeline

The canonical end-to-end path is DVC-driven with one manual handoff: the local LLM classifier runs in Colab, then its classifier-only parquet outputs are downloaded back into this repo.

Final documented artifact:

```bash
reports/pipeline_final_verdict_report.md
```

### 1. Prepare local DVC inputs

```bash
dvc repro build_splits
dvc repro ml_model
dvc repro deberta_model
dvc repro deberta_external@deepset
dvc repro deberta_external@jackhhao
```

These stages produce the split files and DeBERTa predictions required by the Colab handoff and escalation model.

### 2. Run the Colab local LLM classifier

Open and run:

```bash
notebooks/colab_local_llm_classifier.ipynb
```

The notebook must produce classifier-only artifacts with these names:

```bash
data/processed/predictions/llm_predictions_val_colab_local_classifier.parquet
data/processed/predictions/llm_predictions_test_colab_local_classifier.parquet
data/processed/predictions/llm_predictions_unseen_val_colab_local_classifier.parquet
data/processed/predictions/llm_predictions_unseen_test_colab_local_classifier.parquet
data/processed/predictions/llm_predictions_safeguard_test_colab_local_classifier.parquet
data/processed/predictions_external/llm_predictions_external_deepset_colab_local_classifier.parquet
data/processed/predictions_external/llm_predictions_external_jackhhao_colab_local_classifier.parquet
```

### 3. Validate the handoff

```bash
dvc repro validate_colab_handoff
```

If a file is missing or malformed, this stage fails with the exact path to regenerate or download. The pipeline does not fall back to legacy hosted LLM classifier outputs.

### 4. Resume local DVC through final verdict

```bash
dvc repro final_verdict_report
```

This trains the escalation model from validated Colab classifier outputs plus DVC-produced DeBERTa predictions, runs selective judge stages, and writes:

```bash
reports/pipeline_final_verdict_report.md
```

### Main contract

Every `train_escalating_model` input is either DVC-produced or a validated manual Colab handoff artifact. Missing handoff artifacts are errors, not optional skips.
```

- [ ] **Step 2: Remove or demote legacy command examples from README**

Remove primary-path recommendations for:

```text
./run_inference.sh
dvc repro llm_classifier
dvc repro llm_classifier_val
dvc repro research
dvc repro research_val
dvc repro train_risk_model
dvc repro risk_model
dvc repro eval_new
dvc repro eval_new_external@...
python -m src.cli.eval_new
python -m src.cli.run_baseline
python -m src.cli.eval_baselines
```

If a reference remains because the file has not yet been approved for deletion, mark it as legacy and not part of the canonical path.

- [ ] **Step 3: Run README stale-reference checks**

Run:

```bash
rg -n "Two Pipeline Modes|run_inference|llm_classifier_val|train_risk_model|risk_model|eval_new|eval_baselines|test_unseen" README.md
```

Expected: no matches in canonical run instructions. If matches remain in a legacy note, ensure the surrounding text says they are not canonical.

- [ ] **Step 4: Commit README update**

Run:

```bash
git add README.md
git commit -m "Document canonical pipeline path"
```

Expected: commit succeeds.

---

### Task 9: Produce Deletion Candidate List And Stop

**Files:**
- Create: `docs/cleanup/deletion_candidates.md`
- No deletion in this task.

- [ ] **Step 1: Create cleanup docs directory**

Run:

```bash
mkdir -p docs/cleanup
```

Expected: directory exists.

- [ ] **Step 2: Inventory current tracked files**

Run:

```bash
git ls-files > /tmp/llm_gate_tracked_files.txt
```

Expected: tracked file list is available for classification.

- [ ] **Step 3: Create deletion candidate document**

Create `docs/cleanup/deletion_candidates.md` with this structure and fill it using the current repository inventory:

```markdown
# Deletion Candidate List

This document is an approval gate. No file listed here is deleted until the user approves the specific deletion set.

## Classification Rules

- Canonical runtime: required by the fresh DVC + Colab handoff + final verdict path.
- Reusable helper: not directly canonical, but used by kept runtime modules, tests, or notebooks.
- Historical docs: preserved process or research history.
- Stale generated artifact: generated output that is obsolete, duplicated, or no longer part of the canonical output contract.
- Removable legacy path: runtime code, notebook, script, DVC stage, report path, or test that supports only non-main behavior.

## Canonical Runtime

List files and stages that must stay.

## Reusable Helper

List helper files that must stay for canonical runtime or tests.

## Historical Docs

List preserved docs, including `docs/superpowers/` and `openspec/changes/`.

## Stale Generated Artifact Candidates

List generated artifacts that appear stale or duplicated. Include why each is stale and whether it is tracked by git or DVC.

## Removable Legacy Path Candidates

List candidate DVC stages, source files, scripts, notebooks, tests, and docs/report references. Include the reason each is non-main.

## Explicitly Not Deleting Before Approval

State that no deletion has been performed and implementation is paused for approval.
```

- [ ] **Step 4: Use these commands to support classification**

Run:

```bash
dvc stage list
rg -n "run_inference|eval_new|train_risk_model|risk_model|research_external_llm|research_external@|llm_classifier_val|test_unseen|baselines|posthoc_benign" README.md dvc.yaml src tests docs -g '!docs/superpowers/**' -g '!openspec/**'
git ls-files notebooks scripts src/cli src/baselines reports docs research_docs | sort
```

Expected: use outputs to fill the candidate list. Do not delete files.

- [ ] **Step 5: Commit deletion candidate list**

Run:

```bash
git add docs/cleanup/deletion_candidates.md
git commit -m "List cleanup deletion candidates"
```

Expected: commit succeeds.

- [ ] **Step 6: Stop for user approval**

Send the user the path:

```text
docs/cleanup/deletion_candidates.md
```

State clearly:

```text
No deletion has been performed. Please approve the specific deletion candidates before I remove anything.
```

Do not continue to deletion tasks until the user approves.

---

### Task 10: Delete Approved Items Only

**Files:**
- Modify only files explicitly approved from `docs/cleanup/deletion_candidates.md`.
- This task is blocked until user approval.

- [ ] **Step 1: Confirm approval scope**

Before deleting, restate the exact approved files/stages. If approval is partial, delete only the approved subset.

- [ ] **Step 2: Remove approved DVC stages**

Edit `dvc.yaml` to remove only approved non-main stages. Do not remove `validate_colab_handoff`, `train_escalating_model`, judge stages used by `final_verdict_report`, or `final_verdict_report`.

- [ ] **Step 3: Remove approved files**

Use `git rm` only for approved tracked files. For example, if the user approves deleting `run_inference.sh` and `src/cli/eval_new.py`, run:

```bash
git rm run_inference.sh src/cli/eval_new.py
```

Expected: only approved files are staged for deletion.

- [ ] **Step 4: Remove stale references after approved deletions**

Run:

```bash
rg -n "run_inference|eval_new|train_risk_model|risk_model|research_external_llm|llm_classifier_val|test_unseen|baselines|posthoc_benign" README.md dvc.yaml src tests docs -g '!docs/superpowers/**' -g '!openspec/**'
```

Expected: matches are either gone or intentionally preserved in historical docs outside the runtime path.

- [ ] **Step 5: Run verification after deletion**

Run:

```bash
pytest tests/test_validate_colab_handoff.py tests/test_escalating_model.py tests/test_judge_colab_local_predictions.py tests/test_final_verdict_report.py -q
dvc stage list
dvc dag final_verdict_report
dvc status final_verdict_report
```

Expected: tests pass and DVC output is understandable.

- [ ] **Step 6: Commit approved deletion**

Run:

```bash
git add dvc.yaml dvc.lock README.md docs/cleanup/deletion_candidates.md
git status --short
git commit -m "Remove approved legacy pipeline paths"
```

Expected: commit includes only approved deletions and related stale-reference cleanup.

---

### Task 11: Final Verification

**Files:**
- No planned edits unless verification exposes a blocker.

- [ ] **Step 1: Run canonical unit tests**

Run:

```bash
pytest tests/test_validate_colab_handoff.py tests/test_escalating_model.py tests/test_judge_colab_local_predictions.py tests/test_final_verdict_report.py tests/test_colab_local_llm_classifier_notebook.py -q
```

Expected: pass.

- [ ] **Step 2: Run DVC checks**

Run:

```bash
dvc stage list
dvc dag final_verdict_report
dvc status final_verdict_report
```

Expected: commands succeed and the canonical path is clear. If outputs are missing because Colab handoff artifacts are absent, the missing paths must be actionable and exact.

- [ ] **Step 3: Run handoff validation**

Run:

```bash
dvc repro validate_colab_handoff
```

Expected: pass when Colab artifacts exist and match schema; otherwise fail with exact missing or malformed paths.

- [ ] **Step 4: Run final report path when artifacts and API credentials are available**

Run:

```bash
dvc repro final_verdict_report
```

Expected: writes `reports/pipeline_final_verdict_report.md`. If judge API credentials or rate limits block execution, document the exact blocker without changing thresholds, metrics, or evaluation logic.

- [ ] **Step 5: Final status summary**

Run:

```bash
git status --short
```

Expected: only intentional files are modified or untracked. Report any remaining untracked user files separately.

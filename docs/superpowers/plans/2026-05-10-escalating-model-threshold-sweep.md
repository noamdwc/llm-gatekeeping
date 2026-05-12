# Escalating Model Threshold Sweep Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an `unseen_val` threshold sweep for the escalating model.

**Architecture:** Add a pure helper in `src/escalating_model.py` that computes threshold operating-point metrics from scored rows. Update `src.cli.train_escalating_model` to call the helper only for `unseen_val`, write a CSV artifact, and include the sweep table in the markdown report.

**Tech Stack:** Python, pandas, numpy, pytest, existing parquet/CSV research artifacts.

---

## File Structure

- Modify `src/escalating_model.py`: define sweep column names, threshold helper, and report rendering support.
- Modify `src/cli/train_escalating_model.py`: add CLI argument for sweep output, compute the sweep from `unseen_val`, and pass it into the report writer.
- Modify `tests/test_escalating_model.py`: add unit coverage for sweep metrics and CLI artifact/report output.

### Task 1: Threshold Sweep Helper

**Files:**
- Modify: `tests/test_escalating_model.py`
- Modify: `src/escalating_model.py`

- [ ] **Step 1: Write the failing test**

Add a test that imports `THRESHOLD_SWEEP_COLS` and `evaluate_threshold_sweep`, builds a scored frame with `needs_escalation` and `escalation_score`, and asserts the threshold metrics.

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_escalating_model.py::TestPocEvaluation::test_threshold_sweep_reports_operating_point_metrics -v`

Expected: import failure because the helper does not exist.

- [ ] **Step 3: Write minimal implementation**

Add `THRESHOLD_SWEEP_COLS` and `evaluate_threshold_sweep(scored_df, thresholds=None)` to `src/escalating_model.py`. Use default thresholds from `0.0` to `1.0` by `0.05`, inclusive.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_escalating_model.py::TestPocEvaluation::test_threshold_sweep_reports_operating_point_metrics -v`

Expected: PASS.

### Task 2: CLI Artifact and Report Integration

**Files:**
- Modify: `tests/test_escalating_model.py`
- Modify: `src/cli/train_escalating_model.py`
- Modify: `src/escalating_model.py`

- [ ] **Step 1: Write the failing CLI test**

Extend `TestTrainEscalatingModelCli.test_cli_writes_expected_artifacts` to assert that `escalating_model_threshold_sweep_unseen_val.csv` exists, has columns matching `THRESHOLD_SWEEP_COLS`, and that the report contains `Threshold Sweep`.

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_escalating_model.py::TestTrainEscalatingModelCli::test_cli_writes_expected_artifacts -v`

Expected: FAIL because the CSV is not written.

- [ ] **Step 3: Write minimal implementation**

Add a `--threshold-sweep-output` CLI argument defaulting to `data/processed/research/escalating_model_threshold_sweep_unseen_val.csv`. Store the scored `unseen_val` frame, compute `evaluate_threshold_sweep`, write the CSV, and pass the DataFrame into `write_escalating_report`.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_escalating_model.py::TestTrainEscalatingModelCli::test_cli_writes_expected_artifacts -v`

Expected: PASS.

### Task 3: Verification

**Files:**
- No new files.

- [ ] **Step 1: Run focused tests**

Run: `pytest tests/test_escalating_model.py -v`

Expected: all escalating model tests pass.

- [ ] **Step 2: Inspect git diff**

Run: `git diff -- src/escalating_model.py src/cli/train_escalating_model.py tests/test_escalating_model.py docs/superpowers/specs/2026-05-10-escalating-model-threshold-sweep-design.md docs/superpowers/plans/2026-05-10-escalating-model-threshold-sweep.md`

Expected: only threshold-sweep related changes.

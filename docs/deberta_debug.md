# DeBERTa Numeric Debug Guide

## Problem

DeBERTa training can fail with non-finite (NaN/Inf) loss on MPS and CPU devices. Without instrumentation, the pipeline saves a corrupted model and produces all-NaN predictions.

## Debug Modes

### Sanity Forward-Only

Runs forward passes without backward/optimizer to isolate whether NaNs come from the model itself or from the training dynamics.

```bash
python -m src.cli.deberta_classifier --research --cpu \
    --sanity-forward-only --sanity-batches 3 --debug-log-batch-text
```

### Full Debug Instrumentation

Checks tensors at 4 stages per batch: pre-forward, post-forward, post-backward, post-optimizer.

```bash
python -m src.cli.deberta_classifier --research --cpu \
    --debug-numerics --debug-first-n-batches 5
```

### Save Bad Batch Artifacts

Dumps input tensors, loss, logits, and metadata to disk when NaN is detected.

```bash
python -m src.cli.deberta_classifier --research --cpu \
    --debug-numerics --debug-save-bad-batch
```

Artifacts are saved to `artifacts/deberta_classifier/debug/bad_batch_e{E}_s{S}/`.

### Parameter Stats Logging

Log mean/std/min/max for the largest model parameters during debug batches.

```bash
python -m src.cli.deberta_classifier --research --cpu \
    --debug-numerics --debug-log-param-stats --train-only
```

## CLI Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--debug-numerics` | off | Enable numeric debug instrumentation |
| `--debug-first-n-batches N` | 3 | Verbose logging for first N batches |
| `--debug-save-bad-batch` | off | Save artifacts on NaN detection |
| `--debug-stop-on-nan` | on | Stop training on first NaN |
| `--debug-log-param-stats` | off | Log parameter statistics |
| `--debug-log-batch-text` | off | Log decoded batch text |
| `--sanity-forward-only` | off | Forward-only mode (no backward) |
| `--sanity-batches N` | 3 | Number of batches for sanity check |

## Lifecycle Guards

- **Training failure** → `sys.exit(1)`, no model saved, no predictions run
- **Save** → checks all model parameters are finite before writing
- **Load** → warns if loaded model has non-finite parameters
- **Predict** → checks model health before inference

## What Each Stage Diagnoses

| Stage | Detects |
|-------|---------|
| Pre-forward | Corrupt input tensors |
| Post-forward | NaN loss from model computation |
| Post-backward | Exploding/NaN gradients |
| Post-optimizer | Weight corruption after parameter update |

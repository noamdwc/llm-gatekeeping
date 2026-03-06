#!/usr/bin/env bash
#
# run_synth.sh — Run the frozen synthetic_benign DVC stage.
#
# Unfreezes the stage, runs dvc repro, then re-freezes on exit.
# After completion, run `dvc repro` to propagate through the pipeline.
#
# Usage:
#   ./run_synth.sh            # normal run
#   ./run_synth.sh --force    # force re-run
#
set -euo pipefail

DVC_FLAGS=""
if [[ "${1:-}" == "--force" ]]; then
    DVC_FLAGS="--force"
fi

echo "Unfreezing synthetic_benign stage..."
dvc unfreeze synthetic_benign

trap 'echo "Re-freezing synthetic_benign..."; dvc freeze synthetic_benign' EXIT

echo "Running synthetic benign generation..."
dvc repro $DVC_FLAGS synthetic_benign

echo "Done. Run 'dvc repro' to propagate through pipeline."

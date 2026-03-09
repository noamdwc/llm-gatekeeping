#!/usr/bin/env bash
#
# run_synth.sh — Generate synthetic benign prompts (requires NVIDIA_API_KEY).
#
# This is a standalone script, not a DVC stage, because it requires API tokens
# and should only run on demand. After generation, run `dvc repro` to propagate
# the new benign data through the pipeline.
#
# Usage:
#   ./run_synth.sh                        # generate all categories (config quotas)
#   ./run_synth.sh --category C --limit 50  # single category, limited
#
set -euo pipefail

echo "Running synthetic benign generation..."
python -m src.cli.generate_synthetic_benign "$@"

echo "Done. Run 'dvc repro' to propagate through pipeline."

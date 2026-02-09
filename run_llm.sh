#!/usr/bin/env bash
#
# run_llm.sh — Run the full pipeline including frozen LLM stages.
#
# Usage:
#   ./run_llm.sh
#
set -euo pipefail

LLM_STAGES="llm_classifier hybrid_router evaluate"

echo "Unfreezing LLM stages..."
dvc freeze --unfreeze $LLM_STAGES

# Re-freeze on exit (success or failure) so dvc.yaml stays clean
trap 'echo "Re-freezing LLM stages..."; dvc freeze $LLM_STAGES' EXIT

echo "Running full pipeline..."
dvc repro

echo "Done."

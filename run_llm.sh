#!/usr/bin/env bash
#
# run_llm.sh — Run the full research pipeline including frozen LLM stages.
#
# Unfreezes the llm_classifier stage, runs dvc repro, then re-freezes.
# This triggers the full pipeline: preprocess → splits → ml_model →
# llm_classifier → research (+ external datasets).
#
# Usage:
#   ./run_llm.sh
#
set -euo pipefail

LLM_STAGES="llm_classifier"

echo "Unfreezing LLM stages..."
dvc unfreeze $LLM_STAGES

# Re-freeze on exit (success or failure) so dvc.yaml stays clean
trap 'echo "Re-freezing LLM stages..."; dvc freeze $LLM_STAGES' EXIT

echo "Running full research pipeline..."
dvc repro

echo "Done."

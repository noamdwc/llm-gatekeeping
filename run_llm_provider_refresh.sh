#!/usr/bin/env bash
#
# run_llm_provider_refresh.sh — Recompute LLM-dependent DVC stages after changing provider.
#
# Why this exists:
#   DVC does not track LLM_PROVIDER, so switching providers will not invalidate
#   existing LLM outputs automatically. This script forces re-execution from the
#   LLM-producing stages downstream.
#
# Usage:
#   ./run_llm_provider_refresh.sh
#   ./run_llm_provider_refresh.sh --provider openai
#   ./run_llm_provider_refresh.sh --provider nim --config configs/default.yaml
#
set -euo pipefail

PROVIDER="${LLM_PROVIDER:-}"
CONFIG="configs/default.yaml"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --provider)
      PROVIDER="$2"
      shift 2
      ;;
    --config)
      CONFIG="$2"
      shift 2
      ;;
    -h|--help)
      sed -n '2,/^$/{ s/^# //; s/^#//; p }' "$0"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

if [[ -z "$PROVIDER" ]]; then
  echo "Error: provider is required. Pass --provider or set LLM_PROVIDER."
  exit 1
fi

if [[ "$PROVIDER" != "nim" && "$PROVIDER" != "openai" ]]; then
  echo "Error: provider must be 'nim' or 'openai'."
  exit 1
fi

if [[ ! -f "$CONFIG" ]]; then
  echo "Error: config not found: $CONFIG"
  exit 1
fi

LLM_ROOT_STAGES=(
  llm_classifier
  research_external_llm
)

echo "Refreshing DVC outputs for provider: $PROVIDER"
echo "Config: $CONFIG"
echo "Stages: ${LLM_ROOT_STAGES[*]}"

echo "Unfreezing LLM stages..."
dvc unfreeze llm_classifier

trap 'echo "Re-freezing LLM stages..."; dvc freeze llm_classifier' EXIT

echo "Running DVC repro from LLM stages downstream..."
LLM_PROVIDER="$PROVIDER" SKIP_LLM=0 dvc repro --force-downstream "${LLM_ROOT_STAGES[@]}"

echo "Done."

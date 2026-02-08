#!/usr/bin/env bash
#
# run_pipeline.sh — Run the full adversarial-prompt detection pipeline.
#
# Usage:
#   ./run_pipeline.sh [OPTIONS]
#
# Options:
#   --skip-llm          Skip API-dependent steps (llm, hybrid, evaluate)
#   --limit N           Max samples for LLM / hybrid steps (default: 100)
#   --no-wandb          Disable wandb logging
#   --config PATH       Config YAML passed to every step
#   --dynamic           Use dynamic few-shot for LLM classifier
#   --start-from STEP   Resume from a step: preprocess|splits|ml|llm|hybrid|evaluate
#
set -euo pipefail

# ── Defaults ─────────────────────────────────────────────────────────────────
SKIP_LLM=false
LIMIT=100
NO_WANDB=false
CONFIG=""
DYNAMIC=false
START_FROM="preprocess"

# ── Parse arguments ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-llm)   SKIP_LLM=true; shift ;;
    --limit)      LIMIT="$2"; shift 2 ;;
    --no-wandb)   NO_WANDB=true; shift ;;
    --config)     CONFIG="$2"; shift 2 ;;
    --dynamic)    DYNAMIC=true; shift ;;
    --start-from) START_FROM="$2"; shift 2 ;;
    -h|--help)
      sed -n '2,/^$/{ s/^# //; s/^#//; p }' "$0"
      exit 0
      ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

# ── Helpers ──────────────────────────────────────────────────────────────────
STEPS=(preprocess splits ml llm hybrid evaluate)
SECONDS=0

step_index() {
  for i in "${!STEPS[@]}"; do
    [[ "${STEPS[$i]}" == "$1" ]] && echo "$i" && return
  done
  echo "Invalid step: $1. Valid steps: ${STEPS[*]}" >&2; exit 1
}

START_IDX=$(step_index "$START_FROM")

should_run() {
  local idx
  idx=$(step_index "$1")
  [[ "$idx" -ge "$START_IDX" ]]
}

banner() {
  local step_num=$1 total=$2 name=$3
  echo ""
  echo "================================================================"
  echo "  Step ${step_num}/${total}: ${name}"
  echo "================================================================"
}

config_flag() {
  [[ -n "$CONFIG" ]] && echo "--config $CONFIG"
}

# ── Determine total steps ───────────────────────────────────────────────────
if $SKIP_LLM; then
  TOTAL=3
else
  TOTAL=6
fi

STEP_NUM=0

# ── Step 1: Preprocess ──────────────────────────────────────────────────────
if should_run preprocess; then
  STEP_NUM=$((STEP_NUM + 1))
  banner $STEP_NUM $TOTAL "Preprocess"
  # shellcheck disable=SC2046
  python -m src.preprocess $(config_flag)
fi

# ── Step 2: Build Splits ────────────────────────────────────────────────────
if should_run splits; then
  STEP_NUM=$((STEP_NUM + 1))
  banner $STEP_NUM $TOTAL "Build Splits"
  # shellcheck disable=SC2046
  python -m src.build_splits $(config_flag)
fi

# ── Step 3: ML Baseline ─────────────────────────────────────────────────────
if should_run ml; then
  STEP_NUM=$((STEP_NUM + 1))
  banner $STEP_NUM $TOTAL "ML Baseline"
  ML_ARGS=""
  [[ -n "$CONFIG" ]] && ML_ARGS="$ML_ARGS --config $CONFIG"
  $NO_WANDB && ML_ARGS="$ML_ARGS --no-wandb"
  # shellcheck disable=SC2086
  python -m src.ml_baseline $ML_ARGS
fi

# ── API-dependent steps (skipped with --skip-llm) ───────────────────────────
if ! $SKIP_LLM; then

  # ── Step 4: LLM Classifier ────────────────────────────────────────────────
  if should_run llm; then
    STEP_NUM=$((STEP_NUM + 1))
    banner $STEP_NUM $TOTAL "LLM Classifier"
    LLM_ARGS="--split test --limit $LIMIT"
    [[ -n "$CONFIG" ]] && LLM_ARGS="$LLM_ARGS --config $CONFIG"
    $NO_WANDB && LLM_ARGS="$LLM_ARGS --no-wandb"
    $DYNAMIC  && LLM_ARGS="$LLM_ARGS --dynamic"
    # shellcheck disable=SC2086
    python -m src.llm_classifier $LLM_ARGS
  fi

  # ── Step 5: Hybrid Router ─────────────────────────────────────────────────
  if should_run hybrid; then
    STEP_NUM=$((STEP_NUM + 1))
    banner $STEP_NUM $TOTAL "Hybrid Router"
    HYB_ARGS="--limit $LIMIT"
    [[ -n "$CONFIG" ]] && HYB_ARGS="$HYB_ARGS --config $CONFIG"
    $NO_WANDB && HYB_ARGS="$HYB_ARGS --no-wandb"
    # shellcheck disable=SC2086
    python -m src.hybrid_router $HYB_ARGS
  fi

  # ── Step 6: Evaluate ──────────────────────────────────────────────────────
  if should_run evaluate; then
    STEP_NUM=$((STEP_NUM + 1))
    banner $STEP_NUM $TOTAL "Evaluate"
    EVAL_ARGS="--predictions data/processed/predictions_test.csv"
    [[ -n "$CONFIG" ]] && EVAL_ARGS="$EVAL_ARGS --config $CONFIG"
    # shellcheck disable=SC2086
    python -m src.evaluate $EVAL_ARGS
  fi

fi

# ── Done ─────────────────────────────────────────────────────────────────────
echo ""
echo "================================================================"
echo "  Pipeline finished in ${SECONDS}s"
echo "================================================================"

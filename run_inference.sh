#!/usr/bin/env bash
#
# run_inference.sh — Lightweight inference pipeline (no DVC).
#
# Runs the classifier on a data split and generates an evaluation report.
# Assumes ml_baseline.pkl and split parquets already exist (from dvc repro).
#
# Usage:
#   ./run_inference.sh --mode ml --split test               # ML-only, instant
#   ./run_inference.sh --mode hybrid --split test --limit 100  # Hybrid (API tokens)
#   ./run_inference.sh --mode llm --split test --limit 50      # LLM-only (API tokens)
#   ./run_inference.sh --mode ml --split test_unseen           # Generalization test
#
# Options:
#   --mode MODE       ml | hybrid | llm (default: ml)
#   --split SPLIT     test | val | test_unseen (default: test)
#   --limit N         Max samples for LLM/hybrid (default: 100)
#   --config PATH     Config YAML (default: configs/default.yaml)
#   --no-wandb        Disable wandb logging
#   --dynamic         Use dynamic few-shot for LLM classifier
#
set -euo pipefail

# ── Defaults ─────────────────────────────────────────────────────────────────
MODE="ml"
SPLIT="test"
LIMIT=100
CONFIG=""
NO_WANDB=true
DYNAMIC=false

# ── Parse arguments ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode)      MODE="$2"; shift 2 ;;
    --split)     SPLIT="$2"; shift 2 ;;
    --limit)     LIMIT="$2"; shift 2 ;;
    --config)    CONFIG="$2"; shift 2 ;;
    --no-wandb)  NO_WANDB=true; shift ;;
    --dynamic)   DYNAMIC=true; shift ;;
    -h|--help)
      sed -n '2,/^$/{ s/^# //; s/^#//; p }' "$0"
      exit 0
      ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

# ── Validate ─────────────────────────────────────────────────────────────────
VALID_MODES="ml hybrid llm"
if ! echo "$VALID_MODES" | grep -qw "$MODE"; then
  echo "Error: --mode must be one of: $VALID_MODES"
  exit 1
fi

SPLITS_DIR="data/processed/splits"
MODELS_DIR="data/processed/models"

if [[ ! -f "$SPLITS_DIR/${SPLIT}.parquet" ]]; then
  echo "Error: Split not found: $SPLITS_DIR/${SPLIT}.parquet"
  echo "Run 'dvc repro build_splits' first."
  exit 1
fi

if [[ ! -f "$MODELS_DIR/ml_baseline.pkl" ]]; then
  echo "Error: ML model not found: $MODELS_DIR/ml_baseline.pkl"
  echo "Run 'dvc repro ml_model' first."
  exit 1
fi

# ── Helpers ──────────────────────────────────────────────────────────────────
SECONDS=0

config_flag() {
  [[ -n "$CONFIG" ]] && echo "--config $CONFIG"
}

banner() {
  echo ""
  echo "================================================================"
  echo "  $1"
  echo "================================================================"
}

# ── Run inference based on mode ──────────────────────────────────────────────

if [[ "$MODE" == "ml" ]]; then
  banner "ML Inference — split=$SPLIT"
  INF_ARGS="--mode ml --split $SPLIT"
  [[ -n "$CONFIG" ]] && INF_ARGS="$INF_ARGS --config $CONFIG"
  # shellcheck disable=SC2086
  python -m src.cli.infer_split $INF_ARGS

elif [[ "$MODE" == "hybrid" ]]; then
  banner "Hybrid Inference — split=$SPLIT, limit=$LIMIT"
  HYB_ARGS="--limit $LIMIT"
  [[ -n "$CONFIG" ]] && HYB_ARGS="$HYB_ARGS --config $CONFIG"
  $NO_WANDB && HYB_ARGS="$HYB_ARGS --no-wandb"
  # shellcheck disable=SC2086
  python -m src.hybrid_router $HYB_ARGS

elif [[ "$MODE" == "llm" ]]; then
  banner "LLM Inference — split=$SPLIT, limit=$LIMIT"
  LLM_ARGS="--split $SPLIT --limit $LIMIT"
  [[ -n "$CONFIG" ]] && LLM_ARGS="$LLM_ARGS --config $CONFIG"
  $NO_WANDB && LLM_ARGS="$LLM_ARGS --no-wandb"
  $DYNAMIC && LLM_ARGS="$LLM_ARGS --dynamic"
  # shellcheck disable=SC2086
  python -m src.llm_classifier.llm_classifier $LLM_ARGS
fi

# ── Done ─────────────────────────────────────────────────────────────────────
echo ""
echo "================================================================"
echo "  Inference ($MODE) finished in ${SECONDS}s"
echo "================================================================"

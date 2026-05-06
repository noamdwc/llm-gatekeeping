#!/usr/bin/env bash
#
# run_vllm_cpu_docker.sh - Start a local vLLM OpenAI-compatible CPU server.
#
# Defaults are conservative for a 16GB Apple Silicon Mac running Docker.
#
# Usage:
#   ./run_vllm_cpu_docker.sh
#   MODEL_ID=Qwen/Qwen2.5-0.5B-Instruct ./run_vllm_cpu_docker.sh
#
# Client settings:
#   OPENAI_BASE_URL=http://127.0.0.1:8000/v1
#   OPENAI_API_KEY=EMPTY
#
set -euo pipefail

MODEL_ID="${MODEL_ID:-HuggingFaceTB/SmolLM2-360M-Instruct}"
VLLM_IMAGE="${VLLM_IMAGE:-vllm/vllm-openai-cpu:latest-arm64}"
HF_CACHE_DIR="${HF_CACHE_DIR:-$HOME/.cache/huggingface}"
VLLM_CPU_KVCACHE_SPACE=2

mkdir -p "$HF_CACHE_DIR"

docker run --rm -it \
  --platform linux/arm64 \
  -p 8000:8000 \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  -e HF_TOKEN="${HF_TOKEN:-}" \
  -e VLLM_CPU_KVCACHE_SPACE="$VLLM_CPU_KVCACHE_SPACE" \
  --shm-size=2g \
  "$VLLM_IMAGE" \
  --model "$MODEL_ID" \
  --host 0.0.0.0 \
  --port 8000 \
  --max-model-len 1024 \
  --max-num-seqs 2 \
  --tensor-parallel-size 1

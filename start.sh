#!/usr/bin/env bash
set -euo pipefail

: "${MODEL_ID:?set MODEL_ID to the Muse-Glimmer model repo/path}"
: "${API_KEY:?set API_KEY for the backend}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-muse-glimmer}"
export OCR_MODEL_NAME="$SERVED_MODEL_NAME"

# 1. vLLM OCR server, localhost only — the backend reaches it via 127.0.0.1:8000.
python3 -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_ID" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --reasoning-parser muse_glimmer \
  --enable-auto-tool-choice --tool-call-parser muse_glimmer \
  --generation-config auto \
  --host 127.0.0.1 --port 8000 \
  ${VLLM_EXTRA_ARGS:-} &
VLLM_PID=$!

# 2. Block until vLLM answers (weights load can take minutes); bail if it dies.
echo "waiting for vLLM on :8000 ..."
until curl -sf http://127.0.0.1:8000/health >/dev/null; do
  kill -0 "$VLLM_PID" 2>/dev/null || { echo "vLLM exited during startup"; exit 1; }
  sleep 3
done
echo "vLLM ready"

# 3. FastAPI backend (DINOv2 loads on CPU here).
python3 -m uvicorn src.main:app --host 0.0.0.0 --port 8080 &
API_PID=$!

# If either process exits, stop the container so RunPod restarts it.
wait -n "$VLLM_PID" "$API_PID"
echo "a service exited; shutting down"
kill "$VLLM_PID" "$API_PID" 2>/dev/null || true
exit 1

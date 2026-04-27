#!/usr/bin/env bash
set -euo pipefail

HOST_ADDRESS="${HOST_ADDRESS:-0.0.0.0}"
PORT="${PORT:-12205}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

echo "Starting taiko-diffusion backend on http://${HOST_ADDRESS}:${PORT}"
python -m uvicorn webapp.backend.main:app --host "${HOST_ADDRESS}" --port "${PORT}" --reload

#!/usr/bin/env bash
set -euo pipefail
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
RELOAD_FLAG="${RELOAD_FLAG:---reload}"

echo "Starting API server on http://${HOST}:${PORT} (press Ctrl+C to stop)"
python -m uvicorn api.app:app --host "${HOST}" --port "${PORT}" ${RELOAD_FLAG}

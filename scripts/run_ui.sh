#!/usr/bin/env bash
set -euo pipefail
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8501}"

echo "Starting Streamlit UI on http://${HOST}:${PORT} (press Ctrl+C to stop)"
python -m streamlit run webapp/app.py \
  --server.address "${HOST}" \
  --server.port "${PORT}" \
  --server.headless true

#!/usr/bin/env bash
set -euo pipefail
echo "Starting API server on http://localhost:8000 (press Ctrl+C to stop)"
python -m uvicorn api.app:app --reload --port 8000

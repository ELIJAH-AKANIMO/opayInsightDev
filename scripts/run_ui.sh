#!/usr/bin/env bash
set -euo pipefail
echo "Starting Streamlit UI (press Ctrl+C to stop)"
python -m streamlit run webapp/app.py

#!/usr/bin/env bash
set -euo pipefail
VENV_DIR='.venv'

if [ ! -d "$VENV_DIR" ]; then
  python -m venv "$VENV_DIR"
  echo "Created virtual environment at $VENV_DIR"
else
  echo "Virtual environment already exists at $VENV_DIR"
fi

# Use the venv's python to install
VENV_PY="$VENV_DIR/bin/python"
if [ ! -x "$VENV_PY" ]; then
  echo "Could not find $VENV_PY; ensure python is installed and on PATH" >&2
  exit 1
fi

$VENV_PY -m pip install --upgrade pip
$VENV_PY -m pip install -r requirements-web.txt

echo "Installed requirements into $VENV_DIR"
echo "Activate with: source $VENV_DIR/bin/activate"
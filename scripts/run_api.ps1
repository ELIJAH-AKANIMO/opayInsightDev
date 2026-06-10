# Run API (PowerShell)
# Usage: .\scripts\run_api.ps1
# Uses `python -m uvicorn` to avoid relying on uvicorn being on PATH
$ErrorActionPreference = "Stop"
Write-Host "Starting API server on http://localhost:8000 (press Ctrl+C to stop)"
python -m uvicorn api.app:app --reload --port 8000

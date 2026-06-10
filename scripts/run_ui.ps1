# Run Streamlit UI (PowerShell)
# Usage: .\scripts\run_ui.ps1
$ErrorActionPreference = "Stop"
Write-Host "Starting Streamlit UI (press Ctrl+C to stop)"
python -m streamlit run webapp/app.py

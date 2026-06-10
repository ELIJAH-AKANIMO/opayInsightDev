# Create virtual environment and install dependencies (PowerShell)
# Usage: .\scripts\create_venv.ps1
$ErrorActionPreference = 'Stop'
$venvDir = '.venv'

if (-Not (Test-Path $venvDir)) {
    python -m venv $venvDir
    Write-Host "Created virtual environment at $venvDir"
} else {
    Write-Host "Virtual environment already exists at $venvDir"
}

# Use the venv's python to install
$python = Join-Path $venvDir 'Scripts\python.exe'
if (-Not (Test-Path $python)) {
    Write-Error "Could not find $python; ensure python is installed and on PATH"
    exit 1
}

& $python -m pip install --upgrade pip
& $python -m pip install -r requirements-web.txt
Write-Host "Installed requirements into $venvDir"
Write-Host "Activate with: . $venvDir\Scripts\Activate.ps1"
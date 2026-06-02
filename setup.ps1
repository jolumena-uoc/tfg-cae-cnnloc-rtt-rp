# Setup script for the CAE-CNNLoc 2D-Temporal experiments.
#
# Run from PowerShell in this folder:
#   .\setup.ps1
#
# The script creates a local virtual environment in .venv and installs all
# dependencies declared in requirements.txt. Subsequent shells must activate
# the venv with:
#   .\.venv\Scripts\Activate.ps1

$ErrorActionPreference = "Stop"

$pythonExe = "py -3.12"
$venvPath = ".venv"

if (-Not (Test-Path $venvPath)) {
    Write-Host "[1/3] Creating virtual environment in $venvPath ..." -ForegroundColor Cyan
    & py -3.12 -m venv $venvPath
} else {
    Write-Host "[1/3] Reusing existing $venvPath" -ForegroundColor Cyan
}

$venvPython = Join-Path $venvPath "Scripts\python.exe"

Write-Host "[2/3] Upgrading pip..." -ForegroundColor Cyan
& $venvPython -m pip install --upgrade pip

Write-Host "[3/3] Installing project dependencies (takes 3-5 minutes)..." -ForegroundColor Cyan
& $venvPython -m pip install -r requirements.txt

Write-Host ""
Write-Host "Setup complete." -ForegroundColor Green
Write-Host "Activate the environment with:" -ForegroundColor Yellow
Write-Host "    .\.venv\Scripts\Activate.ps1"
Write-Host ""
Write-Host "Sanity check:"
& $venvPython -c "import sys; print('python:', sys.version.split()[0])"
& $venvPython -c "import numpy, pandas, tensorflow as tf; print('numpy:', numpy.__version__, '| pandas:', pandas.__version__, '| tensorflow:', tf.__version__)"

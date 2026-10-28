# One-command setup for Windows PowerShell.
#   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
#   .\setup.ps1
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

# Pick a Python: prefer the py launcher with 3.12 / 3.11, else whatever "python" is.
$py = $null
foreach ($v in "3.12", "3.11", "3.13", "3.10") {
    if (Get-Command py -ErrorAction SilentlyContinue) {
        & py "-$v" --version *> $null
        if ($LASTEXITCODE -eq 0) { $py = @("py", "-$v"); break }
    }
}
if (-not $py) {
    if (Get-Command python -ErrorAction SilentlyContinue) { $py = @("python") }
    else {
        Write-Host "Python not found. Install it with:  winget install -e --id Python.Python.3.12" -ForegroundColor Red
        exit 1
    }
}
$pyExe = $py[0]
$pyArgs = @($py | Select-Object -Skip 1)
Write-Host "Using: $(& $pyExe @pyArgs --version)" -ForegroundColor Cyan

if (-not (Test-Path ".venv")) {
    Write-Host "Creating virtual environment (.venv)..." -ForegroundColor Cyan
    & $pyExe @pyArgs -m venv .venv
}

$venvPy = ".\.venv\Scripts\python.exe"
Write-Host "Installing packages (chromadb, numpy)..." -ForegroundColor Cyan
& $venvPy -m pip install --upgrade pip --quiet
& $venvPy -m pip install -r requirements.txt --quiet
if ($LASTEXITCODE -ne 0) { Write-Host "pip install failed - see docs\04-windows-setup.md#troubleshooting" -ForegroundColor Red; exit 1 }

Write-Host "Downloading the embedding model (one time, ~80 MB)..." -ForegroundColor Cyan
& $venvPy -c "import sys; sys.path.insert(0, 'rag'); from common import get_embedder; get_embedder()"

Write-Host ""
Write-Host "Setup complete. Next:" -ForegroundColor Green
Write-Host "  .\.venv\Scripts\Activate.ps1"
Write-Host "  python rag\step1_embeddings.py"
Write-Host "  python rag\step2_rag_without_vectordb.py"
Write-Host "  python rag\step3_rag_with_chromadb.py"

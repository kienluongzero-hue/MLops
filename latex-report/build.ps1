$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$python = Join-Path $root '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python)) { $python = 'python' }

Push-Location $root
try {
    & $python (Join-Path $PSScriptRoot 'generate_assets.py')
    if ($LASTEXITCODE -ne 0) { throw 'Could not generate report tables and figures.' }
    Push-Location $PSScriptRoot
    try {
        New-Item -ItemType Directory -Force -Path 'build' | Out-Null
        for ($i = 0; $i -lt 2; $i++) {
            & xelatex -interaction=nonstopmode -halt-on-error -file-line-error -output-directory=build main.tex | Out-Null
            if ($LASTEXITCODE -ne 0) { throw 'XeLaTeX failed; see latex-report/build/main.log.' }
        }
    } finally {
        Pop-Location
    }
    $destination = Join-Path $root 'output\pdf'
    New-Item -ItemType Directory -Force -Path $destination | Out-Null
    Copy-Item -LiteralPath (Join-Path $PSScriptRoot 'build\main.pdf') -Destination (Join-Path $destination 'bao_cao_MLOps_bao_tri_du_doan.pdf') -Force
    Write-Host "Created $destination\bao_cao_MLOps_bao_tri_du_doan.pdf"
} finally {
    Pop-Location
}

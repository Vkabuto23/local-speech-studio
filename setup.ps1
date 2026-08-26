param(
    [switch]$CpuOnly,
    [switch]$SkipNeMo
)

$ErrorActionPreference = "Stop"
$ProjectDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$VenvDir = Join-Path $ProjectDir ".venv"
$Python = Join-Path $VenvDir "Scripts\python.exe"

Set-Location $ProjectDir

if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    throw "Python 3.10-3.12 is required. Install it from https://www.python.org/downloads/"
}

if (-not (Test-Path -LiteralPath $Python)) {
    python -m venv $VenvDir
}

& $Python -m pip install --upgrade pip wheel setuptools

if ($CpuOnly) {
    & $Python -m pip install torch==2.10.0 torchaudio==2.10.0 --index-url https://download.pytorch.org/whl/cpu
} else {
    & $Python -m pip install torch==2.10.0 torchaudio==2.10.0 --index-url https://download.pytorch.org/whl/cu128
}

& $Python -m pip install -r (Join-Path $ProjectDir "requirements.txt")
& (Join-Path $ProjectDir "setup_gigaam.ps1") -TargetPython $Python

if (-not $SkipNeMo -and -not $CpuOnly) {
    & $Python -m pip install "nemo_toolkit[asr]==2.7.3"
}

$ConfigPath = Join-Path $ProjectDir "config.json"
if (-not (Test-Path -LiteralPath $ConfigPath)) {
    Copy-Item -LiteralPath (Join-Path $ProjectDir "config.example.json") -Destination $ConfigPath
}

if (-not (Get-Command ffmpeg -ErrorAction SilentlyContinue)) {
    Write-Warning "ffmpeg is not in PATH. Install it with: winget install Gyan.FFmpeg"
}

Write-Host ""
Write-Host "Local Speech Studio is installed."
Write-Host "Run start.bat and open http://127.0.0.1:8015"
Write-Host "Recognition models are downloaded automatically on the first transcription."

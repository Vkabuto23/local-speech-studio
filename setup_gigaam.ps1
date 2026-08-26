param(
    [string]$TargetPython = ""
)

$ErrorActionPreference = "Stop"

$ProjectDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RuntimeDir = Join-Path $ProjectDir "gigaam runtime"
$VenvDir = Join-Path $RuntimeDir ".venv-gigaam"
$GigaAMRevision = "7447938d791c4f3e643386ee22c33777004293a5"

New-Item -ItemType Directory -Force -Path $RuntimeDir | Out-Null

if ($TargetPython) {
    $Python = $TargetPython
} else {
    if (-not (Test-Path -LiteralPath (Join-Path $VenvDir "Scripts\python.exe"))) {
        python -m venv --system-site-packages $VenvDir
    }
    $Python = Join-Path $VenvDir "Scripts\python.exe"
}

& $Python -m pip install --upgrade pip
& $Python -m pip install "https://github.com/salute-developers/GigaAM/archive/$GigaAMRevision.zip"

Write-Host "GigaAM runtime is ready: $Python"

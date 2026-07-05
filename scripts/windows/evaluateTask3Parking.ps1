$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Resolve-Path (Join-Path $ScriptDir "..\..")
Set-Location $ProjectRoot
$env:PYTHONPATH = "$ProjectRoot\src;$env:PYTHONPATH"
if (-not $env:OMP_NUM_THREADS) { $env:OMP_NUM_THREADS = "1" }
if (-not $env:MKL_NUM_THREADS) { $env:MKL_NUM_THREADS = "1" }
if (-not $env:OPENBLAS_NUM_THREADS) { $env:OPENBLAS_NUM_THREADS = "1" }
if (-not $env:NUMEXPR_NUM_THREADS) { $env:NUMEXPR_NUM_THREADS = "1" }
if (-not $env:PYTEST_DISABLE_PLUGIN_AUTOLOAD) { $env:PYTEST_DISABLE_PLUGIN_AUTOLOAD = "1" }
$env:PYTHONUNBUFFERED = "1"

$CheckpointPath = $env:CHECKPOINT
if (-not $CheckpointPath) {
  $CheckpointPath = Get-ChildItem -Path "logs\task3" -Recurse -Directory -Filter "final_checkpoint" -ErrorAction SilentlyContinue | Sort-Object FullName | Select-Object -Last 1 | ForEach-Object { $_.FullName }
}
if (-not $CheckpointPath) {
  Write-Error "Checkpoint not found. Set CHECKPOINT=/path/to/checkpoint"
}
python -m diff_drive_rl.tasks.task3.task3_model_test --checkpoint $CheckpointPath @args

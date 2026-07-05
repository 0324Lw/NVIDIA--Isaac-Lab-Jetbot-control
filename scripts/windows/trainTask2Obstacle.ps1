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

python -m diff_drive_rl.tasks.task2.task2_train --num-envs $(if ($env:NUM_ENVS) { $env:NUM_ENVS } else { "4096" }) @args

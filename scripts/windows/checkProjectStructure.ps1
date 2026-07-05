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

$RequiredPaths = @(
  "src\diff_drive_rl",
  "src\diff_drive_rl\tasks\task1\task1_env.py",
  "src\diff_drive_rl\tasks\task2\task2_env.py",
  "src\diff_drive_rl\tasks\task3\task3_env.py",
  "src\diff_drive_rl\tasks\task4\task4_env.py",
  "tests\task1\task1_env_test.py",
  "tests\task2\task2_world_test.py",
  "tests\task3\task3_world_test.py",
  "tests\task4\task4_world_test.py"
)
foreach ($RequiredPath in $RequiredPaths) {
  if (-not (Test-Path $RequiredPath)) {
    Write-Error "Missing path: $RequiredPath"
  }
}
Write-Host "[OK] project structure check passed"

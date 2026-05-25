param(
    [string]$ProjectRoot = (Resolve-Path ".").Path
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

Write-Host "============================================================"
Write-Host "Diff-Drive UGV Task1 Windows readiness check"
Write-Host "ProjectRoot = $ProjectRoot"
Write-Host "============================================================"

$Required = @(
    "src\diff_drive_rl\tasks\task1\task1_config.py",
    "src\diff_drive_rl\tasks\task1\task1_env.py",
    "src\diff_drive_rl\tasks\task1\task1_train.py",
    "src\diff_drive_rl\tasks\task1\task1_model_test.py"
)

$Missing = 0
foreach ($Path in $Required) {
    $Full = Join-Path $ProjectRoot $Path
    if (Test-Path $Full) {
        Write-Host "[OK] $Path"
    } else {
        Write-Host "[MISSING] $Path"
        $Missing += 1
    }
}

if ($Missing -ne 0) {
    throw "Task1 Windows readiness check failed. Missing count: $Missing"
}

Write-Host "[PASS] Task1 Windows readiness check passed."

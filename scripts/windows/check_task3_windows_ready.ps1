param(
    [string]$ProjectRoot = (Resolve-Path ".").Path
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

Write-Host "============================================================"
Write-Host "Diff-Drive UGV Task3 Windows readiness check"
Write-Host "ProjectRoot = $ProjectRoot"
Write-Host "============================================================"

$Required = @(
    "src\diff_drive_rl\tasks\task3\task3_config.py",
    "src\diff_drive_rl\tasks\task3\task3_env.py",
    "src\diff_drive_rl\tasks\task3\task3_train.py",
    "src\diff_drive_rl\tasks\task3\task3_model_test.py"
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
    throw "Task3 Windows readiness check failed. Missing count: $Missing"
}

Write-Host "[PASS] Task3 Windows readiness check passed."

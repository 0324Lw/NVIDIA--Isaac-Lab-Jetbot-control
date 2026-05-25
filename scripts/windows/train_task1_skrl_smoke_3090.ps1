param(
    [string]$ProjectRoot = (Resolve-Path ".").Path,
    [string]$Python = "python",
    [int]$NumEnvs = 512,
    [int]$TotalSteps = 5000,
    [string]$Device = "cuda:0"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

Set-Location $ProjectRoot
$env:PYTHONPATH = "$ProjectRoot\src;$env:PYTHONPATH"

Write-Host "============================================================"
Write-Host "Diff-Drive UGV Task1 skrl training"
Write-Host "ProjectRoot = $ProjectRoot"
Write-Host "NumEnvs     = $NumEnvs"
Write-Host "TotalSteps  = $TotalSteps"
Write-Host "Device      = $Device"
Write-Host "============================================================"

& $Python "src\diff_drive_rl\tasks\task1\task1_train.py" `
    --num-envs $NumEnvs `
    --total-agent-steps $TotalSteps `
    --save-freq-agent-steps $TotalSteps `
    --test-device $Device `
    --headless

if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

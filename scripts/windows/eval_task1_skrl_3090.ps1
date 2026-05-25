param(
    [string]$ProjectRoot = (Resolve-Path ".").Path,
    [string]$Python = "python",
    [Parameter(Mandatory=$true)][string]$Checkpoint,
    [int]$NumEnvs = 4,
    [int]$Steps = 200,
    [string]$Device = "cuda:0"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

Set-Location $ProjectRoot
$env:PYTHONPATH = "$ProjectRoot\src;$env:PYTHONPATH"

Write-Host "============================================================"
Write-Host "Diff-Drive UGV Task1 skrl evaluation"
Write-Host "ProjectRoot = $ProjectRoot"
Write-Host "Checkpoint  = $Checkpoint"
Write-Host "NumEnvs     = $NumEnvs"
Write-Host "Steps       = $Steps"
Write-Host "Device      = $Device"
Write-Host "============================================================"

& $Python "src\diff_drive_rl\tasks\task1\task1_model_test.py" `
    --checkpoint $Checkpoint `
    --num-envs $NumEnvs `
    --steps $Steps `
    --headless `
    --device $Device

if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

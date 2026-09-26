param(
    [string]$TeamName = 'team_innovators'
)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
python utils\package_submission.py --team-name $TeamName

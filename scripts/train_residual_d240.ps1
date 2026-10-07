param([string]$Python = 'python')

$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $PSScriptRoot
$Builder = Join-Path $Root 'scripts\build_residual_pairs.py'
$Trainer = Join-Path $Root 'scripts\train_residual.py'
$TrainSource = Join-Path $Root 'data\feedback\d240_train.jsonl'
$HoldoutSource = Join-Path $Root 'data\feedback\d240_holdout.jsonl'
$GeneratedDir = Join-Path $Root 'reproduced\d240\feedback\features'
$Train = Join-Path $GeneratedDir 'd240_train_features.jsonl'
$Holdout = Join-Path $GeneratedDir 'd240_holdout_features.jsonl'
$FeatureReport = Join-Path $GeneratedDir 'feature_construction.json'
$Model = Join-Path $Root 'reproduced\d240\feedback\full\ranker.json'
$Report = Join-Path $Root 'reproduced\d240\feedback\full\training.json'
$PythonCommand = Get-Command -Name $Python -ErrorAction SilentlyContinue
if (-not $PythonCommand -and -not (Test-Path -LiteralPath $Python)) {
  throw "Python command or executable not found: $Python"
}
foreach ($Path in @($Builder, $Trainer, $TrainSource, $HoldoutSource)) {
  if (-not (Test-Path -LiteralPath $Path)) { throw "Required path not found: $Path" }
}
New-Item -ItemType Directory -Force -Path $GeneratedDir | Out-Null

& $Python $Builder `
  --train-source $TrainSource `
  --holdout-source $HoldoutSource `
  --train-output $Train `
  --holdout-output $Holdout `
  --report $FeatureReport
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $Python $Trainer `
  --train $Train `
  --holdout $Holdout `
  --model $Model `
  --report $Report `
  --epochs 50 `
  --lr 0.08 `
  --l2 0.0001 `
  --seed 23
exit $LASTEXITCODE

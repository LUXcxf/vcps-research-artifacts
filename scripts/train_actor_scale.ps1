param(
  [Parameter(Mandatory = $true)]
  [ValidateSet('D30', 'D60', 'D120', 'D240')]
  [string]$Scale,
  [Parameter(Mandatory = $true)]
  [string]$ModelPath,
  [string]$Python = 'python'
)

$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $PSScriptRoot
$Config = Get-Content -LiteralPath (Join-Path $Root 'configs\data_scales.json') -Raw | ConvertFrom-Json
$Entry = $Config.scales.$Scale
if (-not $Entry) { throw "Missing scale configuration: $Scale" }

$Data = Join-Path $Root $Entry.actor_data
$Output = Join-Path $Root ('reproduced\' + $Scale.ToLowerInvariant() + '\actor')
$Report = Join-Path $Root ('reproduced\' + $Scale.ToLowerInvariant() + '\actor_training.json')
$Trainer = Join-Path $Root 'scripts\train_actor.py'
$PythonCommand = Get-Command -Name $Python -ErrorAction SilentlyContinue
if (-not $PythonCommand -and -not (Test-Path -LiteralPath $Python)) {
  throw "Python command or executable not found: $Python"
}
if ([System.IO.Path]::IsPathRooted($ModelPath) -and -not (Test-Path -LiteralPath $ModelPath)) {
  throw "Local model path not found: $ModelPath"
}
foreach ($Path in @($Data, $Trainer)) {
  if (-not (Test-Path -LiteralPath $Path)) { throw "Required path not found: $Path" }
}

& $Python -u $Trainer `
  --model-path $ModelPath `
  --data $Data `
  --output-dir $Output `
  --report $Report `
  --epochs $Entry.actor_epochs `
  --batch-size $Config.shared.batch_size `
  --gradient-accumulation-steps $Config.shared.gradient_accumulation_steps `
  --learning-rate $Entry.actor_learning_rate `
  --max-length $Entry.max_length `
  --lora-r $Config.shared.lora_r `
  --lora-alpha $Config.shared.lora_alpha `
  --lora-dropout $Config.shared.lora_dropout `
  --target-modules q_proj k_proj v_proj o_proj `
  --disable-gradient-checkpointing `
  --save-every-optimizer-steps 8 `
  --seed $Config.shared.actor_seed `
  --max-optimizer-steps $Entry.selected_optimizer_step
exit $LASTEXITCODE

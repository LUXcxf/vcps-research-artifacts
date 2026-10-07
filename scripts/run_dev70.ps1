param(
  [ValidateSet('actor', 'structured', 'residual', 'combined')]
  [string]$Variant = 'combined',
  [int]$Limit = 70,
  [int]$StartIndex = 0
)

$ErrorActionPreference = 'Stop'
$OutputEncoding = [System.Text.UTF8Encoding]::new()
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new()

$Branch = Split-Path -Parent $PSScriptRoot
$Python = if ($env:VCPS_PYTHON) { $env:VCPS_PYTHON } else { 'python' }
$Runner = Join-Path $Branch 'scripts\run_unified_eval.py'
$Manifest = Join-Path $Branch 'data\manifests\valid_seen_dev70.jsonl'
$Adapter = Join-Path $Branch 'models\actor_adapter'
$ModelName = switch ($Variant) {
  'actor' { $null }
  'structured' { 'structured_progress_only_d240.json' }
  'residual' { 'entity_tp_learned_semantic_d240.json' }
  'combined' { 'structured_progress_plus_entity_semantic_d240.json' }
  default { 'structured_progress_plus_entity_semantic_d240.json' }
}
$ValueModel = if ($ModelName) { Join-Path $Branch ('models\value_models\' + $ModelName) } else { $null }
$Name = 'd240_vcps_' + $Variant + '_dev70'
$Output = Join-Path $Branch ('reproduced\runs\' + $Name + '_episodes.jsonl')
$Report = Join-Path $Branch ('reproduced\reports\' + $Name + '.json')
$Log = Join-Path $Branch ('reproduced\logs\' + $Name + '_live.log')

$PythonCommand = Get-Command -Name $Python -ErrorAction SilentlyContinue
if (-not $PythonCommand -and -not (Test-Path -LiteralPath $Python)) {
  throw "Python command or executable not found: $Python"
}
foreach ($Path in @($Runner, $Manifest, $Adapter, $ValueModel) | Where-Object { $_ }) {
  if (-not (Test-Path -LiteralPath $Path)) { throw "Required path not found: $Path" }
}
New-Item -ItemType Directory -Force -Path (Split-Path $Output), (Split-Path $Report), (Split-Path $Log) | Out-Null

$Selection = if ($Variant -eq 'actor') { 'admissible_observable_logprob' } else { 'admissible_observable_learned_rerank' }
$Arguments = @(
  '-u', $Runner,
  '--manifest', $Manifest,
  '--adapter-path', $Adapter,
  '--output', $Output,
  '--report', $Report,
  '--action-selection', $Selection,
  '--admissible-candidate-limit', '16',
  '--admissible-score-batch-size', '4',
  '--progress-lexical-weight', '0.0',
  '--progress-value-weight', '0.25',
  '--max-new-tokens', '32',
  '--max-steps', '50',
  '--start-index', $StartIndex.ToString(),
  '--limit', $Limit.ToString(),
  '--fresh-env-per-episode'
)
if ($ValueModel) { $Arguments += @('--progress-value-model', $ValueModel) }

Write-Host "=== Running ${Name}: start=$StartIndex, limit=$Limit ==="
& $Python @Arguments 2>&1 | Tee-Object -FilePath $Log
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

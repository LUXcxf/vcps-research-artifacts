param(
  [int]$Limit = 134,
  [int]$StartIndex = 0
)

$ErrorActionPreference = 'Stop'
$OutputEncoding = [System.Text.UTF8Encoding]::new()
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new()

$Branch = Split-Path -Parent $PSScriptRoot
$Python = if ($env:VCPS_PYTHON) { $env:VCPS_PYTHON } else { 'python' }
$Runner = Join-Path $Branch 'scripts\run_unified_eval.py'
$Manifest = Join-Path $Branch 'data\manifests\valid_unseen134.jsonl'
$Adapter = Join-Path $Branch 'models\actor_adapter'
$ValueModel = Join-Path $Branch 'models\value_models\structured_progress_plus_entity_semantic_d240.json'
$Name = 'frozen_vcps_d240_valid_unseen134'
$Output = Join-Path $Branch ('reproduced\runs\' + $Name + '_episodes.jsonl')
$Report = Join-Path $Branch ('reproduced\reports\' + $Name + '.json')
$Log = Join-Path $Branch ('reproduced\logs\' + $Name + '_live.log')
$Status = Join-Path $Branch ('reproduced\logs\' + $Name + '_status.json')

$PythonCommand = Get-Command -Name $Python -ErrorAction SilentlyContinue
if (-not $PythonCommand -and -not (Test-Path -LiteralPath $Python)) {
  throw "Python command or executable not found: $Python"
}
foreach ($Path in @($Runner, $Manifest, $Adapter, $ValueModel)) {
  if (-not (Test-Path -LiteralPath $Path)) { throw "Required path not found: $Path" }
}
New-Item -ItemType Directory -Force -Path (Split-Path $Output), (Split-Path $Report), (Split-Path $Log) | Out-Null

$ManifestRows = (Get-Content -LiteralPath $Manifest | Where-Object { $_.Trim() }).Count
if ($ManifestRows -ne 134) { throw "Expected 134 manifest rows, found $ManifestRows" }
$ManifestHash = (Get-FileHash -LiteralPath $Manifest -Algorithm SHA256).Hash
$ValueHash = (Get-FileHash -LiteralPath $ValueModel -Algorithm SHA256).Hash

$Arguments = @(
  '-u', $Runner,
  '--manifest', $Manifest,
  '--adapter-path', $Adapter,
  '--progress-value-model', $ValueModel,
  '--output', $Output,
  '--report', $Report,
  '--action-selection', 'admissible_observable_learned_rerank',
  '--admissible-candidate-limit', '16',
  '--admissible-score-batch-size', '4',
  '--progress-lexical-weight', '0.0',
  '--progress-value-weight', '0.25',
  '--max-new-tokens', '32',
  '--max-steps', '50',
  '--start-index', $StartIndex.ToString(),
  '--limit', $Limit.ToString(),
  '--fresh-env-per-episode',
  '--resume-output'
)

@{
  state = 'running'
  updated_at = (Get-Date).ToString('o')
  name = $Name
  evaluation_identity = 'single_frozen_valid_unseen134'
  start_index = $StartIndex
  limit = $Limit
  manifest_rows = $ManifestRows
  manifest_sha256 = $ManifestHash
  value_model_sha256 = $ValueHash
  actor = $Adapter
  manifest = $Manifest
  value_model = $ValueModel
  action_selection = 'admissible_observable_learned_rerank'
  value_weight = 0.25
  candidate_limit = 16
  max_steps = 50
} | ConvertTo-Json | Set-Content -LiteralPath $Status -Encoding utf8

Write-Host "=== Running ${Name}: start=$StartIndex, limit=$Limit ==="
& $Python @Arguments 2>&1 | Tee-Object -FilePath $Log -Append
if ($LASTEXITCODE -ne 0) {
  @{
    state = 'failed'
    updated_at = (Get-Date).ToString('o')
    name = $Name
    exit_code = $LASTEXITCODE
    output = $Output
    report = $Report
  } | ConvertTo-Json | Set-Content -LiteralPath $Status -Encoding utf8
  exit $LASTEXITCODE
}

@{
  state = 'complete'
  updated_at = (Get-Date).ToString('o')
  name = $Name
  evaluation_identity = 'single_frozen_valid_unseen134'
  manifest_sha256 = $ManifestHash
  value_model_sha256 = $ValueHash
  output = $Output
  report = $Report
} | ConvertTo-Json | Set-Content -LiteralPath $Status -Encoding utf8
Write-Host "=== Complete $Name ==="

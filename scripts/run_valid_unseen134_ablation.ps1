param(
  [ValidateSet('actor', 'structured', 'residual', 'all')]
  [string]$Variant = 'all',
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

$Variants = if ($Variant -eq 'all') {
  @('actor', 'structured', 'residual')
} else {
  @($Variant)
}

$ModelByVariant = @{
  actor = $null
  structured = 'structured_progress_only_d240.json'
  residual = 'entity_tp_learned_semantic_d240.json'
}

$PythonCommand = Get-Command -Name $Python -ErrorAction SilentlyContinue
if (-not $PythonCommand -and -not (Test-Path -LiteralPath $Python)) {
  throw "Python command or executable not found: $Python"
}
foreach ($Path in @($Runner, $Manifest, $Adapter)) {
  if (-not (Test-Path -LiteralPath $Path)) { throw "Required path not found: $Path" }
}

$ManifestRows = (Get-Content -LiteralPath $Manifest | Where-Object { $_.Trim() }).Count
if ($ManifestRows -ne 134) { throw "Expected 134 manifest rows, found $ManifestRows" }
$ManifestHash = (Get-FileHash -LiteralPath $Manifest -Algorithm SHA256).Hash

foreach ($CurrentVariant in $Variants) {
  $ModelName = $ModelByVariant[$CurrentVariant]
  $ValueModel = if ($ModelName) { Join-Path $Branch ('models\value_models\' + $ModelName) } else { $null }
  if ($ValueModel -and -not (Test-Path -LiteralPath $ValueModel)) { throw "Required path not found: $ValueModel" }

  $Name = 'd240_vcps_' + $CurrentVariant + '_valid_unseen134'
  $Output = Join-Path $Branch ('reproduced\runs\' + $Name + '_episodes.jsonl')
  $Report = Join-Path $Branch ('reproduced\reports\' + $Name + '.json')
  $Log = Join-Path $Branch ('reproduced\logs\' + $Name + '_live.log')
  $Status = Join-Path $Branch ('reproduced\logs\' + $Name + '_status.json')
  New-Item -ItemType Directory -Force -Path (Split-Path $Output), (Split-Path $Report), (Split-Path $Log) | Out-Null

  $ValueHash = if ($ValueModel) { (Get-FileHash -LiteralPath $ValueModel -Algorithm SHA256).Hash } else { $null }
  $Selection = if ($CurrentVariant -eq 'actor') { 'admissible_observable_logprob' } else { 'admissible_observable_learned_rerank' }
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
    '--fresh-env-per-episode',
    '--resume-output'
  )
  if ($ValueModel) { $Arguments += @('--progress-value-model', $ValueModel) }

  @{
    state = 'running'
    updated_at = (Get-Date).ToString('o')
    name = $Name
    variant = $CurrentVariant
    start_index = $StartIndex
    limit = $Limit
    manifest_rows = $ManifestRows
    manifest_sha256 = $ManifestHash
    value_model_sha256 = $ValueHash
    actor = $Adapter
    manifest = $Manifest
    value_model = $ValueModel
    action_selection = $Selection
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
      variant = $CurrentVariant
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
    variant = $CurrentVariant
    manifest_sha256 = $ManifestHash
    value_model_sha256 = $ValueHash
    output = $Output
    report = $Report
  } | ConvertTo-Json | Set-Content -LiteralPath $Status -Encoding utf8
  Write-Host "=== Complete $Name ==="
}

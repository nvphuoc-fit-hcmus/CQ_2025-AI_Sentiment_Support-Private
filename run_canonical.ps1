# ============================================================
# SAFE-Alert Canonical Training Run
# Run: .\run_canonical.ps1
# ============================================================

$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"
$env:CUBLAS_WORKSPACE_CONFIG = ":4096:8"          # Fix CuBLAS determinism warning
$env:PYTORCH_CUDA_ALLOC_CONF = "max_split_size_mb:128"  # Reduce fragmentation OOM
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

$RUN_DATE = Get-Date -Format "yyyyMMdd_HHmm"
$ARTIFACT_DIR = "services\ai-service\artifacts\canonical_$RUN_DATE"

Set-Location $PSScriptRoot
New-Item -ItemType Directory -Path $ARTIFACT_DIR -Force | Out-Null

# Save provenance
@"
date: $RUN_DATE
gpu: NVIDIA GeForce RTX 4050 Laptop GPU (6GB)
python: 3.11.9
venv: C:\safe_alert_venv
config: train_config_research_best.yaml
symbol: BTCUSDT
horizon: 1h
epochs: 60
n_folds: 5
data_dir: services/ai-service/training_data/v2
articles: 90346
candles: 70261
"@ | Out-File -FilePath "$ARTIFACT_DIR\provenance.txt" -Encoding utf8

Write-Host "[START] Canonical run -> $ARTIFACT_DIR"
Write-Host "[INFO]  Estimated time: 4-8 hours on RTX 4050"
Write-Host ""

C:\safe_alert_venv\Scripts\python.exe -X utf8 `
  services\ai-service\app\v2\pipelines\train_safe_alert.py `
  --config services\ai-service\app\v2\pipelines\train_config_research_best.yaml `
  --walk_forward `
  --symbol BTCUSDT `
  --horizon 1h `
  --epochs 15 `
  --n_folds 2 `
  --artifact_dir $ARTIFACT_DIR `
  2>&1 | Tee-Object -FilePath "$ARTIFACT_DIR\training_log.txt"

Write-Host ""
Write-Host "[DONE] Artifacts saved to: $ARTIFACT_DIR"

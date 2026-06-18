$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8       = "1"
$env:CUBLAS_WORKSPACE_CONFIG = ":4096:8"

$VENV     = "C:\safe_alert_venv\Scripts\python.exe"
$ROOT     = "$PSScriptRoot"
$PIPELINE = "$ROOT\app\v2\pipelines"
$DATA     = "$ROOT\training_data\v2"
$ARTIFACT = "$ROOT\training_data\ablation + baseline"

# ─── 1. most_recent_k_evidence ───────────────────────────────────────────────
Write-Host "`n=== BASELINE 16: most_recent_k_evidence ===" -ForegroundColor Cyan
& $VENV -X utf8 "$PIPELINE\run_baselines.py" `
    --symbol     BTCUSDT `
    --horizon    1h `
    --epochs     40 `
    --batch_size 4 `
    --data_path        "$DATA" `
    --embeddings_path  "$DATA" `
    --artifact_dir     "$ARTIFACT" `
    --baselines        most_recent_k_evidence `
    --output           "$ARTIFACT\result_most_recent_k_evidence.json"

Write-Host "Exit: $LASTEXITCODE" -ForegroundColor $(if ($LASTEXITCODE -eq 0) {"Green"} else {"Red"})

# ─── 2. w/o_bar_sequences ────────────────────────────────────────────────────
Write-Host "`n=== ABLATION: w/o_bar_sequences ===" -ForegroundColor Cyan
& $VENV -X utf8 "$PIPELINE\run_ablation.py" `
    --symbol     BTCUSDT `
    --horizon    1h `
    --epochs     40 `
    --batch_size 4 `
    --data_path        "$DATA" `
    --embeddings_path  "$DATA" `
    --artifact_dir     "$ARTIFACT" `
    --variants         "w/o_bar_sequences" `
    --output           "$ARTIFACT\result_w_o_bar_sequences.json"

Write-Host "Exit: $LASTEXITCODE" -ForegroundColor $(if ($LASTEXITCODE -eq 0) {"Green"} else {"Red"})

Write-Host "`nDone. Files saved to: $ARTIFACT" -ForegroundColor Green

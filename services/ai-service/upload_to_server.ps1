# Upload ai-service code + data lên GPU server
# Chạy script này trên Windows (cần cài OpenSSH hoặc WinSCP CLI)

$SERVER   = "tvquy01@172.29.74.81"
$REMOTE   = "/media/tvquy01"
$LOCAL    = $PSScriptRoot   # thư mục ai-service

Write-Host "=== Upload app/ code ===" -ForegroundColor Cyan
scp -r -P 22 "$LOCAL\app" "${SERVER}:${REMOTE}/ai-service/"

Write-Host "=== Upload training_data/v2 ===" -ForegroundColor Cyan
scp -r -P 22 "$LOCAL\training_data\v2" "${SERVER}:${REMOTE}/ai-service/training_data/"

Write-Host "=== Upload artifacts (fold checkpoints) ===" -ForegroundColor Cyan
scp -r -P 22 "$LOCAL\artifacts" "${SERVER}:${REMOTE}/ai-service/"

Write-Host "=== Upload result files da co ===" -ForegroundColor Cyan
scp -r -P 22 "$LOCAL\training_data\ablation + baseline" "${SERVER}:${REMOTE}/ai-service/results/"

Write-Host "Done! Kiem tra tren server: ls /media/tvquy01/ai-service/" -ForegroundColor Green

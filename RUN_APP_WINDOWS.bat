@echo off
setlocal enabledelayedexpansion

echo.
echo ========================================================================
echo   SAFE-Alert: Crypto Price Prediction Web App
echo ========================================================================
echo.
echo This script will start both Backend and Frontend
echo You need 2 terminals (or split-screen)
echo.
echo INSTRUCTIONS:
echo ─────────────
echo.
echo Step 1: Open Terminal #1 (Cmd or PowerShell)
echo ⏱  Copy and paste:
echo.
echo    cd /d "E:\Khóa luận 1\SA\services\ai-service"
echo    python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
echo.
echo    (Wait until you see "Uvicorn running on http://0.0.0.0:8000")
echo.
echo ─────────────────────────────────────────────────────────────────
echo.
echo Step 2: Open Terminal #2 (new cmd/PowerShell)
echo ⏱  Copy and paste:
echo.
echo    cd /d "E:\Khóa luận 1\SA\web-frontend"
echo    npm install
echo    npm run dev
echo.
echo    (Wait until you see "Local: http://localhost:5173")
echo.
echo ─────────────────────────────────────────────────────────────────
echo.
echo Step 3: Open Browser
echo ⏱  Go to: http://localhost:5173
echo.
echo ========================================================================
echo.
echo WHAT YOU'LL SEE:
echo   - Price chart (real-time)
echo   - AI predictions
echo   - Alert signals
echo   - Selected news articles
echo   - Confidence scores
echo.
echo HOW IT WORKS:
echo   - Frontend (http://localhost:5173)
echo     └─ React UI displaying predictions
echo   - Backend API (http://localhost:8000)
echo     └─ Loads trained model
echo     └─ Runs inference
echo     └─ Returns predictions
echo   - Model weights (already trained)
echo     └─ FIXED (not training live)
echo.
echo Q: Will it train with real data?
echo A: NO! Model weights are already trained.
echo   Frontend just shows inference results.
echo.
echo ========================================================================
echo.
pause

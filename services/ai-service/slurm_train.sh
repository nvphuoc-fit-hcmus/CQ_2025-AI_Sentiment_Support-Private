#!/bin/bash
# ============================================================================
#  SLURM batch script - SAFE-Alert training (train_safe_alert.py)
#  Chay tren server GPU Khoa CNTT (login02).  Nop bang:  sbatch slurm_train.sh
# ============================================================================

#SBATCH --job-name=safe_alert
#SBATCH --gres=gpu:1                 # xin 1 GPU. Doi thanh gpu:2 neu can 2 GPU
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00              # gioi han thoi gian (HH:MM:SS), tang neu can
#SBATCH --output=/media/tvquy01/ai-service/logs/%x_%j.out
#SBATCH --error=/media/tvquy01/ai-service/logs/%x_%j.err
# #SBATCH --partition=               # BO COMMENT + dien ten partition neu `sinfo` yeu cau

set -euo pipefail

PROJECT=/media/tvquy01/ai-service
PIPE=$PROJECT/app/v2/pipelines
mkdir -p "$PROJECT/logs"
cd "$PROJECT"

# --- Kich hoat moi truong Python (chon 1 trong 2 dong duoi) -----------------
source "$PROJECT/venv/bin/activate"          # neu dung venv (xem setup_env.sh)
# source activate safe_alert                 # neu dung conda

# --- Bien moi truong (giong khi chay tren may ca nhan) ----------------------
export PYTHONUTF8=1
export PYTHONIOENCODING=utf-8
export CUBLAS_WORKSPACE_CONFIG=:4096:8

echo "===== GPU info ====="
nvidia-smi
echo "===== Python / Torch ====="
python -c "import torch; print('torch', torch.__version__, '| cuda', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"

# --- Lenh training (giong cau lenh tren Kaggle) -----------------------------
python "$PIPE/train_safe_alert.py" \
    --config "$PIPE/train_config_research_best.yaml" \
    --walk_forward \
    --symbol BTCUSDT \
    --horizon 1h \
    --epochs 40 \
    --n_folds 4 \
    --batch_size 8 \
    --artifact_dir "$PROJECT/artifacts/run_${SLURM_JOB_ID}"

echo "===== DONE. Ket qua nam trong: $PROJECT/artifacts/run_${SLURM_JOB_ID} ====="

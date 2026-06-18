#!/bin/bash
#SBATCH --job-name=safe_missing
#SBATCH --partition=batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/media/tvquy01/logs/missing_%j.log
#SBATCH --error=/media/tvquy01/logs/missing_%j.err

PYTHON=/media/tvquy01/miniconda3/envs/safe_alert/bin/python
WORK=/media/tvquy01
AI=$WORK/ai-service
DATA=$AI/training_data/v2
RESULTS=$WORK/results
PIPELINE=$AI/app/v2/pipelines

export PIP_CACHE_DIR=/media/tvquy01/.cache/pip

mkdir -p $WORK/logs $RESULTS

echo "=== Node: $(hostname) ==="
echo "=== GPU ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

# --- w/o_bar_sequences ---
echo "=== ABLATION: w/o_bar_sequences ==="
$PYTHON -u $PIPELINE/run_ablation.py \
    --symbol BTCUSDT --horizon 1h \
    --epochs 40 --batch_size 64 \
    --data_path $DATA --embeddings_path $DATA \
    --artifact_dir $RESULTS \
    --variants "w/o_bar_sequences" \
    --output $RESULTS/result_w_o_bar_sequences.json
echo "Ablation exit: $?"

echo ""
echo "=== DONE ==="
ls -la $RESULTS/*.json 2>/dev/null

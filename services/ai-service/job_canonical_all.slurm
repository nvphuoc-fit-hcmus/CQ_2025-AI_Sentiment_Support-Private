#!/bin/bash
#SBATCH --job-name=safe_canonical_all
#SBATCH --partition=batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=/media/tvquy01/logs/canonical_all_%j.log
#SBATCH --error=/media/tvquy01/logs/canonical_all_%j.err

PYTHON=/media/tvquy01/miniconda3/envs/safe_alert/bin/python
WORK=/media/tvquy01
AI=$WORK/ai-service
DATA=$AI/training_data/v2
RESULTS=$WORK/results
PIPELINE=$AI/app/v2/pipelines

export PIP_CACHE_DIR=/media/tvquy01/.cache/pip
mkdir -p $WORK/logs $RESULTS

echo "=== Node: $(hostname) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "=== Canonical run: epsilon_h=0.0015 for all ==="
echo ""

# ─── [1/3] 16 BASELINES ──────────────────────────────────────────────────────
echo "=== [1/3] BASELINES (16) ==="
cd $AI
$PYTHON -u $PIPELINE/run_baselines.py \
    --symbol BTCUSDT --horizon 1h \
    --epochs 40 --batch_size 64 \
    --data_path $DATA --embeddings_path $DATA \
    --artifact_dir $RESULTS \
    --output $RESULTS/baseline_results.json
echo "Baselines exit: $?"
echo ""

# ─── [2/3] 9 ABLATION VARIANTS ───────────────────────────────────────────────
echo "=== [2/3] ABLATION (9 variants) ==="
$PYTHON -u $PIPELINE/run_ablation.py \
    --symbol BTCUSDT --horizon 1h \
    --epochs 40 --batch_size 64 \
    --data_path $DATA --embeddings_path $DATA \
    --artifact_dir $RESULTS \
    --variants full_model w/o_selective_news w/o_factor w/o_market \
               w/o_confidence w/o_faithfulness w/o_horizon w/o_lrisk \
               w/o_bar_sequences \
    --output $RESULTS/ablation_results.json
echo "Ablation exit: $?"
echo ""

# ─── [3/3] FAITHFULNESS EVAL ─────────────────────────────────────────────────
echo "=== [3/3] FAITHFULNESS EVAL (epsilon_h=0.0015) ==="
$PYTHON -u $PIPELINE/eval_faithfulness_sec4p4p2.py \
    --model_path $AI/artifacts/fold_4/safe_alert_1h_FINAL.pt \
    --symbol BTCUSDT --horizon 1h \
    --device cuda \
    --epsilon_h_override 0.0015 \
    --output_json $RESULTS/faithfulness_results.json
echo "Faithfulness exit: $?"
echo ""

echo "=== DONE ==="
ls -lh $RESULTS/*.json 2>/dev/null

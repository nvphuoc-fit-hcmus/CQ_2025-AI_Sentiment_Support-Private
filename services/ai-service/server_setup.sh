#!/bin/bash
# Chay tren server sau khi SSH vao
# bash /media/tvquy01/ai-service/server_setup.sh

set -e
WORK=/media/tvquy01
AI=$WORK/ai-service

echo "=== Kiem tra GPU ==="
nvidia-smi

echo "=== Kiem tra Python ==="
python3 --version || python --version

echo "=== Tao virtual environment ==="
cd $WORK
python3 -m venv venv
source venv/bin/activate

echo "=== Cai packages ==="
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install ta==0.11.0 vaderSentiment==3.3.2 pyyaml numpy pandas scikit-learn

echo "=== Kiem tra CUDA ==="
python3 -c "import torch; print('CUDA:', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"

echo "=== Kiem tra data ==="
ls $AI/training_data/v2/

echo "Setup xong!"

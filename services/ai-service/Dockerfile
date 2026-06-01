# ── SAFE-Alert AI Service — reproducible build (P2 #22) ─────────────────────
# Pins: python 3.11.9, torch 2.6.0 CPU, everything else pinned in
# requirements.txt. For reviewer reproducibility:
#   docker build -t safe-alert-ai:2.0 .
#   docker run --rm safe-alert-ai:2.0 pytest tests/
#
# GPU variant: swap base image to pytorch/pytorch:2.6.0-cuda12.1-cudnn9-runtime
# and REMOVE the CPU-wheel install block below. Everything else (pip install
# -r requirements.txt after stripping torch) stays the same.
FROM python:3.11.9-slim

WORKDIR /app

# 1. System libs — build tools + librdkafka for confluent-kafka binary install.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    librdkafka-dev \
    && rm -rf /var/lib/apt/lists/*

# 2. Install the CPU-only PyTorch wheel FIRST so the layer caches independently
# of the rest of requirements.txt. CPU wheel is ~150 MB vs ~2 GB for CUDA.
# Version pinned to keep reproducibility guarantees with requirements.txt.
RUN pip install --no-cache-dir \
    torch==2.6.0 \
    --index-url https://download.pytorch.org/whl/cpu

# 3. Copy pinned requirements and install everything else. Strip `torch` so
# step 2 is not overridden by the indexless PyPI version.
COPY requirements.txt .
RUN sed -i '/^torch==/d' requirements.txt && \
    pip install --no-cache-dir --prefer-binary --timeout 120 --retries 5 -r requirements.txt

# 4. Source code.
COPY app ./app

ENV PYTHONUNBUFFERED=1
# Optional: enable reproducible training runs (slower but deterministic).
# Set at `docker run` time: -e SAFEALERT_DETERMINISTIC=1
ENV SAFEALERT_DETERMINISTIC=0

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8002"]

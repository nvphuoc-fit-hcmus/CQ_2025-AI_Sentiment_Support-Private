FROM python:3.11-slim

WORKDIR /app

# 1. Cài các thư viện hệ thống cần thiết (Cache lớp này)
# Thêm 'librdkafka-dev' để cài confluent-kafka nhanh hơn, không phải build từ source
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    librdkafka-dev \
    && rm -rf /var/lib/apt/lists/*

# 2. MẸO QUAN TRỌNG: Cài PyTorch bản CPU riêng trước (Giảm từ 2GB -> ~150MB)
# Tách ra khỏi requirements.txt để Docker cache lại, không cài lại mỗi khi sửa file req
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# 3. Copy requirements và cài các gói còn lại
COPY requirements.txt .

# Loại bỏ torch trong requirements.txt nếu có (để tránh nó cài đè bản GPU)
# Dùng --prefer-binary để ưu tiên cài bản wheel có sẵn thay vì compile lại (Pandas/Numpy)
RUN sed -i '/torch/d' requirements.txt && \
    pip install --no-cache-dir --prefer-binary --timeout 120 --retries 5 -r requirements.txt

# 4. Copy source code cuối cùng
COPY app ./app

ENV PYTHONUNBUFFERED=1
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8002"]

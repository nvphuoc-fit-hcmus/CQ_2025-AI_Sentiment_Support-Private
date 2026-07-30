# 🚀 Hệ Thống Phân Tích & Dự Đoán Thị Trường Crypto - Aegis

## 📋 Mục Lục
- [Tổng Quan](#-tổng-quan)
- [Kiến Trúc Hệ Thống](#-kiến-trúc-hệ-thống)
- [Các Thành Phần Chính](#-các-thành-phần-chính)
- [Thuật Toán AI & Machine Learning](#-thuật-toán-ai--machine-learning)
- [Bảo Mật với API Gateway](#-bảo-mật-với-api-gateway)
- [HAProxy & WebSocket Scaling](#-haproxy--websocket-scaling)
- [Luồng Dữ Liệu](#-luồng-dữ-liệu)
- [Cài Đặt & Chạy Dự Án](#-cài-đặt--chạy-dự-án)
- [Cấu Trúc Thư Mục](#-cấu-trúc-thư-mục)
- [Monitoring & Logging](#-monitoring--logging)

---

## 🎯 Tổng Quan

**Aegis** là một hệ thống microservices toàn diện dùng để:
- 📊 Thu thập dữ liệu thị trường crypto theo thời gian thực từ Binance
- 📰 Crawl tin tức từ 17+ nguồn uy tín (Cointelegraph, CoinDesk, Blogtienao...)
- 🤖 Phân tích cảm xúc tin tức bằng **FinBERT** (Deep Learning)
- 🔮 Dự đoán giá crypto bằng **Dual-Stream LSTM + Transformer**
- 💬 Giải thích dự đoán bằng tiếng Việt với **Gemini AI**
- 💰 Mô phỏng đầu tư với AI recommendations (VIP feature)
- 💳 Thanh toán tự động qua SePay API
- 🔔 Thông báo real-time qua Email & SSE (Server-Sent Events)

### Công Nghệ Sử Dụng
- **Backend**: Node.js (Express), Python (FastAPI, PyTorch)
- **Frontend**: React + Vite + Zustand
- **Message Broker**: Apache Kafka
- **Databases**: PostgreSQL, TimescaleDB, MongoDB, Redis
- **API Gateway**: Kong Gateway (JWT, Rate Limiting, RBAC)
- **Load Balancer**: HAProxy (Consistent Hashing cho WebSocket)
- **AI/ML**: PyTorch, FinBERT, Gemini API
- **Containerization**: Docker + Docker Compose

---

## 🏗️ Kiến Trúc Hệ Thống

```
┌─────────────────────────────────────────────────────────────────────┐
│                         CLIENT (React App)                          │
│                      http://localhost:5173                          │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    KONG API GATEWAY :8000                           │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │ • JWT Authentication (RS256)                                 │   │
│  │ • Rate Limiting (100 req/min per user)                       │   │
│  │ • RBAC (inject X-User-Id, X-Is-VIP headers)                  │   │
│  │ • Request Size Limiting (1MB)                                │   │
│  │ • CORS Configuration                                         │   │
│  └──────────────────────────────────────────────────────────────┘   │
└─────┬────────┬────────┬────────┬────────┬────────┬────────┬─────────┘
      │        │        │        │        │        │        │
      ▼        ▼        ▼        ▼        ▼        ▼        ▼
   ┌────┐  ┌────┐  ┌────┐  ┌────┐  ┌────┐  ┌────┐  ┌────────┐
   │Auth│  │Core│  │AI  │  │Pay │  │Inv │  │Noti│  │HAProxy │
   │Svc │  │Svc │  │Svc │  │Svc │  │Svc │  │Svc │  │ :8005  │
   └─┬──┘  └─┬──┘  └─┬──┘  └─┬──┘  └─┬──┘  └─┬──┘  └───┬────┘
     │       │       │       │       │       │         │
     │       │       │       │       │       │         ▼
     │       │       │       │       │       │    ┌─────────┐
     │       │       │       │       │       │    │ Stream  │
     │       │       │       │       │       │    │ Service │
     │       │       │       │       │       │    │ (x3)    │
     │       │       │       │       │       │    └────┬────┘
     │       │       │       │       │       │         │
     ▼       ▼       ▼       ▼       ▼       ▼         ▼
┌──────────────────────────────────────────────────────────────┐
│                    KAFKA MESSAGE BROKER                      │
│  Topics: market.prices, news_raw_v2, news_analyzed,          │
│          ai_insights, payment_events, vip_updates            │
└──────────────────────────────────────────────────────────────┘
     │       │       │       │       │       │
     ▼       ▼       ▼       ▼       ▼       ▼
┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐
│Postgres│ │Timescale│ │MongoDB │ │Redis   │ │Payment │
│:5432   │ │DB:5433 │ │:27017  │ │:6379   │ │DB:5435 │
└────────┘ └────────┘ └────────┘ └────────┘ └────────┘
```

### Kiến Trúc Microservices
Hệ thống được thiết kế theo mô hình **Event-Driven Microservices** với:
- ✅ **Loose Coupling**: Services giao tiếp qua Kafka
- ✅ **High Availability**: HAProxy load balancing cho WebSocket
- ✅ **Security-First**: Kong Gateway làm single entry point
- ✅ **Scalability**: Stream service có thể scale horizontal (replicas: 3)

---

## 🔧 Các Thành Phần Chính

### 1️⃣ Infrastructure Services

#### **Kong API Gateway** (Port 8000, 8001)
- **Vai trò**: Single entry point cho tất cả requests từ client
- **Tính năng**:
  - JWT Authentication với RS256 (Public Key Cryptography)
  - Rate Limiting: 100 req/min cho user thường, 30 req/min cho payment
  - RBAC: Inject headers `X-User-Id`, `X-Is-VIP` cho downstream services
  - Request validation: Size limit 1MB, timeout 30s
  - CORS configuration cho `localhost:5173`
- **Config**: `infra/kong_conf/kong.yml`

#### **HAProxy Load Balancer** (Port 8005, 8404)
- **Vai trò**: Load balancing cho WebSocket connections
- **Thuật toán**: **Consistent Hashing** dựa trên `token` parameter
  - Đảm bảo cùng 1 user luôn kết nối đến cùng 1 stream-service instance
  - Giảm thiểu connection drops khi scale
- **Health Check**: HTTP GET `/health` mỗi 3s
- **Monitoring**: Stats page tại `http://localhost:8404/stats`
- **Config**: `haproxy/haproxy.cfg`

#### **Apache Kafka** (Port 9092)
- **Vai trò**: Message broker trung tâm
- **Topics**:
  - `market.prices`: Dữ liệu giá real-time từ Binance
  - `news_raw_v2`: Tin tức thô từ crawler
  - `news_analyzed`: Tin tức đã phân tích sentiment
  - `ai_insights`: Dự đoán từ AI service
  - `payment_events`: Events thanh toán
  - `vip_updates`: Events nâng cấp VIP
- **Retention**: 1 giờ, max 256MB per topic

#### **PostgreSQL** (Port 5432)
- **Database**: `appdb`
- **Sử dụng bởi**: Auth Service
- **Schema**: Users, JWT tokens, VIP status

#### **TimescaleDB** (Port 5433)
- **Database**: `timeseriesdb`
- **Sử dụng bởi**: Core Service
- **Schema**: Historical price data, AI predictions (time-series optimized)

#### **Redis** (Port 6379)
- **Sử dụng bởi**:
  - Crawler Service: Lưu trạng thái crawl
  - Auth Service: Token blacklist (logout)
  - Core Service: Caching

#### **MongoDB** (Port 27018)
- **Database**: `crawler_db`
- **Sử dụng bởi**: Crawler Service
- **Schema**: Raw news articles, crawl history

---

### 2️⃣ Business Services

#### **Auth Service** (Port 8080 - internal)
- **Công nghệ**: Node.js + Express + JWT
- **Chức năng**:
  - Đăng ký, đăng nhập, logout
  - Phát hành JWT token (RS256)
  - Refresh token mechanism (httpOnly cookie)
  - SSE (Server-Sent Events) cho real-time notifications
  - Lắng nghe Kafka topic `vip_updates` → gửi SSE event cho client
- **Endpoints**:
  - `POST /auth/register` - Đăng ký
  - `POST /auth/login` - Đăng nhập
  - `POST /auth/logout` - Đăng xuất
  - `POST /auth/refresh` - Refresh access token
  - `GET /auth/me` - Lấy thông tin user + token mới
  - `GET /auth/events/sse` - SSE stream
  - `GET /auth/public-key` - Public key cho Kong
- **Database**: PostgreSQL (`appdb`)

#### **Core Service** (Port 3000 - internal)
- **Công nghệ**: Node.js + Express + TimescaleDB
- **Chức năng**:
  - Lưu trữ dữ liệu lịch sử giá vào TimescaleDB
  - Lưu trữ AI predictions
  - Cung cấp API lấy historical data
  - Lắng nghe Kafka: `market.prices`, `ai_insights`
- **Endpoints**:
  - `GET /v1/market/history/:symbol` - Lấy historical data
  - `GET /v1/predictions/:symbol` - Lấy AI predictions
  - `GET /health` - Health check
- **Database**: TimescaleDB (`timeseriesdb`)

#### **Stream Service** (Port 3000 - internal, replicas: 3)
- **Công nghệ**: Node.js + Socket.IO + Binance WebSocket
- **Chức năng**:
  - Kết nối Binance WebSocket để nhận dữ liệu giá real-time
  - Phục vụ client qua Socket.IO
  - Đẩy dữ liệu vào Kafka topic `market.prices`
  - **Scaling**: 3 replicas với HAProxy consistent hashing
- **Symbols**: BTCUSDT, ETHUSDT, BNBUSDT, SOLUSDT, DOGEUSDT, ADAUSDT, XRPUSDT, AVAXUSDT, DOTUSDT, POLUSDT
- **Health Check**: `GET /health`

#### **Crawler Service** (Port 8001 - internal)
- **Công nghệ**: Python + FastAPI + BeautifulSoup
- **Chức năng**:
  - Crawl tin tức từ 17+ nguồn RSS/Website
  - Lưu trạng thái vào Redis (tránh duplicate)
  - Lưu raw articles vào MongoDB
  - Đẩy tin tức vào Kafka topic `news_raw_v2`
- **Nguồn tin**: Cointelegraph, CoinDesk, Decrypt, Bitcoin Magazine, TheBlock, Coin68, BlogTienAo, CoinPhoton, TienThuatToan, CryptoSlate, U.Today, AMBCrypto, CryptoPotato, BeInCrypto, CryptoBriefing, CoinJournal, NewsBTC
- **Crawl Interval**: 60 giây
- **Database**: MongoDB (`crawler_db`), Redis

#### **AI Service** (Port 8002 - internal)
- **Công nghệ**: Python + FastAPI + PyTorch + FinBERT + Gemini API
- **Chức năng**:
  - Lắng nghe Kafka: `news_raw_v2`, `market.prices`
  - Phân tích sentiment tin tức bằng **FinBERT**
  - Dự đoán giá bằng **Deep Learning Model** (xem phần [Thuật Toán AI](#-thuật-toán-ai--machine-learning))
  - Giải thích dự đoán bằng **Gemini AI**
  - Đẩy kết quả vào Kafka: `news_analyzed`, `ai_insights`
- **Models**:
  - FinBERT: `ProsusAI/finbert` (Sentiment Analysis)
  - Custom LSTM: `lstm_sentiment_hybrid_v1.pth` (Price Prediction)
  - Gemini: `gemma-3-1b-it` (Explanation Generation)
- **Prediction Interval**: 60 giây
- **Endpoints**:
  - `GET /health` - Health check
  - `GET /v1/predictions/:symbol` - Lấy prediction mới nhất

#### **Payment Service** (Port 3000 - internal)
- **Công nghệ**: Node.js + Express + SePay API
- **Chức năng**:
  - Tạo QR code thanh toán VIP (99,000 VND)
  - Webhook nhận callback từ SePay
  - Verify payment signature
  - Đẩy event `vip_updates` vào Kafka
  - Cập nhật VIP status trong auth-service database
- **Endpoints**:
  - `POST /v1/payments/create` - Tạo payment request
  - `POST /v1/payments/webhook` - SePay webhook
  - `GET /health` - Health check
- **Database**: PostgreSQL (`payment_db`)

#### **Investment Service** (Port 8001 - internal)
- **Công nghệ**: Python + FastAPI
- **Chức năng**:
  - Mô phỏng đầu tư crypto (VIP feature)
  - Phân tích portfolio với AI recommendations
  - Lắng nghe Kafka `ai_insights` để đưa ra khuyến nghị
- **Middleware**: Kiểm tra `X-Is-VIP` header từ Kong
- **Endpoints**:
  - `POST /v1/investments/analyze` - Phân tích đầu tư (timeout 10 phút)
  - `GET /health` - Health check
- **Database**: PostgreSQL (`investment_db`)

#### **Notification Service** (Port 8001 - internal)
- **Công nghệ**: Node.js + Nodemailer
- **Chức năng**:
  - Gửi email thông báo (VIP upgrade, price alerts)
  - Lắng nghe Kafka: `vip_updates`, `price_alerts`
- **SMTP**: Gmail SMTP (cấu hình qua env `SMTP_EMAIL`, `SMTP_PASSWORD`)
- **Database**: PostgreSQL (`notification_db`)

---

## 🤖 Thuật Toán AI & Machine Learning

### 1. Sentiment Analysis với FinBERT

**Model**: `ProsusAI/finbert` (BERT fine-tuned trên financial news)

**Input**: Nội dung tin tức (text)

**Output**: 
- Sentiment: `positive`, `negative`, `neutral`
- Confidence score: 0-1
- Embedding vector: 768 dimensions

**Quy trình**:
```python
# 1. Tokenize text
tokens = tokenizer(news_content, max_length=512, truncation=True)

# 2. Get FinBERT prediction
outputs = finbert_model(**tokens)
sentiment = argmax(outputs.logits)  # 0=positive, 1=negative, 2=neutral

# 3. Get embedding for price prediction model
embedding = outputs.last_hidden_state[:, 0, :]  # [CLS] token embedding
```

---

### 2. Price Prediction với Dual-Stream LSTM + Transformer

**Model**: `AdvancedDualStreamNetwork` (Custom PyTorch model)

**Kiến trúc**:
```
┌─────────────────────────────────────────────────────────────┐
│                  INPUT LAYER                                │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ Price Data   │  │ News Embed   │  │ Coin Index   │      │
│  │ [60, 11]     │  │ [60, 768]    │  │ [1]          │      │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘      │
└─────────┼──────────────────┼──────────────────┼─────────────┘
          │                  │                  │
          ▼                  ▼                  ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────┐
│  Bi-LSTM        │  │  Multi-Head     │  │  Embedding  │
│  (256 units)    │  │  Attention      │  │  (16 dim)   │
└────────┬────────┘  └────────┬────────┘  └──────┬──────┘
         │                    │                   │
         ▼                    ▼                   │
┌─────────────────┐  ┌─────────────────┐         │
│  Transformer    │  │  Temporal       │         │
│  Attention      │  │  Weighting      │         │
└────────┬────────┘  └────────┬────────┘         │
         │                    │                   │
         └──────────┬─────────┘                   │
                    ▼                             │
           ┌─────────────────┐                    │
           │  Cross-Modal    │                    │
           │  Attention      │                    │
           └────────┬────────┘                    │
                    │                             │
                    └──────────┬──────────────────┘
                               ▼
                    ┌─────────────────┐
                    │  Fusion Layer   │
                    │  (384 → 128)    │
                    └────────┬────────┘
                             ▼
                    ┌─────────────────┐
                    │  Shared Layer   │
                    │  (128 dim)      │
                    └────────┬────────┘
                             │
         ┌───────────────────┼───────────────────┐
         ▼                   ▼                   ▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│  Direction Head │ │  Return Head    │ │ Volatility Head │
│  (1h, 24h)      │ │  (1h, 24h)      │ │  (3 classes)    │
└─────────────────┘ └─────────────────┘ └─────────────────┘
```

**Input Features** (60 timesteps × 11 features):
1. OHLCV (Open, High, Low, Close, Volume)
2. RSI (Relative Strength Index - 14 periods)
3. MACD (Moving Average Convergence Divergence)
4. Bollinger Bands (Upper, Lower)
5. SMA 20 (Simple Moving Average)
6. EMA 12 (Exponential Moving Average)

**News Embeddings** (60 timesteps × 768 features):
- Mỗi timestep (5 phút) có 1 vector 768-dim từ FinBERT
- Nếu không có tin: vector zero
- Nếu có nhiều tin: average embeddings

**Output**:
```python
{
    "direction_1h": 0.582,        # Probability UP (58.2%)
    "direction_24h": 0.469,       # Probability UP (46.9%)
    "return_1h": 0.0089,          # Expected return +0.89%
    "return_24h": -0.0156,        # Expected return -1.56%
    "confidence_1h": 0.723,       # Model confidence 72.3%
    "confidence_24h": 0.651,      # Model confidence 65.1%
    "volatility": [0.15, 0.70, 0.15]  # [LOW, MEDIUM, HIGH]
}
```

**Post-Processing**:
```python
# 1. Chuyển probability → direction
if direction_prob > 0.55:
    direction = "UP"
elif direction_prob < 0.45:
    direction = "DOWN"
else:
    direction = "SIDEWAYS"

# 2. Tính giá target
move_percent = (direction_prob - 0.5) * 0.04  # ±2% max
target_price = current_price * (1 + move_percent)

# 3. Điều chỉnh theo volatility
if volatility == "HIGH":
    move_percent *= 1.3
elif volatility == "LOW":
    move_percent *= 0.7

# 4. Safety bounds
move_percent = max(-0.03, min(0.03, move_percent))  # ±3% max
```

**Training**:
- **Dataset**: Historical OHLCV + News từ Kafka (collect bằng `kafka_data_collector.py`)
- **Loss Function**: Multi-task loss (Direction + Return + Volatility)
- **Optimizer**: AdamW với learning rate 0.001
- **Training Script**: `services/ai-service/train.py`
- **Quick Train**: `services/ai-service/quick_train.py`
- **Model Checkpoint**: `ml_models/lstm_sentiment_hybrid_v1.pth`

---

### 3. Explanation Generation với Gemini AI

**Model**: `gemma-3-1b-it` (Gemini API)

**Vai trò**: Gemini **KHÔNG** dự đoán giá, chỉ **GIẢI THÍCH** dự đoán từ Deep Learning model

**Input Prompt**:
```
🎯 ROLE: Bạn là Senior Crypto Market Analyst

📊 NHIỆM VỤ: Phân tích và giải thích dự báo giá cho BTCUSDT

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📈 KẾT QUẢ DỰ BÁO TỪ DEEP LEARNING MODEL
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🔮 DỰ BÁO 1 GIỜ TỚI:
   • Xu hướng: UP
   • Mục tiêu giá: +1.67%
   • Xác suất: 0.582
   • Độ tin cậy: 58.0%
   • Volatility: LOW

📊 CHỈ SỐ KỸ THUẬT HIỆN TẠI:
   • Giá hiện tại: $78,307.65
   • RSI(14): 53.7 → Trung lập
   • MACD: -0.0065 → Tiêu cực
   • Bollinger Bands: [77800 - 79500]

📰 TIN TỨC: [Danh sách tin tức với sentiment]

🎯 PRIMARY DRIVER: TECHNICAL
   → Model đã phân tích và xác định yếu tố chính ảnh hưởng đến giá là: Technical patterns

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📝 YÊU CẦU ĐẦU RA
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Trả về JSON:
{
    "primary_driver": "TECHNICAL_CATALYST",
    "key_event": "Tóm tắt pattern chính",
    "news_citations": ["Trích dẫn nếu có tin"],
    "explanation_vi": "6-8 câu giải thích chi tiết...",
    "causal_chain": {...},
    "sentiment_impact": {...},
    "actionable_advice": "Entry, SL, TP cụ thể..."
}
```

**Output**:
```json
{
  "primary_driver": "TECHNICAL_CATALYST",
  "key_event": "RSI Neutral + MACD Divergence",
  "news_citations": [],
  "explanation_vi": "Dự báo UP +1.67% trong 1 giờ tới dựa trên phân tích kỹ thuật. RSI đang ở mức trung lập 53.7, cho thấy thị trường chưa quá mua hoặc quá bán. MACD âm nhẹ (-0.0065) nhưng đang có dấu hiệu hội tụ, báo hiệu động lực tăng giá sắp tới. Giá hiện tại nằm gần giữa Bollinger Bands, tạo điều kiện cho biến động tăng. Volatility thấp cho thấy rủi ro hạn chế.",
  "causal_chain": {
    "trigger": "Technical pattern convergence",
    "mechanism": "RSI neutral + MACD bullish divergence",
    "outcome": "Short-term upward momentum"
  },
  "sentiment_impact": {
    "news": "NEUTRAL",
    "technical": "SLIGHTLY_BULLISH"
  },
  "actionable_advice": "Entry: $78,000-$78,500 | Stop Loss: $77,500 | Take Profit: $79,500-$80,000"
}
```

**Retry Logic với Exponential Backoff**:
```python
# Xử lý rate limit 429 từ Gemini API
max_retries = 3
for attempt in range(max_retries):
    try:
        response = gemini_client.generate_content(prompt)
        break
    except Exception as e:
        if "429" in str(e):  # Rate limit
            retry_delay = parse_retry_delay(e) or (2 ** attempt)
            await asyncio.sleep(retry_delay)
        else:
            raise
```

---

## 🔒 Bảo Mật với API Gateway

### Kong Gateway Security Features

#### 1. JWT Authentication (RS256)
```yaml
# kong.yml
plugins:
  - name: jwt
    config:
      claims_to_verify: ["exp"]
      key_claim_name: "iss"
      secret_is_base64: false
      anonymous: null  # Không cho phép anonymous access
```

**Quy trình**:
1. Client login → Auth Service phát hành JWT token (RS256)
2. Client gửi request với header `Authorization: Bearer <token>`
3. Kong verify token bằng Public Key từ Auth Service
4. Nếu valid → forward request, nếu invalid → 401 Unauthorized

**JWT Claims**:
```json
{
  "sub": "user_id",
  "email": "user@example.com",
  "is_vip": true,
  "jti": "token_id",
  "iss": "my-app-auth",
  "exp": 1234567890
}
```

#### 2. Rate Limiting
```yaml
# Auth endpoints: Chống brute force
- name: rate-limiting
  config:
    minute: 20   # Max 20 login attempts/minute
    hour: 100    # Max 100 login attempts/hour
    policy: local
    limit_by: ip

# Core API: Per user
- name: rate-limiting
  config:
    minute: 100  # 100 requests/minute per user
    hour: 1000   # 1000 requests/hour per user
    policy: local
    limit_by: credential

# Payment API: Stricter
- name: rate-limiting
  config:
    minute: 30   # 30 requests/minute
    hour: 300    # 300 requests/hour
```

#### 3. RBAC (Role-Based Access Control)
```yaml
# Inject user info vào headers cho downstream services
- name: request-transformer
  config:
    add:
      headers:
        - "X-User-Id:$(claims.sub)"
        - "X-User-Email:$(claims.email)"
        - "X-Is-VIP:$(claims.is_vip)"
        - "X-Token-JTI:$(claims.jti)"
```

**Middleware VIP Check** (Investment Service):
```python
def require_vip(request):
    is_vip = request.headers.get("X-Is-VIP")
    if is_vip != "true":
        raise HTTPException(403, "VIP feature only")
```

#### 4. Request Validation
```yaml
# Giới hạn request body size
- name: request-size-limiting
  config:
    allowed_payload_size: 1  # 1 MB max
```

#### 5. CORS Configuration
```yaml
- name: cors
  config:
    origins: ["http://localhost:5173", "http://myapp.com"]
    methods: ["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"]
    headers: ["Accept", "Content-Type", "Authorization"]
    credentials: true
    max_age: 3600
```

#### 6. Token Refresh Mechanism
```javascript
// store.js - Auto refresh khi 401
authFetch: async (endpoint, options) => {
  let response = await fetch(url, { headers: { Authorization: `Bearer ${token}` } });
  
  if (response.status === 401) {
    // Silent refresh
    const newToken = await refreshAccessToken();
    if (newToken) {
      // Retry với token mới
      response = await fetch(url, { headers: { Authorization: `Bearer ${newToken}` } });
    } else {
      // Logout user
      await logout();
    }
  }
  
  return response;
}
```

#### 7. Token Blacklist (Logout)
```javascript
// Auth Service - Lưu JTI vào Redis khi logout
await redis.setex(`blacklist:${jti}`, ttl, "1");

// Middleware kiểm tra blacklist
if (await redis.exists(`blacklist:${jti}`)) {
  throw new Error("Token revoked");
}
```

---

## ⚖️ HAProxy & WebSocket Scaling

### Vấn Đề Cần Giải Quyết
- **WebSocket** yêu cầu persistent connection
- Khi scale horizontal (3 replicas), cần đảm bảo:
  - Cùng 1 user luôn kết nối đến cùng 1 instance
  - Khi instance die, user tự động reconnect đến instance khác
  - Load distribution đều giữa các instances

### Giải Pháp: Consistent Hashing

**HAProxy Config**:
```haproxy
# haproxy/haproxy.cfg
backend stream_services
    # Consistent hashing dựa trên URL parameter 'token'
    balance url_param token
    hash-type consistent sdbm avalanche
    
    # Health check mỗi 3s
    option httpchk GET /health
    http-check expect status 200
    
    # WebSocket support
    option http-server-close
    option forwardfor
    
    # 10 virtual nodes cho mỗi instance (better distribution)
    server-template stream 10 stream-service:3000 \
        check inter 3s fall 3 rise 2 \
        resolvers docker init-addr none weight 256
```

**Cách hoạt động**:
```
User A (token=abc123) → Hash(abc123) → Instance 1
User B (token=def456) → Hash(def456) → Instance 2
User C (token=ghi789) → Hash(ghi789) → Instance 1

# Khi Instance 2 die:
User B → Hash(def456) → Instance 1 hoặc 3 (rehash)

# Khi Instance 2 recover:
User B → Hash(def456) → Instance 2 (back to original)
```

**Client Connection**:
```javascript
// Frontend - Kết nối qua HAProxy
const socket = io("http://localhost:8005", {
  transports: ["websocket"],
  query: { token: localStorage.getItem("token") }  // ← Dùng cho consistent hashing
});
```

**Monitoring**:
- Stats page: `http://localhost:8404/stats`
- Metrics: Connection count, health status, response time

---

## 🔄 Luồng Dữ Liệu

### 1. Market Data Flow
```
Binance WebSocket
    ↓
Stream Ingester (Python)
    ↓ (Kafka: market.prices)
    ├→ Stream Service (Socket.IO) → Client
    ├→ Core Service → TimescaleDB
    └→ AI Service (buffer for prediction)
```

### 2. News & AI Prediction Flow
```
RSS Feeds (17+ sources)
    ↓
Crawler Service
    ↓ (Kafka: news_raw_v2)
AI Service
    ├→ FinBERT (Sentiment Analysis)
    ├→ LSTM Model (Price Prediction)
    └→ Gemini API (Explanation)
    ↓ (Kafka: ai_insights)
    ├→ Core Service → TimescaleDB
    └→ Client (via API)
```

### 3. Payment & VIP Upgrade Flow
```
Client → Payment Service
    ↓
SePay API (QR Code)
    ↓
User pays via banking app
    ↓
SePay Webhook → Payment Service
    ↓
Verify signature
    ↓ (Kafka: vip_updates)
Auth Service
    ├→ Update VIP status in DB
    └→ Send SSE event to Client
    ↓
Client receives SSE event
    ↓
Call /auth/me to get new token with is_vip=true
    ↓
Update localStorage & Zustand store
    ↓
Access VIP features (Investment Simulator)
```

### 4. Real-time Notification Flow
```
Event occurs (VIP upgrade, price alert)
    ↓ (Kafka: vip_updates, price_alerts)
    ├→ Auth Service → SSE to Client
    └→ Notification Service → Email
```

---

## 🚀 Cài Đặt & Chạy Dự Án

### Yêu Cầu Hệ Thống
- **Docker** >= 20.10
- **Docker Compose** >= 2.0
- **RAM**: Tối thiểu 8GB (khuyến nghị 16GB)
- **Disk**: 10GB free space

### Biến Môi Trường
Tạo file `.env` ở thư mục gốc:
```bash
# Gemini API
GEMINI_API_KEY=your_gemini_api_key_here

# SePay API
SEPAY_API_KEY=your_sepay_api_key_here

# SMTP (Gmail)
SMTP_EMAIL=your_email@gmail.com
SMTP_PASSWORD=your_app_password
```

### Các Bước Chạy

#### 1. Clone Repository
```bash
git clone <repository_url>
cd infra
```

#### 2. Khởi Động Hệ Thống
```bash
# Build và start tất cả services
docker-compose up -d

# Xem logs
docker-compose logs -f

# Xem logs của 1 service cụ thể
docker-compose logs -f ai-service
```

#### 3. Kiểm Tra Health
```bash
# Kong Gateway
curl http://localhost:8000

# HAProxy Stats
open http://localhost:8404/stats

# Auth Service (qua Kong)
curl http://localhost:8000/auth/public-key

# Core Service (qua Kong, cần JWT)
curl -H "Authorization: Bearer <token>" http://localhost:8000/api/v1/health
```

#### 4. Truy Cập Ứng Dụng
- **Frontend**: http://localhost:5173
- **Kong Admin API**: http://localhost:8001
- **HAProxy Stats**: http://localhost:8404/stats

#### 5. Dừng Hệ Thống
```bash
# Dừng tất cả services
docker-compose down

# Dừng và xóa volumes (reset database)
docker-compose down -v
```

---

## 📁 Cấu Trúc Thư Mục

```
infra/
├── docker-compose.yml          # Docker Compose config
├── kong.yml                    # Kong Gateway config (generated)
├── .env                        # Environment variables
│
├── haproxy/
│   └── haproxy.cfg            # HAProxy load balancer config
│
├── infra/
│   ├── kong_conf/
│   │   └── kong.yml           # Kong declarative config template
│   └── generate_kong_yml.py   # Script to generate kong.yml
│
├── ml_models/
│   └── lstm_sentiment_hybrid_v1.pth  # Trained LSTM model (48MB)
│
├── services/
│   ├── auth-service/          # Authentication & JWT
│   │   ├── src/
│   │   ├── Dockerfile
│   │   └── package.json
│   │
│   ├── core-service/          # Historical data & API
│   │   ├── src/
│   │   ├── Dockerfile
│   │   └── package.json
│   │
│   ├── stream-service/        # WebSocket real-time data
│   │   ├── src/
│   │   ├── Dockerfile
│   │   └── package.json
│   │
│   ├── stream-ingester/       # Binance WebSocket → Kafka
│   │   ├── src/
│   │   ├── Dockerfile
│   │   └── package.json
│   │
│   ├── crawler-service/       # News crawler
│   │   ├── app/
│   │   ├── Dockerfile
│   │   └── requirements.txt
│   │
│   ├── ai-service/            # AI predictions
│   │   ├── app/
│   │   ├── train.py           # Training script
│   │   ├── quick_train.py     # Quick training
│   │   ├── kafka_data_collector.py  # Collect training data
│   │   ├── MODEL_ARCHITECTURE_EXPLAINED.md
│   │   ├── PREDICTION_FLOW.md
│   │   ├── GEMINI_ARCHITECTURE.md
│   │   ├── Dockerfile
│   │   └── requirements.txt
│   │
│   ├── payment-service/       # Payment & VIP upgrade
│   │   ├── src/
│   │   ├── Dockerfile
│   │   └── package.json
│   │
│   ├── investment-service/    # Investment simulator (VIP)
│   │   ├── app/
│   │   ├── Dockerfile
│   │   └── requirements.txt
│   │
│   └── notification-service/  # Email notifications
│       ├── src/
│       ├── Dockerfile
│       └── package.json
│
└── web-frontend/              # React frontend
    ├── src/
    │   ├── components/
    │   ├── store.js           # Zustand state management
    │   ├── App.jsx
    │   └── index.css
    ├── Dockerfile
    ├── package.json
    └── vite.config.js
```

---

## 📊 Monitoring & Logging

### Health Checks
Tất cả services đều có endpoint `/health`:
```bash
# Kiểm tra tất cả services
docker-compose ps

# Kiểm tra health của 1 service
docker inspect --format='{{.State.Health.Status}}' <container_name>
```

### Logs
```bash
# Xem logs real-time
docker-compose logs -f

# Xem logs của 1 service
docker-compose logs -f ai-service

# Xem logs với timestamp
docker-compose logs -f --timestamps

# Xem 100 dòng cuối
docker-compose logs --tail=100 ai-service
```

### HAProxy Stats
- URL: http://localhost:8404/stats
- Metrics:
  - Connection count per instance
  - Health status (UP/DOWN)
  - Response time
  - Error rate

### Kong Admin API
```bash
# Xem services
curl http://localhost:8001/services

# Xem routes
curl http://localhost:8001/routes

# Xem plugins
curl http://localhost:8001/plugins

# Xem consumers
curl http://localhost:8001/consumers
```

### Kafka Topics
```bash
# Exec vào Kafka container
docker-compose exec kafka bash

# List topics
kafka-topics --bootstrap-server localhost:9092 --list

# Describe topic
kafka-topics --bootstrap-server localhost:9092 --describe --topic market.prices

# Consume messages
kafka-console-consumer --bootstrap-server localhost:9092 --topic ai_insights --from-beginning
```

### Database
```bash
# PostgreSQL (Auth DB)
docker-compose exec postgres psql -U dev -d appdb

# TimescaleDB (Historical Data)
docker-compose exec timescaledb psql -U dev -d timeseriesdb

# MongoDB (Crawler DB)
docker-compose exec mongo-crawler mongosh

# Redis
docker-compose exec redis redis-cli
```

---

## 🎓 Training Model

### Collect Training Data
```bash
# Chạy data collector (lắng nghe Kafka và lưu vào file)
docker-compose exec ai-service python kafka_data_collector.py

# Data sẽ được lưu vào: services/ai-service/training_data/
```

### Train Model
```bash
# Quick training (test)
docker-compose exec ai-service python quick_train.py

# Full training
docker-compose exec ai-service python train.py

# Model checkpoint sẽ được lưu vào: services/ai-service/checkpoints/best_model.pth
```

### Deploy Model
```bash
# Copy model sang ml_models/
docker-compose exec ai-service bash deploy_model.sh

# Restart AI service để load model mới
docker-compose restart ai-service
```

---

## 🐛 Troubleshooting

### Kafka Connection Issues
```bash
# Kiểm tra Kafka health
docker-compose exec kafka kafka-broker-api-versions --bootstrap-server localhost:9092

# Restart Kafka
docker-compose restart kafka
```

### Kong JWT Issues
```bash
# Kiểm tra public key
curl http://localhost:8000/auth/public-key

# Kiểm tra JWT consumer
curl http://localhost:8001/consumers/shared-auth-consumer/jwt

# Reload Kong config
docker-compose restart kong-init kong
```

### WebSocket Connection Issues
```bash
# Kiểm tra HAProxy stats
open http://localhost:8404/stats

# Kiểm tra stream-service health
docker-compose exec stream-service curl http://localhost:3000/health

# Restart HAProxy
docker-compose restart haproxy
```

### Database Migration Issues
```bash
# Reset database
docker-compose down -v
docker-compose up -d postgres timescaledb

# Chạy migrations (nếu có)
docker-compose exec auth-service npm run migrate
```

---

## 📝 License

MIT License

---

## 👥 Contributors

- **AI/ML Engineer**: Deep Learning model, FinBERT integration
- **Backend Engineer**: Microservices, Kafka, Kong, HAProxy
- **Frontend Engineer**: React, Socket.IO, Zustand
- **DevOps Engineer**: Docker, Docker Compose, CI/CD

---

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/mandeotv1234/SA)
- **Email**: mandeotv1234@gmail.com
---

**🎉 Happy Trading with AI! 🚀**

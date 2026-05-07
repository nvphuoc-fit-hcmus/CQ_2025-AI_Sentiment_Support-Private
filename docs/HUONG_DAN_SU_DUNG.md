# 📘 Hướng Dẫn Sử Dụng — TradeAI (SAFE-Alert Platform)

> **Ứng dụng phân tích dữ liệu đa nguồn để khám phá xu hướng thị trường tài chính**
>
> Đồ án tốt nghiệp — Khoa CNTT, Trường ĐH Khoa Học Tự Nhiên TP.HCM

---

## Mục lục

1. [Yêu cầu hệ thống](#1-yêu-cầu-hệ-thống)
2. [Cài đặt và khởi chạy](#2-cài-đặt-và-khởi-chạy)
3. [Kiến trúc hệ thống](#3-kiến-trúc-hệ-thống)
4. [Hướng dẫn sử dụng giao diện](#4-hướng-dẫn-sử-dụng-giao-diện)
5. [Các tính năng chính](#5-các-tính-năng-chính)
6. [Quản trị hệ thống (Admin)](#6-quản-trị-hệ-thống-admin)
7. [Xử lý sự cố](#7-xử-lý-sự-cố)

---

## 1. Yêu cầu hệ thống

### Phần cứng tối thiểu

| Thành phần | Yêu cầu |
|------------|---------|
| CPU | 4 cores trở lên |
| RAM | 16 GB (khuyến nghị) / 8 GB (tối thiểu) |
| Ổ đĩa | 10 GB trống |
| Mạng | Kết nối Internet ổn định |

### Phần mềm cần cài đặt

| Phần mềm | Phiên bản | Ghi chú |
|-----------|-----------|---------|
| [Docker Desktop](https://www.docker.com/products/docker-desktop/) | ≥ 4.0 | Bao gồm Docker Compose |
| [Node.js](https://nodejs.org/) | ≥ 18.x | Cho frontend development |
| [Git](https://git-scm.com/) | ≥ 2.x | Quản lý source code |

---

## 2. Cài đặt và khởi chạy

### Bước 1: Clone repository

```bash
git clone https://github.com/nvphuoc-fit-hcmus/CQ_2025-AI_Sentiment_Support-Private.git
cd CQ_2025-AI_Sentiment_Support-Private
```

### Bước 2: Cấu hình biến môi trường

```bash
# Sao chép file cấu hình mẫu
cp .env.example .env
```

Mở file `.env` và điền các giá trị:

```env
# API Key cho AI (Gemini/Gemma)
GEMINI_API_KEY=your_gemini_api_key_here

# Thanh toán SePay
SEPAY_API_KEY=your_sepay_api_key_here

# Email thông báo (Gmail App Password)
SMTP_EMAIL=your_email@gmail.com
SMTP_PASSWORD=your_gmail_app_password
```

> **Lưu ý:** Để tạo Gmail App Password, truy cập https://myaccount.google.com/apppasswords

### Bước 3: Khởi chạy Backend (Docker)

```bash
# Khởi chạy toàn bộ 15+ containers
docker compose up -d

# Kiểm tra trạng thái
docker compose ps
```

Đợi khoảng **1–2 phút** cho tất cả services healthy. Thứ tự khởi động:

```
1. Databases    → Kafka, PostgreSQL, TimescaleDB, Redis, MongoDB
2. Core         → Auth, Core Service, Crawler
3. Stream       → Stream Ingester, Stream Service (×3), HAProxy
4. AI & Others  → AI Service, Backtest, Investment, Payment, Notification
5. Gateway      → Kong API Gateway (port 8000)
```

### Bước 4: Khởi chạy Frontend

```bash
cd web-frontend
npm install        # Chỉ cần lần đầu
npm run dev        # Khởi chạy dev server
```

### Bước 5: Truy cập ứng dụng

Mở trình duyệt và truy cập:

| URL | Mô tả |
|-----|-------|
| `http://localhost:5173` | Giao diện web chính |
| `http://localhost:8000` | Kong API Gateway |
| `http://localhost:8001` | Kong Admin API |
| `http://localhost:8404` | HAProxy Stats |

---

## 3. Kiến trúc hệ thống

### Tổng quan Microservices

```
┌─────────────────────────────────────────────────────────────┐
│                    Frontend (React + Vite)                    │
│                    http://localhost:5173                      │
└─────────────────────────┬───────────────────────────────────┘
                          │ HTTP / WebSocket
                          ▼
┌─────────────────────────────────────────────────────────────┐
│              Kong API Gateway (port 8000)                    │
│         JWT RS256 · Rate Limiting · RBAC · CORS             │
└──────┬──────┬──────┬──────┬──────┬──────┬───────────────────┘
       │      │      │      │      │      │
       ▼      ▼      ▼      ▼      ▼      ▼
   ┌──────┐┌──────┐┌──────┐┌──────┐┌──────┐┌──────────┐
   │ Auth ││ Core ││Stream││Invest││ Pay  ││ Backtest │
   │ Svc  ││ Svc  ││ Svc  ││ Svc  ││ Svc  ││   Svc    │
   └──┬───┘└──┬───┘└──┬───┘└──┬───┘└──┬───┘└────┬─────┘
      │       │       │       │       │          │
      ▼       ▼       ▼       ▼       ▼          ▼
   ┌──────────────────────────────────────────────────┐
   │              Apache Kafka (Message Broker)        │
   │     Topics: market.prices, news_raw, ai_insights │
   └──────────────────────────────────────────────────┘
      │          │           │           │
      ▼          ▼           ▼           ▼
   Postgres  TimescaleDB  MongoDB     Redis
   (Users)   (Klines)     (News)     (Cache)
```

### Danh sách Services

| Service | Chức năng | Port nội bộ |
|---------|-----------|-------------|
| **auth-service** | Xác thực JWT RS256, quản lý user, phân quyền VIP/Admin | 8080 |
| **core-service** | API nến giá đa khung thời gian, tin tức, insights | 3000 |
| **stream-ingester** | Kết nối Binance WebSocket, thu thập giá real-time 10 coin | — |
| **stream-service** (×3) | Phân phối giá real-time qua Socket.IO | 3000 |
| **crawler-service** | Crawl tin tức từ 18 nguồn RSS mỗi 60 giây | 8001 |
| **ai-service** | Phục vụ mô hình SAFEAlertNet, API dự đoán | 8002 |
| **backtest-service** | Mô phỏng giao dịch với dữ liệu lịch sử | 8005 |
| **investment-service** | Quản lý danh mục đầu tư ảo | 8001 |
| **payment-service** | Tích hợp thanh toán SePay cho gói VIP | 3000 |
| **notification-service** | Gửi email cảnh báo tín hiệu qua SMTP | — |

---

## 4. Hướng dẫn sử dụng giao diện

### 4.1. Đăng nhập / Đăng ký

1. Truy cập `http://localhost:5173`
2. Nhấn **"Đăng ký ngay"** để tạo tài khoản mới, hoặc nhập email/mật khẩu để đăng nhập
3. Sau khi đăng nhập thành công, hệ thống chuyển đến **Trading Dashboard**

### 4.2. Bố cục Dashboard chính

```
┌─────────────────────────────────────────────────┬──────────┐
│                    Navbar                        │  VIP 👑  │
├────┬────────────────────────────────────────────┼──────────┤
│    │  BTC/USDT (1m)     │  BTC/USDT (1h)       │  🧠 AI   │
│ T  │  ┌──────────────┐  │  ┌──────────────┐    │ Analysis │
│ o  │  │  Biểu đồ 1   │  │  │  Biểu đồ 2   │   │          │
│ o  │  └──────────────┘  │  └──────────────┘    │ Gauge    │
│ l  ├────────────────────┼──────────────────────┤ 82% ▲    │
│ b  │  BTC/USDT (1D)     │  BTC/USDT (1W)       │          │
│ a  │  ┌──────────────┐  │  ┌──────────────┐    │ Factors  │
│ r  │  │  Biểu đồ 3   │  │  │  Biểu đồ 4   │   │ Evidence │
│    │  └──────────────┘  │  └──────────────┘    │ Cards    │
└────┴────────────────────┴──────────────────────┴──────────┘
```

- **Khu vực trái (75%)**: Lưới 2×2 hiển thị 4 biểu đồ nến đồng thời
- **Thanh bên phải (25%)**: Intelligence Hub — Bảng phân tích AI
- **Thanh trên**: Navigation, tìm kiếm symbol, cài đặt

---

## 5. Các tính năng chính

### 5.1. 📊 Biểu đồ đa khung thời gian (Multi-Timeframe Chart)

**Mô tả:** Hiển thị 4 biểu đồ nến (candlestick) đồng thời, mỗi biểu đồ có thể chọn khung thời gian riêng.

**Cách sử dụng:**
- Nhấn các nút **1m / 5m / 15m / 1h / 4h / 1D / 1W / 1M** phía trên mỗi biểu đồ để chuyển khung thời gian
- Cuộn chuột để zoom in/out trên biểu đồ
- Kéo chuột để di chuyển theo thời gian (kéo sang trái sẽ tự động tải thêm dữ liệu lịch sử)

**Chỉ báo kỹ thuật (Technical Indicators):**

| Nút | Chỉ báo | Mô tả |
|-----|---------|-------|
| SMA 20 | Simple Moving Average | Đường trung bình động đơn giản 20 kỳ |
| EMA 12 | Exponential Moving Average | Đường trung bình động hàm mũ 12 kỳ |
| EMA 26 | Exponential Moving Average | Đường trung bình động hàm mũ 26 kỳ |
| BB | Bollinger Bands | Dải Bollinger (20 kỳ, 2 độ lệch chuẩn) |
| RSI | Relative Strength Index | Chỉ số sức mạnh tương đối (14 kỳ) |
| MACD | Moving Average Convergence Divergence | Đường MACD (12, 26, 9) |

**Tín hiệu AI trên biểu đồ:**
- **▲ (mũi tên xanh)**: Tín hiệu TĂNG từ mô hình AI, kèm % độ tin cậy
- **▼ (mũi tên đỏ)**: Tín hiệu GIẢM từ mô hình AI
- **Uncertainty Zone (vùng sọc xám)**: Mô hình chưa đủ tin cậy để phát cảnh báo — nên chờ

### 5.2. 🧠 Intelligence Hub — Bảng phân tích AI (Decision Panel)

Đây là tính năng cốt lõi của hệ thống, trực quan hóa 4 tầng kiến trúc SAFE-Alert.

**Tab "Phân tích AI" (mặc định):**

#### Tầng 4: Confidence Gauge (Đồng hồ độ tin cậy)

- Vòng cung SVG 270° hiển thị độ tin cậy ĉ ∈ [0, 1]
- **Màu xám + "CHỜ TÍN HIỆU"**: Mô hình ở trạng thái Abstain (chưa đủ tin cậy)
- **Màu xanh lá + "▲ TĂNG"**: Tín hiệu mua với độ tin cậy cao
- **Màu đỏ + "▼ GIẢM"**: Tín hiệu bán với độ tin cậy cao
- **Hiệu ứng pulse (nhấp nháy)**: Cảnh báo đang hoạt động

#### Tầng 3: Factor Heatmap (Yếu tố chi phối)

- Hiển thị Top-3 yếu tố thị trường ảnh hưởng lớn nhất đến dự đoán
- Nhãn tiếng Việt (ví dụ: "Dòng vốn tổ chức") kèm phụ đề tiếng Anh (institutional_inflow)
- Thanh ngang thể hiện trọng số tương đối của từng yếu tố

| Yếu tố | Mô tả |
|---------|-------|
| Dòng vốn tổ chức | Hoạt động mua/bán của các quỹ lớn, ETF |
| Dòng vốn ETF | Dòng tiền vào/ra các quỹ Bitcoin ETF |
| Nới lỏng quy định | Tin tức về chính sách pháp lý thuận lợi |
| Siết chặt quy định | Tin tức về siết chặt pháp lý |
| Rủi ro sàn giao dịch | Sự cố hoặc rủi ro từ các sàn lớn |
| Tích lũy cá voi | Hoạt động mua/bán của ví lớn (whales) |
| Đà tăng giá | Xu hướng kỹ thuật tăng |
| Đà giảm giá | Xu hướng kỹ thuật giảm |
| Tin tức vĩ mô | Lãi suất, lạm phát, chính sách tiền tệ |
| Phát triển hệ sinh thái | Cập nhật công nghệ blockchain |

#### Tầng 2: Structured Explainer (Giải thích AI)

- Văn bản giải thích tự nhiên bằng tiếng Việt
- Template: _"Tín hiệu [HƯỚNG] với độ tin cậy [C%] cho [SYMBOL]. Yếu tố chi phối chính là [FACTOR] được hỗ trợ bởi [N] bằng chứng tin tức..."_
- Badge **"Faithful"**: Xác nhận giải thích chỉ dùng bằng chứng đã được mô hình chọn lọc

#### Tầng 1: Evidence Cards (Bằng chứng)

- Danh sách Top-K bài báo mà mô hình đã chọn lọc
- Mỗi card hiển thị: tiêu đề, nguồn (Reuters, CoinDesk, Bloomberg...), link gốc
- Đánh số α1, α2, α3... theo thứ tự relevance score giảm dần

### 5.3. 📈 Watchlist — Bảng giá (Tab "Thị trường")

- Nhấn icon **☰ (List)** trên sidebar để chuyển sang tab Thị trường
- Hiển thị giá real-time của 10 cặp tiền: BTC, ETH, BNB, SOL, XRP, DOGE, ADA, AVAX, DOT, POL
- Nhấn vào một coin để chuyển tất cả 4 biểu đồ sang coin đó
- Ô tìm kiếm phía trên để lọc nhanh

### 5.4. 📰 Tin tức (Tab "Tin tức")

- Nhấn icon **📰 (Newspaper)** trên sidebar
- Danh sách tin tức crypto được crawl tự động từ 18 nguồn mỗi 60 giây
- Mỗi tin kèm nhãn sentiment (Tích cực / Trung lập / Tiêu cực)
- Nhấn vào tin để xem chi tiết

### 5.5. 🔄 Backtest — Mô phỏng giao dịch

- Nhấn menu **"Backtest"** trên thanh navigation
- Chọn chiến lược giao dịch và khoảng thời gian
- Nhấn **"Chạy Backtest"** để mô phỏng
- Kết quả hiển thị: Equity Curve, Net Profit, Win Rate, Max Drawdown, Sharpe Ratio, danh sách giao dịch

### 5.6. 💰 Đầu tư ảo (Investment Simulator)

- Nhấn menu **"Đầu Tư"** trên thanh navigation
- Tạo danh mục đầu tư ảo với vốn ban đầu
- AI tự động phân tích và đề xuất phân bổ tài sản
- Theo dõi hiệu suất danh mục theo thời gian

### 5.7. 👑 Nâng cấp VIP

- Nhấn nút **"VIP"** trên thanh navigation
- Gói VIP mở khóa các tính năng nâng cao:
  - Tín hiệu AI real-time không giới hạn
  - Backtest nâng cao
  - Ưu tiên xử lý AI
- Thanh toán qua SePay (quét QR)

---

## 6. Quản trị hệ thống (Admin)

### Truy cập Admin Panel

- Đăng nhập bằng tài khoản có quyền Admin
- Admin panel cho phép:
  - Quản lý người dùng (xem, khóa, phân quyền)
  - Theo dõi trạng thái hệ thống
  - Xem thống kê sử dụng

---

## 7. Xử lý sự cố

### 7.1. Backend không khởi động

```bash
# Kiểm tra logs của service bị lỗi
docker compose logs <tên-service> --tail 50

# Ví dụ
docker compose logs auth-service --tail 50
docker compose logs ai-service --tail 50

# Khởi động lại một service cụ thể
docker compose restart <tên-service>

# Khởi động lại toàn bộ
docker compose down && docker compose up -d
```

### 7.2. Frontend hiện "503 Error" trên Decision Panel

**Nguyên nhân:** AI Service chưa có dữ liệu thị trường (Kafka chưa sẵn sàng hoặc Stream Ingester chưa chạy).

**Cách xử lý:**
1. Kiểm tra Kafka: `docker compose logs kafka --tail 20`
2. Kiểm tra Stream Ingester: `docker compose logs stream-ingester --tail 20`
3. Đợi 1–2 phút cho dữ liệu được thu thập
4. Nhấn nút **"Thử lại"** trên Decision Panel

> **Lưu ý:** Khi AI backend offline, hệ thống tự động hiển thị dữ liệu demo để minh hoạ giao diện.

### 7.3. Biểu đồ không hiện dữ liệu

**Nguyên nhân:** Core Service hoặc TimescaleDB chưa sẵn sàng.

```bash
# Kiểm tra Core Service
docker compose logs core-service --tail 20

# Kiểm tra TimescaleDB
docker compose logs timescaledb --tail 20
```

### 7.4. Không đăng nhập được

**Nguyên nhân:** Auth Service hoặc PostgreSQL chưa healthy.

```bash
docker compose logs auth-service --tail 30
docker compose logs postgres --tail 20
```

### 7.5. Máy chậm / Hết RAM

Nếu máy có ít RAM (8 GB), có thể chạy các service thiết yếu trước:

```bash
# Chỉ chạy services cần thiết
docker compose up -d kafka postgres timescaledb redis mongo-crawler
docker compose up -d auth-service core-service stream-ingester stream-service
docker compose up -d haproxy kong-init kong

# Sau đó bổ sung thêm nếu cần
docker compose up -d ai-service crawler-service
```

---

## Phụ lục

### A. Các cặp tiền được hỗ trợ

| Symbol | Tên |
|--------|-----|
| BTCUSDT | Bitcoin |
| ETHUSDT | Ethereum |
| BNBUSDT | Binance Coin |
| SOLUSDT | Solana |
| XRPUSDT | Ripple |
| DOGEUSDT | Dogecoin |
| ADAUSDT | Cardano |
| AVAXUSDT | Avalanche |
| DOTUSDT | Polkadot |
| POLUSDT | Polygon |

### B. Khung thời gian được hỗ trợ

| Ký hiệu | Khung thời gian |
|----------|----------------|
| 1m | 1 phút |
| 5m | 5 phút |
| 15m | 15 phút |
| 1h | 1 giờ |
| 4h | 4 giờ |
| 1D | 1 ngày |
| 1W | 1 tuần |
| 1M | 1 tháng |

### C. API Endpoints chính

| Method | Endpoint | Mô tả |
|--------|----------|-------|
| POST | `/auth/login` | Đăng nhập |
| POST | `/auth/register` | Đăng ký |
| GET | `/api/v1/klines` | Lấy dữ liệu nến |
| GET | `/api/v1/news` | Lấy tin tức |
| GET | `/v2/signal/{symbol}` | Lấy tín hiệu AI |
| WS | `/stream-api/socket.io` | WebSocket giá real-time |

---

> **Nhóm phát triển:** 22120269 Nguyễn Hoài Phú · 22120279 Phạm Tài Phúc · 22120285 Nguyễn Văn Phước · 22120292 Nguyễn Hải Quân
>
> **GVHD:** ThS. Trần Văn Quý — Khoa CNTT, Trường ĐH KHTN TP.HCM

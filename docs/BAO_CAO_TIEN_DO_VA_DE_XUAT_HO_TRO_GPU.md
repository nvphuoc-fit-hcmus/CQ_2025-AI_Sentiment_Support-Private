# BÁO CÁO TIẾN ĐỘ ĐỀ TÀI VÀ ĐỀ XUẤT HỖ TRỢ GPU

Kính gửi Thầy,

Nhóm xin gửi báo cáo tổng hợp tiến độ đề tài, các thành phần đã implement theo paper, hiện trạng vận hành hệ thống, khó khăn tài nguyên tính toán, và đề xuất hỗ trợ GPU để hoàn tất giai đoạn training.

## 1. Thông tin chung

- Đề tài: Hệ thống phân tích cảm xúc và dự đoán thị trường crypto theo kiến trúc microservices.
- Mục tiêu: Xây dựng pipeline end-to-end từ thu thập dữ liệu -> phân tích tin tức -> dự đoán tín hiệu -> hiển thị real-time.
- Nền tảng: Node.js + Python FastAPI + Kafka + PostgreSQL/TimescaleDB + MongoDB + Redis + React + Docker Compose.

## 2. Các nội dung đã hoàn thành

### 2.1. Implement kiến trúc hệ thống

Nhóm đã xây dựng đầy đủ hệ thống microservices theo hướng event-driven:

- Gateway và bảo mật:
  - Kong API Gateway (JWT, rate limit, CORS, RBAC headers)
- Truyền dữ liệu realtime:
  - Stream service (Socket.IO, scale replica)
  - HAProxy (consistent hashing cho kết nối realtime)
- Hàng đợi sự kiện:
  - Kafka topics cho market data, news, ai insights, payment events
- Data layer:
  - PostgreSQL, TimescaleDB, MongoDB, Redis
- Business services:
  - Auth, Core, AI, Crawler, Payment, Notification, Investment, Backtest

### 2.2. Implement các module AI theo paper (SAFE-Alert)

Theo đối chiếu với đặc tả paper, nhóm đã implement được các thành phần chính:

- SelectiveNewsEncoder
- FactorModule
- MultiTimescaleMarketEncoder
- CrossAttentionFusion
- Prediction Heads (direction, return, confidence)
- Alert decision gate
- Explanation module
- Multi-objective loss (nhiều thành phần)
- Curriculum training theo giai đoạn
- Walk-forward validation

### 2.3. Pipeline dữ liệu và suy luận

- Thu thập giá real-time từ Binance (nhiều symbol).
- Crawl tin tức đa nguồn và đưa vào pipeline phân tích.
- Tạo feature thị trường + feature tin tức.
- AI service suy luận và đẩy kết quả vào Kafka.
- Core service lưu trữ và cấp dữ liệu cho frontend.

### 2.4. Frontend và tích hợp

- Đã có giao diện web theo dõi coin, khung thời gian, dashboard và các thành phần hiển thị dữ liệu.
- Đã kết nối tới backend stack qua Docker Compose.
- Nhóm đã debug và khôi phục nhiều lỗi hệ thống khi chạy tích hợp (xung đột cổng, service restart, kết nối nội bộ Docker, Redis/Kafka).

## 3. Kết quả hiện tại

- Hệ thống đã chạy được theo hướng full-stack Docker.
- Các service cốt lõi đã dựng thành công (Kafka, Redis, AI, Core, Crawler, DB, v.v.).
- Luồng dữ liệu realtime đã được khôi phục sau khi xử lý xung đột Redis và kết nối stream.
- Nhóm đã có endpoint và pipeline để train/inference và theo dõi trạng thái training.

## 4. Khó khăn lớn nhất: Tài nguyên GPU cho training

Mặc dù đã implement được phần lớn paper, nhóm đang gặp nút thắt ở giai đoạn train full model:

- Mục tiêu cần train đầy đủ các fold/epoch theo thiết kế để đạt chất lượng ổn định.
- Tài nguyên GPU cá nhân không đủ (VRAM và thời gian huấn luyện).
- Đã thử dùng Kaggle GPU nhiều đợt nhưng hết quota/không đủ cho chu trình train đầy đủ.
- Vì vậy, nhóm chưa thể chốt training quy mô lớn để tổng hợp kết quả cuối cùng đúng mức kỳ vọng.

## 5. Đề xuất hỗ trợ từ nhà trường

Nhóm kính đề xuất Thầy hỗ trợ cho mượn tài nguyên GPU của trường trong thời gian ngắn để hoàn tất giai đoạn huấn luyện.

### 5.1. Nhu cầu tối thiểu

- 1 máy GPU (ưu tiên RTX 3090/4090 hoặc tương đương; nếu có A100/V100 là tốt).
- VRAM khuyến nghị: >= 24GB.
- Thời gian sử dụng đề nghị: 1-2 tuần, tập trung train và ablation.
- Môi trường: có thể qua SSH/remote server hoặc workstation tại phòng lab.

### 5.2. Kế hoạch khi được cấp GPU

- Giai đoạn 1: Chuẩn hóa data và chạy precompute.
- Giai đoạn 2: Train full theo lịch walk-forward/curriculum.
- Giai đoạn 3: Đánh giá, đối chiếu metric, thử nghiệm bổ sung.
- Giai đoạn 4: Chốt model, báo cáo kết quả, đóng gói bản demo cuối.

### 5.3. Cam kết

- Nhóm sẽ sử dụng tài nguyên đúng mục đích, ghi nhật ký train đầy đủ.
- Nhóm sẽ nộp kết quả trung gian theo mốc do Thầy quy định.
- Sau khi hoàn tất, nhóm sẽ bàn giao báo cáo kỹ thuật và kết quả thực nghiệm chi tiết.

## 6. Danh mục minh chứng kỹ thuật trong dự án

- Tổng quan hệ thống: `README.md`
- AI service và đối chiếu với paper: `services/ai-service/README.md`
- Cập nhật core phase 2: `services/core-service/README_PHASE2.md`

## 7. Kết luận

Nhóm đã hoàn thành phần lớn implement theo paper và đã đưa được hệ thống vào trạng thái vận hành tích hợp. Vấn đề còn lại chủ yếu là tài nguyên GPU để train đầy đủ và tối ưu hóa kết quả cuối. Nhóm rất mong được Thầy hỗ trợ cho mượn GPU của trường để hoàn tất đề tài đúng tiến độ và chất lượng mong muốn.

Nhóm xin chân thành cảm ơn Thầy.

---

## Phụ lục: Thông tin bổ sung để điền trước khi gửi

- Môn học/Đề tài: ........................................
- Giảng viên hướng dẫn: ..................................
- Tên nhóm: ...............................................
- Thành viên: .............................................
- Ngày gửi báo cáo: ......../......../............

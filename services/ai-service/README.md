# AI Service - SAFE-Alert v2

`ai-service` là service huấn luyện, đánh giá và phục vụ mô hình **SAFE-Alert**
cho bài toán cảnh báo biến động BTCUSDT theo horizon. Code hiện tại tập trung
vào bản paper-final: pipeline train walk-forward, cơ chế selective alert,
giải thích dựa trên news/factor, policy deployable và live inference qua FastAPI.

Service có hai chế độ dùng chính:

- **Research/offline**: precompute dữ liệu, train, walk-forward evaluation,
  ablation, baseline, backtest, export prediction logs/case studies.
- **Runtime/API**: lấy market cache + MongoDB news, chạy inference SAFE-Alert,
  trả tín hiệu `/v2/signal/{symbol}` và lưu latest signal.

> Lưu ý push GitHub: `training_data/`, checkpoint `*.pt`, `artifacts/` và cache
> lớn được ignore. Repo chỉ nên push code, config, test và tài liệu.

---

## 1. Cấu trúc thư mục

```text
ai-service/
├── app/
│   ├── main.py                         # FastAPI app, scheduler, API endpoints
│   ├── market_cache.py                 # Kafka/live candle cache
│   └── v2/
│       ├── constants.py                # Hằng số paper: epsilon_h, K_h, factor defaults
│       ├── factor_ontology.json        # 10 factor classes + keywords
│       ├── DEVIATIONS.md               # Các điểm code khác paper và lý do
│       ├── ERRATUM.md                  # Ghi chú chỉnh paper/thesis nếu cần
│       ├── alerts/
│       │   └── alert_decider.py        # Eq.26 alert gate
│       ├── models/
│       │   └── safe_alert_net.py       # SAFEAlertNet: architecture chính
│       ├── nlp/
│       │   ├── news_selector.py        # Chọn/format news runtime
│       │   └── relevance_scorer.py     # Recency/source/keyword relevance
│       ├── preprocessing/
│       │   ├── market_features.py      # 63-dim scalar features + bar features
│       │   └── news_features.py        # VADER/FinBERT NLP features runtime
│       └── pipelines/
│           ├── train_safe_alert.py              # Train chính
│           ├── train_config_research_best.yaml  # Config mặc định
│           ├── train_config_paper_strict.yaml   # Config so sánh paper-literal
│           ├── safe_alert_dataset.py            # Dataset, labels, leakage guards
│           ├── safe_alert_training_utils.py     # MultiObjectiveLoss + sampler
│           ├── metrics_safe_alert.py            # Metrics, policy search, backtest mini
│           ├── precompute_market_features.py    # Tạo features_precomputed.npy
│           ├── precompute_market_bars.py        # Tạo market_bars*.npz Eq.16
│           ├── precompute_novelty.py            # Tạo article_novelty.npy
│           ├── precompute_factor_labels.py      # Tạo article_factor_labels.npy
│           ├── precompute_entity_sentiment.py   # Tạo article_entity_sentiment.npy
│           ├── gen_emb.py                       # Tạo FinBERT embeddings
│           ├── run_ablation.py                  # Ablation study
│           ├── run_baselines.py                 # Baseline comparison
│           ├── backtest_safe_alert.py           # Backtest checkpoint/policy
│           ├── eval_faithfulness_sec4p4p2.py    # Faithfulness eval
│           ├── case_study_export.py             # Export case studies
│           ├── prediction_logs.py               # Export per-sample prediction logs
│           ├── explanation_generator.py         # Eq.29 template explanation
│           └── live_infer.py                    # Runtime inference adapter
├── tests/                              # Unit/regression tests
├── training_data/                      # Local data, ignored by git
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## 2. Cài đặt

### 2.1. Local Python

Khuyến nghị Python 3.11.

```powershell
cd "e:\Khóa luận 1\SA\services\ai-service"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Nếu dùng GPU CUDA trên Kaggle/Colab, có thể cài PyTorch theo môi trường CUDA
trước, sau đó cài các package còn lại.

### 2.2. Docker CPU

```powershell
docker build -t safe-alert-ai:2.0 .
docker run --rm -p 8002:8002 safe-alert-ai:2.0
```

API chạy tại:

```text
http://localhost:8002/health
```

---

## 3. Dữ liệu đầu vào

Mặc định các script đọc dữ liệu trong:

```text
training_data/v2/
```

Các file quan trọng:

```text
BTCUSDT_1h_ohlcv.csv                 # Candle quyết định chính
BTCUSDT_1m_ohlcv.csv                 # Multi-timeframe input
BTCUSDT_5m_ohlcv.csv
BTCUSDT_15m_ohlcv.csv
BTCUSDT_4h_ohlcv.csv
articles_max.csv                     # News corpus
btcusdt_article_embeddings_max.npy   # Article embeddings, shape (N_articles, 768)
features_precomputed.npy             # Scalar market features, shape (N_candles, 63)
features_precomputed.meta.json       # Metadata alignment check
market_bars.npz                      # Eq.16 bar sequences cho horizon 1h
market_bars_4h.npz                   # Eq.16 bar sequences cho horizon 4h nếu train 4h
article_factor_labels.npy            # Factor pseudo labels, shape (N_articles, 10)
article_entity_sentiment.npy         # Target-based FSA, shape (N_articles, 10)
article_novelty.npy                  # TF-IDF novelty, shape (N_articles,)
```

`training_data/` bị ignore để không push dữ liệu lớn lên GitHub.

---

## 4. Precompute dữ liệu

Các bước dưới đây chạy từ root `ai-service/`.

### 4.1. Article embeddings

```powershell
python app/v2/pipelines/gen_emb.py
```

Output:

```text
training_data/v2/btcusdt_article_embeddings_max.npy
```

Script dùng `ProsusAI/finbert`, fallback sang `bert-base-uncased` nếu FinBERT
không load được.

### 4.2. Scalar market features

```powershell
python app/v2/pipelines/precompute_market_features.py
```

Output:

```text
training_data/v2/features_precomputed.npy
training_data/v2/features_precomputed.meta.json
```

File này giúp train nhanh hơn vì dataset không cần tính technical indicators
trong từng `__getitem__`.

### 4.3. Bar-sequence market input Eq.16

Cho horizon 1h:

```powershell
python app/v2/pipelines/precompute_market_bars.py `
  --data-dir training_data/v2 `
  --symbol BTCUSDT `
  --decision-horizon 1h `
  --bar-seq-len 20 `
  --out training_data/v2/market_bars.npz
```

Cho horizon 4h:

```powershell
python app/v2/pipelines/precompute_market_bars.py `
  --data-dir training_data/v2 `
  --symbol BTCUSDT `
  --decision-horizon 4h `
  --bar-seq-len 20 `
  --out training_data/v2/market_bars_4h.npz
```

Mỗi file chứa tensor 5 timeframe x N candles x 20 bars x 10 features.

### 4.4. Article novelty

```powershell
python app/v2/pipelines/precompute_novelty.py `
  --articles-csv training_data/v2/articles_max.csv `
  --out training_data/v2/article_novelty.npy
```

Novelty là TF-IDF novelty theo paper Section 4.1.3, dùng làm `META[3]`.

### 4.5. Factor labels

Nhanh nhất, không cần API:

```powershell
python app/v2/pipelines/precompute_factor_labels.py `
  --articles_path training_data/v2/articles_max.csv `
  --output_dir training_data/v2 `
  --method keyword
```

Paper-faithful hơn nếu có GPU/API:

```powershell
python app/v2/pipelines/precompute_factor_labels.py `
  --articles_path training_data/v2/articles_max.csv `
  --output_dir training_data/v2 `
  --method qwen
```

Output:

```text
training_data/v2/article_factor_labels.npy
```

### 4.6. Entity sentiment / FSA

```powershell
python app/v2/pipelines/precompute_entity_sentiment.py `
  --articles_path training_data/v2/articles_max.csv `
  --output_dir training_data/v2 `
  --batch_size 16
```

Output:

```text
training_data/v2/article_entity_sentiment.npy
```

---

## 5. Train SAFE-Alert

Lệnh mặc định:

```powershell
python app/v2/pipelines/train_safe_alert.py
```

Mặc định script dùng:

```text
app/v2/pipelines/train_config_research_best.yaml
```

Lệnh rõ ràng hơn:

```powershell
python app/v2/pipelines/train_safe_alert.py `
  --config app/v2/pipelines/train_config_research_best.yaml `
  --symbol BTCUSDT `
  --horizon 1h `
  --data_path training_data/v2 `
  --embeddings_path training_data/v2 `
  --artifact_dir artifacts/research_best_1h
```

Một run nhanh để kiểm tra pipeline:

```powershell
python app/v2/pipelines/train_safe_alert.py `
  --config app/v2/pipelines/train_config_research_best.yaml `
  --epochs 2 `
  --n_folds 1 `
  --artifact_dir artifacts/smoke_1h
```

Nếu muốn đúng paper-literal để so sánh:

```powershell
python app/v2/pipelines/train_safe_alert.py `
  --config app/v2/pipelines/train_config_paper_strict.yaml `
  --artifact_dir artifacts/paper_strict_1h
```

### 5.1. Config chính

`train_config_research_best.yaml` là config mặc định cho bản hiện tại:

- `market_input_mode: hybrid`: scalar features + Eq.16 bar sequences qua gated residual.
- `use_bar_sequences: true`: dùng `market_bars*.npz` nếu có.
- `walk_forward: true`: expanding train -> val -> test.
- `n_folds: 5`: số fold chính thức khi đủ thời gian train.
- `embargo_steps: 24`: tránh leakage giữa train/val/test.
- `balanced_batch_sampler: true`: cân bằng DOWN/NEUTRAL/UP trong train batch.
- `coverage_target: 0.35`: target coverage Eq.37.
- `lambda_vol: 0.0`: Lvol tắt vì không thuộc paper loss.
- `lambda_ret_bin`, `lambda_up_edge`, `lambda_down_edge`: optional heads, mặc định tắt.

CLI luôn override YAML.

### 5.2. Output artifacts

Trong `artifact_dir`, train sẽ sinh:

```text
fold_1/
  safe_alert_1h_best_epoch*.pt
  safe_alert_1h_FINAL.pt
  safe_alert_1h_swa.pt
  safe_alert_1h_swa_policy.json
  training_metrics.json
  case_studies.json
  prediction_logs.jsonl
walk_forward_results.json
safe_alert_btcusdt_1h_policy.json
safe_alert_policy_btcusdt_1h.json
table3_validation_report.json
```

Tên chính xác có thể thay đổi theo horizon/artifact dir, nhưng các nhóm artifact
trên là output chính: checkpoint, policy, metric, case study và prediction log.

---

## 6. Evaluate, ablation, baseline, backtest

### 6.1. Unit tests

```powershell
python -m pytest tests -q
```

Một vài test hay dùng:

```powershell
python -m pytest tests/test_model_forward.py -q
python -m pytest tests/test_safe_alert_losses.py -q
python -m pytest tests/test_stage_aware_scheduler.py -q
```

### 6.2. Ablation study

```powershell
python app/v2/pipelines/run_ablation.py `
  --config app/v2/pipelines/train_config_research_best.yaml `
  --symbol BTCUSDT `
  --horizon 1h `
  --epochs 30 `
  --artifact_dir artifacts/ablation_1h
```

Các variant chính gồm full model và ablation như `w/o_news`, `w/o_factor`,
`w/o_market`, `w/o_confidence`, `w/o_faith`, `w/o_bar_sequences`.

### 6.3. Baseline comparison

```powershell
python app/v2/pipelines/run_baselines.py `
  --config app/v2/pipelines/train_config_research_best.yaml `
  --symbol BTCUSDT `
  --horizon 1h `
  --epochs 30 `
  --artifact_dir artifacts/baselines_1h
```

### 6.4. Backtest

```powershell
python app/v2/pipelines/backtest_safe_alert.py `
  --artifact_dir artifacts/research_best_1h `
  --data_dir training_data/v2 `
  --symbol BTCUSDT `
  --horizon 1h
```

Thêm `--long_only` nếu chỉ muốn backtest long-only.

### 6.5. Faithfulness evaluation

```powershell
python app/v2/pipelines/eval_faithfulness_sec4p4p2.py `
  --model_path artifacts/research_best_1h/fold_1/safe_alert_1h_FINAL.pt `
  --symbol BTCUSDT `
  --horizon 1h `
  --device cuda `
  --batch_size 32
```

---

## 7. Chạy API service

### 7.1. Local

```powershell
uvicorn app.main:app --host 0.0.0.0 --port 8002 --reload
```

Health check:

```powershell
curl http://localhost:8002/health
```

Endpoint chính:

```text
GET  /health
GET  /v2/signal/{symbol}
GET  /v2/signal/{symbol}/cached
POST /v2/signal/run-now
GET  /predictions/latest
GET  /debug/market_cache/{symbol}
GET  /debug/buffer-status
POST /debug/reset-buffer
POST /admin/train-model
GET  /admin/training-status
```

### 7.2. Runtime environment

Các biến môi trường thường dùng:

```text
KAFKA_BROKERS=localhost:9092
MARKET_DATA_TOPIC=market.prices
MARKET_CACHE_GROUP=ai-service-market-cache-v3
MARKET_CACHE_MAX=2000
MONGO_URL=mongodb://mongo-crawler:27017
MONGO_DB=crawler_db
USE_FINBERT=0
SAFEALERT_FAST=0
SAFEALERT_DETERMINISTIC=1
```

`USE_FINBERT=1` giúp runtime NLP sát training hơn nhưng tải model lớn hơn.
Nếu không có MongoDB, runtime có fallback NLP zero/VADER để service không chết.

---

## 8. Mô hình được xây dựng như thế nào

File chính:

```text
app/v2/models/safe_alert_net.py
```

Pipeline forward:

1. **Market encoder**
   - Scalar path: 63-dim technical indicators.
   - Bar path: 5 timeframe x 20 bars x 10 features, Eq.16.
   - Hybrid path: scalar + gated residual từ bar encoder.

2. **Horizon embedding**
   - Horizon thuộc `{15m, 1h, 4h, 24h}`.
   - `K_h` theo paper: 3/4/5/8.

3. **Query projection Eq.9**
   - Query dùng `[market_summary; horizon_embedding]`.
   - Không có learned symbol embedding trong paper-final model.
   - `symbol` vẫn được nhận trong API để giữ compatibility.

4. **Article encoder Eq.8**
   - FinBERT embedding 768-dim + metadata 14-dim.
   - Metadata gồm recency/source/length/novelty + entity sentiment.

5. **Selective attention Eq.10-13**
   - Chọn top-K article theo horizon.
   - Forward giữ hard top-K mask; backward dùng straight-through estimator để selector học được.

6. **Factor module Eq.14-15**
   - 10 factor classes.
   - Không còn factor dropout.
   - `Lfac` supervise bằng `article_factor_labels.npy`.

7. **Market-news-factor fusion Eq.20-22**
   - Cross attention giữa news/factor/market.
   - `z_fac` đi vào forward fusion đúng Eq.22.
   - Gradient từ direction loss về factor module được giảm nhẹ để `Lfac` vẫn là supervision chính.

8. **Prediction heads Eq.23-25**
   - `dir_head`: DOWN/NEUTRAL/UP.
   - `ret_head`: return regression.
   - `conf_head`: confidence sigmoid.
   - Optional heads (`vol`, `ret_bin`, `up_edge`, `down_edge`) tồn tại để tương thích/logging nhưng paper-final config đang tắt loss tương ứng.

9. **Alert policy Eq.26**
   - Alert khi confidence vượt `tau` và max probability vượt `gamma`.
   - `tau/gamma/temperature` fit trên validation, frozen khi test.

---

## 9. Loss và training

Loss chính nằm trong:

```text
app/v2/pipelines/safe_alert_training_utils.py
```

Các thành phần:

```text
Ldir    Direction CE
Lret    SmoothL1 return regression, train-fold robust scaling
Lfac    Factor soft CE
Lsel    Selective attention/cardinality loss
Lcal    Calibration/Brier-style confidence loss
Lfaith  Faithfulness loss, full vs masked prediction
Lrisk   Risk/coverage regularization, target coverage kappa=0.35
```

Training dùng curriculum 3 stage:

- Stage 1: direction/return warmup.
- Stage 2: ramp auxiliary paper losses.
- Stage 3: full objective.

Walk-forward protocol:

```text
train expanding window -> embargo -> validation -> embargo -> test
```

Mỗi fold fit scaler/policy trên train/val tương ứng, không dùng test để tune.

---

## 10. Ý nghĩa các file chính

### API/runtime

- `app/main.py`: FastAPI app, startup/shutdown, scheduler, endpoints.
- `app/market_cache.py`: Kafka candle consumer + in-memory candle cache.
- `app/v2/pipelines/live_infer.py`: adapter runtime từ market/news sang SAFEAlertNet.
- `app/v2/alerts/alert_decider.py`: quyết định alert Eq.26.

### Model/training

- `app/v2/models/safe_alert_net.py`: architecture SAFE-Alert.
- `app/v2/pipelines/train_safe_alert.py`: train loop, walk-forward, checkpoint, policy.
- `app/v2/pipelines/safe_alert_dataset.py`: build sample, labels, article window, quality filter.
- `app/v2/pipelines/safe_alert_training_utils.py`: loss, sampler, curriculum utilities.
- `app/v2/pipelines/metrics_safe_alert.py`: metrics, policy search, calibration, mini-backtest.
- `app/v2/pipelines/utils.py`: artifact/policy helpers.

### Precompute

- `gen_emb.py`: tạo article embeddings.
- `precompute_market_features.py`: tạo scalar features 63-dim.
- `precompute_market_bars.py`: tạo Eq.16 bar sequences.
- `precompute_novelty.py`: tạo TF-IDF novelty.
- `precompute_factor_labels.py`: tạo factor pseudo labels.
- `precompute_entity_sentiment.py`: tạo target-based FSA.

### Evaluation/export

- `run_ablation.py`: ablation study.
- `run_baselines.py`: baseline comparison.
- `backtest_safe_alert.py`: backtest checkpoint + policy.
- `eval_faithfulness_sec4p4p2.py`: faithfulness metrics.
- `case_study_export.py`: JSON case studies cho thesis.
- `prediction_logs.py`: per-sample prediction logs.
- `explanation_generator.py`: deterministic natural-language explanation.

### Docs/config/tests

- `train_config_research_best.yaml`: config mặc định cho run chính.
- `train_config_paper_strict.yaml`: config paper-literal comparison.
- `DEVIATIONS.md`: mapping code vs paper, giải thích deviation.
- `ERRATUM.md`: các điểm paper/thesis cần ghi chú.
- `tests/`: unit/regression tests cho model, loss, dataset, scheduler.

---

## 11. Quy trình build kết quả từ đầu

Nếu bắt đầu từ raw candles + articles:

```powershell
# 1. Cài dependencies
pip install -r requirements.txt

# 2. Tạo embeddings và precompute features
python app/v2/pipelines/gen_emb.py
python app/v2/pipelines/precompute_market_features.py
python app/v2/pipelines/precompute_market_bars.py --data-dir training_data/v2 --symbol BTCUSDT --decision-horizon 1h --out training_data/v2/market_bars.npz
python app/v2/pipelines/precompute_novelty.py --articles-csv training_data/v2/articles_max.csv --out training_data/v2/article_novelty.npy
python app/v2/pipelines/precompute_factor_labels.py --articles_path training_data/v2/articles_max.csv --output_dir training_data/v2 --method keyword
python app/v2/pipelines/precompute_entity_sentiment.py --articles_path training_data/v2/articles_max.csv --output_dir training_data/v2

# 3. Train
python app/v2/pipelines/train_safe_alert.py --config app/v2/pipelines/train_config_research_best.yaml --artifact_dir artifacts/research_best_1h

# 4. Kiểm tra kết quả
Get-Content artifacts/research_best_1h/walk_forward_results.json
Get-Content artifacts/research_best_1h/table3_validation_report.json

# 5. Backtest
python app/v2/pipelines/backtest_safe_alert.py --artifact_dir artifacts/research_best_1h --data_dir training_data/v2 --symbol BTCUSDT --horizon 1h
```

Trên Linux/Kaggle, đổi PowerShell line continuation `` ` `` thành `\`.

---

## 12. Ghi chú paper fidelity

Code hiện tại ưu tiên đủ paper hơn là chỉ tối ưu score một fold. Một số điểm cần
nhớ khi đọc log:

- Coverage target `kappa = 0.35`.
- `Lvol` tắt vì paper không định nghĩa volatility loss.
- `RetBin/EdgeBin` optional và mặc định tắt; log sẽ báo disabled khi lambda bằng 0.
- `factor_dropout` đã bỏ vì không thuộc paper.
- Symbol embedding đã bỏ; Eq.9 query đi qua market summary.
- `DEVIATIONS.md` là file cần đọc khi so sánh từng Eq. với implementation.

---

## 13. Troubleshooting

### `market_bars.npz` missing

Train vẫn có thể fallback scalar/hybrid degraded, nhưng để đủ Eq.16 nên chạy:

```powershell
python app/v2/pipelines/precompute_market_bars.py --data-dir training_data/v2 --symbol BTCUSDT --decision-horizon 1h --out training_data/v2/market_bars.npz
```

### `features_precomputed.npy` missing

Train sẽ chậm hơn nhiều. Chạy:

```powershell
python app/v2/pipelines/precompute_market_features.py
```

### `article_factor_labels.npy` missing

`Lfac` sẽ không có pseudo-label precomputed tốt. Chạy:

```powershell
python app/v2/pipelines/precompute_factor_labels.py --articles_path training_data/v2/articles_max.csv --output_dir training_data/v2 --method keyword
```

### PyTorch/pytest thiếu dependency

Kiểm tra lại environment:

```powershell
python -m pip install -r requirements.txt
python -m pytest tests -q
```

### Muốn train nhanh nhưng không cần reproducibility tuyệt đối

```powershell
$env:SAFEALERT_FAST="1"
python app/v2/pipelines/train_safe_alert.py
```

Mặc định nên giữ deterministic khi cần bảo vệ kết quả.

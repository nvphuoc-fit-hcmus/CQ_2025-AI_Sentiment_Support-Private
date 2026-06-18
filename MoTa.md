# Mô tả paper SAFE-Alert theo hướng dễ review

Tài liệu này mô tả lại paper SAFE-Alert theo cách dễ hiểu, để người có domain knowledge về finance, NLP và trading review ý tưởng trước khi viết manuscript chính thức cho Expert Systems with Applications (ESWA).

Tinh thần của tài liệu là **happy path nếu thực nghiệm thành công**: chúng ta mô tả câu chuyện khoa học mong muốn, các contribution, quy trình thực nghiệm, và những ví dụ cụ thể. Các con số cuối cùng vẫn cần được chốt bằng một lần chạy canonical run có config, seed, data hash và artifact rõ ràng.

---

## 1. Ý tưởng ngắn gọn của paper

SAFE-Alert không chỉ hỏi:

> "Giá BTC sẽ tăng hay giảm?"

mà hỏi:

> "Có nên phát cảnh báo giao dịch lúc này không, bằng chứng nào đang ủng hộ, yếu tố tài chính nào chi phối, và model có đủ tự tin để hành động không?"

Đây là sự khác biệt quan trọng. Nhiều model dự báo tài chính chỉ đưa ra prediction mỗi candle, ví dụ `UP`, `DOWN`, hoặc `NEUTRAL`. Trong thực tế trading, điều quan trọng hơn là:

- Lúc nào không nên hành động.
- Tin nào thật sự liên quan đến tài sản.
- Tin đó tác động qua kênh tài chính nào.
- Model có đang quá tự tin vào một tín hiệu yếu hay không.
- Explanation có phản ánh đúng quá trình suy luận của model hay chỉ là câu văn nghe hợp lý.

SAFE-Alert vì vậy được đặt như một hệ thống **news-to-alert**, không phải chỉ là **news-to-price prediction**.

---

## 2. Ví dụ trực quan đầu tiên

Giả sử trong 1 giờ gần nhất, hệ thống thu được 6 tin về BTC:

| Tin | Nội dung rút gọn | Nhận xét domain |
|---|---|---|
| n1 | Bitcoin spot ETF ghi nhận dòng tiền vào 900 triệu USD | Rất liên quan, có thể bullish |
| n2 | Một sàn giao dịch tạm dừng rút tiền do bảo trì | Liên quan, có thể gây risk-off |
| n3 | Fed phát tín hiệu giữ lãi suất cao lâu hơn | Macro risk, có thể bearish |
| n4 | Bài blog nói Bitcoin là tương lai của tiền tệ | Chung chung, ít tín hiệu ngắn hạn |
| n5 | Whale chuyển BTC lên exchange | Có thể bearish nếu gắn với bán ra |
| n6 | Tin cũ về halving được đăng lại | Có thể bị trùng lặp, novelty thấp |

Model all-news thông thường có thể average tất cả tin này thành một sentiment score. Vấn đề là tin n4 và n6 có thể gây nhiễu, tin n2 và n5 có tác động khác nhau, tin n1 và n3 tác động qua hai cơ chế tài chính trái ngược nhau.

SAFE-Alert làm khác:

1. Chọn Top-K tin quan trọng nhất theo horizon, ví dụ `n1`, `n3`, `n5`.
2. Gán các tin này vào factor:
   - `n1` -> `etf_flow`
   - `n3` -> `macro_uncertainty`
   - `n5` -> `whale_accumulation` hoặc `exchange_risk`
3. Kết hợp với market context:
   - RSI, MACD, volatility, volume spike, multi-timescale momentum.
4. Sinh dự báo:
   - Direction: `UP`
   - Expected return: `+0.8%`
   - Confidence: `0.78`
5. Áp dụng alert gate:
   - Nếu confidence và max probability đều vượt ngưỡng -> phát cảnh báo.
   - Nếu prediction có vẻ đúng nhưng confidence thấp -> abstain.
6. Sinh explanation:
   - "Tín hiệu UP được hỗ trợ bởi dòng tiền ETF mạnh, nhưng rủi ro macro vẫn tồn tại; model chỉ phát alert vì market momentum và volume đang cùng pha."

Mục tiêu của paper là chứng minh cách làm này tốt hơn việc đọc tất cả tin như nhau hoặc chỉ đưa ra sentiment score phẳng.

---

## 3. Bài toán nghiên cứu

### 3.1. Input

Tại mỗi thời điểm candle `t`, hệ thống có:

- Dữ liệu thị trường OHLCV đến trước hoặc tại thời điểm `t`.
- Tập tin tức đã xuất bản trước `t`.
- Metadata của tin:
  - thời gian đăng,
  - nguồn tin,
  - độ dài,
  - độ mới,
  - embedding FinBERT,
  - factor label,
  - target-based sentiment theo factor.

### 3.2. Output

Model sinh ra:

- Direction: `DOWN`, `NEUTRAL`, `UP`.
- Expected return.
- Confidence.
- Alert decision: `ALERT` hoặc `ABSTAIN`.
- Selected news.
- Top financial factors.
- Natural-language explanation.

### 3.3. Nhãn dự báo

Nhãn dự báo được tạo theo horizon:

- `15m`: nhìn trước 1 bước 15 phút.
- `1h`: nhìn trước 1 bước 1 giờ.
- `4h`: nhìn trước 4 candle 1h.
- `24h`: nhìn trước 24 candle 1h.

Để tránh look-ahead, giá thực thi nên là open của candle tiếp theo. Ví dụ:

```text
Tại candle t:
  model chỉ thấy dữ liệu trước/tại t
  nếu có alert, lệnh được giả định vào tại open(t+1)
  lợi nhuận được tính từ open(t+1) đến close(t+h)
```

Đây là điểm quan trọng với finance reviewer, vì nếu dùng close của candle hiện tại hoặc tin tức cùng timestamp không rõ ràng, kết quả có thể bị nghi look-ahead bias.

---

## 4. Cách tiếp cận SAFE-Alert

SAFE-Alert gồm 5 ý tưởng chính.

### 4.1. Selective news: không phải tin nào cũng quan trọng như nhau

Với mỗi horizon, model chọn số bài khác nhau:

| Horizon | Số bài nên chọn | Lý do |
|---|---:|---|
| 15m | 3 | Chỉ tin rất mới và rất mạnh mới tác động kịp |
| 1h | 4 | Tin nóng và phản ứng ngắn hạn |
| 4h | 5 | Cần thêm context rộng hơn |
| 24h | 8 | Tin macro và narrative dài hơn |

Ví dụ:

```text
Nếu horizon = 1h:
  Chọn 4 tin quan trọng nhất.

Nếu horizon = 24h:
  Chọn 8 tin vì tin macro, regulation, ETF flow có thể tác động chậm hơn.
```

Happy path mong muốn:

- So với all-news average, selective news giảm nhiễu.
- Explanation ngắn hơn, đúng bằng chứng hơn.
- Khi xóa các tin được chọn, prediction của model thay đổi rõ hơn so với xóa tin random.

### 4.2. Factor-grounded reasoning: giải thích bằng yếu tố tài chính

Thay vì nói:

> "Sentiment tích cực nên BTC tăng."

SAFE-Alert nói:

> "Tín hiệu tăng được hỗ trợ bởi `etf_flow` và `institutional_inflow`; rủi ro chính là `macro_uncertainty`."

Ontology hiện tại gồm 10 factor:

| Factor | Ý nghĩa | Ví dụ tin |
|---|---|---|
| `institutional_inflow` | Dòng tiền tổ chức | MicroStrategy mua BTC |
| `etf_flow` | Dòng tiền ETF | Bitcoin ETF inflow/outflow |
| `regulatory_easing` | Pháp lý nới lỏng | SEC phê duyệt sản phẩm mới |
| `regulatory_tightening` | Pháp lý siết chặt | Kiện tụng, cấm, enforcement |
| `exchange_risk` | Rủi ro sàn giao dịch | Hack, halt withdrawal |
| `liquidity_squeeze` | Squeeze, liquidation, leverage | Funding rate quá nóng |
| `whale_accumulation` | Cá voi, on-chain, holder | Whale tích lũy hoặc nạp sàn |
| `macro_uncertainty` | Fed, CPI, recession, USD | Fed hawkish |
| `protocol_upgrade` | Nâng cấp giao thức | Halving, fork, mainnet |
| `network_outage` | Sự cố mạng | Congestion, outage, fee spike |

Happy path mong muốn:

- Factor giúp reviewer finance thấy explanation có ý nghĩa kinh tế.
- Factor consistency cao hơn khi so với pseudo-label/human-label.
- Bỏ factor module thì F1 có thể giảm nhẹ, nhưng explanation và faithfulness giảm rõ.

### 4.3. Multi-timescale market context: tin tức phải được đặt vào thị trường

Một tin tốt không luôn tạo tín hiệu mua.

Ví dụ:

```text
Tin: ETF inflow mạnh.
Market context A:
  BTC vượt kháng cự, volume tăng, funding bình thường.
  -> SAFE-Alert có thể phát UP alert.

Market context B:
  BTC đang dump mạnh, volatility tăng, funding quá nóng, liquidation cascade.
  -> SAFE-Alert có thể abstain hoặc dự báo rủi ro cao.
```

Do đó SAFE-Alert không chỉ dùng news embedding. Model kết hợp:

- technical indicators,
- momentum,
- volatility,
- volume,
- multi-timescale market encoder,
- cross-attention giữa news/factor và market context.

Happy path mong muốn:

- Market-only baseline đã mạnh, nhưng SAFE-Alert vượt market-only trong những giai đoạn tin tức có tác động.
- News-only hoặc all-news model kém ổn định hơn vì không biết trạng thái thị trường.

### 4.4. Confidence-aware alerting: dự báo khác với hành động

Model có thể dự báo `UP`, nhưng không phát alert nếu độ tin cậy thấp.

Điều kiện alert:

```text
alert = 1 nếu:
  confidence >= tau
  và max(direction_probability) >= gamma
```

Ví dụ:

| Prediction | Confidence | Max prob | Kết quả |
|---|---:|---:|---|
| UP | 0.82 | 0.76 | ALERT |
| UP | 0.48 | 0.74 | ABSTAIN |
| DOWN | 0.80 | 0.51 | ABSTAIN |
| NEUTRAL | 0.88 | 0.70 | Thường không nên trade |

Happy path mong muốn:

- Always-alert có coverage cao nhưng precision thấp.
- SAFE-Alert coverage vừa phải nhưng Alert Precision và Sharpe tốt hơn.
- Reviewer thấy đây là decision-support system có kiểm soát risk, không chỉ là classifier.

### 4.5. Faithfulness: explanation phải có tác động thật

Một explanation nghe hay chưa chắc faithful. SAFE-Alert cần được test bằng các phép:

| Test | Cách hiểu đơn giản | Happy path |
|---|---|---|
| Deletion / Comprehensiveness | Xóa tin model nói là quan trọng | Confidence giảm rõ |
| Insertion | Chỉ đưa tin được chọn vào từ context rỗng | Prediction tốt lên |
| Sufficiency | Giữ lại tin được chọn, bỏ tin còn lại | Chất lượng không giảm nhiều |
| Factor consistency | Top factor có khớp label/reference không | Jaccard/overlap cao |

Ví dụ:

```text
Full input:
  p(UP) = 0.74

Xóa 2 tin selected:
  p(UP) = 0.41

Giữ chỉ 2 tin selected:
  p(UP) = 0.70

Diễn giải:
  Hai tin selected thật sự là bằng chứng chính.
```

Nếu xóa tin selected mà prediction không đổi, explanation có nguy cơ chỉ là trang trí.

---

## 5. Contribution dự kiến của paper

### Contribution 1: Chuyển từ sentiment forecasting sang evidence-based alerting

Nhiều paper lấy sentiment score rồi đưa vào LSTM/Transformer để dự báo giá. SAFE-Alert đặt vấn đề khác:

```text
Không chỉ "sentiment là positive hay negative?"
Mà là:
  Tin nào đang tác động?
  Tác động qua factor nào?
  Thị trường có ủng hộ tác động đó không?
  Có đủ tự tin để phát alert không?
```

Contribution này cần được viết rõ để tránh bị reviewer xem là "thêm attention vào model".

### Contribution 2: Kiến trúc hợp nhất prediction, explanation, calibration và alert policy

SAFE-Alert không tách riêng:

- model dự báo,
- model giải thích,
- threshold alert,
- faithfulness check.

Tất cả nằm trong một pipeline:

```text
News + Market
  -> Select evidence
  -> Infer financial factors
  -> Fuse with market context
  -> Predict direction/return/confidence
  -> Decide alert/abstain
  -> Explain using selected evidence and factors
```

Đây là điểm hợp với ESWA: hệ thống hỗ trợ quyết định thông minh, có thể triển khai, không chỉ là benchmark model.

### Contribution 3: Protocol pseudo-online cho trading decision

Mỗi fold:

```text
Train window -> embargo -> validation window -> embargo -> test window
```

Threshold `tau`, `gamma`, temperature scaling phải được tune trên validation, sau đó freeze khi test. Nếu làm đúng, reviewer sẽ tin hơn vì policy không được chọn bằng test set.

### Contribution 4: Đánh giá đa chiều

Paper không chỉ báo cáo Accuracy:

- Forecast:
  - Accuracy,
  - Macro-F1,
  - MCC,
  - AUC.
- Calibration:
  - ECE,
  - Brier score.
- Alert:
  - Alert Precision,
  - Coverage,
  - Selective Risk.
- Trading utility:
  - Sharpe,
  - Sortino,
  - Calmar,
  - Max Drawdown,
  - PnL sau cost.
- Explanation:
  - deletion,
  - insertion,
  - sufficiency,
  - factor consistency.

Happy path là SAFE-Alert không nhất thiết thắng tất cả metric, nhưng thắng theo câu chuyện hợp lý:

```text
Market-only có thể khá mạnh về F1.
All-news có thể thêm signal nhưng nhiễu.
SAFE-Alert thắng rõ về alert precision, faithfulness, drawdown, và coverage-risk trade-off.
```

---

## 6. Tại sao có khả năng phù hợp Expert Systems with Applications

### 6.1. Fit với scope của ESWA

ESWA ưu tiên các hệ thống intelligent/expert system có ứng dụng rõ trong industry, finance, stock trading, risk assessment, data mining và text mining. SAFE-Alert có các điểm fit:

- Bài toán finance/trading rõ ràng.
- Có NLP, ML, time-series, explanation, decision support.
- Có hệ thống end-to-end gắn với microservices, Kafka, crawler, AI service, frontend.
- Có protocol đánh giá gần với sử dụng thực tế.

### 6.2. Không nên claim quá đà

Không nên nói:

> "Chúng tôi phát minh Transformer mới."

Nên nói:

> "Chúng tôi đề xuất một framework decision-support hợp nhất selective evidence, factor-grounded explanation, calibrated alerting và pseudo-online evaluation cho financial news-to-alert."

Đây là claim hợp lý hơn và khó bị bắt bẻ hơn.

### 6.3. Reviewer ESWA có thể thích gì

Reviewer ESWA thường sẽ quan tâm:

- Hệ thống có giải quyết bài toán ứng dụng thật không?
- Có methodology rõ và reproducible không?
- Có baseline đủ mạnh không?
- Có ablation tách contribution không?
- Có practical implication không?
- Có nói rõ limitation không?

SAFE-Alert có cơ hội nếu paper chứng minh được:

```text
1. Selective news thật sự giảm noise.
2. Factor explanation thật sự faithful.
3. Alert gate thật sự cải thiện precision/utility.
4. Protocol không leakage.
5. System có thể chạy gần real-time.
```

---

## 7. Happy-path experimental story

Nếu mọi thứ thành công, câu chuyện kết quả nên như sau:

### 7.1. Forecasting

SAFE-Alert đạt Macro-F1 và MCC cao hơn:

- market-only,
- all-news fusion,
- sentiment+market,
- soft-attention news,
- price momentum,
- selective forecasting baseline.

Diễn giải mong muốn:

```text
Market features là signal nền tảng.
Tin tức chỉ có ích khi được chọn lọc.
All-news đưa thêm nhiều nhiễu vì tin không liên quan.
SAFE-Alert giữ lại tin có tác động, nên F1/MCC tốt hơn.
```

### 7.2. Alerting

SAFE-Alert có coverage gần target, ví dụ 30-40%, nhưng Alert Precision cao hơn Always-Alert.

Diễn giải:

```text
Trong trading, không phát signal cũng là một quyết định.
Model tốt không cần trade mỗi candle.
Alert gate giúp bỏ qua thời điểm bất định cao.
```

### 7.3. Utility

SAFE-Alert có Sharpe/Sortino tốt hơn và Max Drawdown thấp hơn baselines.

Diễn giải:

```text
F1 cao chưa chắc trading tốt.
SAFE-Alert tối ưu hơn ở tầng decision: chỉ vào lệnh khi đủ tin cậy.
Do đó có thể có ít lệnh hơn nhưng chất lượng lệnh tốt hơn.
```

### 7.4. Explanation

SAFE-Alert có deletion/insertion/sufficiency tốt hơn:

```text
Khi xóa selected news, confidence giảm.
Khi giữ selected news, model vẫn giữ được phần lớn signal.
Factor top-3 khớp với annotation/reference tốt hơn random.
```

Diễn giải:

```text
Explanation không chỉ là generated text.
Bằng chứng được dùng trực tiếp trong forward pass.
```

### 7.5. Ablation

Bảng ablation happy path:

| Biến thể | Kết quả mong đợi | Ý nghĩa |
|---|---|---|
| Full SAFE-Alert | Tốt nhất tổng thể | Tất cả module bổ sung nhau |
| w/o selective news | F1 và faithfulness giảm | Chọn lọc tin có ích |
| w/o factor | Explanation/factor consistency giảm | Factor layer có ý nghĩa |
| w/o market | F1/Sharpe giảm mạnh | Market context là nền tảng |
| w/o confidence | Coverage/precision xấu hơn | Alert gate cần confidence |
| w/o faithfulness | Explanation kém faithful | Loss faithfulness có tác dụng |
| w/o Lrisk | Coverage lệch target | Selective risk giúp policy ổn định |
| w/o horizon | Multi-horizon generalization kém hơn | Horizon conditioning cần thiết |

---

## 8. Quy trình thực nghiệm từng bước

Phần này viết theo hướng để người khác làm lại. Command có thể cần điều chỉnh theo máy GPU và đường dẫn data.

### Bước 0: Chốt canonical run

Trước khi viết paper, cần chốt một run chính thức:

```text
paper_artifacts/
  safe_alert_btcusdt_1h_run_YYYYMMDD/
    config.json
    git_commit.txt
    data_hashes.json
    walk_forward_results.json
    baseline_results.json
    ablation_results.json
    faithfulness_results.json
    tables/
    figures/
```

Lý do: hiện repo có nhiều artifact khác nhau. Một artifact nói dataset 70,260 samples, artifact khác nói 14,187 samples. Paper không nên trộn hai nguồn này. Cần chốt một canonical run và chỉ dùng nó cho bảng/figure chính.

### Bước 1: Chuẩn bị dữ liệu OHLCV và news

Input cần có:

- OHLCV theo symbol/horizon, ví dụ BTCUSDT 1h.
- Tin tức crypto/finance có timestamp.
- Source, title, content.

Ví dụ record OHLCV:

```csv
timestamp,open,high,low,close,volume
2024-01-15 10:00:00,42000,42400,41800,42350,1234.5
```

Ví dụ record news:

```csv
timestamp,source,title,content
2024-01-15 09:35:00,CoinDesk,"Bitcoin ETF sees strong inflow","..."
```

Yêu cầu quan trọng:

- Timestamp phải cùng timezone hoặc được normalize.
- Tin tức tại thời điểm `t` chỉ được dùng nếu published before candle decision time.
- Duplicate news nên được loại hoặc đánh dấu novelty.

### Bước 2: Tạo market features

Chạy precompute market features:

```bash
cd services/ai-service
python app/v2/pipelines/precompute_market_features.py
```

Output mong đợi:

```text
training_data/v2/features_precomputed.npy
training_data/v2/features_precomputed.meta.json
```

Feature gồm:

- 5 timescale groups x 12 indicators.
- 3 cross-timeframe features.
- Tổng 63 features.

Ví dụ cách hiểu:

```text
Scale ngắn hạn:
  RSI ngắn, MACD ngắn, volatility ngắn.

Scale dài hơn:
  RSI dài, momentum dài, volatility dài.

Cross-scale:
  Thị trường có đang đồng pha giữa ngắn hạn và dài hạn không?
```

### Bước 3: Tạo FinBERT embeddings cho article

Chạy embedding pipeline:

```bash
cd services/ai-service
python app/v2/pipelines/gen_emb.py
```

Output mong đợi:

```text
training_data/v2/btcusdt_article_embeddings_max.npy
```

Mỗi article có vector 768 chiều.

Ví dụ:

```text
"Bitcoin ETF inflows surge..."
  -> FinBERT embedding 768-dim
```

Embedding này không phải sentiment score duy nhất. Nó là biểu diễn ngữ nghĩa của bài viết, sau đó SAFE-Alert mới học chọn lọc và hợp nhất.

### Bước 4: Tạo factor labels và target-based sentiment

Tạo factor labels:

```bash
cd services/ai-service
python app/v2/pipelines/precompute_factor_labels.py \
  --method keyword
```

Nếu có API và muốn artifact mạnh hơn cho paper:

```bash
python app/v2/pipelines/precompute_factor_labels.py \
  --method gemini
```

Tạo entity sentiment:

```bash
python app/v2/pipelines/precompute_entity_sentiment.py
```

Output mong đợi:

```text
article_factor_labels.npy
article_entity_sentiment.npy
```

Ví dụ:

```text
Title: "Spot Bitcoin ETF inflows hit record high"

Factor distribution:
  etf_flow: 0.72
  institutional_inflow: 0.18
  macro_uncertainty: 0.02
  others: small

Entity sentiment:
  etf_flow: +0.65
  institutional_inflow: +0.40
  macro_uncertainty: 0.00
```

Ghi chú quan trọng cho paper:

- Nếu dùng keyword labels, nên nói là pseudo-label.
- Nếu dùng LLM labels, cần có audit/human validation.
- Nếu claim factor reasoning mạnh, nên có ít nhất một mẫu manual review.

### Bước 5: Build dataset theo candle-time

SAFEAlertDataset sẽ tạo sample theo candle:

```text
Sample tại candle t:
  market_features(t)
  articles in [t - lookback, t)
  direction label from future horizon
  return label from execution price
```

Ví dụ:

```text
Candle t = 10:00
Lookback = 24h

Tin được phép:
  09:35, 08:10, hôm qua 16:00

Tin không được phép:
  10:00 nếu timestamp bằng candle và không rõ thứ tự công bố
  10:05 vì là tương lai
```

Đây là bước reviewer finance sẽ soi kỹ.

### Bước 6: Train SAFE-Alert walk-forward

Command chính:

```bash
cd services/ai-service
python app/v2/pipelines/train_safe_alert.py \
  --walk_forward \
  --symbol BTCUSDT \
  --horizon 1h \
  --epochs 60 \
  2>&1 | tee training_log.txt
```

Protocol:

```text
Fold 1:
  train early period
  validation next period
  test next period

Fold 2:
  train expands
  validation slides forward
  test slides forward
```

Có embargo:

```text
train -> 24 candles embargo -> validation -> 24 candles embargo -> test
```

Happy path:

- F1 tăng dần khi train window lớn hơn.
- Metrics không collapse ở fold mới.
- Coverage gần target.

### Bước 7: Tune threshold trên validation, freeze trên test

Trong mỗi fold:

```text
1. Train model trên train.
2. Dùng validation để chọn:
   - tau: confidence threshold
   - gamma: direction probability threshold
   - temperature: calibration scalar
3. Freeze tau/gamma/temperature.
4. Evaluate trên test.
```

Ví dụ:

```json
{
  "tau": 0.504,
  "gamma": 0.547,
  "temperature": 1.8,
  "policy_source": "validation_grid_search"
}
```

Điều cần tránh:

```text
Không chọn threshold trên test set.
Không report policy được tune sau khi nhìn test.
```

### Bước 8: Chạy baseline comparison

Command:

```bash
cd services/ai-service
python app/v2/pipelines/run_baselines.py \
  --symbol BTCUSDT \
  --horizon 1h \
  --epochs 30
```

Output mong đợi:

```text
artifacts/colab_run/baseline_results.json
```

Baseline cần có:

| Baseline | Câu hỏi nó trả lời |
|---|---|
| market_only | Chỉ market có đủ không? |
| all_news_fusion | Dùng tất cả tin có tốt không? |
| sentiment_market | Sentiment scalar có đủ không? |
| nsm_style | Soft attention all-news có đủ không? |
| llm_factor | Factor static có đủ không? |
| sep_style | Selective prediction post-hoc có đủ không? |
| temperature_scaled | Calibration post-hoc có đủ không? |
| always_alert | Nếu lúc nào cũng alert thì sao? |
| current_price_predictor | Momentum đơn giản có cạnh tranh được không? |

Happy path:

```text
SAFE-Alert không chỉ thắng trivial baseline,
mà phải thắng hoặc có trade-off tốt hơn các baseline gần với literature.
```

### Bước 9: Chạy ablation study

Command:

```bash
cd services/ai-service
python app/v2/pipelines/run_ablation.py \
  --symbol BTCUSDT \
  --horizon 1h \
  --epochs 30
```

Output mong đợi:

```text
artifacts/colab_run/ablation_results.json
```

Mục tiêu:

- Chứng minh mỗi module có vai trò.
- Nếu một module không ảnh hưởng gì, cần xem lại claim.

Ví dụ diễn giải:

```text
w/o selective news:
  F1 giảm nhẹ, faithfulness giảm mạnh.

w/o confidence:
  Coverage tăng nhưng Alert Precision giảm.

w/o market:
  F1 và Sharpe giảm mạnh.
```

### Bước 10: Chạy faithfulness evaluation

Command:

```bash
cd services/ai-service
python app/v2/pipelines/eval_faithfulness_sec4p4p2.py
```

Output mong đợi:

```text
artifacts/colab_run/faithfulness_results.json
```

Cần có random baseline:

```text
Selected evidence của SAFE-Alert
vs
Random K articles trong cùng window
vs
Most recent K articles
vs
Highest source credibility K articles
```

Happy path:

```text
SAFE selected articles có deletion drop cao hơn random.
SAFE selected articles có sufficiency drop thấp hơn random.
Top factors ổn định hơn khi perturb article window.
```

### Bước 11: Tổng hợp table và figure cho paper

Table để đưa vào paper:

1. Dataset statistics:
   - số candles,
   - số articles,
   - time range,
   - symbols,
   - horizons,
   - class distribution.
2. Main forecast result:
   - Accuracy,
   - Macro-F1,
   - MCC,
   - AUC.
3. Alert result:
   - Precision,
   - Coverage,
   - Sharpe,
   - Sortino,
   - MDD.
4. Ablation:
   - full vs w/o modules.
5. Faithfulness:
   - deletion,
   - insertion,
   - sufficiency,
   - factor consistency.

Figure để đưa vào paper:

1. SAFE-Alert architecture.
2. Walk-forward split diagram.
3. Coverage-risk curve.
4. Calibration reliability diagram.
5. Faithfulness deletion/insertion bar chart.
6. Case study timeline: news -> factors -> alert.

---

## 9. Nhiều ví dụ minh họa để domain expert review

### Ví dụ 1: ETF inflow bullish

Input:

```text
News:
  "US spot Bitcoin ETFs record $1.2B net inflow"

Market:
  BTC trên EMA 20 và EMA 50
  volume tăng
  volatility vừa phải
```

Expected SAFE-Alert behavior:

```text
Selected news:
  ETF inflow article

Top factor:
  etf_flow
  institutional_inflow

Prediction:
  UP

Alert:
  ALERT nếu confidence cao

Explanation:
  Dòng tiền ETF mạnh kết hợp với momentum thị trường ủng hộ khả năng tăng ngắn hạn.
```

Domain expert can review:

- ETF inflow có nên bullish trong horizon 1h không?
- Có cần lag effect 4h/24h thay vì 1h không?
- Nếu ETF inflow đã được market price-in thì có nên alert không?

### Ví dụ 2: Exchange hack bearish

Input:

```text
News:
  "Major exchange suspends withdrawals after suspected exploit"

Market:
  BTC volume bán tăng
  spread/volatility tăng
```

Expected behavior:

```text
Factor:
  exchange_risk

Prediction:
  DOWN

Alert:
  ALERT nếu market cùng xác nhận risk-off.
  ABSTAIN nếu tin chỉ ảnh hưởng altcoin nhỏ và BTC market không phản ứng.
```

Good explanation:

```text
Tin tức về rủi ro sàn giao dịch làm tăng khả năng bán tháo ngắn hạn; tín hiệu được xác nhận bởi volume và volatility tăng.
```

Bad explanation:

```text
Sentiment negative nên giá giảm.
```

Lý do bad: quá chung chung, không nói cơ chế tài chính.

### Ví dụ 3: Fed hawkish nhưng BTC không giảm ngay

Input:

```text
News:
  "Fed signals higher-for-longer interest rates"

Market:
  BTC đang sideway
  volume thấp
  không có breakdown
```

Expected behavior:

```text
Factor:
  macro_uncertainty

Prediction:
  DOWN hoặc NEUTRAL

Alert:
  Có thể ABSTAIN vì market chưa confirm.
```

Điểm hay:

```text
SAFE-Alert không nên máy móc short mọi tin macro negative.
Confidence gate giúp tránh overtrade.
```

### Ví dụ 4: Tin tốt nhưng funding quá nóng

Input:

```text
News:
  "Bitcoin ETF inflows remain positive"

Market:
  funding rate rất cao
  long liquidation risk tăng
  RSI ngắn hạn overbought
```

Expected behavior:

```text
Factors:
  etf_flow positive
  liquidity_squeeze risk

Prediction:
  NEUTRAL hoặc abstain

Explanation:
  Tin ETF tích cực nhưng rủi ro squeeze và overbought làm tín hiệu không đủ an toàn để phát alert.
```

Điểm cần review:

- Ontology có cần tách `liquidity_squeeze` thành long/short squeeze không?
- Funding rate có nằm trong market feature chưa?

### Ví dụ 5: Whale nạp BTC lên exchange

Input:

```text
News/on-chain:
  "Whale transfers 8,000 BTC to Binance"

Market:
  price gần resistance
  volume tăng nhưng candle yếu
```

Expected behavior:

```text
Factor:
  whale_accumulation hoặc exchange_risk tùy nội dung

Prediction:
  DOWN nếu context là potential sell pressure

Alert:
  Chỉ alert nếu confidence đủ cao.
```

Domain issue:

```text
Whale transfer không luôn bearish.
Cần phân biệt exchange inflow, exchange outflow, OTC, internal transfer.
```

Đây là điểm domain expert có thể giúp cải thiện ontology.

### Ví dụ 6: Protocol upgrade

Input:

```text
News:
  "Ethereum mainnet upgrade scheduled successfully"

Symbol:
  ETHUSDT
```

Expected behavior:

```text
Factor:
  protocol_upgrade

Prediction:
  UP nếu market support
  ABSTAIN nếu tin đã được price-in
```

Điểm review:

- Với BTC, protocol upgrade hiếm hơn.
- Với ETH/SOL, factor này quan trọng hơn.
- Paper nếu chỉ test BTC thì factor `protocol_upgrade` có thể ít được validate.

### Ví dụ 7: Regulation easing

Input:

```text
News:
  "Court rules in favor of crypto exchange in regulatory dispute"
```

Expected behavior:

```text
Factor:
  regulatory_easing

Prediction:
  UP cho market-wide crypto nếu context tốt
```

Cần review:

- Tin regulation có ảnh hưởng market-wide hay chỉ một token?
- Cần entity linking để biết tin tác động BTC, ETH, XRP hay toàn market.

### Ví dụ 8: Regulation tightening

Input:

```text
News:
  "SEC files lawsuit against major crypto platform"
```

Expected behavior:

```text
Factor:
  regulatory_tightening

Prediction:
  DOWN hoặc ABSTAIN tùy market reaction
```

Good explanation:

```text
Rủi ro pháp lý tăng có thể làm giảm appetite với crypto assets; model phát alert vì đồng thời market momentum đang suy yếu.
```

### Ví dụ 9: Tin lặp lại, novelty thấp

Input:

```text
News 1:
  "Bitcoin ETF inflow hits record"

News 2:
  "BTC ETF sees record inflows"

News 3:
  "Record inflow into spot Bitcoin ETF"
```

Expected behavior:

```text
Model không nên xem đây là 3 bằng chứng độc lập mạnh như nhau.
Novelty/redundancy nên làm giảm trọng số các bản tin trùng lặp.
```

Điểm review:

- Cần duplicate detection tốt hơn không?
- Crawler có loại duplicate theo title/content hash chưa?

### Ví dụ 10: Model đúng direction nhưng trade vẫn lỗ

Input:

```text
Prediction:
  UP đúng direction

Actual:
  Giá tăng nhẹ +0.05%

Transaction cost:
  0.10%
```

Kết quả:

```text
Direction correct nhưng net return âm.
```

Ý nghĩa:

```text
F1 không đủ cho trading.
Cần report net returns sau cost, Sharpe, drawdown.
```

### Ví dụ 11: High precision nhưng Sharpe thấp

Tình huống:

```text
Alert Precision = 80%
Nhưng 20% sai rơi vào những candle biến động lớn.
```

Kết quả:

```text
Precision cao nhưng Sharpe thấp, drawdown lớn.
```

Diễn giải:

```text
Trading utility phụ thuộc magnitude, volatility và tail risk.
Paper cần giải thích nếu có fold như vậy.
```

### Ví dụ 12: Abstain là quyết định đúng

Input:

```text
News:
  Mixed ETF inflow positive, Fed negative, exchange risk unclear

Market:
  sideways, low volume

Prediction:
  UP probability = 0.46
  DOWN probability = 0.38
  NEUTRAL probability = 0.16
  confidence = 0.44
```

Expected behavior:

```text
ABSTAIN.
```

Explanation:

```text
Model nhận thấy có nhiều tin trái chiều và confidence không đạt ngưỡng.
```

Đây là case rất quan trọng cho ESWA vì nó cho thấy hệ thống hỗ trợ quyết định có trách nhiệm.

---

## 10. Những bảng kết quả nên có trong happy path

### Bảng 1: Main forecasting result

| Model | Macro-F1 | MCC | AUC | ECE |
|---|---:|---:|---:|---:|
| Market-only | TBD | TBD | TBD | TBD |
| All-news fusion | TBD | TBD | TBD | TBD |
| Sentiment+market | TBD | TBD | TBD | TBD |
| Soft-attention news | TBD | TBD | TBD | TBD |
| SAFE-Alert | Best/TBD | Best/TBD | Best/TBD | Low/TBD |

### Bảng 2: Alerting and trading utility

| Model | Alert Precision | Coverage | Sharpe | MDD |
|---|---:|---:|---:|---:|
| Always-alert | Low/TBD | 100% | TBD | TBD |
| Selective baseline | TBD | target | TBD | TBD |
| SAFE-Alert | High/TBD | target | High/TBD | Lower/TBD |

### Bảng 3: Ablation

| Variant | Macro-F1 | Alert Precision | Sharpe | Faithfulness |
|---|---:|---:|---:|---:|
| Full SAFE-Alert | Best/TBD | Best/TBD | Best/TBD | Best/TBD |
| w/o selective news | Down/TBD | Down/TBD | Down/TBD | Down/TBD |
| w/o factor | Slight down/TBD | TBD | TBD | Down/TBD |
| w/o confidence | TBD | Down/TBD | Down/TBD | TBD |
| w/o faithfulness | Similar/TBD | TBD | TBD | Down/TBD |

### Bảng 4: Case study qualitative

| Time | Selected news | Factor | Prediction | Alert | Explanation quality |
|---|---|---|---|---|---|
| t1 | ETF inflow | `etf_flow` | UP | ALERT | Good |
| t2 | Exchange hack | `exchange_risk` | DOWN | ALERT | Good |
| t3 | Mixed macro | `macro_uncertainty` | UP | ABSTAIN | Good because uncertain |

---

## 11. Checklist cho domain expert review

### 11.1. Finance/trading review

- Neutral band hiện tại có hợp lý theo horizon không?
- Return target nên dùng close-to-close, open-to-close, hay next-open-to-horizon-close?
- Transaction cost 0.1% có thực tế cho Binance spot/futures không?
- Có cần thêm slippage theo volatility/volume không?
- Short position có hợp lệ với setup sản phẩm không?
- Funding rate, open interest, liquidation có nên thành market features bắt buộc không?
- Sharpe tính trên hourly returns có nên annualize hay report per-trade Sharpe riêng?
- Drawdown tính trên alert returns đã đúng với cash exposure chưa?

### 11.2. NLP/factor review

- 10 factor hiện tại đã đủ cho crypto chưa?
- Có cần thêm factor:
  - stablecoin depeg,
  - miner selling,
  - exchange reserve,
  - derivatives funding,
  - geopolitical risk,
  - risk-on/risk-off equities?
- Keyword labels có quá yếu cho claim factor reasoning không?
- Cần bao nhiêu mẫu human audit để thuyết phục?
- Entity sentiment theo factor có bị sai khi câu có sarcasm, negation, hoặc conditional statement không?

### 11.3. Experimental design review

- Walk-forward split có dài đủ cho từng regime không?
- Có cần test nhiều asset ngoài BTC không?
- Có cần test nhiều horizon không?
- Baseline đã đủ mạnh chưa?
- Có cần statistical significance test không?
- Có cần stress test theo market regime:
  - bull,
  - bear,
  - sideways,
  - crash,
  - ETF approval period?

### 11.4. ESWA claim review

- Claim "strong accept" có nên hạ thành "competitive for ESWA" không?
- Contribution có quá gần với engineering system không, hay đủ research novelty?
- Paper có cần thêm formalization về selective risk và faithfulness không?
- Có cần release code/data subset để tăng reproducibility không?

---

## 12. Những rủi ro cần ghi rõ để tránh bị reviewer bắt

### Rủi ro 1: Artifact không nhất quán

Hiện có artifact/report khác nhau về dataset size và metric. Trước khi submit:

```text
Chỉ dùng một canonical artifact.
Tất cả table/figure phải sinh từ artifact đó.
```

### Rủi ro 2: Factor label heuristic

Nếu dùng keyword label, reviewer có thể nói factor reasoning chỉ là rules. Cách giảm rủi ro:

- Gọi là weak supervision/pseudo-label.
- Có LLM-label variant.
- Có human audit.
- Báo cáo label quality.

### Rủi ro 3: Baseline chưa đủ mạnh

ESWA reviewer sẽ không chấp nhận nếu chỉ so với trivial baseline. Cần có:

- market-only strong baseline,
- all-news fusion,
- sentiment+market,
- attention baseline,
- recent time-series baseline,
- selective prediction baseline.

### Rủi ro 4: Trading result bị overfitting

Cần tránh:

- chọn threshold trên test,
- tune quá nhiều policy rồi lấy policy tốt nhất,
- chỉ report PnL mà không report drawdown,
- không tính cost.

### Rủi ro 5: Explanation nghe hay nhưng không faithful

Cần có:

- deletion,
- insertion,
- sufficiency,
- random evidence baseline,
- case study có perturbation.

---

## 13. Một outline paper để viết tiếp

1. Introduction
   - Vấn đề: financial news forecasting nhiễu, giải thích yếu, alert nhiều.
   - Motivation: trading cần decision support, not prediction every candle.
   - Contributions.

2. Related Work
   - Financial sentiment analysis.
   - Cryptocurrency forecasting with news and technical indicators.
   - Selective prediction and calibration.
   - Explainable AI/faithfulness in finance.

3. Method: SAFE-Alert
   - Problem definition.
   - SelectiveNewsEncoder.
   - FactorModule.
   - MultiTimescaleMarketEncoder.
   - CrossAttentionFusion.
   - Confidence-aware alerting.
   - Explanation and faithfulness loss.

4. Experimental Setup
   - Dataset.
   - Label construction.
   - Walk-forward protocol.
   - Baselines.
   - Metrics.
   - Implementation details.

5. Results
   - Main forecasting results.
   - Alerting/utility results.
   - Ablation.
   - Faithfulness.
   - Case studies.

6. Discussion
   - Why selective evidence matters.
   - Why confidence gate matters.
   - Practical deployment.
   - Limitations.

7. Conclusion
   - SAFE-Alert as explainable, calibrated news-to-alert decision support.

---

## 14. Câu chuyện một câu để giữ paper đúng hướng

Nếu cần tóm tắt paper bằng một câu:

> SAFE-Alert là một framework hỗ trợ quyết định tài chính gần thời gian thực, biến dòng tin tức dày đặc thành các cảnh báo có chọn lọc, có giải thích theo yếu tố tài chính, và có kiểm soát độ tin cậy trước khi hành động.

Nếu cần tóm tắt novelty bằng một câu:

> Điểm mới không nằm ở một encoder riêng lẻ, mà nằm ở việc kết hợp selective evidence, factor-grounded explanation, calibrated abstention và pseudo-online trading evaluation trong cùng một hệ thống có thể triển khai.

Nếu cần tóm tắt happy path bằng một câu:

> Khi thực nghiệm thành công, SAFE-Alert sẽ không chỉ dự báo đúng hơn, mà còn phát ít tín hiệu nhiễu hơn, giải thích faithful hơn, và tạo trade-off tốt hơn giữa coverage, precision và risk-adjusted return.


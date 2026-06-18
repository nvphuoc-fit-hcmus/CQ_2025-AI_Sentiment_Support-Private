# Lộ trình cải tiến SAFE-Alert

Tài liệu này mô tả cách nâng cấp dự án SAFE-Alert hiện tại. Nội dung dựa trên `MoTa.md`, tài liệu trong `docs/`, source code hiện tại ở `services/ai-service`, artifact hiện có, và các tài liệu uy tín về financial NLP, selective prediction, calibration, time-series forecasting, multimodal finance, faithfulness evaluation và forecast evaluation.

---

## 1. Mục tiêu nâng cấp

### 1.1. Định vị paper sau khi nâng cấp

SAFE-Alert nên được định vị là một **intelligent decision support system** cho giao dịch crypto/finance, không chỉ là một mô hình dự báo giá.

Thay vì claim:

> Chúng tôi đề xuất một encoder mới cho dự báo giá Bitcoin.

Paper nên claim:

> Chúng tôi đề xuất một hệ thống cảnh báo giao dịch dựa trên bằng chứng, kết hợp chọn lọc tin tức, suy luận theo yếu tố tài chính, ngữ cảnh thị trường đa khung thời gian, confidence-aware abstention, và đánh giá faithfulness của explanation trong giao thức walk-forward pseudo-online.

Định vị này phù hợp hơn với ESWA vì ESWA ưu tiên hệ thống thông minh có thiết kế, triển khai, kiểm thử, ứng dụng thực tế và guideline sử dụng.

### 1.2. Luận điểm khoa học cần chứng minh

Muốn paper đủ mạnh, SAFE-Alert cần chứng minh 5 luận điểm:

| Luận điểm | Cần chứng minh bằng gì |
|---|---|
| Selective news giảm nhiễu so với all-news | Baseline all-news, random-K, most-recent-K, source-priority-K |
| Factor-grounded reasoning tạo explanation có ý nghĩa tài chính | Factor audit, factor consistency, domain expert review |
| Confidence-aware alerting tốt hơn luôn phát tín hiệu | Alert precision, coverage, selective risk curve, Sharpe/drawdown |
| Multi-timescale market context giúp ổn định dự báo | Market-only baseline, no-market ablation, horizon ablation |
| Explanation faithful hơn explanation hậu nghiệm | Deletion, insertion, sufficiency, comprehensiveness, random evidence comparison |

Nếu chỉ báo cáo Accuracy/F1, paper sẽ giống nhiều nghiên cứu price prediction thông thường. Điểm mạnh cần đẩy là **decision support có abstention và bằng chứng kiểm chứng được**.

---

## 2. Chẩn đoán hiện trạng dự án

### 2.1. Những điểm đã có và nên giữ

Repo hiện tại đã có nền tảng tốt:

- `MoTa.md` mô tả rõ story paper theo hướng news-to-alert, không chỉ news-to-price.
- `services/ai-service/README.md` mô tả kiến trúc SAFE-Alert v2, walk-forward, baseline, ablation và faithfulness.
- `services/ai-service/app/v2/pipelines/train_safe_alert.py` đã có walk-forward expanding window với embargo.
- `services/ai-service/app/v2/pipelines/safe_alert_dataset.py` dùng cửa sổ tin tức quá khứ, loại same-bar ambiguous leakage bằng điều kiện article time `< candle time`.
- Label sử dụng giá thực thi `open` của candle kế tiếp, phù hợp hơn với backtest thực tế.
- Model có chọn Top-K news theo horizon, factor module, market encoder, confidence head, policy threshold.
- `run_baselines.py` định nghĩa 14 baseline.
- `run_ablation.py` định nghĩa 7 ablation variant.
- Training/validation đã có các metric như Macro-F1, MCC, ECE, AUC, alert precision, coverage, Sharpe, Sortino, Calmar, max drawdown, faithfulness proxy.

Đây là nền đủ tốt để phát triển thành paper hệ thống.

### 2.2. Các rủi ro hiện tại nếu submit ngay

| Rủi ro | Vì sao nguy hiểm với reviewer |
|---|---|
| Artifact không nhất quán | `colab_run` báo 70,260 samples, `v2` báo 14,187 samples. Paper không được trộn hai nguồn này. |
| Baseline artifact chưa thấy | README nói có baseline, nhưng folder artifact hiện tại chưa có `baseline_results.json`. |
| Ablation artifact chưa thấy | README nói có ablation, nhưng folder artifact hiện tại chưa có `ablation_results.json`. |
| Faithfulness artifact chưa thấy | README nói output `faithfulness_results.json`, nhưng artifact hiện tại chưa có file này. |
| Factor labels còn phụ thuộc keyword/weak supervision | Reviewer finance/NLP có thể hỏi label quality và ontology validation. |
| Faithfulness script có insertion synthetic | `eval_faithfulness_sec4p4p2.py` đang mô tả insertion lift với synthetic positive article, chưa đủ thuyết phục bằng insertion dùng evidence thật. |
| Chưa đủ multi-asset/multi-horizon | Nếu chỉ BTCUSDT 1h, claim generalizable còn yếu. |
| Trading evaluation cần thực tế hơn | Cần fee, slippage, turnover, drawdown, statistical significance. |

### 2.3. Trạng thái artifact hiện có

Các artifact hiện thấy:

```text
services/ai-service/artifacts/colab_run/walk_forward_results.json
services/ai-service/artifacts/colab_run/table3_validation_report.json
services/ai-service/artifacts/colab_run/safe_alert_btcusdt_1h_policy.json
services/ai-service/artifacts/colab_run/safe_alert_policy_btcusdt_1h.json
services/ai-service/artifacts/v2/walk_forward_results.json
services/ai-service/artifacts/v2/safe_alert_btcusdt_1h_FINAL.pt
```

Điểm cần ghi rõ trong paper package:

- `colab_run/walk_forward_results.json` có `dataset_len = 70260`.
- `v2/walk_forward_results.json` có `dataset_len = 14187`.
- Chưa thấy `baseline_results.json`, `ablation_results.json`, `faithfulness_results.json` trong các artifact hiện tại.

Kết luận: bước đầu tiên không phải train thêm ngay, mà là **chốt canonical run và provenance**.

---

## 3. Chuẩn ESWA cần đạt

### 3.1. ESWA muốn gì

Theo aims & scope chính thức, ESWA tập trung vào expert/intelligent systems được thiết kế, phát triển, kiểm thử, triển khai hoặc quản lý trong các lĩnh vực thực tế, trong đó có finance và stock trading. ESWA cũng nhấn mạnh genuine innovation, không chỉ đổi tên khái niệm cũ bằng thuật ngữ mới.

Vì vậy, paper SAFE-Alert nên tránh kể câu chuyện như sau:

> Chúng tôi thêm attention vào FinBERT để dự báo BTC.

Nên kể như sau:

> Chúng tôi thiết kế một decision support pipeline có khả năng abstain, giải thích bằng factor tài chính, chọn bằng chứng theo horizon, và kiểm tra xem bằng chứng được giải thích có thật sự ảnh hưởng đến dự báo hay không.

### 3.2. Claim nên dùng

Claim an toàn và mạnh:

- SAFE-Alert là một framework tích hợp cho evidence-based financial alerting.
- Novelty nằm ở sự kết hợp có kiểm chứng giữa selective news, factor-grounded reasoning, confidence-aware abstention và faithfulness evaluation.
- Hệ thống được đánh giá bằng giao thức walk-forward pseudo-online, không shuffle random.
- Đánh giá bao gồm cả predictive metrics, alert metrics, calibration, trading utility và explanation faithfulness.
- Kiến trúc có thể triển khai trong microservice/Kafka hiện tại, nhưng paper tập trung vào scientific validation.

Claim nên tránh:

- "Dự báo giá Bitcoin chính xác tuyệt đối."
- "Model tạo lợi nhuận chắc chắn."
- "Encoder mới vượt mọi SOTA."
- "Explanation là nguyên nhân thật tuyệt đối."
- "Strong accept chắc chắn."

---

## 4. Roadmap nâng cấp từng giai đoạn

## Giai đoạn 0: Chốt canonical run và provenance

### Mục tiêu

Tạo một nguồn kết quả duy nhất cho paper. Không trộn artifact 70,260 samples và 14,187 samples.

### Việc cần làm

1. Chọn canonical dataset và canonical artifact.
2. Ghi rõ lý do chọn:
   - Nếu chọn 70,260: vì dữ liệu rộng hơn, có walk-forward đầy đủ hơn.
   - Nếu chọn 14,187: vì đây là artifact v2 cuối cùng có checkpoint đi kèm.
3. Lưu metadata cho run:
   - commit hash,
   - data hash,
   - config hash,
   - seed,
   - thời gian chạy,
   - thiết bị chạy,
   - version Python/PyTorch,
   - command log,
   - model checkpoint,
   - metric JSON.
4. Tạo cấu trúc paper artifact:

```text
paper_artifacts/
  safe_alert_eswa_canonical_<date>/
    README.md
    run_config.yaml
    data_manifest.json
    command_log.txt
    walk_forward_results.json
    baseline_results.json
    ablation_results.json
    faithfulness_results.json
    trading_results.json
    figures/
    tables/
```

### Lệnh tham chiếu hiện có

```bash
cd services/ai-service

python app/v2/pipelines/train_safe_alert.py \
  --walk_forward \
  --symbol BTCUSDT \
  --horizon 1h \
  --epochs 60 \
  --artifact_dir artifacts/colab_run
```

### Tiêu chí nghiệm thu

| Tiêu chí | Đạt khi |
|---|---|
| Không còn mâu thuẫn sample count | Paper chỉ dùng một canonical dataset length |
| Có provenance | Người khác biết chính xác config và data nào tạo ra bảng |
| Có command log | Có thể rerun pipeline |
| Có frozen policy | Threshold `tau`, `gamma`, temperature được tune trên validation và freeze trên test |

### Ví dụ dễ hiểu

Nếu bảng chính của paper lấy Macro-F1 từ artifact 70,260 samples nhưng backtest lấy Sharpe từ artifact 14,187 samples, reviewer có thể xem đó là cherry-picking. Canonical run giúp tránh vấn đề này.

---

## Giai đoạn 1: Nâng cấp dữ liệu và data card

### Mục tiêu

Chứng minh SAFE-Alert không chỉ hoạt động trên một cấu hình BTCUSDT 1h duy nhất.

### Nâng cấp đề xuất

| Thành phần | Hiện tại | Nâng cấp |
|---|---|---|
| Asset | Chủ yếu BTCUSDT | BTC, ETH, SOL, BNB hoặc top crypto có thanh khoản |
| Horizon | Chủ yếu 1h | 15m, 1h, 4h, 24h |
| News quality | Có news + embeddings | Dedupe, source audit, timestamp audit, entity linking |
| Market regimes | Chưa tách rõ | Bull, bear, sideway, crash, ETF, Fed, exchange crisis |
| Documentation | README kỹ thuật | Data card cho paper |

### Các bước chi tiết

1. Liệt kê toàn bộ nguồn dữ liệu đang dùng:
   - OHLCV,
   - news articles,
   - embeddings,
   - factor labels,
   - entity sentiment,
   - market features.
2. Kiểm tra timestamp:
   - timezone,
   - missing timestamp,
   - duplicate timestamp,
   - news timestamp sau candle có bị lọt vào sample không.
3. Dedupe news:
   - trùng title,
   - trùng URL nếu có,
   - trùng nội dung gần giống,
   - tin đăng lại nhiều lần.
4. Entity linking:
   - tin nói về BTC thì gắn BTC,
   - tin nói về Ethereum thì không tự động dùng cho BTC trừ khi có factor macro/systemic,
   - tin Fed/CPI có thể dùng cho nhiều asset.
5. Tạo data card:
   - thời gian bắt đầu/kết thúc,
   - số candle,
   - số bài viết,
   - số nguồn tin,
   - số bài/candle trung bình,
   - phân bố factor,
   - phân bố label `DOWN/NEUTRAL/UP`,
   - tỷ lệ no-news candles,
   - tỷ lệ duplicate bị loại.

### Output cần có

```text
paper_artifacts/<run_id>/data_manifest.json
paper_artifacts/<run_id>/data_card.md
paper_artifacts/<run_id>/dataset_stats.json
paper_artifacts/<run_id>/dataset_quality_report.md
```

### Tiêu chí nghiệm thu

| Tiêu chí | Đạt khi |
|---|---|
| Không nghi look-ahead | News window dùng dữ liệu trước candle, không lấy same-bar ambiguous news |
| Có multi-asset hoặc giải thích vì sao chưa có | Nếu không đủ dữ liệu, paper phải nói rõ giới hạn |
| Có multi-horizon hoặc giải thích scope | Ít nhất 1h và 4h nên có nếu muốn claim horizon-aware |
| Có data quality report | Reviewer thấy dữ liệu được audit chứ không scrape rồi train ngay |

### Ví dụ minh họa

Tin:

```text
"Ethereum staking withdrawals surge after network upgrade"
```

Không nên tự động dùng như tin mạnh cho BTC nếu không có bằng chứng market-wide. Nếu entity linking yếu, model có thể đưa tin ETH vào BTC và tạo explanation sai. Sau nâng cấp, tin này chỉ dùng cho BTC khi factor là systemic crypto liquidity hoặc market-wide risk.

---

## Giai đoạn 2: Nâng cấp factor ontology và chất lượng label

### Mục tiêu

Biến factor-grounded reasoning thành contribution đáng tin, không chỉ là keyword matching.

### Hiện trạng

`services/ai-service/app/v2/factor_ontology.json` và pipeline factor hiện có 10 factor. `precompute_factor_labels.py` hỗ trợ nhiều method:

- `keyword`,
- `zero_shot`,
- `llm_api`,
- `gemini`,
- `mistral`,
- `blend`.

Tuy nhiên default vẫn là `keyword`, và dataset có fallback keyword/title matching khi thiếu precomputed labels. Đây là điểm cần nâng cấp trước khi claim explanation theo factor.

### Ontology đề xuất cho paper

| Nhóm factor | Factor nên có | Ví dụ |
|---|---|---|
| Institutional flow | `etf_flow`, `institutional_inflow` | ETF inflow/outflow, corporate treasury buy |
| Regulation | `regulatory_easing`, `regulatory_tightening` | SEC approval, enforcement, ban |
| Exchange/system risk | `exchange_risk`, `custody_risk` | hack, withdrawal halt, insolvency rumor |
| Liquidity/derivatives | `liquidity_squeeze`, `funding_leverage`, `liquidation_pressure` | funding nóng, OI tăng, liquidation cascade |
| On-chain/whale | `whale_accumulation`, `miner_selling`, `exchange_inflow` | whale deposit to exchange |
| Macro | `macro_uncertainty`, `risk_on_liquidity`, `dollar_rates` | Fed hawkish, CPI surprise, DXY spike |
| Protocol/network | `protocol_upgrade`, `network_outage`, `security_exploit` | fork, outage, exploit |
| Stablecoin/systemic | `stablecoin_risk`, `market_structure_risk` | depeg, reserve concern |

Không nhất thiết phải tăng số factor ngay trong model v1. Nếu tăng ontology làm code đổi nhiều, paper có thể giữ 10 factor nhưng bổ sung mô tả, mapping và audit. Nếu thay đổi factor dimension, cần migration model và artifact mới.

### Quy trình label mạnh hơn

1. Chạy `keyword` để có baseline nhanh.
2. Chạy `blend` để dùng zero-shot khi confidence cao, keyword khi confidence thấp.
3. Lấy mẫu audit thủ công 500-1000 articles.
4. Domain expert gán:
   - primary factor,
   - secondary factor nếu có,
   - expected direction với BTC/ETH,
   - confidence,
   - ghi chú nếu tin không liên quan asset.
5. Đo:
   - agreement giữa keyword/blend/LLM và expert,
   - macro-F1 factor,
   - confusion matrix,
   - coverage theo factor,
   - tỷ lệ factor không rõ.
6. Dùng audit set để calibrate hoặc chọn method label chính.

### Lệnh tham chiếu

```bash
cd services/ai-service

python app/v2/pipelines/precompute_factor_labels.py \
  --method keyword

python app/v2/pipelines/precompute_factor_labels.py \
  --method blend \
  --batch_size 8 \
  --blend_max_prob 0.7
```

### Tiêu chí nghiệm thu

| Tiêu chí | Đạt khi |
|---|---|
| Ontology được domain expert đồng ý | Không có factor quá mơ hồ hoặc trùng nghĩa nghiêm trọng |
| Có audit set | Ít nhất 500 articles nếu nguồn lực hạn chế, tốt hơn là 1000+ |
| Có metric label quality | Factor macro-F1/agreement được báo cáo |
| Có confusion matrix | Biết factor nào hay nhầm với factor nào |
| Có examples | Paper đưa 5-10 ví dụ factor đúng/sai |

### Ví dụ minh họa

Tin:

```text
"Bitcoin ETF records $900M net inflow while BTC breaks above key resistance"
```

Label tốt:

```text
primary_factor: etf_flow
secondary_factor: institutional_inflow
expected_direction: bullish
confidence: high
```

Tin:

```text
"Major exchange pauses withdrawals after suspicious wallet activity"
```

Label tốt:

```text
primary_factor: exchange_risk
secondary_factor: liquidity_squeeze
expected_direction: bearish
confidence: high
```

Tin:

```text
"Analyst says Bitcoin may become the future of money"
```

Label tốt:

```text
primary_factor: none_or_low_signal
expected_direction: neutral
confidence: low
```

Nếu ontology không có trạng thái low-signal, model có thể ép tin chung chung vào factor sai. Vì vậy data card nên báo cáo tỷ lệ low-signal/noisy articles.

---

## Giai đoạn 3: Nâng cấp mô hình SAFE-Alert

### Mục tiêu

Giữ kiến trúc chính, nhưng bổ sung các tín hiệu giúp model chọn evidence tốt hơn và abstain đáng tin hơn.

### 3.1. Entity-aware selective news

Hiện model chọn tin theo embedding/meta/factor. Nên thêm entity-linking để giảm tin không liên quan.

Ví dụ:

```text
Tin A: "Solana network outage causes transaction failures"
Asset: BTCUSDT
```

Không nên là Top-K evidence cho BTC trừ khi market đang có systemic crypto risk. Nếu không có entity-aware selection, explanation có thể nghe hợp lý nhưng sai trọng tâm.

Output mong muốn:

```text
article_asset_tags: ["SOL"]
systemic_relevance: low
btc_relevance: low
eligible_for_btc_topk: false
```

### 3.2. Recency, novelty và source reliability

Dataset hiện có metadata về recency, source credibility, length, rank. Nên biến chúng thành phần được audit rõ:

| Signal | Ý nghĩa |
|---|---|
| Recency | Tin mới hơn thường quan trọng hơn trong horizon ngắn |
| Novelty | Tin đăng lại hoặc duplicate nên giảm trọng số |
| Source reliability | Nguồn tin đáng tin hơn nên được ưu tiên |
| Entity relevance | Tin đúng asset nên ưu tiên |
| Factor confidence | Factor label chắc chắn hơn nên ưu tiên |

Ví dụ:

```text
n1: ETF inflow từ nguồn chính thống, mới 5 phút, unique -> weight cao
n2: blog bullish chung chung, đăng lại từ tuần trước -> weight thấp
n3: tin Fed hawkish, nguồn uy tín, market-wide -> weight trung bình/cao
```

### 3.3. Nâng return/risk head

Nếu expected return correlation yếu, không nên ép paper claim regression mạnh. Có thể nâng cấp theo hướng:

- ordinal magnitude bins:
  - strong down,
  - mild down,
  - neutral,
  - mild up,
  - strong up.
- quantile return:
  - q10,
  - q50,
  - q90.
- downside risk:
  - probability of adverse move,
  - expected shortfall proxy,
  - max adverse excursion nếu có intra-horizon data.

Điểm mạnh ESWA là decision support, nên return head nên phục vụ alert policy, không nhất thiết phải dự báo giá chính xác từng điểm.

### 3.4. Confidence gate nâng cấp

Hiện pipeline đã có calibration và threshold policy. Nên formalize thành selective prediction:

```text
alert nếu:
  max_probability >= tau
  confidence >= gamma
  expected_utility_after_cost > 0
ngược lại:
  abstain
```

Nâng cấp đề xuất:

- tune `tau`, `gamma` chỉ trên validation,
- freeze policy trên test,
- báo cáo risk-coverage curve,
- chọn target coverage trước khi nhìn test,
- thêm conformal/selective risk guarantee nếu đủ thời gian.

### Tiêu chí nghiệm thu

| Tiêu chí | Đạt khi |
|---|---|
| Top-K ít tin sai asset hơn | Entity-relevance audit cải thiện |
| Alert ít hơn nhưng chính xác hơn | Alert precision tăng khi coverage giảm hợp lý |
| Calibration tốt | ECE thấp và reliability diagram hợp lý |
| Return/risk head có ích | Utility/backtest cải thiện so với chỉ direction |

---

## Giai đoạn 4: Nâng cấp baseline để reviewer khó bác

### Mục tiêu

Chứng minh SAFE-Alert không chỉ thắng baseline yếu.

### Baseline hiện có trong repo

`services/ai-service/app/v2/pipelines/run_baselines.py` đã định nghĩa 14 baseline:

| Nhóm | Baseline |
|---|---|
| Market-only | `market_only`, `raw_prob_threshold`, `temperature_scaled`, `current_price_predictor` |
| All-news | `all_news_fusion`, `all_news_llm_expl` |
| Sentiment/news fusion | `sentiment_market`, `nsm_style`, `finin_style`, `interleaved` |
| Factor/selective | `llm_factor`, `sep_style`, `selective_forecasting` |
| Trivial policy | `always_alert` |

Cần chạy thật và lưu artifact:

```text
paper_artifacts/<run_id>/baseline_results.json
```

### Baseline nên bổ sung

| Baseline | Lý do cần có |
|---|---|
| LightGBM/XGBoost market-only | Strong tabular baseline cho market features |
| TFT | Multi-horizon interpretable time-series baseline |
| PatchTST | Transformer time-series baseline hiện đại |
| iTransformer | Strong modern time-series backbone |
| FinBERT all-news average | Kiểm tra selective news có hơn all-news sentiment không |
| MFB-style multimodal fusion | ESWA đã có paper multimodal Bitcoin fusion, nên cần đối chiếu |
| Random-K evidence | Chứng minh chọn tin không phải ngẫu nhiên |
| Most-recent-K evidence | Chứng minh không chỉ chọn tin mới nhất |
| Source-priority-K evidence | Chứng minh không chỉ chọn nguồn uy tín |

### Lệnh tham chiếu hiện có

```bash
cd services/ai-service

python app/v2/pipelines/run_baselines.py \
  --symbol BTCUSDT \
  --horizon 1h \
  --epochs 30 \
  --artifact_dir artifacts/colab_run
```

Chạy nhanh để smoke test:

```bash
python app/v2/pipelines/run_baselines.py \
  --symbol BTCUSDT \
  --horizon 1h \
  --epochs 3 \
  --baselines always_alert market_only all_news_fusion \
  --artifact_dir artifacts/colab_run
```

### Bảng paper mong muốn

| Model | Macro-F1 | MCC | ECE | Alert Precision | Coverage | Sharpe | Max DD |
|---|---:|---:|---:|---:|---:|---:|---:|
| Market-only MLP | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| XGBoost market-only | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| All-news FinBERT | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| MFB-style fusion | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| PatchTST | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| iTransformer | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| SAFE-Alert | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

Không nên điền số từ nhiều run khác nhau. Mọi dòng trong bảng chính phải đến từ canonical run.

### Tiêu chí nghiệm thu

| Tiêu chí | Đạt khi |
|---|---|
| Có baseline mạnh | Không chỉ so với always-alert hoặc market-only đơn giản |
| Có baseline gần literature | Có multimodal/all-news/time-series modern baseline |
| Cùng protocol | Tất cả dùng same walk-forward split và same test windows |
| Có artifact | Kết quả lưu JSON, có thể trace về command |

---

## Giai đoạn 5: Nâng cấp ablation study

### Mục tiêu

Chứng minh từng component của SAFE-Alert có đóng góp riêng.

### Ablation hiện có

`services/ai-service/app/v2/pipelines/run_ablation.py` đã định nghĩa:

| Variant | Ý nghĩa |
|---|---|
| `w/o_selective_news` | Uniform average over all articles, bỏ Top-K gating |
| `w/o_factor` | Tắt factor module |
| `w/o_market` | Tắt market encoder |
| `w/o_confidence` | Tắt confidence head |
| `w/o_faithfulness` | Tắt faithfulness loss |
| `w/o_horizon` | Tắt horizon embedding |
| `w/o_lrisk` | Tắt selective risk loss |

### Lệnh tham chiếu

```bash
cd services/ai-service

python app/v2/pipelines/run_ablation.py \
  --symbol BTCUSDT \
  --horizon 1h \
  --epochs 30 \
  --artifact_dir artifacts/colab_run
```

### Ablation nên bổ sung sau khi nâng cấp model

| Variant mới | Câu hỏi trả lời |
|---|---|
| `w/o_entity_linking` | Entity relevance có giảm tin sai asset không? |
| `w/o_novelty` | Novelty có giảm duplicate/noise không? |
| `w/o_source_reliability` | Source credibility có giúp chọn evidence không? |
| `w/o_calibration` | Temperature/calibration có giảm ECE và cải thiện alert policy không? |
| `w/o_factor_audit_labels` | Label factor audit có hơn keyword-only không? |

### Bảng paper mong muốn

| Variant | Macro-F1 | MCC | Alert Precision | Coverage | Faithfulness | Sharpe |
|---|---:|---:|---:|---:|---:|---:|
| Full SAFE-Alert | TBD | TBD | TBD | TBD | TBD | TBD |
| w/o selective news | TBD | TBD | TBD | TBD | TBD | TBD |
| w/o factor | TBD | TBD | TBD | TBD | TBD | TBD |
| w/o market | TBD | TBD | TBD | TBD | TBD | TBD |
| w/o confidence | TBD | TBD | TBD | TBD | TBD | TBD |
| w/o faithfulness | TBD | TBD | TBD | TBD | TBD | TBD |
| w/o horizon | TBD | TBD | TBD | TBD | TBD | TBD |
| w/o Lrisk | TBD | TBD | TBD | TBD | TBD | TBD |

### Happy-path kết quả cần thấy

- Bỏ selective news: F1/alert precision giảm, deletion/insertion faithfulness giảm.
- Bỏ factor: predictive metrics có thể giảm ít, nhưng explanation quality và factor consistency giảm rõ.
- Bỏ market: model dễ bị tin tức nhiễu chi phối.
- Bỏ confidence: coverage tăng nhưng alert precision và Sharpe giảm.
- Bỏ faithfulness loss: explanation nghe hợp lý nhưng deletion/insertion kém hơn.
- Bỏ horizon: performance kém hơn ở multi-horizon.
- Bỏ Lrisk: classification có thể tương tự nhưng trading utility giảm.

### Tiêu chí nghiệm thu

| Tiêu chí | Đạt khi |
|---|---|
| Mỗi component có bằng chứng | Ít nhất một metric chính giảm khi bỏ component |
| Ablation cùng split | Không dùng split khác với full model |
| Không cherry-pick | Báo cả metric tăng và giảm |
| Có confidence interval | Bootstrap hoặc fold-wise std |

---

## Giai đoạn 6: Nâng cấp faithfulness evaluation

### Mục tiêu

Chứng minh explanation và selected news không chỉ là câu chuyện hậu nghiệm, mà có ảnh hưởng thật đến output model.

### Hiện trạng

Trong `train_safe_alert.py`, validation đã có hướng đúng hơn: deletion, selected-only, no-article baseline, sufficiency và insertion gain. Tuy nhiên `eval_faithfulness_sec4p4p2.py` vẫn mô tả insertion lift với synthetic positive article. Với paper ESWA, nên chuẩn hóa evaluation độc lập theo evidence thật.

### Faithfulness tests cần có

| Test | Cách làm | Kỳ vọng |
|---|---|---|
| Deletion/comprehensiveness | Xóa Top-K selected news | Confidence/probability giảm |
| Sufficiency | Chỉ giữ Top-K selected news | Prediction giữ được phần lớn |
| Insertion | Từ no-news thêm Top-K selected news | Confidence đúng hướng tăng |
| Random-K comparison | Thay Top-K bằng K tin random | SAFE Top-K tốt hơn random |
| Most-recent-K comparison | Thay Top-K bằng K tin mới nhất | SAFE Top-K tốt hơn chỉ recency |
| Source-priority-K comparison | Chọn K tin nguồn uy tín nhất | SAFE Top-K tốt hơn chỉ source |
| Explanation stability | Nhiễu nhỏ input không đổi explanation quá mạnh | Explanation ổn định |

### Ví dụ minh họa

Một sample dự báo `UP` vì chọn 3 tin:

```text
n1: ETF inflow mạnh
n2: MicroStrategy mua BTC
n3: BTC breakout với volume cao
```

Faithfulness tốt:

```text
full prediction: UP, confidence 0.82
remove n1,n2,n3: UP hoặc NEUTRAL, confidence 0.55
keep only n1,n2,n3: UP, confidence 0.77
replace by random news: confidence 0.58
```

Faithfulness yếu:

```text
full prediction: UP, confidence 0.82
remove selected news: UP, confidence 0.81
keep only selected news: DOWN, confidence 0.50
```

Trong trường hợp yếu, explanation không faithful dù nghe hợp lý.

### Lệnh tham chiếu hiện có

```bash
cd services/ai-service

python app/v2/pipelines/eval_faithfulness_sec4p4p2.py \
  --model_path artifacts/v2/safe_alert_btcusdt_1h_FINAL.pt \
  --symbol BTCUSDT \
  --horizon 1h \
  --device cpu \
  --output_json artifacts/colab_run/faithfulness_results.json
```

### Nâng cấp cần implement sau tài liệu này

Script faithfulness độc lập nên nhận thêm:

```text
--evidence_baselines random most_recent source_priority no_news
--num_samples 1000
--bootstrap 1000
--case_study_csv paper_artifacts/<run_id>/faithfulness_case_studies.csv
```

Đây là đề xuất cho giai đoạn code sau, không phải thay đổi trong tài liệu hiện tại.

### Tiêu chí nghiệm thu

| Tiêu chí | Đạt khi |
|---|---|
| Không dùng synthetic evidence làm kết quả chính | Insertion dùng selected real news và no-news baseline |
| Có random/recent/source baseline | Chứng minh selector học được gì đó |
| Có case study | Domain expert kiểm tra examples |
| Có failure cases | Paper trung thực về explanation sai/yếu |

---

## Giai đoạn 7: Nâng cấp trading evaluation

### Mục tiêu

Chứng minh alert policy có ý nghĩa tài chính sau chi phí, không chỉ tăng F1.

### Metric nên báo cáo

| Nhóm | Metric |
|---|---|
| Forecast | Accuracy, Macro-F1, MCC, AUC |
| Calibration | ECE, Brier, reliability diagram |
| Alert policy | Alert precision, coverage, risk-coverage curve |
| Trading | Net return, Sharpe, Sortino, Calmar, max drawdown |
| Execution | Turnover, number of trades, average holding time |
| Robustness | Fold-wise mean/std, bootstrap CI |
| Cost sensitivity | fee/slippage grid |

### Backtest cần thực tế hơn

Backtest nên có tối thiểu:

```text
entry_price = open(t+1)
exit_price = close(t+h)
fee_bps = configurable
slippage_bps = configurable
position = +1 for UP alert, -1 for DOWN alert, 0 for abstain/neutral
```

Nếu short không thực tế với spot, cần báo cáo hai setting:

- long-only,
- long-short futures-like.

### Cost sensitivity table

| Cost setting | Fee bps | Slippage bps | Sharpe | Max DD | Net return |
|---|---:|---:|---:|---:|---:|
| optimistic | 2 | 1 | TBD | TBD | TBD |
| realistic | 5 | 5 | TBD | TBD | TBD |
| conservative | 10 | 10 | TBD | TBD | TBD |

### Statistical tests

Nên bổ sung:

- paired bootstrap theo test windows để có confidence interval,
- Diebold-Mariano cho predictive loss khi so sánh forecast models,
- caution về multiple testing nếu thử nhiều strategy,
- không dùng p-value một cách trang trí.

### Ví dụ minh họa

Model A:

```text
Coverage: 90%
Alert Precision: 52%
Sharpe: 0.05
Max DD: -35%
```

Model B:

```text
Coverage: 30%
Alert Precision: 68%
Sharpe: 0.40
Max DD: -12%
```

Trong decision support, Model B có thể tốt hơn dù phát ít tín hiệu hơn. Đây chính là câu chuyện của SAFE-Alert: **abstain khi không đủ chắc chắn**.

### Tiêu chí nghiệm thu

| Tiêu chí | Đạt khi |
|---|---|
| Có net metrics sau chi phí | Không chỉ gross return |
| Có cost sensitivity | Kết quả không sụp khi fee/slippage tăng nhẹ |
| Có fold-wise robustness | Không phụ thuộc một fold may mắn |
| Có comparison với always-alert/buy-hold | Reviewer thấy policy có ích |

---

## Giai đoạn 8: Reproducible paper package

### Mục tiêu

Mọi bảng và hình trong paper phải sinh được từ artifact canonical, không điền tay.

### Cấu trúc đề xuất

```text
paper_artifacts/<run_id>/
  README.md
  data_card.md
  data_manifest.json
  run_config.yaml
  command_log.txt
  walk_forward_results.json
  baseline_results.json
  ablation_results.json
  faithfulness_results.json
  trading_results.json
  tables/
    table1_dataset_stats.csv
    table2_main_results.csv
    table3_baselines.csv
    table4_ablation.csv
    table5_faithfulness.csv
    table6_trading_cost_sensitivity.csv
  figures/
    fig1_architecture.png
    fig2_walk_forward_protocol.png
    fig3_risk_coverage_curve.png
    fig4_reliability_diagram.png
    fig5_faithfulness_deletion_insertion.png
    fig6_case_study_timeline.png
```

### Paper tables cần có

| Table | Nội dung |
|---|---|
| Table 1 | Dataset statistics theo asset/horizon |
| Table 2 | Main SAFE-Alert walk-forward results |
| Table 3 | Baseline comparison |
| Table 4 | Ablation study |
| Table 5 | Faithfulness evaluation |
| Table 6 | Trading evaluation sau cost |
| Table 7 | Domain expert factor audit |

### Paper figures cần có

| Figure | Nội dung |
|---|---|
| Fig. 1 | SAFE-Alert end-to-end architecture |
| Fig. 2 | Walk-forward split với embargo |
| Fig. 3 | Alert risk-coverage curve |
| Fig. 4 | Calibration/reliability diagram |
| Fig. 5 | Deletion/insertion faithfulness |
| Fig. 6 | Case study timeline: news -> factor -> alert -> market move |

### Tiêu chí nghiệm thu

| Tiêu chí | Đạt khi |
|---|---|
| Reproducible | Bảng/hình sinh lại được từ JSON |
| Traceable | Mỗi số trong paper có source artifact |
| Không manual editing | Không copy số từ notebook rời |
| Có README rerun | Người khác biết chạy lại lệnh nào |

---

## 5. Quy trình thí nghiệm end-to-end đề xuất

Đây là happy path sau khi đã nâng cấp dữ liệu và factor labels.

### Bước 1: Chuẩn bị dữ liệu

Input:

```text
OHLCV candles
news articles
source metadata
asset/entity mapping
```

Output:

```text
articles_clean.csv
candles_clean.csv
data_manifest.json
data_card.md
```

Checklist:

- Không duplicate nặng.
- Timestamp chuẩn timezone.
- Không dùng tin sau thời điểm dự báo.
- Có label distribution.
- Có no-news candle rate.

### Bước 2: Tạo market features

Lệnh tham chiếu:

```bash
cd services/ai-service

python app/v2/pipelines/precompute_market_features.py
```

Output mong muốn:

```text
market_features.npy
market_features_meta.json
```

Checklist:

- Feature chỉ dùng quá khứ và hiện tại, không dùng future.
- Có thống kê NaN/inf.
- Có feature dimension đúng với model hiện tại.

### Bước 3: Tạo FinBERT embeddings

Lệnh tham chiếu:

```bash
cd services/ai-service

python app/v2/pipelines/gen_emb.py
```

Output mong muốn:

```text
btcusdt_article_embeddings_max.npy
embedding_manifest.json
```

Checklist:

- Số row embedding khớp số article.
- Embedding không toàn zero.
- Model FinBERT và version được ghi lại.

### Bước 4: Tạo factor labels

Lệnh tham chiếu:

```bash
cd services/ai-service

python app/v2/pipelines/precompute_factor_labels.py \
  --method blend \
  --batch_size 8 \
  --blend_max_prob 0.7
```

Output mong muốn:

```text
article_factor_labels.npy
factor_label_report.json
factor_audit_sample.csv
```

Checklist:

- Distribution không collapse vào một factor.
- Có audit set.
- Có confusion matrix nếu có expert labels.

### Bước 5: Tạo entity/factor sentiment

Lệnh tham chiếu:

```bash
cd services/ai-service

python app/v2/pipelines/precompute_entity_sentiment.py
```

Output mong muốn:

```text
article_entity_sentiment.npy
entity_sentiment_report.json
```

Checklist:

- Sentiment gắn factor/asset, không chỉ sentiment chung.
- Fed hawkish không bị hiểu là positive chỉ vì câu văn không tiêu cực.
- Exchange hack phải negative với exchange/system risk.

### Bước 6: Train SAFE-Alert walk-forward

Lệnh tham chiếu:

```bash
cd services/ai-service

python app/v2/pipelines/train_safe_alert.py \
  --walk_forward \
  --symbol BTCUSDT \
  --horizon 1h \
  --epochs 60 \
  --artifact_dir artifacts/colab_run
```

Output mong muốn:

```text
walk_forward_results.json
safe_alert_btcusdt_1h_FINAL.pt
safe_alert_btcusdt_1h_policy.json
```

Checklist:

- Validation tune `tau`, `gamma`, temperature.
- Test dùng frozen validation policy.
- Có fold-wise metrics.
- Có embargo.

### Bước 7: Chạy baseline

Lệnh tham chiếu:

```bash
cd services/ai-service

python app/v2/pipelines/run_baselines.py \
  --symbol BTCUSDT \
  --horizon 1h \
  --epochs 30 \
  --artifact_dir artifacts/colab_run
```

Output mong muốn:

```text
baseline_results.json
```

Checklist:

- Baseline dùng cùng split.
- Threshold tune trên validation.
- Không dùng test để chọn threshold.
- Có strong baselines.

### Bước 8: Chạy ablation

Lệnh tham chiếu:

```bash
cd services/ai-service

python app/v2/pipelines/run_ablation.py \
  --symbol BTCUSDT \
  --horizon 1h \
  --epochs 30 \
  --artifact_dir artifacts/colab_run
```

Output mong muốn:

```text
ablation_results.json
```

Checklist:

- Mỗi ablation thay đúng một component.
- Full model và ablation cùng protocol.
- Báo cả metric prediction, alert, trading, faithfulness.

### Bước 9: Chạy faithfulness độc lập

Lệnh tham chiếu:

```bash
cd services/ai-service

python app/v2/pipelines/eval_faithfulness_sec4p4p2.py \
  --model_path artifacts/v2/safe_alert_btcusdt_1h_FINAL.pt \
  --symbol BTCUSDT \
  --horizon 1h \
  --device cpu \
  --output_json artifacts/colab_run/faithfulness_results.json
```

Output mong muốn sau khi nâng cấp:

```text
faithfulness_results.json
faithfulness_case_studies.csv
```

Checklist:

- Deletion dùng selected news thật.
- Insertion dùng no-news -> selected-news thật.
- Có random/recent/source baselines.
- Có case study và failure cases.

### Bước 10: Chạy trading evaluation

Output mong muốn:

```text
trading_results.json
cost_sensitivity.csv
risk_coverage_curve.csv
```

Checklist:

- Entry dùng `open(t+1)`.
- Exit theo horizon.
- Có fee/slippage.
- Có long-only và long-short nếu cần.
- Có turnover và max drawdown.

### Bước 11: Sinh bảng/hình paper

Output mong muốn:

```text
tables/*.csv
figures/*.png
figures/*.svg
```

Checklist:

- Mỗi bảng có source JSON.
- Không copy số thủ công.
- Figure có caption giải thích rõ.

---

## 6. Ví dụ paper story sau khi nâng cấp thành công

### Case 1: ETF inflow, market context ủng hộ

Input:

```text
Tin 1: Bitcoin ETF records $900M net inflow.
Tin 2: BTC breaks resistance with above-average volume.
Tin 3: Fed comments neutral, no new macro shock.
Market: momentum 1h và 4h cùng dương, volatility không quá cao.
```

SAFE-Alert:

```text
selected_news: Tin 1, Tin 2
factors: etf_flow, institutional_inflow
direction: UP
confidence: high
decision: ALERT
```

Giải thích tốt:

```text
Model phát alert UP vì dòng tiền ETF và momentum thị trường cùng pha.
Tin macro không đủ mạnh để phủ định tín hiệu.
```

Faithfulness kỳ vọng:

```text
Xóa Tin 1 và Tin 2 -> confidence giảm rõ.
Giữ chỉ Tin 1 và Tin 2 -> direction vẫn UP.
Random news -> confidence thấp hơn.
```

### Case 2: Exchange hack, market context xấu

Input:

```text
Tin 1: Major exchange reports suspicious wallet outflows.
Tin 2: Withdrawals temporarily paused.
Market: volume spike, downside momentum, funding overheated.
```

SAFE-Alert:

```text
selected_news: Tin 1, Tin 2
factors: exchange_risk, liquidity_squeeze
direction: DOWN
confidence: high
decision: ALERT
```

Giải thích tốt:

```text
Rủi ro sàn giao dịch và khả năng squeeze thanh khoản làm xác suất DOWN tăng.
Market context xác nhận bằng volume spike và momentum âm.
```

### Case 3: Tin tốt nhưng market context xấu

Input:

```text
Tin 1: A company announces small BTC purchase.
Tin 2: Fed signals rates higher for longer.
Market: BTC dưới MA, volatility tăng, funding nóng.
```

SAFE-Alert tốt:

```text
direction: UP hoặc NEUTRAL yếu
confidence: low
decision: ABSTAIN
```

Đây là điểm quan trọng: SAFE-Alert không phải cứ sentiment tốt là mua. Nếu context mâu thuẫn, abstain là hành vi đúng.

### Case 4: Model all-news bị nhiễu, selective news tốt hơn

Input:

```text
10 tin trong window:
  2 tin ETF thật sự quan trọng
  2 tin macro quan trọng
  3 tin blog chung chung
  2 tin đăng lại
  1 tin altcoin không liên quan
```

All-news baseline:

```text
Average embedding làm tín hiệu bị pha loãng.
Explanation khó chỉ ra evidence chính.
```

SAFE-Alert:

```text
Chọn 4 tin quan trọng nhất.
Loại hoặc giảm trọng số tin trùng/tin không liên quan.
Explanation ngắn và kiểm chứng được bằng deletion/insertion.
```

### Case 5: Confidence gate cứu paper khỏi overtrading

Input:

```text
Direction logits hơi nghiêng UP.
Max probability = 0.46
Confidence = 0.41
```

Policy:

```text
tau = 0.55
gamma = 0.50
decision = ABSTAIN
```

Nếu bỏ confidence gate:

```text
Model phát nhiều alert yếu.
Coverage tăng nhưng alert precision và Sharpe giảm.
```

Đây là ablation quan trọng cho ESWA vì nó chứng minh decision support cần biết khi nào không hành động.

---

## 7. Checklist để biết paper đã đủ mạnh hơn chưa

### 7.1. Data và protocol

- [ ] Chỉ dùng một canonical run.
- [ ] Có data hash và config hash.
- [ ] Có walk-forward split với embargo.
- [ ] News window không bị leakage.
- [ ] Có entry price `open(t+1)`.
- [ ] Có data card.
- [ ] Có label distribution.
- [ ] Có source/news quality audit.

### 7.2. Model và method

- [ ] Selective news có formalization.
- [ ] Factor ontology có mô tả domain rõ.
- [ ] Confidence gate có validation-tuned policy.
- [ ] Calibration được báo cáo bằng ECE/Brier/reliability diagram.
- [ ] Return/risk head phục vụ alert utility.
- [ ] Explanation được kiểm tra bằng perturbation, không chỉ generated text.

### 7.3. Baseline

- [ ] Có market-only strong baseline.
- [ ] Có all-news baseline.
- [ ] Có sentiment-market baseline.
- [ ] Có modern time-series baseline.
- [ ] Có selective forecasting baseline.
- [ ] Có random/recent/source evidence baseline.
- [ ] Có MFB-style hoặc multimodal finance baseline nếu đủ thời gian.

### 7.4. Ablation

- [ ] Bỏ selective news.
- [ ] Bỏ factor.
- [ ] Bỏ market.
- [ ] Bỏ confidence.
- [ ] Bỏ faithfulness loss.
- [ ] Bỏ horizon conditioning.
- [ ] Bỏ selective risk loss.
- [ ] Có fold-wise variance hoặc confidence interval.

### 7.5. Faithfulness

- [ ] Deletion/comprehensiveness.
- [ ] Sufficiency.
- [ ] True insertion từ no-news baseline.
- [ ] Random evidence comparison.
- [ ] Most-recent evidence comparison.
- [ ] Source-priority evidence comparison.
- [ ] Case studies có domain expert review.
- [ ] Failure cases được báo cáo trung thực.

### 7.6. Trading utility

- [ ] Có fee.
- [ ] Có slippage.
- [ ] Có turnover.
- [ ] Có Max Drawdown.
- [ ] Có Sharpe/Sortino/Calmar.
- [ ] Có cost sensitivity.
- [ ] Có long-only và long-short nếu phù hợp.
- [ ] Không claim lợi nhuận chắc chắn.

---

## 8. Tiêu chí "đủ cạnh tranh" cho ESWA

### 8.1. Mức tối thiểu nên đạt

Paper có thể được xem là đủ cạnh tranh nếu:

- SAFE-Alert vượt market-only và all-news baseline trên Macro-F1/MCC hoặc có trade-off alert tốt hơn rõ ràng.
- Alert Precision cao hơn Always-Alert và raw threshold baseline ở coverage hợp lý.
- ECE thấp hoặc calibration cải thiện so với uncalibrated model.
- Backtest sau cost không sụp hoàn toàn và drawdown thấp hơn baseline chọn lọc yếu.
- Faithfulness metrics tốt hơn random/recent/source evidence.
- Ablation cho thấy ít nhất 3 component chính có đóng góp.
- Artifact canonical tái lập được.

### 8.2. Mức strong paper nên hướng đến

Paper sẽ thuyết phục hơn nhiều nếu:

- Có multi-asset hoặc ít nhất multi-horizon.
- Có domain expert audit cho factor labels và explanation case studies.
- Có statistical confidence interval.
- Có robustness theo market regime.
- Có cost sensitivity sau fee/slippage.
- Có public/reproducible artifact hoặc ít nhất private artifact package đầy đủ.

### 8.3. Red flags phải tránh

- Kết quả chính lấy từ nhiều artifact không cùng run.
- Baseline train ít epoch hơn SAFE-Alert mà không giải thích.
- Threshold tune trên test.
- Shuffle random time-series split.
- Chỉ báo cáo Accuracy.
- Không có transaction cost.
- Explanation chỉ là LLM text, không có faithfulness test.
- Factor labels không được audit nhưng claim causal/economic reasoning quá mạnh.

---

## 9. Cách viết contribution sau khi nâng cấp

### Contribution 1: Evidence-selective financial alerting

Nên viết:

> We formulate financial news modeling as confidence-aware alerting rather than unconditional price prediction, allowing the system to abstain under weak or conflicting evidence.

Ý nghĩa:

- Đây là điểm khác price prediction thông thường.
- Kết nối trực tiếp với selective classification/reject option.

### Contribution 2: Factor-grounded explanation

Nên viết:

> We ground selected news in a finance-oriented factor ontology and evaluate whether these factors align with expert-audited market drivers.

Ý nghĩa:

- Không chỉ sentiment positive/negative.
- Có thể review bằng domain knowledge.

### Contribution 3: Unified evaluation

Nên viết:

> We evaluate the system under a walk-forward protocol using predictive, calibration, alert-policy, trading-utility and explanation-faithfulness metrics.

Ý nghĩa:

- Reviewer thấy paper có evaluation toàn diện.

### Contribution 4: Practical decision support pipeline

Nên viết:

> We demonstrate how the framework can be integrated into a deployable microservice pipeline while preserving a reproducible pseudo-online experimental protocol.

Ý nghĩa:

- Phù hợp ESWA vì có application và system angle.

---

## 10. Nguồn học thuật và official để justify roadmap

Các nguồn này nên được đưa vào phần Related Work hoặc Method Motivation của manuscript:

| Chủ đề | Nguồn | Dùng để justify |
|---|---|---|
| ESWA scope | https://www.sciencedirect.com/journal/expert-systems-with-applications | ESWA phù hợp intelligent systems trong finance/stock trading |
| FinBERT | https://arxiv.org/abs/1908.10063 | Domain-specific language model cho financial sentiment |
| Selective classification | https://papers.neurips.cc/paper/7073-selective-classification-for-deep-neural-networks | Reject option, risk-coverage, abstention |
| Calibration | https://proceedings.mlr.press/v70/guo17a.html | Temperature scaling, ECE, calibrated confidence |
| Temporal Fusion Transformer | https://arxiv.org/abs/1912.09363 | Multi-horizon interpretable time-series baseline |
| PatchTST | https://openreview.net/forum?id=Jbdc0vTOcol | Modern Transformer time-series baseline |
| iTransformer | https://openreview.net/forum?id=JePfAI8fah | Strong time-series forecasting backbone |
| MFB Bitcoin ESWA | https://www.sciencedirect.com/science/article/pii/S0957417424023820 | Multimodal Bitcoin price prediction baseline trong ESWA |
| ERASER faithfulness | https://aclanthology.org/2020.acl-main.408/ | Comprehensiveness, sufficiency, rationale evaluation |
| Diebold-Mariano | https://www.tandfonline.com/doi/abs/10.1080/07350015.1995.10524599 | So sánh predictive accuracy |

### Cách dùng nguồn trong paper

- FinBERT không phải novelty, chỉ là encoder nền cho financial text.
- Selective classification là nền tảng lý thuyết cho abstention.
- Calibration là lý do cần temperature scaling và ECE.
- TFT/PatchTST/iTransformer là baseline hoặc related work time-series.
- MFB là baseline/related work gần nhất cho multimodal Bitcoin forecasting trong ESWA.
- ERASER là nền tảng để đánh giá faithfulness bằng deletion/sufficiency.
- Diebold-Mariano dùng thận trọng cho forecast comparison, không dùng để phóng đại trading result.

---

## 11. Kết luận thực thi

SAFE-Alert hiện đã có một hướng paper hợp lý: chuyển từ dự báo giá sang cảnh báo giao dịch có chọn lọc, có bằng chứng và có abstention. Tuy nhiên để đạt mức cạnh tranh cao tại ESWA, dự án cần nâng cấp từ prototype/happy-path thành một experimental package có provenance, baseline mạnh, ablation đầy đủ, faithfulness thật, trading evaluation sau chi phí và factor audit có domain knowledge.

Thứ tự ưu tiên nên là:

1. Chốt canonical run và artifact provenance.
2. Chạy lại baseline/ablation/faithfulness để có artifact thật.
3. Nâng factor labels bằng audit set.
4. Bổ sung evidence baselines cho selector.
5. Bổ sung trading cost sensitivity.
6. Nếu còn thời gian, mở rộng multi-asset/multi-horizon và thêm modern time-series baselines.

Nếu các bước này thành công, paper sẽ có câu chuyện rõ ràng cho ESWA: một hệ thống expert/intelligent decision support trong finance, có triển khai thực tế, có đánh giá pseudo-online, có explanation kiểm chứng được, và có utility metric gần với nhu cầu giao dịch thật.

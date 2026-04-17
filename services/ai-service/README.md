# AI Service — SAFE-Alert v2

Microservice dự đoán tín hiệu giao dịch tiền mã hóa sử dụng kiến trúc **SAFE-Alert** (Sentiment-Aware Financial Event Alert). Kết hợp dữ liệu thị trường OHLCV từ Kafka với phân tích tin tức NLP từ MongoDB để tạo tín hiệu BUY/SELL/HOLD cho BTCUSDT với khả năng giải thích (explainability) có thể kiểm chứng theo PDF Section 4.4.2.

---

## Mục lục

1. [Kiến trúc mô hình](#1-kiến-trúc-mô-hình)
2. [Implement chi tiết từng module](#2-implement-chi-tiết-từng-module)
3. [Multi-objective Loss](#3-multi-objective-loss-eq3037)
4. [3-Stage Curriculum Training](#4-3-stage-curriculum-training)
5. [Walk-Forward Protocol](#5-walk-forward-cross-validation-protocol)
6. [Model Selection & Alert Policy](#6-model-selection--alert-policy)
7. [Kết quả Walk-Forward (5 Folds)](#7-kết-quả-walk-forward-5-folds)
8. [Thảo luận kết quả](#8-thảo-luận-kết-quả)
9. [Ablation Study (PDF Section 5.2)](#9-ablation-study-pdf-section-52)
10. [Baseline Comparison (PDF Section 4.3 / 5.1)](#10-baseline-comparison-pdf-section-43--51)
11. [Faithfulness Evaluation (PDF Section 4.4.2)](#11-faithfulness-evaluation-pdf-section-442)
12. [Cách chạy](#12-cách-chạy)
13. [Live Inference Pipeline](#13-live-inference-pipeline)
14. [API Endpoints](#14-api-endpoints)
15. [Cấu trúc thư mục](#15-cấu-trúc-thư-mục)
16. [Cài đặt & Dependencies](#16-cài-đặt--dependencies)

---

## 1. Kiến trúc mô hình

### 1.1 Tổng quan

SAFEAlertNet gồm 6 sub-module, implement theo đúng PDF Section 3. Luồng tính toán:

```
─────────────────────────────────────────────────────────────────────────────────
INPUT
─────────────────────────────────────────────────────────────────────────────────
market_feat   (B, 63)    5 timeframes × 12 indicators + 3 cross-timeframe
article_emb   (B, K, 768) FinBERT embeddings, L2-normalized
article_meta  (B, K, 14) recency, length, source_cred, novelty + 10 FSA scores
horizon       str         "1h" | "4h" | "15m" | "24h"
─────────────────────────────────────────────────────────────────────────────────

          horizon → h_emb_table (Embedding[4, 16]) → h_emb (B, 16)
                                  │
                    ┌─────────────┴──────────────────────────────┐
                    ▼                                            ▼
         Phi_mkt (Eq.16-19)                         Phi_sel (Eq.8-13)
  MultiTimescaleMarketEncoder              SelectiveNewsEncoder
         ↓ splits 63-dim into 5×15               ↓ encodes each article
         ↓ 5 independent encoders                  with h_emb concat
         Z_mkt (B,64), u_stack (B,5,64)         e_i (B,K,128)
                    │                                   │
                    │                    SelectiveAttention (Eq.9-13)
                    │                    ↓ attention scores a_i ∈ ℝ
                    │                    ↓ temp-softmax α = softmax(a/τ)
                    │                    ↓ TopK mask → α̃
                    │                    z_news (B,128)
                    │                                   │
                    │                    Phi_fac (Eq.14-15)
                    │                    ↓ p^fac_i = softmax(W_fac·e_i)  (B,K,10)
                    │                    ↓ z_fac = U·Σα̃_i·p^fac_i        (B,64)
                    │                                   │
                    └───────────────────┬───────────────┘
                                        ▼
                          Phi_fus (Eq.20-22) CrossAttentionFusion
                          CrossAttn(Q=z_news, K/V=u_stack)  → z_news' (B,64)
                          CrossAttn(Q=z_fac,  K/V=u_stack)  → z_fac'  (B,64)
                          MLP([z_news'; z_fac'; Z_mkt; h_emb]) → z_fus (B,64)
                                        │
                          Dropout(0.3)
                                        │
                          Phi_pred (Eq.23-25)
                          ├─► dir_head  (64→3)   → dir_logits    cross-entropy target
                          ├─► ret_head  (64→1)   → ret_pred       regression target
                          └─► conf_head (64→1)   → confidence     sigmoid ∈ (0,1)
                                        │
                          Alert Decision (Eq.26)
                          alert = 1[ conf ≥ τ  AND  max(softmax(dir_logits)) ≥ γ ]
                                        │
                          Phi_exp (Eq.27-29) ExplanationModule
                          selected_news, top_factors, nl_explanation
─────────────────────────────────────────────────────────────────────────────────
```

### 1.2 Dimension Map

| Tensor | Shape | Mô tả |
|--------|-------|-------|
| `h_emb` | `(B, 16)` | Learned horizon embedding |
| `e_i` | `(B, K, 128)` | Article representations |
| `alpha_tilde` | `(B, K)` | Sparse attention weights (TopK masked) |
| `z_news` | `(B, 128)` | News context vector |
| `p_fac` | `(B, K, 10)` | Per-article factor distributions |
| `z_fac` | `(B, 64)` | Factor embedding vector |
| `Z_mkt` | `(B, 64)` | Horizon-weighted market encoding |
| `u_stack` | `(B, 5, 64)` | Per-timeframe market encodings |
| `z_news'` | `(B, 64)` | News after cross-attention with market |
| `z_fac'` | `(B, 64)` | Factor after cross-attention with market |
| `z_fus` | `(B, 64)` | Final fused representation |
| `dir_logits` | `(B, 3)` | DOWN(0) / NEUTRAL(1) / UP(2) |
| `ret_pred` | `(B, 1)` | Expected 1h log-return |
| `confidence` | `(B,)` | Alert confidence ∈ (0,1) |

### 1.3 K_h Map (articles per horizon)

| Horizon | K_h | Lý do |
|---------|-----|-------|
| 15m | 3 | Short horizon — ít bài báo ảnh hưởng |
| 1h | 4 | Horizon train chính trong thesis |
| 4h | 5 | Medium horizon |
| 24h | 8 | Long horizon — nhiều context hơn |

### 1.4 Factor Ontology (10 classes)

| Index | Factor Name | Keyword Examples |
|-------|-------------|-----------------|
| 0 | `institutional_inflow` | institution, grayscale, microstrategy, treasury, bitcoin purchase |
| 1 | `etf_flow` | etf, spot etf, bitcoin etf, sec approval, inflow, outflow |
| 2 | `regulatory_easing` | approved, legal, regulatory clarity, licensed, framework |
| 3 | `regulatory_tightening` | ban, crackdown, illegal, sanction, sec sue, enforcement |
| 4 | `exchange_risk` | hack, exploit, exchange down, withdrawal halt, ftx, rug |
| 5 | `liquidity_squeeze` | liquidity, leverage, liquidation, margin call, cascade |
| 6 | `whale_accumulation` | whale, on-chain, address, wallet, accumulate, large buy |
| 7 | `macro_uncertainty` | inflation, fed, interest rate, recession, fomc, yield |
| 8 | `protocol_upgrade` | upgrade, fork, halving, taproot, layer2, lightning |
| 9 | `network_outage` | outage, congestion, fees spike, mempool, hash rate, 51% |

Factor labels được precompute bằng LLM (Mistral/Gemini) → soft distribution (B, K, 10), không phải one-hot, để model học xác suất thay vì decision boundary cứng.

---

## 2. Implement chi tiết từng module

### 2.1 ArticleEncoder — Eq.8

```python
# e_i = Phi_news(x_i, m_i, h)
# Input: emb (B,K,768) + meta (B,K,14) + h_emb (B,16) broadcast to (B,K,16)
# → concat → (B,K,798) → Linear(798,256) → LayerNorm → GELU → Dropout(0.2) → Linear(256,128) → LayerNorm → GELU
# Output: (B, K, 128)
```

- Horizon embedding được broadcast và concat theo chiều bài báo — mỗi bài báo "biết" horizon mà model đang predict.
- LayerNorm sau mỗi layer: tránh internal covariate shift khi batch nhỏ (article window thưa).
- GELU thay ReLU: gradient không chết ở vùng âm, quan trọng khi meta features có giá trị âm (FSA sentiment ∈ [-1, +1]).

### 2.2 SelectiveAttention — Eq.9-13

```python
# a_i = v^T · tanh(W_e·e_i + W_q·q + W_m·m_i)     Eq.9
# α   = softmax(a / τ, masked)                       Eq.11 — τ learnable, clamp[0.1, 5.0]
# α̃  = α ⊙ I(i ∈ TopK(a, K_h))                    Eq.12 — TopK on raw scores a, NOT α
# z_news = Σ α̃_i · e_i                               Eq.13
```

- `W_e` (128→64), `W_q` (64→64), `W_m` (14→64), `v` (64→1): tất cả linear, không bias, để không gian Q/K/V đồng nhất chiều.
- Temperature `τ` là learnable parameter (init=1.0, clamp [0.1, 5.0]), khác với một số implementation dùng fixed temperature.
- TopK mask dùng `scores_masked >= threshold - 1e-5` (dấu `>=` với epsilon để handle ties đúng với PDF notation).
- Soft gates `σ(a_i)` được trả ra riêng cho Lsel gradient — lý do: softmax α̃ luôn sum = K_h (constant), nên `(Σα̃ - K_h)² = 0` mọi lúc → zero gradient. Sigmoid gates không bị ràng buộc này.

### 2.3 FactorModule — Eq.14-15

```python
# p^fac_i = softmax(W_fac · e_i + b_fac)    Eq.14 — W_fac: Linear(128, 10)
# z_fac   = U · Σ(α̃_i · p^fac_i)           Eq.15 — U: Linear(10, 64, bias=False)
```

- `W_fac`: Linear(128→10) — per-article factor classifier.
- `U` (embedding matrix): Linear(10→64, no bias) — projects factor probability vector sang continuous factor embedding space.
- Lfac supervise `p^fac_i` trực tiếp bằng soft cross-entropy với LLM-generated labels → U học geometry của factor space gián tiếp qua gradient backprop.

### 2.4 MultiTimescaleMarketEncoder — Eq.16-19

```python
# Feature layout của 63-dim market vector:
#   dims  0-11 : 1m  timeframe (RSI_1m, MACD_1m, BB_1m, ...)
#   dims 12-23 : 5m  timeframe (RSI_5m, MACD_5m, ...)
#   dims 24-35 : 15m timeframe
#   dims 36-47 : 1h  timeframe
#   dims 48-59 : 4h  timeframe
#   dims 60-62 : cross-timeframe (RSI_agg, Momentum_cross, Volume_trend)
#
# Mỗi encoder nhận: tf_feat(12) + cross(3) = 15 dims → LINEAR ENCODER(15→64→LayerNorm→GELU)
#
# u_δ = Encoder_δ(feat_δ)                  Eq.17
# score_δ = w_δ^T tanh(W_δ·u_δ + W_h·h)  Eq.18 — horizon-conditioned
# β = softmax([score_1,...,score_5])        Eq.18
# Z_mkt = Σ β_δ · u_δ                      Eq.19
```

- 5 independent encoders, mỗi cái nhận 15 features thực sự khác nhau (không dùng cùng 1 vector 5 lần).
- Cross-timeframe features (dims 60-62) được concat vào mỗi timeframe → mỗi encoder có cả local và global context.
- `u_stack = stack(u_1,...,u_5)` → (B, 5, 64): dùng làm K/V trong CrossAttentionFusion (không phải z_mkt đơn lẻ vì `softmax(1 token) = 1.0` → output = V = z_mkt, news contribution = 0).

**12 Indicators per timeframe:** RSI, MACD, MACD_signal, BB_position, BB_width, EMA, Stochastic_K, Stochastic_D, Volume_ratio, ATR, ROC, CCI.

### 2.5 CrossAttentionFusion — Eq.20-22

```python
# proj_news: Linear(128→64)
# proj_fac:  Linear(64→64)
# ca_news:   MultiheadAttention(embed_dim=64, num_heads=4, kdim=64, vdim=64)
# ca_fac:    MultiheadAttention(embed_dim=64, num_heads=min(4,64//16), ...)
#
# z̃_news = CrossAttn(Q=proj_news(z_news).unsqueeze(1), K=u_stack, V=u_stack).squeeze(1)  Eq.20
# z̃_fac  = CrossAttn(Q=proj_fac(z_fac).unsqueeze(1),  K=u_stack, V=u_stack).squeeze(1)  Eq.21
# z_fus   = MLP([z̃_news; z̃_fac; Z_mkt; h_emb])                                          Eq.22
#         = Sequential(Linear(64+64+64+16=208, 128), GELU, Dropout(0.1), Linear(128, 64), GELU)
```

- CrossAttn với 5 K/V tokens (u_stack) → attention weights có ý nghĩa (không phải scalar = 1).
- News và Factor attend vào market timeframe khác nhau — z̃_news có thể attend nhiều vào short-term (1m, 5m) khi breaking news, z̃_fac attend vào long-term (4h) khi macro event.
- h_emb concat vào MLP input → fusion biết horizon ngay tại điểm kết hợp.

### 2.6 Prediction Heads — Eq.23-25

```python
self.dir_head  = nn.Linear(64, 3)   # Eq.23: logits cho DOWN/NEUTRAL/UP
self.ret_head  = nn.Linear(64, 1)   # Eq.24: expected log-return
self.conf_head = nn.Linear(64, 1)   # Eq.25: confidence → sigmoid → [0,1]
nn.init.zeros_(self.ret_head.bias)  # zero-init: tránh spurious positive offset trước training
```

### 2.7 Alert Decision — Eq.26

```python
# alert = 1[ conf ≥ τ_h  AND  max(softmax(dir_logits)) ≥ γ_h ]
```

- **Hai điều kiện độc lập:** τ kiểm soát confidence head, γ kiểm soát certainty về hướng. Model có thể confident (ĉ cao) nhưng vẫn không chắc về hướng (max_prob thấp) — AND gate chặn cả hai.
- τ, γ được tìm bằng grid search trên val set mỗi fold (không phải fixed).

### 2.8 ExplanationModule — Eq.27-29

```python
# Eq.27: S^news = {n_i | i ∈ TopK(α̃, K_h)}
# Eq.28: S^fac  = TopP(Σ α̃_i · p^fac_i, P=3)
# Eq.29: Phi_txt = template NL string
```

Output ví dụ:
```
"Tin hieu: BTC co kha nang tang (UP) trong 1h toi.
Do tin cay: 0.72.
Bang chung chinh: Bitcoin ETF inflows surge to $1.2B; Fed signals pause.
Yeu to chi phoi: etf_flow va macro_uncertainty."
```

---

## 3. Multi-objective Loss (Eq.30-37)

### 3.1 Công thức tổng

```
L = λ₁·Ldir + λ₂·Lret + λ₃·Lfac + λ₄·Lsel + λ₅·Lcal + λ₆·Lfaith + λ₇·Lrisk    (Eq.30)
```

### 3.2 Chi tiết từng thành phần

#### Ldir — Eq.31 (Direction Classification)

```python
# Cross-entropy với sqrt-softened class weights (xử lý imbalance DOWN/NEUTRAL/UP)
cw = compute_class_weight("balanced", classes=[0,1,2], y=train_labels)  # sklearn
cw_sqrt = np.sqrt(cw)  # sqrt để làm mềm (không penalize quá mạnh NEUTRAL)
Ldir = F.cross_entropy(dir_logits, dir_labels, weight=cw_sqrt)
```

- Class weights dùng sqrt (không phải balanced full weight) — thử nghiệm cho thấy full weight làm mô hình bỏ NEUTRAL hoàn toàn (macro F1 giảm).
- Ldir là term chính (λ=1.0) — không thay đổi qua 3 stages.

#### Lret — Eq.32 (Return Regression)

```python
# SmoothL1 trên return đã chuẩn hóa theo train std
scale = ret_train_std.clamp(min=1e-6)
Lret = F.smooth_l1_loss(ret_pred / scale, ret_labels / scale)
```

- SmoothL1 (Huber loss) thay MSE: ít nhạy cảm với outlier return (crypto có nhiều spike).
- Chuẩn hóa bằng training std: giữ gradient magnitude ổn định khi return scale khác nhau giữa bull/bear market.
- **Kết quả:** RetCorr ≈ 0 (BTC 1h return quá noisy, signal/noise ratio thấp, xem mục 8.8).

#### Lfac — Eq.33 (Factor Prediction)

```python
# Soft CE: L_fac = -Σ_i (α̃_i / Σα̃) · Σ_c ỹ_{i,c} · log p^fac_{i,c}
# Per-article log_softmax → masked average qua K articles

log_p = F.log_softmax(fac_probs, dim=-1)    # (B, K, 10)
per_article_ce = -(fac_labels * log_p).sum(dim=-1)  # (B, K)
# Article mask → average chỉ trên bài báo thực (không padding)
Lfac = (per_article_ce * article_mask).sum(dim=1) / article_mask.sum(dim=1).clamp(min=1.0)
```

- **Bug quan trọng tránh được:** Không average logits trước rồi mới log_softmax (Jensen's inequality: log_softmax(mean(p_i)) ≠ mean(log_softmax(p_i)) → gradient yếu đi ~1/K).
- Label smoothing có thể bật (`factor_label_smoothing > 0`) khi labels có nhiễu từ LLM.

#### Lsel — Eq.34 (Selection Regularization)

```python
# L_sel = (Σ σ(a_i) - K_h)² + η · Σ α̃_i · log(α̃_i)
#   Term 1: ép tổng sigmoid gates ≈ K_h  (training selection sparsity)
#   Term 2: entropy regularization trên α̃ (phân tán attention, tránh collapse)

# soft_gates = σ(a_i): (B, K), target = min(n_valid_articles, K_h)
target_k = article_mask.float().sum(dim=1).clamp(max=K_h)
sel_term = (soft_gates.sum(dim=1) - target_k) ** 2

# Entropy: dùng α̃ normalized (not soft_gates)
alpha_norm = alpha_tilde / alpha_tilde.sum(dim=1, keepdim=True).clamp(min=1e-8)
neg_entropy = (alpha_norm.clamp(min=1e-8) * torch.log(alpha_norm.clamp(min=1e-8))).sum(dim=1)
Lsel = (sel_term + eta * neg_entropy).mean()   # eta=0.1, clamp[-50, 50]
```

#### Lcal — Eq.35 (Confidence Calibration)

```python
# Brier score: L_cal = (ĉ - 1[ŷ=y])²
# calibration_targets = 1 khi prediction đúng, 0 khi sai
conf_c = confidence.clamp(1e-7, 1-1e-7)  # tránh log(0) trong diagnostics
Lcal = ((conf_c - calibration_targets.float()) ** 2).mean()
```

- Mục tiêu: `ĉ → P(correct)` — confidence head học cách tự đánh giá xem mình đúng hay sai.
- Khác ECE (evaluation metric): Lcal là **training loss** trực tiếp tối ưu Brier score, ECE chỉ dùng khi evaluate.

#### Lfaith — Eq.36 (Faithfulness Loss)

```python
# L_faith = (1/|S_art|) Σ_{i∈S_art} max(0, m - (p̂_full,i - p̂_masked,i))
# m = 0.15 (margin), p̂ = max(softmax(dir_logits))

full_max   = softmax(dir_logits).max(dim=-1)[0]         # (B,)
masked_max = softmax(masked_dir_logits).max(dim=-1)[0]  # (B,) — no-article forward
gap        = full_max - masked_max
has_art    = article_mask.any(dim=1).float()             # (B,) — skip no-article samples
Lfaith     = (clamp(0.15 - gap, min=0) * has_art).sum() / has_art.sum().clamp(min=1)
```

- **forward_masked():** chạy model với `article_mask = all zeros` → `has_any = 0` → `z_news = 0, z_fac = 0` → prediction dựa hoàn toàn vào market. Gap = đóng góp thuần của articles.
- Chỉ tính cho samples có bài báo thực (`has_art` mask) — no-article samples đều cho gap≈0, đưa vào sẽ push margin về 0.15 cho tất cả và làm loãng gradient.

#### Lrisk — Eq.37 (Selective Risk + Coverage)

```python
# L_risk = (Σ ĉ_i · ℓ_i) / (Σ ĉ_i)  +  μ · max(0, κ - mean(ĉ))
# ĉ: confidence (clamp min=0.01 cho stability)
# ℓ: CE loss per-sample (cached từ Ldir forward, không recompute)
# κ: 0.35 (coverage target), μ: 0.02

conf  = confidence.clamp(min=0.01)  # floor để tránh Σĉ ≈ 0 khi model uncertain
denom = conf.sum() + 1e-8
risk_term   = (ce_per_sample * conf).sum() / denom     # weighted average CE by confidence
cov_penalty = max(0, coverage_target - conf.mean())    # penalty khi ít alert quá
Lrisk = risk_term + 0.02 * cov_penalty
```

- **Semantic:** model học gán confidence cao cho samples mà nó predict đúng, thấp cho sai → confidence-weighted CE thấp khi sorting đúng.
- Coverage penalty ép `mean(ĉ) ≥ κ=0.35` — nếu model không đủ alert, bị penalize.

### 3.3 Numerical Stability

| Constant | Giá trị | Mục đích |
|----------|---------|---------|
| `_EPS` | `1e-8` | Tổng quát — chia không |
| `_CONF_MIN` | `1e-7` | Clamp confidence thấp cho Lcal |
| `_CONF_MAX` | `1-1e-7` | Clamp confidence cao cho Lcal |
| `_LRISK_CONF_MIN` | `0.01` | Floor cho Lrisk denominator (floor at 0.01·B) |
| `_LSEL_CLAMP` | `50.0` | Clamp Lsel để tránh explosion khi sparse news |
| tau clamp | `[0.1, 5.0]` | Clamp attention temperature τ |

### 3.4 NaN/Inf Guard

Mỗi forward pass kiểm tra validity:
```python
invalid = nan(dir_logits).any() | inf(dir_logits).any() | nan(ret_pred) | ...
if valid_idx.sum() == 0: return zero loss (no update)
```

Nếu model parameters bị NaN (có thể xảy ra do gradient explosion trước clip), `_sanitize_model_parameters()` clamp tất cả params về `[-1e4, 1e4]`.

---

## 4. 3-Stage Curriculum Training

### 4.1 Lambda Schedule (60 epochs)

```
Epoch Range │ Stage │  λ₁   λ₂   λ₃   λ₄   λ₅   λ₆   λ₇
            │       │  Ldir  Lret  Lfac  Lsel  Lcal  Lfa  Lrisk
────────────┼───────┼──────────────────────────────────────────
ep 1–12     │  S1   │  1.0  0.5  0.00  0.2  0.00  0.00  0.00
ep 13–36    │  S2   │  1.0  0.5  ramp  0.2  ramp  ramp  ramp
ep 37–60    │  S3   │  1.0  0.5  0.30  0.2  0.10  0.10  0.05
────────────┴───────┴──────────────────────────────────────────
```

**Stage 2 ramp (linear interpolation S1 → S2 targets):**
- λ₃ (Lfac):  0.00 → 0.30 (full weight by ep36)
- λ₅ (Lcal):  0.00 → 0.02 (small — tránh temperature drift)
- λ₆ (Lfaith):0.00 → 0.10
- λ₇ (Lrisk): 0.00 → 0.01 (small — early coverage signal)

**Tại sao S1 = 20% (không phải 30%):**
- Với 30%: Stage 1 = 18 epochs, Stage 2 chỉ có 18 epochs → Lfac từ 0 → max trong 18 epochs quá ngắn → factor head không kịp học trước khi Stage 3 bắt đầu.
- Với 20%: Stage 2 = 24 epochs → factor head có thêm 6 epochs, đủ để Lfac hội tụ ban đầu.

**Tại sao có λ₅=0.02 trong Stage 2:**
- Không có Lcal, `Ldir` đẩy model overconfident (entropy giảm, logits scale tăng).
- Temperature scaling compensate bằng cách tăng temperature từ 0.6 → 2.8 trong Stage 2.
- Với λ₅=0.02 nhỏ: temperature drift 0.6 → ~1.2 (kiểm soát được) → Stage 3 không cần "undo" quá nhiều.

**Tại sao có λ₇=0.01 trong Stage 2:**
- Confidence head không nhận gradient trong S1. Trong S2, chỉ có Lfaith trực tiếp.
- Lrisk nhỏ giúp confidence head học phân biệt correct/incorrect từ early hơn.

### 4.2 Optimizer & Scheduler

```python
optimizer = AdamW(
    params = model.parameters() + loss_fn.parameters(),
    lr = 3e-4,
    weight_decay = 1e-5,
)
scheduler = CosineAnnealingWarmRestarts(T_0=10, T_mult=2, eta_min=1e-7)
# T_0=10: warm restart mỗi 10 epochs (lúc training mới bắt đầu)
# T_mult=2: mỗi restart, period tăng gấp đôi (10→20→40)
# Khi Stage 3 bắt đầu: LR × 0.5 (fine-tuning mode)
```

### 4.3 Gradient Accumulation & Clipping

```python
_ACCUM_STEPS = 2   # effective batch = batch_size × 2
grad_clip = 1.0    # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
```

Gradient clipping trước `optimizer.step()` — tránh explosion khi Lfac bắt đầu (epoch 13, gradient đột ngột từ 0 → nonzero).

### 4.4 Early Stopping

```python
_ES_PATIENCE = 12  # Stage 3 only, theo model_score
```

Best model được lưu theo `model_score` (không phải val_loss) tại mỗi epoch Stage 3. Nếu 12 epochs liên tiếp không cải thiện → dừng.

### 4.5 Input Normalization

- **Market features:** StandardScaler fit trên train set, apply trên val/test. Scaler được lưu theo fold để tránh data leakage.
- **Return targets:** chuẩn hóa theo `ret_std` của train set (dùng cho Lret), không apply vào model input.
- **Article embeddings:** L2-normalized trước khi lưu vào `.npy` file.
- **Temperature scaling:** Grid search `[0.6, 0.75, 0.9, 1.0, 1.1, 1.25, 1.5, 1.8, 2.2, 2.8, 3.5]` trên val set, freeze trước khi dùng cho test.

---

## 5. Walk-Forward Cross-Validation Protocol

### 5.1 Design

```
Expanding window (PDF Eq.1): mỗi fold mở rộng train set thêm 1 window
Embargo: 24 candles (= 24h) giữa train↔val và val↔test để tránh leakage
                                                             từ carry-forward positions

Total: 70,260 candles  →  5 folds đều nhau (≈14,052 candles/fold)
```

### 5.2 Exact Splits

| Fold | Train [start, end] | Val [start, end] | Test [start, end] | Train Size | Val Size | Test Size |
|------|--------------------|-------------------|-------------------|-----------|---------|----------|
| 1 | [0, 28103] | [28128, 32318] | [32343, 36534] | 28,104 | 4,191 | 4,192 |
| 2 | [0, 32342] | [32367, 36557] | [36582, 40773] | 32,343 | 4,191 | 4,192 |
| 3 | [0, 36581] | [36606, 40796] | [40821, 45012] | 36,582 | 4,191 | 4,192 |
| 4 | [0, 40820] | [40845, 45035] | [45060, 49251] | 40,821 | 4,191 | 4,192 |
| 5 | [0, 45059] | [45084, 49274] | [49299, 53490] | 45,060 | 4,191 | 4,192 |

**Embargo = 24 steps:** `train_end + 24 = val_start`, `val_end + 24 = test_start`.

### 5.3 Policy Freezing

Thresholds τ, γ, temperature **được fit hoàn toàn trên val set** của từng fold, sau đó **frozen** khi evaluate trên test set. Mỗi fold có policy riêng:

| Fold | τ (conf) | γ (max_prob) | Temperature |
|------|---------|------------|------------|
| 1 | 0.529 | 0.504 | 2.8 |
| 2 | 0.504 | 0.547 | 1.8 |
| 3 | 0.497 | 0.677 | 1.1 |
| 4 | 0.455 | 0.631 | 1.25 |
| 5 | 0.493 | 0.581 | 1.8 |

---

## 6. Model Selection & Alert Policy

### 6.1 Model Selection Score (PDF Section 4.5.1)

```python
# Score = 0.40·F1 + 0.35·Sharpe_norm - 0.15·ECE_penalty + 0.10·MCC_norm
mcc_norm   = clip((mcc + 1) / 2, 0, 1)           # [-1,1] → [0,1]
ece_norm   = clip(1 - ece, 0, 1)                  # lower ECE → higher contribution
sharpe_norm = clip(alert_sharpe / 2.0, -1, 1)     # normalize, allow negative

score = 0.40*f1 + 0.35*sharpe_norm - 0.15*(1-ece_norm) + 0.10*mcc_norm
```

Rationale weights: F1 là chỉ số chính (40%), Sharpe quan trọng thứ hai (35% — đây là trading system), ECE penalty nhỏ hơn (15%), MCC bổ trợ robustness (10%).

### 6.2 Alert Policy Grid Search

```python
# τ grid: percentiles [20,35,50,60,65,75,80,85,90] của distribution confidence trên val
# γ grid: percentiles [20,35,50,55,65,70,75,80]    của distribution max_prob trên val
# Seed: τ_seed = p65(confidence), γ_seed = p55(max_prob) → P(pass)≈35%×45%≈16%, gần κ=35%

# Policy objective:
# objective = model_score + utility_bonus - coverage_penalty - negative_sharpe_penalty
#
# coverage_penalty:
#   if coverage < 0.30: penalty = 0.50*gap + 0.60*I(gap > 0.05)   (strong under-penalty)
#   if 0.30 ≤ coverage ≤ 0.40: 0 (tolerance band ±5%)
#   if coverage > 0.40: penalty = 0.20*(coverage - 0.40)

# Chọn policy tốt nhất TRONG số compliant (coverage ≥ 30%)
# Fallback: highest-coverage nếu không tìm thấy compliant
```

### 6.3 Deployment Model Selection

```python
# Chọn fold có val_model_score cao nhất
selected_deployment_fold = 2   # score = 0.358 (highest)
deployment_policy = {
    "tau": 0.504, "gamma": 0.547, "temperature": 1.8,
    "coverage": 0.375, "alert_precision": 0.667, "val_sharpe": 0.485,
}
```

---

## 7. Kết quả Walk-Forward (5 Folds)

**Dataset:** BTCUSDT 1h candles, 70,260 samples, expanding window với 24-step embargo.
**Training:** 60 epochs/fold, 3-stage curriculum, early stopping patience=12 (Stage 3 only).

### 7.1 Test Set — Forecast & Calibration Metrics

| Fold | Acc | Macro F1 | AUC | MCC | ECE | Brier |
|------|-----|---------|-----|-----|-----|-------|
| Fold 1 | 47.7% | 0.454 | 0.647 | 0.204 | **0.011** | — |
| Fold 2 | 53.8% | 0.538 | 0.719 | 0.309 | 0.022 | — |
| Fold 3 | **66.2%** | **0.570** | **0.762** | **0.346** | 0.024 | — |
| Fold 4 | 55.4% | 0.551 | 0.738 | 0.325 | 0.034 | — |
| Fold 5 | 60.3% | 0.563 | 0.746 | 0.338 | 0.042 | — |
| **AVG** | **56.7%** | **0.535** | **0.722** | **0.304** | **0.027** | — |

### 7.2 Test Set — Alert & Financial Metrics

| Fold | Precision | Coverage | Sharpe | Sortino | Calmar | MDD | PnL |
|------|-----------|----------|--------|---------|--------|-----|-----|
| Fold 1 | 0.589 | 33.0% | 0.366 | 0.831 | 331.9 | -10.0% | +525% |
| Fold 2 | **0.652** | 37.5% | 0.386 | 0.888 | 397.5 | **-7.1%** | +505% |
| Fold 3 | **0.807** | 40.4% | 0.043 | 0.142 | 6.5 | -24.2% | +30% |
| Fold 4 | 0.705 | 33.7% | **0.388** | **1.257** | **456.4** | **-4.8%** | +356% |
| Fold 5 | 0.721 | 38.3% | 0.164 | 0.401 | 60.7 | -10.7% | +119% |
| **AVG** | **0.695** | **36.6%** | **0.269** | **0.704** | **250.6** | **-11.4%** | **+307%** |

> **Coverage target:** κ = 0.35 (PDF). Kết quả avg = 36.6% — đạt trong khoảng ±1.6%.

### 7.3 Faithfulness Metrics (Test Set)

| Fold | Comprehensiveness | Insertion Gain | Sufficiency Drop | Factor Consistency |
|------|-----------------|----------------|-----------------|-------------------|
| Fold 1 | 0.0648 | 0.0804 | 0.0199 | **0.624** |
| Fold 2 | 0.0654 | 0.0736 | 0.0135 | 0.555 |
| Fold 3 | 0.0615 | 0.0686 | 0.0128 | 0.317 |
| Fold 4 | 0.0556 | 0.0658 | 0.0119 | 0.346 |
| Fold 5 | 0.0445 | 0.0546 | 0.0183 | 0.305 |
| **AVG** | **0.0583** | **0.0686** | **0.0153** | **0.429** |

### 7.4 Model Scores & Deployment

| Fold | Val Score | Test Score | Deployed |
|------|----------|-----------|---------|
| Fold 1 | 0.292 | 0.304 | — |
| **Fold 2** | **0.358** | **0.345** | **YES** |
| Fold 3 | 0.240 | 0.299 | — |
| Fold 4 | 0.261 | 0.349 | — |
| Fold 5 | 0.272 | 0.315 | — |
| **AVG** | **0.285** | **0.322** | — |

---

## 8. Thảo luận Kết quả

### 8.1 Coverage Target — Thành công sau Session 19

Coverage avg 36.6% đạt mục tiêu κ=35%. Đây là cải thiện căn bản từ phiên bản cũ (7–12%). Hai fix chính:

**Fix 1 — Hạ seed percentile:**
- `_TAU_PERCENTILE: 85→65`: P(conf≥p65)≈35% (thay vì p85≈15%) → seed τ gần target ngay từ đầu
- `_GAMMA_PERCENTILE: 80→55`: P(max_prob≥p55)≈45% → AND-gate: 35%×45%≈16% (cao hơn p80 case: 15%×20%=3%)
- Grid mở rộng để bao phủ vùng lân cận seed tốt hơn (thêm 60, 65 vào tau grid; 55 vào gamma grid)

**Fix 2 — Tăng penalty under-coverage 3.2×:**
- Cũ: penalty=0.036 cho coverage=12% < utility_bonus Sharpe≈0.04–0.06 → high-Sharpe 7% coverage được chọn
- Mới: penalty=0.115 cho coverage=12% → luôn lớn hơn utility_bonus → grid buộc phải chọn policy có coverage ≥ 30%

### 8.2 F1 Tăng Dần Theo Fold (Expanding Window Effect)

| Fold | Train Size | F1 | Gain |
|------|-----------|----|----|
| 1 | 28,104 | 0.454 | baseline |
| 2 | 32,343 | 0.538 | +0.084 |
| 3 | 36,582 | 0.570 | +0.032 |
| 4 | 40,821 | 0.551 | -0.019 |
| 5 | 45,060 | 0.563 | +0.012 |

F1 tăng 0.454→0.570 từ Fold 1 đến Fold 3, sau đó ổn định ~0.55–0.57. Expanding window giúp model thấy nhiều market regime hơn (bear 2020, bull 2021, crash 2022, recovery 2023). Fold 4 hơi giảm do test period (2024 early) có characteristics khác với training.

### 8.3 Fold 3 — Precision Cao nhưng Sharpe Thấp (Paradox)

| | F1 | Precision | γ threshold | Sharpe | MDD |
|-|----|-----------|------------|--------|-----|
| Fold 3 | 0.570 | **0.807** | **0.677** | 0.043 | -24.2% |
| Fold 4 | 0.551 | 0.705 | 0.631 | **0.388** | -4.8% |

**Cơ chế:** γ=0.677 (cao nhất) → model chỉ alert khi max_prob ≥ 0.677 → precision 80.7% nhưng rất ít trade và những trade này vào thời điểm BTC có volatility cao (late 2023). Một trade lớn thua lỗ (MDD -24.2%) xóa tích lũy → Sharpe 0.043.

γ cao không phải artifact — đó là policy mà grid search tìm thấy tối ưu trên val set của Fold 3 (late-2023 market). Val set period và test period có cùng regime bất ổn, nhưng tail risk không predict được → Sharpe thấp trên test.

**Kết luận:** Precision 80.7% là thật (80% alert đúng chiều), nhưng trong thị trường volatile, đúng chiều không đủ nếu magnitude nhỏ mà tail loss lớn. Sharpe capture điều này, F1 không.

### 8.4 Fold 4 — Risk-Adjusted Return Tốt Nhất

Fold 4 có Sharpe=0.388, Sortino=1.257, MDD=-4.84% — tốt nhất trong 5 folds. Test period của Fold 4 tương ứng với Bitcoin bull market 2024 (Q1-Q2), nơi UP signals có momentum rõ hơn. γ=0.631 (trung bình) cho phép alert thường xuyên hơn Fold 3 mà vẫn đủ precise (70.5%).

### 8.5 ECE Tăng Dần Theo Fold

| Fold | ECE | Temperature |
|------|-----|------------|
| 1 | 0.011 | 2.8 |
| 2 | 0.022 | 1.8 |
| 3 | 0.024 | 1.1 |
| 4 | 0.034 | 1.25 |
| 5 | 0.042 | 1.8 |

Temperature optimal giảm từ 2.8→1.1→1.25 (không monotone) — phản ánh mỗi fold test period có distribution khác. Fold 1: model overconfident nhiều (cần temp=2.8 để soften), Fold 3: model ít overconfident hơn (temp=1.1). ECE tăng cho thấy calibration khó hơn khi market 2024-2025 volatile hơn training data 2020-2023.

### 8.6 Faithfulness — Tin Tức Có Đóng Góp Thực Sự

- **Comprehensiveness 0.058:** Xóa K=4 selected articles, max_prob giảm 5.8pp trung bình. Nếu model không dùng news, giá trị phải ≈ 0.
- **Insertion gain 0.069:** Thêm bài báo positive (embedding = mean positive articles), confidence tăng 6.9pp. Insertion > Comprehensiveness (6.9 vs 5.8) nghĩa là thêm tin tốt có tác động lớn hơn xóa tin xấu — phù hợp với crypto market (news-driven spikes mạnh hơn news-driven drops).
- **Sufficiency drop 0.015:** Chỉ giữ K=4 selected articles, accuracy giảm 1.5pp so với dùng tất cả articles. Phi_sel đã chọn đúng những bài quan trọng nhất — most of the signal là trong K=4 bài này.
- **Factor consistency 0.429:** Thấp hơn kỳ vọng (target > 0.5). Fold 1-2 (0.55-0.62) tốt, Fold 3-5 (0.30-0.35) yếu. Nguyên nhân: articles 2024-2025 có embedding distribution khác với training data (FinBERT trained đến 2023), factor head bị confused bởi OOD embeddings.

### 8.7 Stage 3 — Điểm Yếu Trong Curriculum

Qua log training, best checkpoint thường là epoch đầu tiên Stage 3 (epoch 37). Hai cơ chế:

**Cơ chế 1 — Lambda Jump:** Lcal 0.02→0.10 (5×) và Lrisk 0.01→0.05 (5×) tại epoch 37, đồng thời LR giảm 50%. Model không kịp thích nghi → validation metrics giảm ngay sau ep37, không hồi phục trong patience=12 epochs.

**Cơ chế 2 — Distribution Shift:** Lfac qua Stage 2 thay đổi shared encoder → distribution of confidence scores và max_prob thay đổi → policy (τ, γ) fit trên val ở epoch 36 trở nên suboptimal ở Stage 3 representations. Model cần re-search policy nhưng policy bị frozen từ best Stage 3 epoch.

**Giải pháp đề xuất (future work):** Smooth transition 5-10 epochs khi vào Stage 3 thay vì step function. Ước tính cải thiện Sharpe +0.03–0.05 dựa trên gradient analysis.

### 8.8 Lret Không Hội Tụ

RetCorr ≈ 0 trên tất cả folds và epochs. BTC 1h return có signal/noise ratio thấp:
- Autocorrelation của BTC 1h return ≈ -0.02 (gần random walk)
- Ldir gradient mạnh hơn Lret (~10×) → optimizer ưu tiên hướng, ignore magnitude
- Model thực tế: dự đoán đúng hướng (F1=0.535) nhưng không biết magnitude

Giải pháp tiềm năng: dùng 4h hoặc 24h return (ít noisy hơn), hoặc auxiliary target là volatility thay vì return.

### 8.9 So sánh với Mục Tiêu PDF

| Metric | PDF Target | Kết quả | Đánh giá |
|--------|-----------|---------|---------|
| Macro F1 | > 0.35 | **0.535** (avg) | Vượt 53%, tất cả fold > 0.45 |
| Alert Coverage | κ ≈ 0.35 | **0.366** (avg) | Đạt, variance ±3.5% |
| Alert Precision | cao | **0.695** (avg) | 69.5% alert đúng chiều |
| Sharpe | > 0 | **0.269** (avg) | Đạt 4/5 folds (Fold 3: 0.043 borderline) |
| ECE | < 0.05 | **0.027** (avg) | Đạt thoải mái, avg < threshold |
| AUC | > 0.65 | **0.722** (avg) | Vượt mục tiêu |
| MCC | > 0.20 | **0.304** (avg) | Đạt tốt, balanced prediction |
| RetCorr | > 0 | ≈ 0 | **Không đạt** — BTC 1h quá noisy |
| Factor Consistency | > 0.5 | 0.429 (avg) | **Biên yếu** — OOD embeddings late folds |

---

## 9. Ablation Study (PDF Section 5.2)

Chạy bằng `run_ablation.py` — 7 variants × 30 epochs × 5 folds. Mỗi variant thay đổi đúng 1 component.

| # | Variant | Thay đổi | Metrics bị ảnh hưởng dự kiến |
|---|---------|----------|------------------------------|
| 1 | `w/o_selective_news` | α̃ → uniform average ALL articles (no TopK) | F1↓, Comprehensiveness↓, Coverage giảm vì noise articles |
| 2 | `w/o_factor` | z_fac = 0 (FactorModule disabled) | Factor Consistency=N/A, Sharpe↓ (mất ontology signal) |
| 3 | `w/o_market` | z_mkt = z_fac = 0 (market encoder disabled) | F1↓↓, Sharpe↓↓ (thị trường là signal chính) |
| 4 | `w/o_confidence` | conf = 0.5 cố định (conf head disabled) | Coverage 50% cố định, Alert Precision mất đi |
| 5 | `w/o_faithfulness` | λ₆ = 0.0 (Lfaith disabled) | Comprehensiveness↓, Insertion Gain↓ |
| 6 | `w/o_horizon` | h_emb = 0 (no horizon conditioning) | F1↓ ở 4h/24h horizon (kém generalize hơn) |
| 7 | `w/o_lrisk` | λ₇ = 0.0 (Lrisk disabled) | Coverage giảm về ~10-15% (không có coverage penalty) |

**Implementation:**
- `ablation_flag` string được truyền vào `model.forward(ablation=...)` → zeroes out đúng component.
- `lambda6_force` / `lambda7_force`: ghi đè lambda sau mỗi `_update_loss_weights_for_stage()`.
- `AblationTrainer` extends `SAFEAlertTrainer`, inject flag vào mỗi forward pass mà không sửa weight tensors.

---

## 10. Baseline Comparison (PDF Section 4.3 / 5.1)

Chạy bằng `run_baselines.py` — 14 baselines, train trên train split, tune trên val, report trên test.

| # | Baseline | Architecture | Thành phần khác biệt |
|---|----------|-------------|----------------------|
| 1 | `market_only` | MLP(63→128→128→3) + conf_head(128→1) | Không dùng news, baseline thuần market |
| 2 | `all_news_fusion` | mean(all embeddings) + MLP(768+63→256→3) | News nhưng không selective |
| 3 | `always_alert` | Trivial: predict majority class, alert=1 | No training |
| 4 | `raw_prob_threshold` | market_only + conf=max(softmax) | Không có dedicated confidence head |
| 5 | `sentiment_market` | 8-dim sentiment bottleneck + market MLP | 8-dim VADER/FinBERT mean → dense → concat market |
| 6 | `nsm_style` | Soft dot-product attention ALL articles | Attention không TopK, query = pooled market |
| 7 | `llm_factor` | Pre-computed factor labels (10-dim) + market | Neural factor decomposition → static LLM label vector |
| 8 | `sep_style` | MLP(news+market) + max-softmax threshold | No Lcal, no Lrisk, threshold tuned post-hoc |
| 9 | `finin_style` | MultiheadCrossAttn(news, market) | Full cross-attention, no selective Top-K |
| 10 | `interleaved` | (article, market) pair → mask-weighted agg | Pair-wise encoding thay joint encoding |
| 11 | `temperature_scaled` | market_only + post-hoc temperature (Guo 2017) | Temperature scaling sau training, không trong loss |
| 12 | `selective_forecasting` | all_news_fusion + coverage-optimized threshold | Coverage constraint tại inference, không trong training |
| 13 | `current_price_predictor` | Price momentum (close/open - 1) | Zero ML — price change direction làm prediction |
| 14 | `all_news_llm_expl` | all_news_fusion + LLM explanation post-hoc | News fusion + Gemini explain (no SHAP) |

**Shared hyperparameters:** `hidden=128`, `dropout=0.3`, `n_classes=3`, `accum_steps=2`.

---

## 11. Faithfulness Evaluation (PDF Section 4.4.2)

### 11.1 Comprehensiveness (Deletion Test)

```python
# gap = max_prob(full) - max_prob(masked_selected_only)
# Báo cáo: mean(gap) — higher = selected articles matter more
```

Mask TOP-K selected articles (theo α̃), chạy forward_masked(). Nếu news có ý nghĩa: max_prob giảm khi bỏ đi. Avg gap = 0.058.

### 11.2 Insertion Gain (True Insertion Baseline)

```python
# no_article: all-zeros mask → has_any=0 → z_news=0, z_fac=0 → market-only prediction
# insertion_gain = selected_conf - no_article_conf
# Cả hai path qua cùng CrossAttentionFusion → gain là contribution thuần của articles
```

Khác occlusion: insertion đo "articles thêm bao nhiêu so với không có gì" thay vì "selected articles thêm bao nhiêu so với unselected". Avg gain = 0.069.

### 11.3 Sufficiency Drop

```python
# selected_only: chỉ giữ K=4 selected articles, zero-out phần còn lại
# sufficiency_drop = max_prob(full) - max_prob(selected_only)
# Lower = better: Phi_sel đã chọn đủ thông tin
```

Avg drop = 0.015 — model giữ 98.5% signal chỉ với 4/nhiều articles. Phi_sel hoạt động hiệu quả.

### 11.4 Factor Consistency

```python
# Test: thêm Gaussian noise vào article embeddings (σ=0.1)
# Đo % samples mà argmax(factor distribution) không đổi
# Higher = factor head ổn định hơn với noise trong embedding space
```

Avg = 0.429. Fold 1-2 (~0.55-0.62) tốt, giảm ở Fold 3-5 (~0.30-0.35) do OOD embedding space.

### 11.5 Attention-Gap Correlation

```python
# Pearson correlation giữa attention weights α̃_i và |gap_i|
# (gap_i khi mask article i riêng lẻ)
# Higher = attention weights align với actual prediction impact
```

### 11.6 Chạy Faithfulness Eval

```bash
python app/v2/pipelines/eval_faithfulness_sec4p4p2.py \
    --symbol BTCUSDT \
    --horizon 1h \
    --device cuda      # hoặc cpu
```

Output: `artifacts/colab_run/faithfulness_results.json`

---

## 12. Cách Chạy

### 12.1 Preprocessing (chạy 1 lần, output đã có sẵn trong `training_data/v2/`)

```bash
cd services/ai-service

# Bước 1: FinBERT embeddings (768-dim) — cần GPU, ~30 phút
# Output: training_data/v2/btcusdt_article_embeddings_max.npy  (90346, 768)
python app/v2/pipelines/gen_emb.py

# Bước 2: Market features 63-dim — CPU, ~5 phút
# Output: training_data/v2/features_precomputed.npy  (70260, 63)
python app/v2/pipelines/precompute_market_features.py

# Bước 3: Factor labels (cần MISTRAL_API_KEY hoặc GEMINI_API_KEY) — ~2 giờ
# Output: training_data/v2/article_factor_labels.npy  (90346, 10)
python app/v2/pipelines/precompute_factor_labels.py

# Bước 4: Entity sentiment FSA (ProsusAI/finbert) — GPU, ~1 giờ
# Output: training_data/v2/article_entity_sentiment.npy  (90346, 14)
python app/v2/pipelines/precompute_entity_sentiment.py
```

### 12.2 Training — Walk-Forward 5 Folds (Kaggle / Colab GPU)

```bash
# Main training (quan trọng nhất) — ~8-10 giờ trên T4 GPU
# Lưu log để visualize training curves sau này
python app/v2/pipelines/train_safe_alert.py \
    --walk_forward \
    --symbol BTCUSDT \
    --horizon 1h \
    --epochs 60 \
    2>&1 | tee training_log.txt

# Xem kết quả:
cat artifacts/colab_run/walk_forward_results.json
cat artifacts/colab_run/table3_validation_report.json
```

**Kaggle Notebook (thêm ký tự `!` prefix):**
```python
# Cell 1: setup
import subprocess
proc = subprocess.Popen([
    "python", "app/v2/pipelines/train_safe_alert.py",
    "--walk_forward", "--symbol", "BTCUSDT", "--horizon", "1h", "--epochs", "60"
], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
for line in proc.stdout:
    print(line, end="")
```

### 12.3 Experiments (chạy sau khi có checkpoint từ training)

```bash
# Ablation study — 7 variants × 30 epochs/fold × 5 folds (~4 giờ T4)
python app/v2/pipelines/run_ablation.py \
    --symbol BTCUSDT \
    --horizon 1h \
    --epochs 30
# Output: artifacts/colab_run/ablation_results.json

# Baseline comparison — 14 baselines (~3 giờ T4)
python app/v2/pipelines/run_baselines.py \
    --symbol BTCUSDT \
    --horizon 1h \
    --epochs 30
# Chạy subset nhanh:
python app/v2/pipelines/run_baselines.py \
    --baselines market_only sentiment_market temperature_scaled all_news_fusion
# Output: artifacts/colab_run/baseline_results.json

# Backtest trên test set
python app/v2/pipelines/backtest_safe_alert.py \
    --symbol BTCUSDT \
    --horizon 1h
# Output: artifacts/colab_run/backtest_results.json

# Faithfulness evaluation
python app/v2/pipelines/eval_faithfulness_sec4p4p2.py \
    --symbol BTCUSDT \
    --horizon 1h
# Output: artifacts/colab_run/faithfulness_results.json
```

### 12.4 Visualize Kết Quả

```bash
# SVG charts — zero dependencies (chạy kể cả khi numpy/matplotlib hỏng)
# Output: artifacts/colab_run/figures/*.svg (12 charts, mở bằng browser)
python app/v2/visualization/generate_svg_charts.py

# Matplotlib PNG charts (cần numpy + matplotlib)
# Output: artifacts/colab_run/figures/*.png (7 charts)
python app/v2/visualization/visualize_results.py

# Training curves từ log file
python app/v2/visualization/visualize_training_curves.py \
    --log training_log.txt \
    --fold 4              # specific fold, hoặc bỏ để plot tất cả
# Output: artifacts/colab_run/figures/fig_training_fold4.png
```

### 12.5 Live Service (Docker)

```bash
# Từ thư mục gốc SA/
docker compose up -d ai-service

# Xem log realtime
docker logs sa-ai-service-1 -f

# Test endpoint
curl http://localhost:8002/health
curl http://localhost:8002/v2/signal/BTCUSDT
```

### 12.6 Local Dev (không Docker)

```bash
cd services/ai-service
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8002 --reload
```

---

## 13. Live Inference Pipeline

### 13.1 Luồng Dữ Liệu

```
Kafka topic: market.prices
      │  250 candles 1h BTCUSDT
      ▼
build_market_features()
      │  63-dim tensor từ 250 candles × 12 indicators × 5 timeframes
      ▼
MongoDB crawler_db (24h window)
      │  bài báo gần nhất 24h, filter theo symbol relevance
      ▼
VADER/FinBERT scoring → article_meta (K, 14)
FinBERT embedding lookup → article_emb (K, 768)
      │
      └──────────────────────────────────────────────┐
                                                      ▼
                                      SAFEAlertNet.forward(
                                          market_feat,
                                          horizon="1h",
                                          article_emb,
                                          article_meta,
                                      )
                                                      │
                                        alert_decision(tau=0.504, gamma=0.547)
                                                      │
                                    ┌─────────────────┴─────────────────┐
                                 horizon_1h                          horizon_4h
                                    │                                    │
                                    └──────────────┬─────────────────────┘
                                                   ▼
                                    Kafka: ai_insights → core-service → frontend
```

### 13.2 Fallbacks

| Tình huống | Fallback |
|------------|---------|
| MongoDB không kết nối | articles = [], model chạy market-only (z_news=z_fac=0) |
| Model chưa load | Return HOLD, confidence=0 |
| Kafka không có đủ 250 candles | Dùng data cũ trong cache, cảnh báo |

### 13.3 Inference Time

~2–5 giây per request (bao gồm MongoDB fetch + FinBERT embedding lookup + model forward).

---

## 14. API Endpoints

| Method | Endpoint | Mô tả | Latency |
|--------|----------|-------|---------|
| `GET` | `/health` | Health check | <10ms |
| `GET` | `/v2/signal/{symbol}` | Live inference. `symbol` ∈ {BTCUSDT, ETHUSDT} | 2–5s |
| `GET` | `/v2/signal/{symbol}/cached` | Lấy kết quả từ APScheduler cache (hourly refresh) | <50ms |
| `POST` | `/v2/signal/run-now` | Trigger inference thủ công tất cả symbols | 2–5s |
| `GET` | `/debug/market_cache/{symbol}` | Xem candles đang cache trong bộ nhớ | <10ms |
| `POST` | `/internal/analyze-investment` | API nội bộ cho Investment Service | 2–5s |
| `POST` | `/admin/train-model` | Trigger training (background process) | async |
| `GET` | `/admin/training-status` | Trạng thái training hiện tại | <10ms |

**Response đầy đủ `/v2/signal/BTCUSDT`:**
```json
{
  "timestamp": "2026-04-17T10:00:00Z",
  "symbol": "BTCUSDT",
  "data_source": "live",
  "alert": {
    "alert": true,
    "level": "high",
    "signal": "BUY"
  },
  "horizon_1h": {
    "signal": "BUY",
    "confidence": 0.72,
    "probs": { "DOWN": 0.12, "NEUTRAL": 0.18, "UP": 0.70 },
    "should_alert": true,
    "selected_news": [
      "Bitcoin ETF inflows surge to $1.2B daily",
      "Fed signals pause on rate hikes at FOMC meeting"
    ],
    "top_factors": [
      { "factor": "etf_flow", "score": 0.68 },
      { "factor": "macro_uncertainty", "score": 0.41 },
      { "factor": "institutional_inflow", "score": 0.31 }
    ],
    "explanation": "Tin hieu: BTC co kha nang tang (UP) trong 1h toi. Do tin cay: 0.72. Bang chung chinh: Bitcoin ETF inflows surge to $1.2B daily. Yeu to chi phoi: etf_flow va macro_uncertainty."
  },
  "horizon_4h": {
    "signal": "BUY",
    "confidence": 0.61,
    "probs": { "DOWN": 0.15, "NEUTRAL": 0.24, "UP": 0.61 },
    "should_alert": false
  },
  "top_inputs": {
    "rsi_14": 62.3,
    "macd_hist": 1.87,
    "bb_position": 0.72,
    "volume_ratio": 1.43,
    "ema_cross": 0.31
  }
}
```

---

## 15. Cấu Trúc Thư Mục

```
ai-service/
│
├── app/
│   ├── main.py                              # FastAPI app, APScheduler (hourly inference)
│   ├── market_cache.py                      # Kafka consumer → in-memory candle buffer (250 candles)
│   │
│   └── v2/
│       ├── factor_ontology.json             # 10 factor names + keywords (loaded by safe_alert_net.py)
│       │
│       ├── models/
│       │   └── safe_alert_net.py            # SAFEAlertNet + 6 sub-modules (Eq.8–29)
│       │                                    # ArticleEncoder, SelectiveAttention, FactorModule,
│       │                                    # MultiTimescaleMarketEncoder, CrossAttentionFusion,
│       │                                    # ExplanationModule, SAFEAlertNet
│       │
│       ├── alerts/
│       │   └── alert_decider.py             # Eq.26: alert = 1[conf≥τ AND max_prob≥γ]
│       │
│       ├── nlp/
│       │   ├── news_selector.py             # Top-K article selection cho live inference
│       │   └── relevance_scorer.py          # VADER / FinBERT scoring
│       │
│       ├── preprocessing/
│       │   ├── market_features.py           # 63-dim feature engineering từ OHLCV
│       │   └── news_features.py             # Article metadata extraction
│       │
│       ├── pipelines/
│       │   ├── live_infer.py                # Live inference adapter (entry point từ main.py)
│       │   ├── utils.py                     # ARTIFACT_DIR, load_safe_alert_policy()
│       │   │
│       │   ├── train_safe_alert.py          # SAFEAlertTrainer + walk-forward CV
│       │   │                                # 3-stage curriculum, early stopping, policy search
│       │   ├── safe_alert_training_utils.py # MultiObjectiveLoss (Eq.30–37)
│       │   ├── safe_alert_dataset.py        # SAFEAlertDataset — per-candle article window
│       │   ├── metrics_safe_alert.py        # 7 metric groups + model_score + search_alert_policy
│       │   │
│       │   ├── run_ablation.py              # 7 ablation variants (PDF Section 5.2)
│       │   ├── run_baselines.py             # 14 baseline models (PDF Section 4.3 / 5.1)
│       │   ├── backtest_safe_alert.py       # Test set backtest với transaction costs
│       │   ├── eval_faithfulness_sec4p4p2.py # Faithfulness tests (PDF Section 4.4.2)
│       │   │
│       │   ├── gen_emb.py                   # FinBERT embedding generation (768-dim)
│       │   ├── precompute_market_features.py # 63-dim market feature precomputation
│       │   ├── precompute_factor_labels.py  # LLM factor label generation (Mistral/Gemini)
│       │   ├── precompute_entity_sentiment.py # FSA entity sentiment (ProsusAI/finbert)
│       │   └── train_fresh.py               # Wrapper: xóa checkpoint → gọi train_safe_alert
│       │
│       └── visualization/
│           ├── generate_svg_charts.py       # Pure Python SVG (zero dependencies, 12 charts)
│           ├── visualize_results.py         # Matplotlib PNG (7 charts, cần numpy)
│           └── visualize_training_curves.py # Per-epoch training curves từ log file
│
├── artifacts/
│   └── colab_run/                           # Output của training Kaggle/Colab
│       ├── walk_forward_results.json        # 5-fold test metrics (main result file)
│       ├── table3_validation_report.json    # Deployment policy (Fold 2 selected)
│       ├── ablation_results.json            # 7-variant ablation (sau khi chạy run_ablation)
│       ├── baseline_results.json            # 14-baseline comparison (sau run_baselines)
│       ├── backtest_results.json            # Test set backtest
│       ├── faithfulness_results.json        # Faithfulness evaluation
│       ├── fold_1/                          # Fold 1 checkpoint + training_metrics.json
│       ├── fold_2/                          # Fold 2 checkpoint (DEPLOYED)
│       ├── fold_3/
│       ├── fold_4/
│       ├── fold_5/
│       └── figures/                         # Charts output (SVG / PNG)
│           ├── fig1_classification_metrics.svg
│           ├── fig2_sharpe_sortino.svg
│           ├── fig3_coverage.svg
│           ├── fig4_pnl.svg
│           ├── fig5_max_drawdown.svg
│           ├── fig6_ece_calibration.svg
│           ├── fig7_policy_params.svg
│           ├── fig8_faithfulness.svg
│           ├── fig9_alert_precision.svg
│           ├── fig10_val_vs_test_score.svg
│           ├── fig11_radar_chart.svg
│           └── fig12_summary_table.svg
│
├── training_data/
│   └── v2/
│       ├── articles_max.csv                 # 90,346 bài báo (2020–2025), semicolon separator
│       ├── tweets.csv                       # Twitter data (semicolon, utf-8-sig)
│       ├── btcusdt_training_dataset_v2.csv  # 70,260 candles với targets
│       ├── btcusdt_article_embeddings_max.npy  # (90346, 768) FinBERT, L2-normalized
│       ├── article_factor_labels.npy        # (90346, 10) soft factor distributions từ LLM
│       ├── article_entity_sentiment.npy     # (90346, 14) FSA: [recency,len,cred,novelty,10×FSA]
│       └── features_precomputed.npy         # (70260, 63) market features, precomputed
│
├── Dockerfile
├── requirements.txt
└── run_visualize.bat                        # Windows: fix numpy rồi chạy visualize_results.py
```

---

## 16. Cài đặt & Dependencies

### 16.1 Environment Variables

| Biến | Mặc định | Mô tả |
|------|----------|-------|
| `KAFKA_BROKERS` | `kafka:9092` | Kafka broker address |
| `MARKET_DATA_TOPIC` | `market.prices` | Topic nhận 1h candle data |
| `AI_INSIGHTS_TOPIC` | `ai_insights` | Topic publish signals |
| `MONGO_URL` | `mongodb://mongo-crawler:27017` | MongoDB cho news |
| `MONGO_DB` | `crawler_db` | Database name |
| `MONGO_COLLECTION` | `articles` | Collection name |
| `USE_FINBERT` | `0` | `1` = dùng FinBERT (440MB) thay VADER (4MB) |
| `MISTRAL_API_KEY` | — | Cho precompute_factor_labels.py |
| `GEMINI_API_KEY` | — | Alternative LLM (Gemini Flash) |

### 16.2 Dependencies Chính

| Package | Version | Mục đích |
|---------|---------|---------|
| `torch` | ≥2.0 | SAFEAlertNet, training, inference |
| `transformers` | ≥4.30 | FinBERT (ProsusAI/finbert) |
| `fastapi` | ≥0.100 | REST API framework |
| `uvicorn` | ≥0.22 | ASGI server |
| `scikit-learn` | ≥1.3 | Baselines, class weights, StandardScaler |
| `xgboost` | ≥1.7 | Baseline 5+ |
| `lightgbm` | ≥4.0 | Baseline 5+ |
| `confluent-kafka` | ≥2.0 | Kafka consumer/producer |
| `pymongo` | ≥4.0 | MongoDB news storage |
| `ta` | ≥0.10 | Technical analysis indicators (RSI, MACD, BB, ...) |
| `vaderSentiment` | ≥3.3 | Fast news scoring (fallback khi không có FinBERT) |
| `apscheduler` | ≥3.10 | Hourly inference scheduler |
| `shap` | ≥0.42 | Model explainability (auxiliary, không dùng trong training) |
| `numpy` | ≥1.24 | Array ops, feature precomputation |
| `pandas` | ≥2.0 | Data loading, preprocessing |
| `matplotlib` | ≥3.7 | Visualization (optional, PNG charts) |

### 16.3 Kiểm tra Môi Trường

```bash
# Kiểm tra GPU available
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"

# Kiểm tra model load
python -c "
from app.v2.models.safe_alert_net import SAFEAlertNet
m = SAFEAlertNet(); print('OK — params:', sum(p.numel() for p in m.parameters()))
"
# Expected: ~2.5M parameters

# Kiểm tra dataset
python -c "
from app.v2.pipelines.safe_alert_dataset import SAFEAlertDataset
ds = SAFEAlertDataset(symbol='BTCUSDT', horizon='1h')
print('Dataset size:', len(ds), '| Sample keys:', list(ds[0].keys()))
"
```

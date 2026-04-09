"""
NLP feature extraction for V2 pipeline.

Inspired by crypto-price-forecasting research (Section 3 — NLP Models):
  - Approach 0: VADER — fast dictionary-based, score = pos - neg in [-1, 1]
  - Approach 1: FinBERT — pretrained financial news sentiment (Araci 2019)
      model: ProsusAI/finbert (~440MB, free)
      score = P(positive) - P(negative)  ∈ [-1, 1]
  - Approach 2: BART-Large-MNLI — zero-shot coin-specific bullishness
      hypothesis = "This example is bullish for {coin}."
      score = P(entailment) - P(contradiction)
  - Approach 3: Fine-tuned RoBERTa — supervised on price target (optional)

Section 6 shows NLP data contributes 12.7% (BTC) and 22.2% (ETH) of XGBoost feature
importance — making it the 3rd most important feature category after technical indicators
and transaction/balance data.

Design decisions:
  - VADER always available (no GPU, no large download)
  - FinBERT optional (use_finbert=False by default) — ~440MB, free HuggingFace, financial domain
  - BART-MNLI optional (use_bart=False by default) — 1.6GB model
  - All scorers are lazy-loaded singletons
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("v2.news_features")


# ──────────────────────────────────────────────────────────────
# VADER (Section 3, approach 0)
# ──────────────────────────────────────────────────────────────

def get_vader_score(text: str) -> float:
    """
    VADER sentiment: positive_score - negative_score ∈ [-1, 1].
    CPU-only, no model download required beyond small dictionary.
    """
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        if not isinstance(text, str) or not text.strip():
            return 0.0
        scores = SentimentIntensityAnalyzer().polarity_scores(text)
        return float(scores["pos"] - scores["neg"])
    except ImportError:
        logger.debug("vaderSentiment not installed, returning 0.0")
        return 0.0
    except Exception:
        return 0.0


# ──────────────────────────────────────────────────────────────
# BART-MNLI zero-shot bullishness (Section 3, approach 2)
# ──────────────────────────────────────────────────────────────

class BartBullishnessScorer:
    """
    Coin-specific zero-shot bullishness scorer using BART-Large-MNLI.

    Directly mirrors BartMNLI from reference repo:
      hypothesis = "This example is bullish for {coin_name}."
      score = softmax(logits)[2] - softmax(logits)[0]  # entailment - contradiction

    Returns score ∈ [-1, 1]:
      > 0  → text is bullish for the coin
      < 0  → text is bearish for the coin
    """

    def __init__(self):
        self._model = None
        self._tokenizer = None
        self._device = None

    def _load(self) -> bool:
        if self._model is not None:
            return True
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            self._tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-mnli")
            self._model = (
                AutoModelForSequenceClassification
                .from_pretrained("facebook/bart-large-mnli")
                .to(self._device)
            )
            logger.info("BART-MNLI loaded on %s", self._device)
            return True
        except Exception as e:
            logger.warning("Could not load BART-MNLI: %s", e)
            self._model = None
            return False

    def get_score(self, text: str, coin_name: str = "Bitcoin") -> float:
        if not self._load():
            return 0.0
        if not isinstance(text, str) or not text.strip():
            return 0.0
        try:
            from scipy.special import softmax
            hypothesis = f"This example is bullish for {coin_name}."
            encoded = self._tokenizer.encode(
                text, hypothesis,
                return_tensors="pt",
                truncation=True,
                max_length=1024,
            ).to(self._device)
            output = self._model(encoded)[0][0].detach().cpu().numpy()
            probs = softmax(output)
            # indices: 0 = contradiction, 1 = neutral, 2 = entailment
            return float(probs[2] - probs[0])
        except Exception as e:
            logger.warning("BART score error: %s", e)
            return 0.0


# ──────────────────────────────────────────────────────────────
# FINBERT (Section 3, approach 1 — thay Twitter-RoBERTa)
# ──────────────────────────────────────────────────────────────

class FinBertScorer:
    """
    Sentiment scorer dùng ProsusAI/finbert — được train trên Reuters + Bloomberg.

    Tham chiếu: Araci, D. (2019). FinBERT: Financial Sentiment Analysis
    with Pre-trained Language Models. https://arxiv.org/abs/1908.10063

    Phù hợp hơn Twitter-RoBERTa cho financial news (CoinDesk, CoinTelegraph...)
    vì được train đúng domain: financial text, không phải social media.

    Labels: positive=0, negative=1, neutral=2
    score = P(positive) - P(negative) ∈ [-1, 1]
      > 0  → tin tức tích cực (bullish)
      < 0  → tin tức tiêu cực (bearish)
      ≈ 0  → trung tính

    Model ~440MB, download 1 lần về HuggingFace cache.
    """

    MODEL_NAME = "ProsusAI/finbert"

    def __init__(self):
        self._model = None
        self._tokenizer = None
        self._device = None

    def _load(self) -> bool:
        if self._model is not None:
            return True
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            self._tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)
            self._model = (
                AutoModelForSequenceClassification
                .from_pretrained(self.MODEL_NAME)
                .to(self._device)
            )
            self._model.eval()
            logger.info("FinBERT loaded on %s", self._device)
            return True
        except Exception as e:
            logger.warning("Could not load FinBERT: %s", e)
            self._model = None
            return False

    def get_score(self, text: str) -> float:
        """
        Trả về P(positive) - P(negative) ∈ [-1, 1].
        FinBERT giới hạn 512 tokens — đủ cho headline + lead paragraph.
        """
        if not self._load():
            return 0.0
        if not isinstance(text, str) or not text.strip():
            return 0.0
        try:
            import torch
            inputs = self._tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True,
            ).to(self._device)
            with torch.no_grad():
                logits = self._model(**inputs).logits[0]
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            # FinBERT labels: positive=0, negative=1, neutral=2
            return float(probs[0] - probs[1])
        except Exception as e:
            logger.warning("FinBERT score error: %s", e)
            return 0.0

    def get_scores_batch(self, texts: list[str], batch_size: int = 16) -> list[float]:
        """
        Batch inference — nhanh hơn nhiều so với gọi từng text.
        batch_size=16 (nhỏ hơn RoBERTa) vì BERT dùng nhiều RAM hơn.
        """
        if not self._load():
            return [0.0] * len(texts)
        results = []
        try:
            import torch
            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]
                batch = [t if isinstance(t, str) and t.strip() else " " for t in batch]
                inputs = self._tokenizer(
                    batch,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512,
                    padding=True,
                ).to(self._device)
                with torch.no_grad():
                    logits = self._model(**inputs).logits
                probs = torch.softmax(logits, dim=-1).cpu().numpy()
                # positive=0, negative=1, neutral=2
                scores = (probs[:, 0] - probs[:, 1]).tolist()
                results.extend(scores)
        except Exception as e:
            logger.warning("FinBERT batch error: %s", e)
            results.extend([0.0] * (len(texts) - len(results)))
        return results


# Lazy singletons
_finbert_scorer: Optional[FinBertScorer] = None
_bart_scorer: Optional[BartBullishnessScorer] = None


def get_finbert_score(text: str) -> float:
    """Sentiment score từ FinBERT. Load model lần đầu gọi."""
    global _finbert_scorer
    if _finbert_scorer is None:
        _finbert_scorer = FinBertScorer()
    return _finbert_scorer.get_score(text)


def get_bart_bullish_score(text: str, coin_name: str = "Bitcoin") -> float:
    """Get coin-specific bullishness score. Loads BART-MNLI on first call."""
    global _bart_scorer
    if _bart_scorer is None:
        _bart_scorer = BartBullishnessScorer()
    return _bart_scorer.get_score(text, coin_name)


# ──────────────────────────────────────────────────────────────
# COIN NAME MAPPING
# ──────────────────────────────────────────────────────────────

COIN_NAME_MAP: dict[str, str] = {
    "BTCUSDT": "Bitcoin",
    "ETHUSDT": "Ethereum",
    "BNBUSDT": "BNB",
    "SOLUSDT": "Solana",
    "ADAUSDT": "Cardano",
    "XRPUSDT": "XRP",
    "DOTUSDT": "Polkadot",
    "DOGEUSDT": "Dogecoin",
    "AVAXUSDT": "Avalanche",
    "MATICUSDT": "Polygon",
}


# ──────────────────────────────────────────────────────────────
# BATCH NLP SCORING
# ──────────────────────────────────────────────────────────────

def add_nlp_scores(
    news_df: pd.DataFrame,
    symbol: str = "BTCUSDT",
    use_finbert: bool = False,
    use_bart: bool = False,
) -> pd.DataFrame:
    """
    Add NLP scores to a news DataFrame.

    Columns added:
      - vader_score    : VADER pos-neg ∈ [-1, 1]  (always)
      - finbert_score  : FinBERT P(positive)-P(negative) ∈ [-1, 1]  (if use_finbert=True)
      - bullish_score  : BART-MNLI entailment-contradiction ∈ [-1, 1]  (if use_bart=True)
      - nlp_score      : combined score (weighted average của các model có sẵn)

    Trọng số kết hợp:
      - Chỉ VADER:               nlp_score = vader_score
      - VADER + FinBERT:         nlp_score = 0.3*vader + 0.7*finbert  (FinBERT domain-specific)
      - VADER + BART:            nlp_score = 0.5*vader + 0.5*bart
      - VADER + FinBERT + BART:  nlp_score = 0.2*vader + 0.5*finbert + 0.3*bart

    Args:
        news_df      : DataFrame with at least one of [title, text, content]
        symbol       : Trading symbol for BART coin-specific hypothesis
        use_finbert  : Load FinBERT (~440MB, financial domain, recommended)
        use_bart     : Load BART-MNLI (~1.6GB, coin-specific bullishness)
    """
    out = news_df.copy()

    # Pick best available text column
    if "title" in out.columns:
        text_col = "title"
    elif "text" in out.columns:
        text_col = "text"
    elif "content" in out.columns:
        text_col = "content"
    else:
        out["vader_score"] = 0.0
        out["nlp_score"] = 0.0
        return out

    texts = out[text_col].fillna("").astype(str)

    # VADER — always computed (fast, CPU-only)
    out["vader_score"] = texts.apply(get_vader_score)

    if use_finbert:
        global _finbert_scorer
        if _finbert_scorer is None:
            _finbert_scorer = FinBertScorer()
        logger.info("Computing FinBERT scores (%d articles)...", len(out))
        out["finbert_score"] = _finbert_scorer.get_scores_batch(texts.tolist())

    if use_bart:
        coin_name = COIN_NAME_MAP.get(symbol, "Bitcoin")
        logger.info("Computing BART-MNLI bullishness for %s (%d articles)...", coin_name, len(out))
        out["bullish_score"] = texts.apply(lambda t: get_bart_bullish_score(t, coin_name))

    # Kết hợp theo trọng số
    if use_finbert and use_bart:
        out["nlp_score"] = (
            0.2 * out["vader_score"]
            + 0.5 * out["finbert_score"]
            + 0.3 * out["bullish_score"]
        )
    elif use_finbert:
        out["nlp_score"] = 0.3 * out["vader_score"] + 0.7 * out["finbert_score"]
    elif use_bart:
        out["nlp_score"] = 0.5 * out["vader_score"] + 0.5 * out["bullish_score"]
    else:
        out["nlp_score"] = out["vader_score"]

    return out


# ──────────────────────────────────────────────────────────────
# WINDOW AGGREGATION
# ──────────────────────────────────────────────────────────────

def aggregate_nlp_window(news_window: pd.DataFrame) -> dict:
    """
    Aggregate NLP scores for a time window of news articles into scalar features.

    Feature design inspired by Section 3 post-processing (3_nlp_models/4_processing):
    - Mean score (trend direction)
    - Std score (divergence of opinions)
    - Max/min (extreme sentiment events)
    - Bullish/bearish ratio (directional balance)
    - Max absolute score (strength of signal)

    Returns:
        dict of scalar features ready to merge into training dataset row.
    """
    if news_window.empty:
        return {
            "news_count": 0,
            "vader_mean": 0.0,
            "vader_std": 0.0,
            "vader_max": 0.0,
            "vader_min": 0.0,
            "finbert_mean": 0.0,
            "bullish_ratio": 0.0,
            "bearish_ratio": 0.0,
            "nlp_score_mean": 0.0,
            "nlp_score_max_abs": 0.0,
        }

    n = len(news_window)

    # Use vader_score if available, else sentiment_score (legacy column from V1)
    if "vader_score" in news_window.columns:
        vader = news_window["vader_score"].fillna(0.0).astype(float)
    elif "sentiment_score" in news_window.columns:
        vader = news_window["sentiment_score"].fillna(0.0).astype(float)
    else:
        vader = pd.Series([0.0] * n)

    nlp = (
        news_window["nlp_score"].fillna(0.0).astype(float)
        if "nlp_score" in news_window.columns
        else vader
    )

    # FinBERT mean nếu có
    finbert_mean = 0.0
    if "finbert_score" in news_window.columns:
        finbert_mean = float(news_window["finbert_score"].fillna(0.0).astype(float).mean())

    return {
        "news_count": n,
        "vader_mean": float(vader.mean()),
        "vader_std": float(vader.std()) if n > 1 else 0.0,
        "vader_max": float(vader.max()),
        "vader_min": float(vader.min()),
        "finbert_mean": finbert_mean,
        "bullish_ratio": float((vader > 0.1).sum() / n),
        "bearish_ratio": float((vader < -0.1).sum() / n),
        "nlp_score_mean": float(nlp.mean()),
        "nlp_score_max_abs": float(nlp.abs().max()),
    }


# ──────────────────────────────────────────────────────────────
# LAG FEATURES FOR NLP (Section 4)
# ──────────────────────────────────────────────────────────────

def add_nlp_lag_features(
    df: pd.DataFrame,
    nlp_cols: list[str] | None = None,
    lags: list[int] | None = None,
) -> pd.DataFrame:
    """
    Add lagged NLP features to dataset.

    Research (Section 4 Granger causality) found sentiment scores at lags 4-6 to be
    Granger-causal for BTC price (e.g., btc_news_bart_mnli_bullish_score_10,
    btc_tweets_roberta_finetuned_score_0 through _13).

    For hourly candles: lags 1-3 cover the recent 1-3 hour sentiment history.
    """
    if nlp_cols is None:
        nlp_cols = ["vader_mean", "finbert_mean", "nlp_score_mean", "bullish_ratio", "bearish_ratio"]
    if lags is None:
        lags = [1, 2, 3]

    out = df.copy()
    for col in nlp_cols:
        if col not in out.columns:
            continue
        for lag in lags:
            out[f"{col}_lag{lag}"] = out[col].shift(lag)
    return out

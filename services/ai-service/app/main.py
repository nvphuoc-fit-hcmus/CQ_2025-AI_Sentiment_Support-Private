import os
import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from threading import Thread
from app.market_cache import start_market_cache_thread

# V1 modules (optional — may not be present in all deployments)
try:
    from app.kafka_consumer import start_consumer as _start_consumer
    _HAS_KAFKA_CONSUMER = True
except ImportError:
    _HAS_KAFKA_CONSUMER = False
    _start_consumer = None

try:
    from app.kafka_producer import close_producer, create_startup_topics
    _HAS_KAFKA_PRODUCER = True
except ImportError:
    _HAS_KAFKA_PRODUCER = False
    close_producer = lambda: None
    create_startup_topics = lambda: None

try:
    from app.modules.causal import schedule_causal_job, run_causal_now, analyze_event_causal
    _HAS_CAUSAL = True
except ImportError:
    _HAS_CAUSAL = False
    schedule_causal_job = lambda: None
    run_causal_now = lambda: {}
    analyze_event_causal = lambda p: {}

try:
    from app.modules.sentiment import analyze_sentiment_text
    _HAS_SENTIMENT = True
except ImportError:
    _HAS_SENTIMENT = False
    analyze_sentiment_text = lambda t: {"error": "sentiment module not available"}

try:
    from app.news_crawler import start_news_crawler, stop_news_crawler
    _HAS_NEWS_CRAWLER = True
except ImportError:
    _HAS_NEWS_CRAWLER = False
    start_news_crawler = lambda: None
    stop_news_crawler  = lambda: None

logger = logging.getLogger("ai-service")

SUPPORTED_SYMBOLS = ["BTCUSDT", "ETHUSDT"]

app = FastAPI(title="AI Service — SAFE-Alert v2")

# FIXED: Add CORS middleware to allow frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (localhost:5173, etc)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



@app.on_event("startup")
def startup_event():
    # Ensure Kafka topics exist (skip if Kafka not available or times out)
    if _HAS_KAFKA_PRODUCER:
        from threading import Thread
        def _setup_kafka():
            try:
                create_startup_topics()
            except Exception as e:
                logger.warning(f"Failed to create startup topics (Kafka may not be available): {e}")

        # Run in background thread with timeout
        t = Thread(target=_setup_kafka, daemon=True)
        t.start()
    else:
        logger.warning("kafka_producer module not found — Kafka producer skipped")

    if _HAS_KAFKA_CONSUMER:
        t = Thread(target=_start_consumer, daemon=True)
        t.start()
    else:
        logger.warning("kafka_consumer module not found — V1 Kafka consumer skipped")

    # start market cache consumer to populate recent klines from Kafka
    start_market_cache_thread()
    schedule_causal_job()

    # SAFE-Alert v2: start live news crawler (populates MongoDB for NLP features)
    if _HAS_NEWS_CRAWLER:
        start_news_crawler()

    # SAFE-Alert v2: schedule hourly inference for all supported symbols
    _start_v2_scheduler()


@app.on_event('shutdown')
def shutdown_event():
    try:
        close_producer()
    except Exception:
        pass
    stop_news_crawler()
    _stop_v2_scheduler()


# ──────────────────────────────────────────────────────────────
# SAFE-Alert v2 — Scheduler
# ──────────────────────────────────────────────────────────────

_v2_scheduler = None


def _v2_result_to_kafka_payload(sym: str, result: dict) -> dict:
    """
    Transform V2 SAFE-Alert inference result into the payload format expected by
    core-service AiInsightsConsumer and InsightsList frontend component.

    Frontend fields used:
      payload.meta.{symbol, market_sentiment_label, analyzed_articles}
      payload.predictions[0].{symbol, current_price, explanation, forecast, causal_analysis,
                               news_impact_analysis, technical_indicators}
    """
    h1h   = result.get("horizon_1h", {})
    h4h   = result.get("horizon_4h", {})
    alert = result.get("alert", {})
    top   = result.get("top_inputs", {})

    # Current price from market cache
    current_price = 0.0
    try:
        from app.market_cache import get_candles
        candles = get_candles(sym, limit=1, interval="1h")
        if candles:
            current_price = float(candles[-1].get("close", 0.0))
    except Exception:
        pass

    # Signal → direction label
    def signal_to_direction(sig: str) -> str:
        return {"BUY": "UP", "SELL": "DOWN"}.get(sig, "SIDEWAYS")

    def signal_to_sentiment(sig: str) -> str:
        return {"BUY": "BULLISH", "SELL": "BEARISH"}.get(sig, "NEUTRAL")

    sig_1h  = h1h.get("signal", "HOLD")
    sig_4h  = h4h.get("signal", "HOLD")
    conf_1h = float(h1h.get("confidence", 0.0))
    conf_4h = float(h4h.get("confidence", 0.0))
    prob_1h = float(h1h.get("final_prob", 0.5))
    prob_4h = float(h4h.get("final_prob", 0.5))

    # Expected price: use prob deviation as a proxy for return magnitude
    # BUY: slight positive offset; SELL: slight negative; HOLD: flat
    def expected_price(current: float, prob: float, sig: str) -> float:
        if current <= 0:
            return current
        mag = abs(prob - 0.5) * 0.04   # max ~2% at full confidence
        if sig == "BUY":
            return round(current * (1 + mag), 2)
        if sig == "SELL":
            return round(current * (1 - mag), 2)
        return round(current, 2)

    exp_1h = expected_price(current_price, prob_1h, sig_1h)
    exp_4h = expected_price(current_price, prob_4h, sig_4h)
    chg_4h = ((exp_4h - current_price) / current_price * 100) if current_price > 0 else 0.0

    # Volatility bucket from bb_width proxy (not available directly → use confidence)
    volatility = "low" if conf_1h < 0.25 else ("high" if conf_1h > 0.55 else "medium")

    # Top selected news articles
    selected_news = h1h.get("selected_news", [])
    key_event = selected_news[0]["title"] if selected_news else None

    # Consensus sentiment
    consensus_signal = alert.get("signal") or sig_1h
    sentiment_label  = signal_to_sentiment(consensus_signal)

    news_count = int(top.get("news_count") or 0)

    return {
        "type": "aggregated_prediction",
        "meta": {
            "symbol":                 sym,
            "market_sentiment_label": sentiment_label,
            "analyzed_articles":      news_count,
            "model":                  h1h.get("model_used", "SAFE-Alert v2"),
            "data_source":            "live",
            "alert_level":            alert.get("level", "abstain"),
        },
        "predictions": [
            {
                "symbol":        sym,
                "current_price": current_price,
                "explanation":   h1h.get("explanation", ""),
                "forecast": {
                    "next_1h": {
                        "direction":      signal_to_direction(sig_1h),
                        "expected_price": exp_1h,
                        "confidence":     round(conf_1h * 100, 1),
                        "volatility":     volatility,
                        "prob_up":        round(prob_1h, 4),
                    },
                    "next_4h": {
                        "direction":           signal_to_direction(sig_4h),
                        "expected_price":      exp_4h,
                        "price_change_percent": round(chg_4h, 2),
                        "expected_range": {
                            "low":  round(min(current_price, exp_4h) * 0.98, 2),
                            "high": round(max(current_price, exp_4h) * 1.02, 2),
                        },
                        "confidence": round(conf_4h * 100, 1),
                    },
                    "next_24h": {
                        "direction":           signal_to_direction(sig_4h),
                        "expected_price":      exp_4h,
                        "price_change_percent": round(chg_4h, 2),
                        "expected_range": {
                            "low":  round(min(current_price, exp_4h) * 0.98, 2),
                            "high": round(max(current_price, exp_4h) * 1.02, 2),
                        },
                        "confidence": round(conf_4h * 100, 1),
                        "note": "Backward-compatible alias of next_4h.",
                    },
                },
                "causal_analysis": {
                    "key_event":      key_event,
                    "explanation_vi": h1h.get("explanation", ""),
                    "shap_factors":   h1h.get("factors_text", ""),
                },
                "news_impact_analysis": {
                    "overall_sentiment": sentiment_label,
                    "vader_mean":        top.get("vader_mean"),
                    "bullish_ratio":     top.get("bullish_ratio"),
                    "selected_news":     selected_news,
                },
                "technical_indicators": {
                    "rsi":          top.get("rsi_14"),
                    "macd_hist":    top.get("macd_hist"),
                    "bb_pos":       top.get("bb_pos"),
                    "stoch_rsi":    top.get("stoch_rsi"),
                    "volume_spike": top.get("volume_spike"),
                },
            }
        ],
    }


def _run_v2_inference_all():
    """Scheduled job: run live inference for all supported symbols."""
    from app.v2.pipelines.live_infer import run_live_inference
    from app.kafka_producer import publish_insight
    for sym in SUPPORTED_SYMBOLS:
        try:
            result = run_live_inference(sym)
            alert  = result.get("alert", {})
            logger.info(
                "[v2-scheduler] %s → signal=%s  alert=%s  level=%s",
                sym,
                result.get("horizon_1h", {}).get("signal"),
                alert.get("alert"),
                alert.get("level"),
            )
            # Publish to Kafka → core-service → frontend
            kafka_payload = _v2_result_to_kafka_payload(sym, result)
            publish_insight(kafka_payload)
            logger.info("[v2-scheduler] %s published to ai_insights topic", sym)
        except Exception as e:
            logger.warning("[v2-scheduler] %s failed: %s", sym, e)


def _start_v2_scheduler():
    global _v2_scheduler
    try:
        from apscheduler.schedulers.background import BackgroundScheduler
        _v2_scheduler = BackgroundScheduler()
        # Run every hour at minute 2 (gives Kafka time to flush the candle)
        _v2_scheduler.add_job(_run_v2_inference_all, "cron", minute=2)
        _v2_scheduler.start()
        logger.info("SAFE-Alert v2 scheduler started (runs hourly at :02)")
    except Exception as e:
        logger.warning("Could not start v2 scheduler: %s", e)


def _stop_v2_scheduler():
    global _v2_scheduler
    if _v2_scheduler:
        try:
            _v2_scheduler.shutdown(wait=False)
        except Exception:
            pass


@app.get('/health')
def health():
    return {"alive": True}


# ──────────────────────────────────────────────────────────────
# SAFE-Alert v2 — Endpoints
# ──────────────────────────────────────────────────────────────

@app.get("/v2/signal/{symbol}")
def v2_signal_live(symbol: str):
    """
    SAFE-Alert v2: Run live inference for a symbol.

    Flow: market_cache → features → SAFEAlertNet → Eq.26 alert decision.
    Response time: ~2-5 seconds depending on market/news availability.
    """
    sym = symbol.upper()
    if sym not in SUPPORTED_SYMBOLS:
        raise HTTPException(
            status_code=400,
            detail=f"Symbol '{sym}' not supported. Choose from: {SUPPORTED_SYMBOLS}",
        )
    try:
        from app.v2.pipelines.live_infer import run_live_inference
        return run_live_inference(sym)
    except ValueError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.exception("v2 live inference error for %s", sym)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/v2/signal/{symbol}/cached")
def v2_signal_cached(symbol: str):
    """
    SAFE-Alert v2: Return last cached signal (from hourly scheduler).
    Much faster than /v2/signal/{symbol} — no model computation.
    """
    sym = symbol.upper()
    from app.v2.pipelines.live_infer import get_cached_signal
    cached = get_cached_signal(sym)
    if cached is None:
        raise HTTPException(
            status_code=404,
            detail=f"No cached signal for {sym}. Call /v2/signal/{sym} first or wait for scheduler.",
        )
    return cached


@app.post("/v2/signal/run-now")
def v2_run_now():
    """
    SAFE-Alert v2: Manually trigger inference for all supported symbols immediately.
    Runs in background thread — returns immediately.
    """
    t = Thread(target=_run_v2_inference_all, daemon=True)
    t.start()
    return {"status": "ok", "message": f"Inference triggered for {SUPPORTED_SYMBOLS}"}


@app.get('/predictions/latest')
def get_latest_prediction():
    """Get the latest Deep Learning prediction result."""
    try:
        from app.modules.news_aggregator import get_last_result
        return get_last_result()
    except ImportError:
        raise HTTPException(status_code=503, detail="news_aggregator module not available")


@app.get('/debug/market_cache/{symbol}')
def debug_market_cache(symbol: str):
    """Return recent cached candles for a symbol (oldest->newest). Dev-only endpoint."""
    try:
        from app.market_cache import get_candles
        candles = get_candles(symbol.upper(), 200)
        return {"symbol": symbol.upper(), "count": len(candles), "candles": candles}
    except Exception as e:
        return {"error": str(e)}


@app.post('/sentiment/')
def sentiment_endpoint(text: str):
    return analyze_sentiment_text(text)


@app.post('/causal/run-now')
def causal_now():
    res = run_causal_now()
    return {"status": "ok", "result": res}


@app.post('/causal/for-news')
def causal_for_news(payload: dict):
    """Run causal analysis for a provided news payload (best-effort).

    Example payload shape:
    {
      "url": "...",
      "title": "...",
      "published_at": "2025-11-28T08:00:00Z",
      "raw": { ... }
    }
    """
    try:
        res = analyze_event_causal(payload)
        return {"status": "ok", "insight": res}
    except Exception as e:
        return {"status": "error", "reason": str(e)}


@app.get('/debug/buffer-status')
def get_buffer_status():
    """Get current news buffer status."""
    try:
        from app.modules.news_aggregator import get_buffer_status
        return get_buffer_status()
    except ImportError:
        raise HTTPException(status_code=503, detail="news_aggregator module not available")


@app.post('/debug/reset-buffer')
def reset_news_buffer():
    """Reset the news buffer and seen URLs. Use this to clear duplicate detection."""
    try:
        from app.modules.news_aggregator import reset_buffer
        reset_buffer()
        return {"status": "ok", "message": "News buffer and seen URLs cleared"}
    except ImportError:
        raise HTTPException(status_code=503, detail="news_aggregator module not available")


@app.post('/internal/analyze-investment')
def api_analyze_investment(payload: dict):
    """
    Internal API for Investment Service to get specific advice.
    Payload: { symbol, amount, buy_price, target_sell_time, current_time }
    """
    try:
        from app.modules.investment_advisor import analyze_investment
        return analyze_investment(payload)
    except ImportError:
        raise HTTPException(status_code=503, detail="investment_advisor module not available")


@app.post('/admin/train-model')
def trigger_model_training():
    """
    Manually trigger model training (admin only).
    This will collect data from Kafka and train the model in background.
    """
    try:
        from background_trainer import get_background_trainer
        trainer = get_background_trainer()
        
        if trainer.is_training:
            return {"status": "error", "message": "Training already in progress"}
        
        trainer.run_once()
        return {
            "status": "ok", 
            "message": "Model training started in background",
            "last_training": trainer.last_training_time
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.get('/admin/training-status')
def get_training_status():
    """Get current training status."""
    try:
        from background_trainer import get_background_trainer
        trainer = get_background_trainer()
        return {
            "is_training": trainer.is_training,
            "last_training_time": trainer.last_training_time,
            "training_interval_hours": trainer.training_interval / 3600
        }
    except Exception as e:
        return {"error": str(e)}

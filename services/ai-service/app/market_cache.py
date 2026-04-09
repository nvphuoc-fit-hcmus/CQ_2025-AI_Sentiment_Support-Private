from collections import defaultdict, deque
import os
import json
import threading
import time
from confluent_kafka import Consumer

# Simple in-memory market data cache populated from Kafka topic (market_data)
# Keeps recent 1m candles per symbol in a deque for fast lookup by ai-service.

KAFKA_BROKER = os.getenv("KAFKA_BROKERS", "localhost:9092")
MARKET_TOPIC = os.getenv("MARKET_DATA_TOPIC", "market.prices")  # Match stream-ingester topic
GROUP_ID = os.getenv("MARKET_CACHE_GROUP", "ai-service-market-cache-v3")
MAX_PER_SYMBOL = int(os.getenv("MARKET_CACHE_MAX", "2000"))

_cache = defaultdict(lambda: deque(maxlen=MAX_PER_SYMBOL))
_running = False


def _start_consumer():
    global _running
    conf = {
        "bootstrap.servers": KAFKA_BROKER,
        "group.id": GROUP_ID,
        "auto.offset.reset": "earliest",
    }
    consumer = Consumer(conf)
    consumer.subscribe([MARKET_TOPIC])
    _running = True
    try:
        while _running:
            msg = consumer.poll(1.0)
            if msg is None:
                continue
            if msg.error():
                # ignore partition EOF
                continue
            try:
                payload = msg.value().decode("utf-8")
                j = json.loads(payload)
                # Support multiple incoming shapes. Common shapes:
                # 1) {"symbol":"BTCUSDT", "interval":"1m", "time":..., "close":...}
                # 2) {"symbol":"BTCUSDT", "kline":{...}, "interval": "1m"}
                
                # unwrap nested 'data' (some producers wrap messages)
                if isinstance(j, dict) and 'data' in j and isinstance(j['data'], dict):
                    j = j['data']

                # primary symbol and interval
                sym = j.get('symbol') or j.get('s')
                interval = j.get('interval') or j.get('i') or '1m' # Default to 1m if missing

                # extract full OHLCV
                t = None
                open_p = None
                high_p = None
                low_p = None
                close = None
                vol = None
                
                # if kline nested
                k = j.get('kline') or j.get('k')
                if isinstance(k, dict):
                    t = k.get('closeTime') or k.get('close_time') or k.get('openTime') or k.get('open_time') or k.get('t') or k.get('T')
                    close = k.get('close') or k.get('c')
                    open_p = k.get('open') or k.get('o')
                    high_p = k.get('high') or k.get('h')
                    low_p = k.get('low') or k.get('l')
                    vol = k.get('volume') or k.get('v')
                    # Interval might be inside kline object in some streams
                    if not interval:
                        interval = k.get('interval') or k.get('i')
                else:
                    t = j.get('time') or j.get('ts') or j.get('open_time') or j.get('t')
                    close = j.get('close') or j.get('c') or j.get('price')
                    open_p = j.get('open') or j.get('o') or close 
                    high_p = j.get('high') or j.get('h') or close
                    low_p = j.get('low') or j.get('l') or close
                    vol = j.get('volume') or j.get('v') or 0

                # normalize numeric types and timestamps (ms -> s)
                try:
                    if t is not None:
                        t = int(t)
                        if t > 1_000_000_000_000: t = int(t / 1000)
                        
                    close = float(close) if close is not None else 0.0
                    open_p = float(open_p) if open_p is not None else close
                    high_p = float(high_p) if high_p is not None else close
                    low_p = float(low_p) if low_p is not None else close
                    vol = float(vol) if vol is not None else 0.0
                except Exception:
                    continue

                if sym and t:
                    # Key format: SYMBOL_INTERVAL (e.g., BTCUSDT_1h)
                    key = f"{sym.upper()}_{interval}"
                    
                    _cache[key].append({
                        "time": int(t),
                        "open": open_p, 
                        "high": high_p, 
                        "low": low_p, 
                        "close": close, 
                        "volume": vol
                    })

            except Exception:
                continue
    finally:
        try:
            consumer.close()
        except Exception:
            pass


_PREFETCH_SYMBOLS   = ["BTCUSDT", "ETHUSDT"]
_PREFETCH_INTERVALS = ["1h", "1d"]   # intervals needed by V2 models
_PREFETCH_LIMIT     = 250


def _prefetch_from_binance(symbols=None, intervals=None, limit=_PREFETCH_LIMIT):
    """
    Fetch historical klines directly from Binance REST API and seed the cache.
    Runs at startup so that V2 inference has enough candles even after a restart.
    Silently skips on network error.
    """
    import urllib.request
    syms  = symbols  or _PREFETCH_SYMBOLS
    ivals = intervals or _PREFETCH_INTERVALS

    for sym in syms:
        for interval in ivals:
            key = f"{sym.upper()}_{interval}"
            if len(_cache[key]) >= limit:
                continue   # already populated (e.g. from Kafka earliest offset)
            try:
                url = (
                    f"https://api.binance.com/api/v3/klines"
                    f"?symbol={sym.upper()}&interval={interval}&limit={limit}"
                )
                with urllib.request.urlopen(url, timeout=10) as resp:
                    rows = json.loads(resp.read())
                # Each row: [openTime,open,high,low,close,vol,closeTime,...]
                for row in rows:
                    t_ms = int(row[6])              # closeTime ms
                    _cache[key].append({
                        "time":   int(t_ms / 1000),
                        "open":   float(row[1]),
                        "high":   float(row[2]),
                        "low":    float(row[3]),
                        "close":  float(row[4]),
                        "volume": float(row[5]),
                    })
                print(f"[market_cache] prefetched {len(rows)} {interval} candles for {sym}")
            except Exception as e:
                print(f"[market_cache] prefetch failed for {sym}/{interval}: {e}")


def start_market_cache_thread():
    # Seed cache with historical data before starting live consumer
    _prefetch_from_binance()
    t = threading.Thread(target=_start_consumer, daemon=True)
    t.start()


def get_candles(symbol: str, limit: int, interval: str = "15min"):
    """
    Return up to `limit` dicts (time, open, high, low, close, volume) for given symbol and interval.
    
    Args:
        symbol: e.g. "BTCUSDT"
        limit: max number of candles
        interval: e.g. "1m", "1h", "1d". Note: "15min" in pandas -> "15m" in binance usually.
                  This function expects the interval format used in Kafka topics (e.g., "1h").
                  Users must ensure mapping.
    """
    # Quick fix for mapping Pandas offsets to Binance intervals if needed
    # But for now assuming exact match.
    # Note: inference.py passes env TIMEFRAME which might be "15min" (pandas style) or "15m" (binance style).
    
    # Map common pandas aliases to likely binance keys if not found
    key = f"{symbol.upper()}_{interval}"
    deq = _cache.get(key)
    
    if not deq:
        # Try fallback mapping if key not found (e.g. 15min vs 15m)
        if "min" in interval:
            alt_interval = interval.replace("min", "m")
            deq = _cache.get(f"{symbol.upper()}_{alt_interval}")
            
    if not deq:
        return []
        
    res = list(deq)
    # Sort just in case? Usually appended in order.
    # res.sort(key=lambda x: x['time']) 
    
    if limit and len(res) > limit:
        return res[-limit:]
    return res

import os
import asyncio
import hashlib
import logging
import time
from functools import partial
from urllib.parse import urlparse, urljoin
from datetime import datetime, timezone, timedelta

from fastapi import FastAPI
from pydantic import BaseModel
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import redis
from email.utils import parsedate_to_datetime

from app.db import init_db, get_active_sources, update_source_status, save_article, get_db
from app.services.discovery import get_links_from_rss, get_links_from_html
from app.services.extractor import smart_extract
from app.kafka_producer import produce_news, close_producer, create_startup_topics

LOG = logging.getLogger("crawler.main")
if not logging.getLogger().hasHandlers():
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s"
    )
    
    # Disable noisy debug logs from libraries
    logging.getLogger("pymongo").setLevel(logging.WARNING)
    logging.getLogger("pymongo.connection").setLevel(logging.WARNING)
    logging.getLogger("pymongo.command").setLevel(logging.WARNING)
    logging.getLogger("pymongo.topology").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("trafilatura").setLevel(logging.WARNING)
    logging.getLogger("trafilatura.main_extractor").setLevel(logging.WARNING)
    logging.getLogger("trafilatura.readability_lxml").setLevel(logging.WARNING)
    logging.getLogger("trafilatura.external").setLevel(logging.WARNING)
    
    # Keep crawler, gemini, and extraction logs at INFO
    logging.getLogger("crawler").setLevel(logging.INFO)
    logging.getLogger("crawler.gemini").setLevel(logging.INFO)
    logging.getLogger("crawler.extraction").setLevel(logging.INFO)

app = FastAPI(title="Crawler Service")

class CrawlRequest(BaseModel):
    url: str

# Redis & Config
REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379/0")
CRAWL_INTERVAL = int(os.getenv("CRAWL_INTERVAL_SECONDS", "300"))
REDIS_TTL = int(os.getenv("CRAWLER_REDIS_TTL", str(7 * 24 * 3600)))
CRAWLER_PER_HOST_DELAY = float(os.getenv("CRAWLER_PER_HOST_DELAY", "2.0"))
CRAWLER_HOST_COOLDOWN = int(os.getenv('CRAWLER_HOST_COOLDOWN_SECONDS', '3600'))

redis_client = redis.from_url(REDIS_URL)

_per_host_last_request: dict[str, float] = {}
_host_block_until: dict[str, float] = {}

def _make_session() -> requests.Session:
    s = requests.Session()
    retries = Retry(total=2, backoff_factor=0.5, status_forcelist=[500, 502, 503, 504], allowed_methods=["HEAD", "GET", "OPTIONS"])
    adapter = HTTPAdapter(max_retries=retries)
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    s.headers.update({
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Upgrade-Insecure-Requests": "1"
    })
    return s

session = _make_session()

def _url_hash(url: str) -> str:
    return hashlib.sha256(url.encode('utf-8')).hexdigest()

def _is_crypto_related(title: str, content: str) -> bool:
    """Check if article contains crypto-related keywords."""
    title_lower = title.lower()
    content_lower = content.lower()
    
    crypto_keywords = [
        'bitcoin', 'btc', 'ethereum', 'eth', 'crypto', 'cryptocurrency',
        'blockchain', 'binance', 'bnb', 'solana', 'sol', 'cardano', 'ada',
        'ripple', 'xrp', 'dogecoin', 'doge', 'polkadot', 'dot', 'avalanche',
        'avax', 'polygon', 'matic', 'defi', 'nft', 'web3', 'altcoin',
        'stablecoin', 'usdt', 'usdc', 'mining', 'wallet', 'exchange',
        'token', 'coin', 'satoshi', 'halving', 'bull market', 'bear market'
    ]
    
    return any(kw in title_lower or kw in content_lower for kw in crypto_keywords)

async def process_article_task(item: dict):
    """Async task to extract and publish article."""
    url = item.get('link') or item.get('url')
    if not url: return
    
    # Blacklist: Skip Google News redirect URLs (cannot be parsed)
    if 'news.google.com/rss/articles/' in url:
        LOG.debug(f"Skipping Google News redirect: {url[:80]}...")
        return

    key = f"crawler:seen:{_url_hash(url)}"
    if redis_client.exists(key):
         LOG.debug(f"Skipping seen url: {url}")
         return

    # Check directly provided content first (RSS optimization)
    # DISABLE RSS SHORTCUT to force HTML Fetching & Structure Learning behaviors
    # If we have title + long content, we can skip fetching!
    # if item.get('manual_content') and item.get('manual_title') and len(item['manual_content']) > 500:
    #     title_lower = item['manual_title'].lower()
    #     # ... (logic commented out) ...
    #     LOG.info(f"Using direct RSS content for {url}")
    #     produce_news(payload)
    #     redis_client.set(key, 1, ex=REDIS_TTL)
    #     return

    # Fallback to standard Fetch & Extract
    host = urlparse(url).netloc
    
    # Check block
    now = time.monotonic()
    if block_until := _host_block_until.get(host):
        if now < block_until:
            return

    loop = asyncio.get_running_loop()
    try:
        resp = await loop.run_in_executor(None, partial(session.get, url, timeout=20))
        if resp.status_code == 429:
             _host_block_until[host] = time.monotonic() + 60
             return
        resp.raise_for_status()
        html = resp.text
        
        # 2. Extract
        data = await loop.run_in_executor(None, partial(smart_extract, url, html))
        
        if not data or not data.get('title'):
             LOG.warning(f"Skipping {url}: extraction incomplete")
             return
        
        # CRYPTO KEYWORD FILTER
        if not _is_crypto_related(data.get('title', ''), data.get('content', '')):
            LOG.info(f"Skipping non-crypto article: {data.get('title', '')[:50]}... (Extracted but filtered)")
            return
             
        # 3. Publish
        # import time <-- REMOVED
        payload = {
            'source': host.replace("www.", ""),
            'url': url,
            'title': data.get('title'),
            'content': data.get('content'),
            'published_at': data.get('date'),  # Original publish time from article
            'crawl_timestamp': time.time(),    # When we crawled it
            'symbol': data.get('symbols', ['BTCUSDT'])[0] if data.get('symbols') else None,
            'symbols': data.get('symbols', ['BTCUSDT']),  # All related symbols
            'category': data.get('category', 'General'),
            'sentiment': data.get('sentiment'),
            'relevance_score': data.get('relevance_score', 0.5)
        }
        
        produce_news(payload)
        save_article(payload)  # Save detailed content to Mongo
        LOG.info(f"Published & Saved: {payload['title']} ({url})")
        
        redis_client.set(key, 1, ex=REDIS_TTL)
        
    except Exception as e:
        LOG.error(f"Error processing {url}: {e}")

async def crawl_cycle():
    """Discover links from all active sources."""
    sources = get_active_sources()
    if not sources:
        LOG.info("No active sources in DB (crawl_sources collection).")
        return

    LOG.info(f"Starting discovery on {len(sources)} sources.")
    all_items = []
    
    for src in sources:
        url = src['url']
        src_type = src.get('source_type', 'html')
        
        # Check host blocks
        host = urlparse(url).netloc
        now = time.monotonic()
        if block_until := _host_block_until.get(host):
            if now < block_until:
                continue

        # Politeness wait
        last = _per_host_last_request.get(host)
        if last:
            wait = CRAWLER_PER_HOST_DELAY - (now - last)
            if wait > 0:
                await asyncio.sleep(wait)

        # Discovery
        found_items = []
        try:
            loop = asyncio.get_running_loop()
            if src_type == 'rss':
                found_items = await loop.run_in_executor(None, partial(get_links_from_rss, url, session))
            else:
                # HTML discovery requies fetch
                resp = await loop.run_in_executor(None, partial(session.get, url, timeout=20))
                _per_host_last_request[host] = time.monotonic()
                if resp.status_code == 200:
                    found_items = await loop.run_in_executor(None, partial(get_links_from_html, url, resp.text))
            
            update_source_status(url, error=False)
        except Exception as e:
            LOG.error(f"Discovery failed for {url}: {e}")
            update_source_status(url, error=True)
            continue
            
        all_items.extend(found_items)

    # Process links (Dedup happens in process_article_task via Redis)
    # Be gentle with concurrency
    sem = asyncio.Semaphore(10)
    
    async def _sem_task(item):
        async with sem:
            await process_article_task(item)
            
    await asyncio.gather(*[_sem_task(i) for i in all_items])

_bg_task = None

@app.on_event('startup')
async def startup_event():
    LOG.info("Crawler Service v2.0 Starting...")
    init_db()
    create_startup_topics()
    
    loop = asyncio.get_running_loop()
    async def _loop():
        while True:
            try:
                await crawl_cycle()
            except Exception:
                LOG.exception("Cycle failed")
            await asyncio.sleep(CRAWL_INTERVAL)
            
    global _bg_task
    _bg_task = loop.create_task(_loop())

@app.on_event('shutdown')
async def shutdown_event():
    if _bg_task:
        _bg_task.cancel()
    close_producer()
    session.close()
    LOG.info("Crawler Stopped.")

@app.post('/crawl/')
async def crawl_endpoint(req: CrawlRequest):
    asyncio.create_task(process_article_task({'link': req.url}))
    return {"status": "queued"}

@app.get('/health')
def health():
    return {"status": "ok"}


@app.get('/news/latest')
def get_latest_news(limit: int = 100):
    """Fallback endpoint: return latest crawled articles from MongoDB."""
    try:
        lim = max(1, min(int(limit or 100), 500))
        db = get_db()
        docs = list(
            db.news_articles.find(
                {},
                {
                    "_id": 0,
                    "url": 1,
                    "source": 1,
                    "title": 1,
                    "sentiment": 1,
                    "published_at": 1,
                    "created_at": 1,
                },
            )
            .sort("created_at", -1)
            .limit(lim)
        )

        rows = []
        for doc in docs:
            sentiment = doc.get("sentiment")
            sentiment_score = 0.0
            if isinstance(sentiment, (int, float)):
                sentiment_score = float(sentiment)
            elif isinstance(sentiment, dict):
                if isinstance(sentiment.get("score"), (int, float)):
                    sentiment_score = float(sentiment.get("score"))
                else:
                    pos = float(sentiment.get("positive", 0) or 0)
                    neg = float(sentiment.get("negative", 0) or 0)
                    sentiment_score = pos - neg

            rows.append(
                {
                    "time": doc.get("published_at") or doc.get("created_at"),
                    "url": doc.get("url"),
                    "source": doc.get("source"),
                    "title": doc.get("title"),
                    "sentiment_score": sentiment_score,
                    "raw_score": sentiment,
                }
            )

        return {
            "count": len(rows),
            "rows": rows,
            "total": len(rows),
            "source": "crawler_fallback",
        }
    except Exception as e:
        LOG.error(f"Failed to fetch fallback news: {e}")
        return {"count": 0, "rows": [], "total": 0, "source": "crawler_fallback_error"}

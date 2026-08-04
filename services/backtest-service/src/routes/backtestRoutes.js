const express = require('express');
const router = express.Router();
const db = require('../config/db');
const BacktestEngine = require('../backtest/engine');
const { loadHistoricalPredictions } = require('../backtest/historical-predictions');
const axios = require('axios');
const jwt = require('jsonwebtoken');

// Helper: Fetch candles from Binance API
const fetchBinanceCandles = async (symbol, interval, startTime, endTime) => {
    let allCandles = [];
    let currentStart = new Date(startTime).getTime();
    const endTimestamp = new Date(endTime).getTime();

    // Safety limit to prevent infinite loops
    const MAX_REQUESTS = 50;
    let requests = 0;

    console.log(`[Backtest] Fetching Binance data for ${symbol} (${interval}) from ${startTime} to ${endTime}`);

    while (currentStart < endTimestamp && requests < MAX_REQUESTS) {
        try {
            const url = `https://api.binance.com/api/v3/klines`;
            const params = {
                symbol: symbol.toUpperCase(),
                interval: interval,
                startTime: currentStart,
                endTime: endTimestamp,
                limit: 1000
            };

            const res = await axios.get(url, { params });
            const data = res.data;

            if (!data || data.length === 0) break;

            const candles = data.map(c => ({
                time: new Date(c[0]),
                open: parseFloat(c[1]),
                high: parseFloat(c[2]),
                low: parseFloat(c[3]),
                close: parseFloat(c[4]),
                volume: parseFloat(c[5])
            }));

            allCandles = allCandles.concat(candles);

            const lastOpenTime = data[data.length - 1][0];
            if (lastOpenTime >= endTimestamp) break;

            currentStart = lastOpenTime + 1;
            requests++;

            // Rate limiting: wait 100ms between requests
            await new Promise(r => setTimeout(r, 100));
        } catch (e) {
            console.error('[Backtest] Binance fetch error:', e.message);
            break;
        }
    }

    return allCandles;
};

// Helper: Resample candles to larger timeframes
const resampleCandles = (candles, timeframe) => {
    if (!timeframe || timeframe === '1h') return candles;

    const timeMap = {
        '15m': 15 * 60 * 1000,
        '30m': 30 * 60 * 1000,
        '1h': 60 * 60 * 1000,
        '4h': 4 * 60 * 60 * 1000,
        '12h': 12 * 60 * 60 * 1000,
        '1d': 24 * 60 * 60 * 1000,
        '1w': 7 * 24 * 60 * 60 * 1000
    };

    const intervalMs = timeMap[timeframe];

    if (intervalMs < 60 * 60 * 1000) {
        console.warn(`[Backtest] Cannot resample 1h DB data to ${timeframe}. returning empty.`);
        return [];
    }

    if (!intervalMs) return candles;

    const resampled = [];
    let currentBucket = null;

    for (const candle of candles) {
        const candleTime = new Date(candle.time).getTime();
        const bucketStartTime = Math.floor(candleTime / intervalMs) * intervalMs;

        if (!currentBucket || currentBucket.time !== bucketStartTime) {
            if (currentBucket) {
                currentBucket.time = new Date(currentBucket.time);
                resampled.push(currentBucket);
            }

            currentBucket = {
                time: bucketStartTime,
                open: Number(candle.open),
                high: Number(candle.high),
                low: Number(candle.low),
                close: Number(candle.close),
                volume: Number(candle.volume)
            };
        } else {
            currentBucket.high = Math.max(currentBucket.high, Number(candle.high));
            currentBucket.low = Math.min(currentBucket.low, Number(candle.low));
            currentBucket.close = Number(candle.close);
            currentBucket.volume += Number(candle.volume);
        }
    }

    if (currentBucket) {
        currentBucket.time = new Date(currentBucket.time);
        resampled.push(currentBucket);
    }

    return resampled;
};

// Middleware: Auth
const requireAuth = (req, res, next) => {
    const kongUserId = req.headers['x-user-id'];
    if (kongUserId) {
        req.userId = kongUserId;
        return next();
    }

    const authHeader = req.headers.authorization;
    if (authHeader && authHeader.startsWith('Bearer ')) {
        const token = authHeader.split(' ')[1];
        try {
            const decoded = jwt.decode(token);
            if (decoded && decoded.sub) {
                req.userId = decoded.sub;
                return next();
            }
        } catch (e) {
            console.error('Error decoding token', e);
        }
    }

    return res.status(401).json({ error: 'Unauthorized: Missing User Identity' });
};

// GET /v1/backtest/history
router.get('/history', requireAuth, async (req, res) => {
    try {
        const { rows } = await db.query(`
      SELECT id, strategy_name, symbol, start_date, end_date, win_rate, net_profit_percent, created_at
      FROM backtest_results
      WHERE user_id = $1
      ORDER BY created_at DESC
      LIMIT 20
    `, [req.userId]);

        res.json(rows);
    } catch (err) {
        console.error('Error fetching backtest history:', err);
        res.status(500).json({ error: 'Internal Server Error' });
    }
});

// GET /v1/backtest/:id
router.get('/:id', requireAuth, async (req, res) => {
    try {
        const { rows } = await db.query(`
      SELECT * FROM backtest_results
      WHERE id = $1 AND user_id = $2
    `, [req.params.id, req.userId]);

        if (rows.length === 0) {
            return res.status(404).json({ error: 'Backtest not found' });
        }

        res.json(rows[0]);
    } catch (err) {
        console.error('Error fetching backtest detail:', err);
        res.status(500).json({ error: 'Internal Server Error' });
    }
});

// POST /v1/backtest/run
router.post('/run', requireAuth, async (req, res) => {
    const { strategy, symbol, start_date, end_date, initial_capital } = req.body;

    if (!strategy || !symbol || !start_date || !end_date) {
        return res.status(400).json({ error: 'Missing required parameters' });
    }

    try {
        console.log(`[BACKTEST] Starting backtest for ${symbol} from ${start_date} to ${end_date}`);

        // --- 1. FETCH CANDLES (Binance -> DB Fallback) ---
        let processedCandles = [];
        const timeframe = strategy.timeframe || '1h';

        // Try Binance first
        try {
            processedCandles = await fetchBinanceCandles(symbol, timeframe, start_date, end_date);
            if (processedCandles.length > 0) {
                console.log(`[Backtest] Loaded ${processedCandles.length} candles from Binance.`);
            }
        } catch (binanceErr) {
            console.warn('[Backtest] Binance fetch failed:', binanceErr.message);
        }

        // Fallback to TimescaleDB (market_klines table in timeseriesdb)
        if (processedCandles.length === 0) {
            console.log('[Backtest] Fetching from TimescaleDB (Fallback)...');

            // Connect to TimescaleDB for market data
            const { Pool } = require('pg');
            const tsPool = new Pool({
                host: process.env.TIMESCALE_HOST || 'timescaledb',
                port: 5432,
                database: 'timeseriesdb',
                user: process.env.TIMESCALE_USER || 'dev',
                password: process.env.TIMESCALE_PASSWORD || 'dev'
            });

            try {
                const candlesResult = await tsPool.query(
                    `SELECT * FROM market_klines 
                   WHERE symbol = $1 AND time >= $2 AND time <= $3 ORDER BY time ASC`,
                    [symbol.toUpperCase(), start_date, end_date]
                );

                processedCandles = resampleCandles(candlesResult.rows, timeframe);
                console.log(`[Backtest] Loaded ${processedCandles.length} candles from TimescaleDB.`);
            } finally {
                await tsPool.end();
            }
        }

        if (processedCandles.length < 50) {
            return res.status(400).json({ error: "Insufficient historical data (min 50 candles required)." });
        }

        // --- 2. FETCH AI PREDICTIONS & NEWS ---
        // For backtest, we need ALL predictions in the date range, not just latest
        // Query TimescaleDB directly for historical predictions
        const { Pool } = require('pg');
        const tsPool = new Pool({
            host: process.env.TIMESCALE_HOST || 'timescaledb',
            port: 5432,
            database: 'timeseriesdb',
            user: process.env.TIMESCALE_USER || 'dev',
            password: process.env.TIMESCALE_PASSWORD || 'dev'
        });

        let predictions = [];
        try {
            const predictionsQuery = `
              SELECT 
                time,
                symbol,
                payload->'predictions'->0 as prediction_data
              FROM ai_insights
              WHERE type = 'aggregated_prediction'
                AND time >= $1 AND time <= $2
                AND symbol = $3
              ORDER BY time ASC
            `;

            const predictionsRes = await tsPool.query(predictionsQuery, [start_date, end_date, symbol.toUpperCase()]);

            predictions = predictionsRes.rows.map(row => ({
                time: row.time,
                forecast: row.prediction_data.forecast,
                volatility: row.prediction_data.volatility,
                direction: row.prediction_data.forecast?.next_1h?.direction,
                confidence: row.prediction_data.forecast?.next_1h?.confidence
            }));

            console.log(`[BACKTEST] Loaded ${predictions.length} predictions from TimescaleDB`);
        } catch (predErr) {
            console.error('[BACKTEST] Failed to fetch predictions:', predErr.message);
        }

        const modelPredictions = loadHistoricalPredictions({
            symbol,
            startDate: start_date,
            endDate: end_date,
        });
        if (modelPredictions.length > 0) {
            predictions = modelPredictions;
            console.log(`[BACKTEST] Loaded ${predictions.length} causal SAFE-Alert predictions from model replay`);
        }

        // Fetch news sentiment data from TimescaleDB
        let newsData = [];
        try {
            const newsPool = new Pool({
                host: process.env.TIMESCALE_HOST || 'timescaledb',
                port: 5432,
                database: 'timeseriesdb',
                user: process.env.TIMESCALE_USER || 'dev',
                password: process.env.TIMESCALE_PASSWORD || 'dev'
            });

            const newsQuery = `
              SELECT time, sentiment_score, title, url, raw_score
              FROM news_sentiment
              WHERE time >= $1 AND time <= $2
                AND (
                  raw_score->'symbols' ? $3
                  OR raw_score->'symbols' ? 'ALL'
                  OR raw_score->'coins' ? $3
                  OR raw_score->>'tag' ILIKE '%' || $3 || '%'
                  OR raw_score->>'origin' = 'articles_max.csv'
                  OR raw_score IS NULL
                )
              ORDER BY time ASC
            `;

            const newsRes = await newsPool.query(newsQuery, [start_date, end_date, symbol.toUpperCase()]);
            newsData = newsRes.rows;
            console.log(`[BACKTEST] Loaded ${newsData.length} news items from TimescaleDB`);

            await newsPool.end();
        } catch (newsErr) {
            console.warn('[BACKTEST] Failed to fetch news data:', newsErr.message);
        }

        console.log(`[BACKTEST] Aux Data: ${predictions.length} predictions, ${newsData.length} news items`);

        // --- 3. RUN ENGINE ---
        const engine = new BacktestEngine(
            strategy,
            {
                candles: processedCandles,
                predictions: predictions,
                news: newsData
            },
            initial_capital || 10000
        );

        const results = engine.run();

        if (results.error) {
            return res.status(400).json({ error: results.error });
        }

        const timeframeMs = {
            '15m': 15 * 60 * 1000, '30m': 30 * 60 * 1000,
            '1h': 60 * 60 * 1000, '4h': 4 * 60 * 60 * 1000,
            '12h': 12 * 60 * 60 * 1000, '1d': 24 * 60 * 60 * 1000,
            '1w': 7 * 24 * 60 * 60 * 1000,
        };
        const bucketMs = timeframeMs[timeframe] || timeframeMs['1h'];
        const newsBuckets = new Map();
        for (const item of newsData) {
            const itemTime = new Date(item.time).getTime();
            if (!Number.isFinite(itemTime)) continue;
            const bucketTime = Math.floor(itemTime / bucketMs) * bucketMs;
            if (!newsBuckets.has(bucketTime)) {
                newsBuckets.set(bucketTime, {
                    time: new Date(bucketTime).toISOString(),
                    count: 0, sentiment_sum: 0, articles: [],
                });
            }
            const bucket = newsBuckets.get(bucketTime);
            bucket.count += 1;
            bucket.sentiment_sum += Number(item.sentiment_score || 0);
            if (bucket.articles.length < 5) {
                bucket.articles.push({
                    title: item.title, url: item.url,
                    sentiment_score: Number(item.sentiment_score || 0),
                    content: item.raw_score?.content || item.raw_score?.summary || '',
                });
            }
        }
        results.news_count = newsData.length;
        results.news_timeline = [...newsBuckets.values()].map(bucket => ({
            time: bucket.time,
            count: bucket.count,
            average_sentiment: bucket.count ? bucket.sentiment_sum / bucket.count : 0,
            articles: bucket.articles,
        })).sort((a, b) => new Date(a.time) - new Date(b.time));

        // --- 4. SAVE RESULTS ---
        const insertQuery = `
              INSERT INTO backtest_results (
                user_id, strategy_name, strategy_config, symbol, start_date, end_date, initial_capital,
                total_trades, winning_trades, losing_trades, win_rate, 
                total_profit, total_loss, net_profit, net_profit_percent,
                max_drawdown, sharpe_ratio, 
                trades, equity_curve, execution_time_ms, data_points_analyzed,
                news_count, news_timeline
              ) VALUES (
                $1, $2, $3, $4, $5, $6, $7,
                $8, $9, $10, $11,
                $12, $13, $14, $15,
                $16, $17,
                $18, $19, $20, $21, $22, $23
              ) RETURNING id
            `;

        const saved = await db.query(insertQuery, [
            req.userId,
            strategy.name || 'Untitled Strategy',
            JSON.stringify(strategy),
            symbol,
            start_date,
            end_date,
            initial_capital,
            results.total_trades,
            results.winning_trades,
            results.losing_trades,
            results.win_rate,
            results.total_profit,
            results.total_loss,
            results.net_profit,
            results.net_profit_percent,
            results.max_drawdown,
            results.sharpe_ratio,
            JSON.stringify(results.trades),
            JSON.stringify(results.equity_curve),
            results.execution_time_ms,
            results.data_points_analyzed,
            results.news_count,
            JSON.stringify(results.news_timeline)
        ]);

        console.log(`[BACKTEST] Completed successfully. ID: ${saved.rows[0].id}`);

        res.json({ status: 'success', results, id: saved.rows[0].id });


    } catch (err) {
        console.error('Error running backtest:', err);
        res.status(500).json({ error: 'An unexpected error occurred', details: err.message });
    }
});

module.exports = router;

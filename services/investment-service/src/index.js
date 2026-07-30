const express = require('express');
const { Pool } = require('pg');
const { Kafka } = require('kafkajs');
const WebSocket = require('ws');
const cron = require('node-cron');
const http = require('http');
const axios = require('axios');

const app = express();
const server = http.createServer(app);
const wss = new WebSocket.Server({ server });

app.use(express.json());

// Database connection
const pool = new Pool({
    host: process.env.POSTGRES_HOST || 'localhost',
    port: 5432,
    database: process.env.POSTGRES_DB || 'investment_db',
    user: process.env.POSTGRES_USER || 'dev',
    password: process.env.POSTGRES_PASSWORD || 'dev',
});

// Kafka setup
const kafka = new Kafka({
    clientId: 'investment-service',
    brokers: (process.env.KAFKA_BROKERS || 'localhost:9092').split(',')
});

const producer = kafka.producer();
const consumer = kafka.consumer({ groupId: 'investment-service-group-v3' });

// Pending requests map
const pendingAnalysisRequests = new Map(); // requestId -> { resolve, reject, timeout }

// Initialize database
async function initDB() {
    const client = await pool.connect();
    try {
        await client.query(`
      CREATE TABLE IF NOT EXISTS investments (
        id SERIAL PRIMARY KEY,
        user_id VARCHAR(255) NOT NULL,
        symbol VARCHAR(20) NOT NULL,
        usdt_amount DECIMAL(20, 8) NOT NULL,
        coin_amount DECIMAL(20, 8) NOT NULL,
        buy_price DECIMAL(20, 8) NOT NULL,
        buy_time TIMESTAMP NOT NULL DEFAULT NOW(),
        sell_price DECIMAL(20, 8),
        sell_time TIMESTAMP,
        target_sell_time TIMESTAMP NOT NULL,
        status VARCHAR(20) NOT NULL DEFAULT 'active',
        ai_prediction JSONB,
        ai_advice TEXT,
        actual_profit_usdt DECIMAL(20, 8),
        predicted_profit_usdt DECIMAL(20, 8),
        created_at TIMESTAMP NOT NULL DEFAULT NOW(),
        updated_at TIMESTAMP NOT NULL DEFAULT NOW()
      );

      CREATE INDEX IF NOT EXISTS idx_investments_user ON investments(user_id);
      CREATE INDEX IF NOT EXISTS idx_investments_status ON investments(status);
      CREATE INDEX IF NOT EXISTS idx_investments_target_sell ON investments(target_sell_time);
    `);
        console.log('[DB] Investment database initialized');
    } finally {
        client.release();
    }
}

// Get current price from stream service
async function getCurrentPrice(symbol) {
    try {
        const response = await axios.get(`http://core-service:3000/v1/klines?symbol=${symbol}&interval=1m&limit=1`);
        if (response.data && response.data.length > 0) {
            return parseFloat(response.data[0].close);
        }
        throw new Error('No price data found');
    } catch (error) {
        console.error(`[ERROR] Failed to get price for ${symbol}:`, error.message);
        return 50000 + Math.random() * 1000;
    }
}

// Kafka consumer for investment analysis results
async function consumeKafkaTopics() {
    try {
        await consumer.connect();
        // Subscribe only to investment analysis results
        await consumer.subscribe({ topics: ['investment.analysis.result'], fromBeginning: false });

        await consumer.run({
            eachMessage: async ({ topic, partition, message }) => {
                try {
                    const payload = JSON.parse(message.value.toString());

                    if (topic === 'investment.analysis.result') {
                        // Handle analysis result from AI service
                        const requestId = payload.requestId;
                        if (requestId && pendingAnalysisRequests.has(requestId)) {
                            const { resolve, timeout } = pendingAnalysisRequests.get(requestId);
                            clearTimeout(timeout);
                            pendingAnalysisRequests.delete(requestId);
                            resolve(payload);
                            console.log(`[KAFKA] Resolved analysis request ${requestId}`);
                        }
                    }
                } catch (error) {
                    console.error('[KAFKA ERROR]', error);
                }
            },
        });
        console.log('[KAFKA] Consumer connected and listening to investment.analysis.result');
    } catch (err) {
        console.warn('[KAFKA] Consumer setup failed (non-fatal):', err.message);
        console.warn('[KAFKA] Service will continue without Kafka consumer - analysis will use fallback mode');
    }
}

// Fetch latest prediction for a specific symbol from Core Service
async function fetchPredictionForSymbol(symbol) {
    try {
        // Query for latest prediction of this specific symbol
        const url = `http://core-service:3000/v1/insights/internal?type=aggregated_prediction&limit=10`;
        const res = await axios.get(url);

        if (res.data && res.data.rows && res.data.rows.length > 0) {
            // Find the most recent prediction for this symbol
            for (const row of res.data.rows) {
                if (row.payload && row.payload.predictions && Array.isArray(row.payload.predictions)) {
                    const pred = row.payload.predictions.find(p => p.symbol === symbol);
                    if (pred) {
                        console.log(`[AI-FETCH] Found prediction for ${symbol} from ${row.time}`);
                        return pred;
                    }
                }
            }
            console.warn(`[AI-FETCH] No prediction found for ${symbol} in last 10 predictions`);
            return null;
        }

        console.warn('[AI-FETCH] No predictions available in database');
        return null;
    } catch (e) {
        console.error(`[AI-FETCH ERROR] Failed to fetch prediction for ${symbol}:`, e.message);
        return null;
    }
}

// Fallback: query ai-service in-memory cache directly
async function fetchPredictionFromAIService(symbol) {
    try {
        const res = await axios.get('http://ai-service:8002/predictions/latest', { timeout: 5000 });
        const data = res.data;
        // data = { meta: {...}, predictions: [...] } or { status: 'no_prediction_yet' }
        if (data && Array.isArray(data.predictions)) {
            const pred = data.predictions.find(p => p.symbol === symbol);
            if (pred) {
                console.log(`[AI-FETCH-DIRECT] Found prediction for ${symbol} from ai-service cache`);
                return normalizePrediction(pred);
            }
            console.warn(`[AI-FETCH-DIRECT] Symbol ${symbol} not in ai-service cache (${data.predictions.length} predictions available)`);
        } else {
            console.warn('[AI-FETCH-DIRECT] ai-service has no predictions yet:', data && data.status);
        }
        return null;
    } catch (e) {
        console.warn(`[AI-FETCH-DIRECT] Failed to query ai-service directly:`, e.message);
        return null;
    }
}

// NEW: Fetch SAFE-Alert real-time signal with technical indicators
async function fetchSAFEAlertSignal(symbol, targetSellTime = null) {
    try {
        let res;
        try {
            res = await axios.get(`http://ai-service:8002/v2/signal/${symbol}/cached`, { timeout: 5000 });
            console.log(`[SAFE-ALERT] Using cached multi-horizon signal for ${symbol}`);
        } catch (cachedError) {
            console.warn(`[SAFE-ALERT] Cached signal unavailable for ${symbol}, running live inference`);
            res = await axios.get(`http://ai-service:8002/v2/signal/${symbol}`, { timeout: 30000 });
        }
        const data = res.data;

        if (data && data.horizon_1h && data.horizon_4h) {
            const h1 = data.horizon_1h;
            const h4 = data.horizon_4h;
            console.log(
                `[SAFE-ALERT] ${symbol}: 1h=${h1.signal}/${Number(h1.confidence || 0).toFixed(3)}, `
                + `4h=${h4.signal}/${Number(h4.confidence || 0).toFixed(3)}`
            );

            const directionValue = (signal) => signal === 'BUY' ? 1 : (signal === 'SELL' ? -1 : 0);
            const toDirection = (score) => score > 0.08 ? 'UP' : (score < -0.08 ? 'DOWN' : 'SIDEWAYS');
            const probabilityEdge = (horizon) => {
                const up = Number(horizon.probs?.UP);
                const down = Number(horizon.probs?.DOWN);
                if (Number.isFinite(up) && Number.isFinite(down)) return up - down;
                return directionValue(horizon.signal) * Number(horizon.confidence || 0);
            };
            const horizonChange = (horizon, maxMovePercent) => {
                return Math.max(-1, Math.min(1, probabilityEdge(horizon))) * maxMovePercent;
            };

            const hoursToTarget = targetSellTime
                ? Math.max(0, (new Date(targetSellTime).getTime() - Date.now()) / 3_600_000)
                : 1;
            const weights = hoursToTarget <= 2
                ? { h1: 0.7, h4: 0.3 }
                : (hoursToTarget <= 6 ? { h1: 0.4, h4: 0.6 } : { h1: 0.25, h4: 0.75 });

            const h1Change = horizonChange(h1, 2);
            const h4Change = horizonChange(h4, 4);
            const consensusScore = probabilityEdge(h1) * weights.h1 + probabilityEdge(h4) * weights.h4;
            const signalsAgree = directionValue(h1.signal) === directionValue(h4.signal);
            const confidence = Math.max(0, Math.min(
                1,
                (Number(h1.confidence || 0) * weights.h1 + Number(h4.confidence || 0) * weights.h4)
                * (signalsAgree ? 1 : 0.78)
            ));
            const direction = toDirection(consensusScore);
            const changePercent = parseFloat((h1Change * weights.h1 + h4Change * weights.h4).toFixed(2));
            const reason = signalsAgree
                ? `Hai khung 1H và 4H đồng thuận ${direction === 'UP' ? 'tăng' : (direction === 'DOWN' ? 'giảm' : 'trung tính')}.`
                : 'Hai khung 1H và 4H chưa đồng thuận; độ tin cậy đã được điều chỉnh giảm.';

            // Map to investment format
            return {
                symbol,
                direction,
                confidence: confidence,
                change_percent: changePercent,
                reason,
                causal_factor: 'SAFE-Alert Multi-Horizon',
                technical_indicators: data.top_inputs,
                selected_news: [...(h1.selected_news || []), ...(h4.selected_news || [])],
                alert_level: data.alert.level,
                is_safe_alert: true,
                consensus: {
                    agrees: signalsAgree,
                    score: parseFloat(consensusScore.toFixed(4)),
                    hours_to_target: parseFloat(hoursToTarget.toFixed(2)),
                    weights,
                },
                forecast: {
                    next_1h: {
                        direction: toDirection(directionValue(h1.signal)),
                        confidence: Number(h1.confidence || 0),
                        price_change_percent: parseFloat(h1Change.toFixed(2)),
                    },
                    next_4h: {
                        direction: toDirection(directionValue(h4.signal)),
                        confidence: Number(h4.confidence || 0),
                        price_change_percent: parseFloat(h4Change.toFixed(2)),
                    },
                },
            };
        }

        console.warn('[SAFE-ALERT] Invalid response format:', data);
        return null;
    } catch (e) {
        console.warn(`[SAFE-ALERT] Failed to fetch signal for ${symbol}:`, e.message);
        return null;
    }
}


// Get AI prediction for specific symbol
// Normalize a raw prediction object (from inference engine) to add top-level shorthand fields
function normalizePrediction(pred) {
    if (!pred) return null;
    const next1h = (pred.forecast && pred.forecast.next_1h) ? pred.forecast.next_1h : {};
    // Add top-level shorthand fields used by generateAdvice() fallback
    if (pred.direction === undefined) pred.direction = next1h.direction || 'SIDEWAYS';
    if (pred.confidence === undefined) pred.confidence = (next1h.confidence || 0) / 100; // normalize 0-100 → 0-1
    if (pred.change_percent === undefined) pred.change_percent = next1h.price_change_percent || 0;
    if (pred.reason === undefined) pred.reason = pred.explanation || '';
    if (pred.causal_factor === undefined) pred.causal_factor = (pred.causal_analysis && pred.causal_analysis.primary_driver) || '';
    return pred;
}

async function getAIPredictionForSymbol(symbol) {
    // 1. Try core-service database (primary source)
    let pred = await fetchPredictionForSymbol(symbol);
    if (pred) return normalizePrediction(pred);

    // 2. Fall back to ai-service in-memory cache (works even when core-service DB is empty)
    console.warn(`[AI] No prediction in DB for ${symbol}, trying ai-service cache...`);
    pred = await fetchPredictionFromAIService(symbol);
    if (pred) return pred;

    console.warn(`[AI] No prediction available for ${symbol} from any source`);
    return null;
}

function buildLocalFallbackPrediction(symbol) {
    // Lightweight deterministic fallback so local dev can still use investment analysis
    // when AI pipeline is unavailable.
    const now = new Date().toISOString();
    const pseudo = Array.from(symbol).reduce((acc, ch) => acc + ch.charCodeAt(0), 0);
    const raw = ((pseudo % 7) - 3) * 0.15; // roughly -0.45% .. +0.45%
    const change = Math.max(-0.8, Math.min(0.8, raw));
    const direction = change > 0.1 ? 'UP' : (change < -0.1 ? 'DOWN' : 'SIDEWAYS');

    return {
        symbol,
        direction,
        confidence: 0.55,
        change_percent: change,
        reason: 'AI service unavailable - using local fallback estimation.',
        causal_factor: 'LOCAL_FALLBACK',
        is_fallback: true,
        forecast: {
            next_1h: {
                direction,
                price_change_percent: change,
                confidence: 55
            }
        },
        meta: {
            timestamp: now,
            source: 'local_fallback'
        }
    };
}

// Helper: Perform Investment Analysis Logic
async function analyzeInvestmentLogic(symbol, usdt_amount, target_sell_time) {
    const buyPrice = await getCurrentPrice(symbol);
    const coinAmount = usdt_amount / buyPrice;

    // PRIMARY: Try SAFE-Alert real-time signal FIRST
    let aiPred = await fetchSAFEAlertSignal(symbol, target_sell_time);
    if (aiPred) {
        console.log(`[INVESTMENT] Using SAFE-Alert prediction for ${symbol}`);
    } else {
        // SECONDARY: Try legacy AI service
        console.log(`[INVESTMENT] SAFE-Alert unavailable, trying legacy AI service...`);
        aiPred = await getAIPredictionForSymbol(symbol);
        if (!aiPred) {
            aiPred = buildLocalFallbackPrediction(symbol);
            console.warn(`[AI] Using local fallback prediction for ${symbol}`);
        }
    }
    let aiAnalysis = null;
    try {
        if (!aiPred.is_fallback) {
            const requestId = `${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
            const analysisPayload = {
                requestId,
                symbol,
                amount: parseFloat(usdt_amount),
                buy_price: buyPrice,
                target_sell_time,
                current_time: new Date().toISOString(),
                market_prediction: aiPred
            };

            // Send to Kafka
            await producer.send({
                topic: 'investment.analysis.request',
                messages: [{ key: requestId, value: JSON.stringify(analysisPayload) }]
            });

            // Wait for reply with bounded timeout
            aiAnalysis = await new Promise((resolve, reject) => {
                const timeout = setTimeout(() => {
                    if (pendingAnalysisRequests.has(requestId)) {
                        pendingAnalysisRequests.delete(requestId);
                        resolve(null);
                        console.warn(`[KAFKA TIMEOUT] Analysis request ${requestId} timed out`);
                    }
                }, 8000); // 8 seconds timeout (ai-service uses cached V2 signal)

                pendingAnalysisRequests.set(requestId, { resolve, reject, timeout });
            });
        }
    } catch (err) {
        console.error('[AI ANALYSIS ERROR]', err.message);
    }

    // Process Result
    let predictedPrice, predictedProfitUsdt, aiAdvice, predictedPercent;

    if (aiAnalysis && !aiAnalysis.error) {
        // Got response from AI service
        predictedPrice = aiAnalysis.predicted_price;
        predictedProfitUsdt = aiAnalysis.predicted_profit_usdt;
        aiAdvice = aiAnalysis.advice;
        predictedPercent = aiAnalysis.predicted_profit_percent;

        const displayDirection = aiAnalysis.details?.direction || aiPred.direction;
        const displayConfidence = aiAnalysis.details?.confidence || aiPred.confidence;

        // Enrich aiPred
        aiPred.direction = displayDirection;
        aiPred.confidence = displayConfidence;
        aiPred.change_percent = predictedPercent;
    } else {
        // Fallback calculation
        predictedPrice = buyPrice + (buyPrice * ((aiPred.change_percent || 0) / 100));
        predictedProfitUsdt = (predictedPrice - buyPrice) * coinAmount;
        predictedPercent = aiPred.change_percent || 0;

        aiAdvice = generateAdvice(aiPred, buyPrice, usdt_amount);
    }

    return {
        buyPrice,
        coinAmount,
        aiPred,
        aiAdvice,
        predictedPrice,
        predictedProfitUsdt,
        predictedPercent
    };
}

// POST /v1/investments/analyze - Preview analysis only
app.post('/v1/investments/analyze', async (req, res) => {
    const { symbol, usdt_amount, target_sell_time } = req.body;
    if (!symbol || !usdt_amount || !target_sell_time) {
        return res.status(400).json({ error: 'Missing required fields' });
    }

    try {
        const result = await analyzeInvestmentLogic(symbol, usdt_amount, target_sell_time);

        res.json({
            ai_recommendation: {
                advice: result.aiAdvice,
                predicted_price: result.predictedPrice,
                predicted_profit_usdt: result.predictedProfitUsdt,
                predicted_profit_percent: result.predictedPercent,
                confidence: result.aiPred.confidence,
                direction: result.aiPred.direction,
                causal_factor: result.aiPred.causal_factor,
                buy_price: result.buyPrice,
                forecast: result.aiPred.forecast,
                consensus: result.aiPred.consensus,
                reason: result.aiPred.reason,
            }
        });
    } catch (err) {
        if (err.message === 'AI_SERVICE_UNAVAILABLE') {
            return res.status(503).json({
                error: `Chưa có dự đoán AI cho ${symbol}. Vui lòng thử lại sau.`,
                error_code: 'AI_SERVICE_UNAVAILABLE',
                details: 'No AI prediction available for this symbol yet. Please try again later.'
            });
        }
        res.status(500).json({ error: err.message });
    }
});

// POST /v1/investments - Create new investment simulation
app.post('/v1/investments', async (req, res) => {
    const { user_id, symbol, usdt_amount, target_sell_time, ai_analysis } = req.body;

    if (!user_id || !symbol || !usdt_amount || !target_sell_time) {
        return res.status(400).json({ error: 'Missing required fields' });
    }

    try {
        let buyPrice, coinAmount, aiPred, aiAdvice, predictedProfitUsdt, predictedPrice, predictedPercent;

        if (ai_analysis) {
            console.log(`[INVESTMENT] Using pre-calculated analysis for ${symbol}`);
            buyPrice = ai_analysis.buy_price || await getCurrentPrice(symbol);
            coinAmount = usdt_amount / buyPrice;
            aiAdvice = ai_analysis.advice;
            predictedProfitUsdt = ai_analysis.predicted_profit_usdt;
            predictedPrice = ai_analysis.predicted_price;
            predictedPercent = ai_analysis.predicted_profit_percent;

            // Reconstruct aiPred for DB
            aiPred = {
                symbol,
                direction: ai_analysis.direction,
                confidence: ai_analysis.confidence,
                change_percent: predictedPercent,
                causal_factor: ai_analysis.causal_factor,
                reason: ai_analysis.reason,
                forecast: ai_analysis.forecast,
                consensus: ai_analysis.consensus,
            };
        } else {
            console.log(`[INVESTMENT] Performing new analysis for ${symbol}`);
            const analysis = await analyzeInvestmentLogic(symbol, usdt_amount, target_sell_time);
            buyPrice = analysis.buyPrice;
            coinAmount = analysis.coinAmount;
            aiPred = analysis.aiPred;
            aiAdvice = analysis.aiAdvice;
            predictedProfitUsdt = analysis.predictedProfitUsdt;
            predictedPrice = analysis.predictedPrice;
            predictedPercent = analysis.predictedPercent;
        }

        // Insert investment
        const result = await pool.query(`
      INSERT INTO investments (
        user_id, symbol, usdt_amount, coin_amount, buy_price, target_sell_time,
        ai_prediction, ai_advice, predicted_profit_usdt
      ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
      RETURNING *
    `, [user_id, symbol, usdt_amount, coinAmount, buyPrice, target_sell_time,
            JSON.stringify(aiPred), aiAdvice, predictedProfitUsdt]);

        const investment = result.rows[0];

        // Send notification to user via WebSocket
        sendToUser(user_id, {
            type: 'investment_created',
            investment: investment,
            ai_recommendation: {
                advice: aiAdvice,
                predicted_price: predictedPrice,
                predicted_profit_usdt: predictedProfitUsdt,
                predicted_profit_percent: predictedPercent,
                confidence: aiPred.confidence,
                direction: aiPred.direction
            }
        });

        res.json({
            investment: investment,
            ai_recommendation: {
                advice: aiAdvice,
                predicted_price: predictedPrice,
                predicted_profit_usdt: predictedProfitUsdt,
                predicted_profit_percent: predictedPercent,
                confidence: aiPred.confidence,
                direction: aiPred.direction,
                causal_factor: aiPred.causal_factor,
                reason: aiPred.reason,
                forecast: aiPred.forecast,
                consensus: aiPred.consensus,
            }
        });

    } catch (err) {
        console.error('[ERROR] Create investment failed:', err.message);
        res.status(500).json({ error: err.message });
    }
});

// SSE Clients map
const sseClients = new Map(); // userId -> [{ res, id }]

// GET /v1/investments/events - SSE Endpoint
app.get('/v1/investments/events', (req, res) => {
    const userId = req.query.user_id;
    if (!userId) return res.status(400).send('Missing user_id');

    // Headers for SSE
    res.writeHead(200, {
        'Content-Type': 'text/event-stream',
        'Cache-Control': 'no-cache',
        'Connection': 'keep-alive',
        'X-Accel-Buffering': 'no', // Disable buffering for Nginx/Kong
        'Content-Encoding': 'identity', // Disable compression for SSE
        'Access-Control-Allow-Origin': '*' // Ensure CORS works for SSE
    });

    // Send initial connection message and ping to keep alive
    res.write(`data: ${JSON.stringify({ type: 'connected' })}\n\n`);

    const heartbeat = setInterval(() => {
        res.write(': heartbeat\n\n'); // SSE comment to keep connection alive
    }, 30000);

    // Add client to map
    if (!sseClients.has(userId)) {
        sseClients.set(userId, []);
    }
    const clientId = Date.now();
    sseClients.get(userId).push({ res, id: clientId });

    console.log(`[SSE] User ${userId} connected. Total clients: ${sseClients.get(userId).length}`);

    // Remove client on close
    req.on('close', () => {
        clearInterval(heartbeat);
        const clients = sseClients.get(userId) || [];
        sseClients.set(userId, clients.filter(c => c.id !== clientId));
        if (sseClients.get(userId).length === 0) {
            sseClients.delete(userId);
        }
        console.log(`[SSE] User ${userId} disconnected. Remaining clients: ${sseClients.get(userId)?.length || 0}`);
    });
});

// Helper: Send message to user via SSE
function sendToUser(userId, data) {
    const clients = sseClients.get(userId);
    if (!clients || clients.length === 0) return;

    console.log(`[SSE] Sending ${data.type} to user ${userId} (${clients.length} clients)`);
    clients.forEach(client => {
        try {
            client.res.write(`data: ${JSON.stringify(data)}\n\n`);
        } catch (err) {
            console.error(`[SSE ERROR] Send failed:`, err.message);
        }
    });
}

// GET /v1/investments/:user_id - Get user's investments with pagination
app.get('/v1/investments/:user_id', async (req, res) => {
    try {
        const userId = req.params.user_id;
        const page = parseInt(req.query.page) || 1;
        const limit = parseInt(req.query.limit) || 10;
        const offset = (page - 1) * limit;

        if (page < 1 || limit < 1) {
            return res.status(400).json({ error: 'Page and limit must be positive integers' });
        }

        // Get total count
        const countResult = await pool.query(
            'SELECT COUNT(*) FROM investments WHERE user_id = $1',
            [userId]
        );
        const total = parseInt(countResult.rows[0].count);

        // Get paginated data
        const result = await pool.query(
            'SELECT * FROM investments WHERE user_id = $1 ORDER BY created_at DESC LIMIT $2 OFFSET $3',
            [userId, limit, offset]
        );

        res.json({
            investments: result.rows,
            pagination: {
                total,
                page,
                limit,
                totalPages: Math.ceil(total / limit)
            }
        });
    } catch (error) {
        console.error('[ERROR] Get investments failed:', error);
        res.status(500).json({ error: 'Internal server error' });
    }
});

// POST /v1/investments/:id/sell - Manually sell investment
app.post('/v1/investments/:id/sell', async (req, res) => {
    const { id } = req.params;

    try {
        const investment = await pool.query(
            'SELECT * FROM investments WHERE id = $1 AND status = $2',
            [id, 'active']
        );

        if (investment.rows.length === 0) {
            return res.status(404).json({ error: 'Investment not found or already closed' });
        }

        await closeInvestment(investment.rows[0]);

        res.json({ message: 'Investment closed successfully' });
    } catch (error) {
        console.error('[ERROR] Sell investment failed:', error);
        res.status(500).json({ error: 'Internal server error' });
    }
});

// Helper: Close investment and calculate results
async function closeInvestment(inv) {
    const sellPrice = await getCurrentPrice(inv.symbol);
    const sellValueUsdt = sellPrice * inv.coin_amount;
    const actualProfitUsdt = sellValueUsdt - inv.usdt_amount;
    const actualProfitPercent = (actualProfitUsdt / inv.usdt_amount) * 100;

    await pool.query(`
    UPDATE investments 
    SET sell_price = $1, sell_time = NOW(), status = 'closed', 
        actual_profit_usdt = $2, updated_at = NOW()
    WHERE id = $3
  `, [sellPrice, actualProfitUsdt, inv.id]);

    const accuracy = calculateAccuracy(actualProfitUsdt, inv.predicted_profit_usdt);

    // Publish to Kafka
    await producer.send({
        topic: 'investment_results',
        messages: [{
            key: inv.user_id,
            value: JSON.stringify({
                investment_id: inv.id,
                user_id: inv.user_id,
                symbol: inv.symbol,
                usdt_invested: parseFloat(inv.usdt_amount),
                actual_profit_usdt: parseFloat(actualProfitUsdt),
                actual_profit_percent: actualProfitPercent,
                predicted_profit_usdt: parseFloat(inv.predicted_profit_usdt),
                ai_accuracy: accuracy,
                buy_price: parseFloat(inv.buy_price),
                sell_price: parseFloat(sellPrice),
                buy_time: inv.buy_time,
                sell_time: new Date(),
                result: actualProfitUsdt >= 0 ? 'profit' : 'loss'
            })
        }]
    });

    // Send WebSocket notification
    sendToUser(inv.user_id, {
        type: 'investment_closed',
        investment_id: inv.id,
        symbol: inv.symbol,
        result: actualProfitUsdt >= 0 ? 'profit' : 'loss',
        actual_profit_usdt: actualProfitUsdt,
        actual_profit_percent: actualProfitPercent,
        predicted_profit_usdt: inv.predicted_profit_usdt,
        ai_accuracy: accuracy,
        message: `Đầu tư ${inv.symbol} đã đóng. ${actualProfitUsdt >= 0 ? 'Lời' : 'Lỗ'} ${Math.abs(actualProfitUsdt).toFixed(2)} USDT (${actualProfitPercent.toFixed(2)}%)`
    });

    console.log(`[CLOSE] Investment ${inv.id} closed. Profit: ${actualProfitUsdt} USDT`);
}

// Helper: Generate AI advice in Vietnamese
function generateAdvice(aiPred, buyPrice, usdtAmount) {
    const { change_percent, confidence, direction, reason, causal_factor } = aiPred;

    let advice = '';
    const isActionable = confidence >= 0.55 && aiPred.consensus?.agrees !== false;

    if (direction === 'UP' && isActionable) {
        advice = `✅ XU HƯỚNG TĂNG ĐƯỢC XÁC NHẬN\n`;
        advice += `Dự đoán giá sẽ TĂNG ${change_percent.toFixed(2)}% (độ tin cậy ${(confidence * 100).toFixed(1)}%)\n`;
        advice += `Lợi nhuận dự kiến: ${(usdtAmount * change_percent / 100).toFixed(2)} USDT\n`;
    } else if (direction === 'DOWN' && isActionable) {
        advice = `⚠️ XU HƯỚNG GIẢM ĐƯỢC XÁC NHẬN\n`;
        advice += `Dự đoán giá sẽ GIẢM ${Math.abs(change_percent).toFixed(2)}% (độ tin cậy ${(confidence * 100).toFixed(1)}%)\n`;
        advice += `Rủi ro lỗ: ${Math.abs(usdtAmount * change_percent / 100).toFixed(2)} USDT\n`;
    } else {
        advice = `⚠️ TÍN HIỆU CHƯA ĐỦ RÕ RÀNG\n`;
        advice += `Kết quả tổng hợp có độ tin cậy ${(confidence * 100).toFixed(1)}%`;
        advice += aiPred.consensus?.agrees === false ? ' và hai khung thời gian chưa đồng thuận.\n' : '.\n';
        advice += `Nên tiếp tục quan sát thay vì xem đây là tín hiệu xác nhận.\n`;
    }

    if (reason) {
        advice += `\nLý do: ${reason}`;
    }
    if (causal_factor) {
        advice += `\nNguyên nhân: ${causal_factor}`;
    }

    return advice;
}

// Helper: Calculate accuracy
function calculateAccuracy(actual, predicted) {
    if (predicted === 0) return actual === 0 ? 100 : 0;
    const error = Math.abs(actual - predicted) / Math.abs(predicted);
    return Math.max(0, Math.min(100, (1 - error) * 100));
}


// Background job: Auto-close investments at target time
cron.schedule('* * * * *', async () => {
    console.log('[CRON] Checking for investments to auto-close...');

    try {
        const result = await pool.query(`
      SELECT * FROM investments 
      WHERE status = 'active' AND target_sell_time <= NOW()
    `);

        for (const inv of result.rows) {
            await closeInvestment(inv);
        }

        if (result.rows.length > 0) {
            console.log(`[CRON] Auto-closed ${result.rows.length} investments`);
        }
    } catch (error) {
        console.error('[CRON ERROR]', error);
    }
});

// Health check
app.get('/health', (req, res) => {
    res.json({ status: 'ok', service: 'investment-service' });
});

// Start server
const PORT = process.env.PORT || 8001;
server.listen(PORT, async () => {
    try {
        await initDB();
    } catch (err) {
        console.error('[STARTUP] DB init failed:', err.message);
    }
    try {
        await producer.connect();
    } catch (err) {
        console.warn('[STARTUP] Kafka producer connect failed (non-fatal):', err.message);
    }
    await consumeKafkaTopics();
    console.log(`[INVESTMENT SERVICE] Running on port ${PORT}`);
    console.log(`[WEBSOCKET] Ready for connections`);
});

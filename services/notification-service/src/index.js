const express = require('express');
const { Pool } = require('pg');
const { Kafka } = require('kafkajs');
const nodemailer = require('nodemailer');
const cors = require('cors');

const app = express();
app.use(express.json());
app.use(cors());

// --- CONFIG ---
// Use local DB for caching user settings (synced via Kafka) and email history
const POSTGRES_CONNECTION = {
    user: process.env.POSTGRES_USER || 'dev',
    host: process.env.POSTGRES_HOST || 'notification-db',
    database: process.env.POSTGRES_DB || 'notification_db',
    password: process.env.POSTGRES_PASSWORD || 'dev',
    port: 5432,
};

const KAFKA_BROKER = process.env.KAFKA_BROKERS || 'kafka:9092';

// Set up your Gmail credentials in docker-compose environment variables
const SMTP_CONFIG = {
    service: 'gmail',
    auth: {
        user: process.env.SMTP_EMAIL || 'your_email@gmail.com',
        pass: process.env.SMTP_PASSWORD || 'your_app_password'
    }
};

const FE_URL = process.env.FE_URL || 'http://localhost:5173';

// --- DB SETUP ---
const pool = new Pool(POSTGRES_CONNECTION);
const authPool = new Pool({
    connectionString: process.env.AUTH_DATABASE_URL || 'postgresql://dev:dev@postgres:5432/appdb'
});

async function initDB() {
    try {
        // Table 1: Cache user notification settings (synced from auth-service via Kafka)
        await pool.query(`
            CREATE TABLE IF NOT EXISTS user_notification_settings (
                user_id VARCHAR(255) PRIMARY KEY,
                email VARCHAR(255) NOT NULL,
                prediction_symbols TEXT[] DEFAULT '{}',
                investment_enabled BOOLEAN DEFAULT FALSE,
                updated_at TIMESTAMP DEFAULT NOW()
            );
        `);

        // Table 2: Email history for tracking sent emails
        await pool.query(`
            CREATE TABLE IF NOT EXISTS email_history (
                id SERIAL PRIMARY KEY,
                user_id VARCHAR(255) NOT NULL,
                email VARCHAR(255) NOT NULL,
                type VARCHAR(50) NOT NULL, -- 'PREDICTION' or 'INVESTMENT'
                symbol VARCHAR(20),
                subject TEXT,
                sent_at TIMESTAMP DEFAULT NOW()
            );
        `);

        await pool.query(`CREATE INDEX IF NOT EXISTS idx_email_history_user ON email_history(user_id);`);
        await pool.query(`CREATE INDEX IF NOT EXISTS idx_email_history_sent ON email_history(sent_at);`);

        console.log('[DB] Notification tables ready (settings cache + email history)');
    } catch (e) {
        console.error('[DB ERROR]', e);
    }
}

// --- EMAILER ---
const transporter = nodemailer.createTransport(SMTP_CONFIG);

async function sendWithBrevo(to, subject, htmlContent) {
    const apiKey = String(process.env.BREVO_API_KEY || '').trim();
    if (!apiKey) return false;

    const senderEmail = String(
        process.env.BREVO_SENDER_EMAIL || process.env.SMTP_EMAIL || ''
    ).trim();
    if (!senderEmail) {
        throw new Error('BREVO_SENDER_EMAIL is not configured');
    }

    const response = await fetch('https://api.brevo.com/v3/smtp/email', {
        method: 'POST',
        headers: {
            accept: 'application/json',
            'content-type': 'application/json',
            'api-key': apiKey
        },
        body: JSON.stringify({
            sender: {
                name: process.env.BREVO_SENDER_NAME || 'Aegis',
                email: senderEmail
            },
            to: [{ email: to }],
            subject,
            htmlContent
        }),
        signal: AbortSignal.timeout(Number(process.env.BREVO_TIMEOUT_MS || 12000))
    });

    if (!response.ok) {
        const detail = await response.text();
        throw new Error(`Brevo API ${response.status}: ${detail.slice(0, 300)}`);
    }

    console.log(`[EMAIL] Sent via Brevo to ${to}: ${subject}`);
    return true;
}

async function sendEmail(to, subject, htmlContent) {
    try {
        if (await sendWithBrevo(to, subject, htmlContent)) return;

        if (!process.env.SMTP_EMAIL || !process.env.SMTP_PASSWORD) {
            throw new Error('Neither Brevo nor SMTP email delivery is configured');
        }

        await transporter.sendMail({
            from: `"Aegis" <${SMTP_CONFIG.auth.user}>`,
            to,
            subject,
            html: htmlContent
        });
        console.log(`[EMAIL] Sent via SMTP to ${to}: ${subject}`);
    } catch (e) {
        console.error(`[EMAIL ERROR] Failed to send to ${to}:`, e.message);
        throw e;
    }
}

// --- HTML TEMPLATES ---
const getPredictionTemplate = (symbol, prediction, advice) => `
    <div style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto; border: 1px solid #e0e0e0; border-radius: 8px; overflow: hidden;">
        <div style="background-color: #1e293b; color: white; padding: 20px; text-align: center;">
            <h2 style="margin: 0;">🚀 Cơ Hội Mới: ${symbol}</h2>
        </div>
        <div style="padding: 24px; background-color: #f8fafc;">
            <div style="display: flex; justify-content: space-between; margin-bottom: 20px;">
                <div style="text-align: center; flex: 1; padding: 10px; background: white; border-radius: 8px; margin-right: 10px;">
                    <div style="color: #64748b; font-size: 12px;">XU HƯỚNG</div>
                    <div style="color: ${prediction.direction === 'UP' ? '#22c55e' : '#ef4444'}; font-weight: bold; font-size: 18px;">
                        ${prediction.direction === 'UP' ? 'TĂNG 📈' : 'GIẢM 📉'}
                    </div>
                </div>
                <div style="text-align: center; flex: 1; padding: 10px; background: white; border-radius: 8px; margin-left: 10px;">
                    <div style="color: #64748b; font-size: 12px;">MỤC TIÊU</div>
                    <div style="color: #3b82f6; font-weight: bold; font-size: 18px;">
                        ${(prediction.change_percent || 0).toFixed(2)}%
                    </div>
                </div>
            </div>
            
            <div style="background: white; padding: 16px; border-radius: 8px; border-left: 4px solid #3b82f6; margin-bottom: 20px;">
                <h3 style="margin-top: 0; color: #334155; font-size: 16px;">💡 Phân Tích AI</h3>
                <p style="color: #475569; line-height: 1.6; white-space: pre-wrap;">${advice}</p>
            </div>

            <div style="text-align: center;">
                <a href="${FE_URL}" style="display: inline-block; background-color: #2563eb; color: white; padding: 12px 24px; text-decoration: none; border-radius: 6px; font-weight: bold;">
                    Mở App Để Giao Dịch
                </a>
            </div>
        </div>
        <div style="background-color: #f1f5f9; padding: 12px; text-align: center; color: #94a3b8; font-size: 12px;">
            Crypto Investment Simulator - AI Powered
        </div>
    </div>
`;

const getInvestmentResultTemplate = (inv, result) => {
    const isProfit = result.actual_profit_usdt >= 0;
    const color = isProfit ? '#22c55e' : '#ef4444';

    return `
    <div style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto; border: 1px solid #e0e0e0; border-radius: 8px; overflow: hidden;">
        <div style="background-color: ${isProfit ? '#14532d' : '#450a0a'}; color: white; padding: 20px; text-align: center;">
            <h2 style="margin: 0;">${isProfit ? '🎉 Chốt Lời Thành Công!' : '⚠️ Đã Cắt Lỗ'} - ${inv.symbol}</h2>
        </div>
        <div style="padding: 24px; background-color: #f8fafc;">
            <div style="text-align: center; margin-bottom: 24px;">
                <div style="font-size: 14px; color: #64748b;">Lợi Nhuận Thực Tế</div>
                <div style="font-size: 32px; font-weight: bold; color: ${color};">
                    ${isProfit ? '+' : ''}${parseFloat(result.actual_profit_usdt).toFixed(2)} USDT
                    <span style="font-size: 16px; color: ${color};">
                        (${parseFloat(result.actual_profit_percent).toFixed(2)}%)
                    </span>
                </div>
            </div>

            <div style="background: white; padding: 20px; border-radius: 8px; margin-bottom: 20px; box-shadow: 0 1px 3px rgba(0,0,0,0.1);">
                <table style="width: 100%; border-collapse: collapse;">
                    <tr style="border-bottom: 1px solid #f1f5f9;">
                        <td style="padding: 8px; color: #64748b;">Giá Mua</td>
                        <td style="padding: 8px; text-align: right; font-family: monospace;">$${parseFloat(inv.buy_price).toLocaleString()}</td>
                    </tr>
                    <tr style="border-bottom: 1px solid #f1f5f9;">
                        <td style="padding: 8px; color: #64748b;">Giá Bán</td>
                        <td style="padding: 8px; text-align: right; font-family: monospace;">$${parseFloat(inv.sell_price).toLocaleString()}</td>
                    </tr>
                    <tr>
                        <td style="padding: 8px; color: #64748b;">AI Dự Báo</td>
                        <td style="padding: 8px; text-align: right;">
                             Dự kiến lãi <b>${parseFloat(inv.predicted_profit_usdt).toFixed(2)} USDT</b>
                             <br/>
                             <span style="font-size: 11px; color: ${result.ai_accuracy > 80 ? 'green' : 'orange'}">Độ chính xác: ${parseFloat(result.ai_accuracy).toFixed(1)}%</span>
                        </td>
                    </tr>
                </table>
            </div>

            <div style="text-align: center;">
                <a href="${FE_URL}" style="display: inline-block; background-color: #1e293b; color: white; padding: 12px 24px; text-decoration: none; border-radius: 6px; font-weight: bold;">
                    Xem Chi Tiết
                </a>
            </div>
        </div>
    </div>
    `;
};

// --- KAFKA CONSUMER ---
const kafka = new Kafka({ clientId: 'notification-service', brokers: KAFKA_BROKER.split(',') });
const consumer = kafka.consumer({ groupId: 'notification-service-group' });

async function startConsumer() {
    await consumer.connect();
    await consumer.subscribe({ topics: ['ai_insights', 'investment_results', 'user_settings_updated'], fromBeginning: false });

    await consumer.run({
        eachMessage: async ({ topic, partition, message }) => {
            try {
                const payload = JSON.parse(message.value.toString());

                console.log(`[KAFKA] Received from ${topic}:`, JSON.stringify(payload).substring(0, 200));

                if (topic === 'ai_insights') {
                    // AI Service sends: { meta: {...}, predictions: [...] }
                    const predictions = payload.predictions || [];
                    console.log(`[KAFKA] Processing ${predictions.length} predictions`);
                    for (const pred of predictions) {
                        await processPredictionNotification(pred);
                    }
                }
                else if (topic === 'investment_results') {
                    console.log('[KAFKA] Processing investment result');
                    await processInvestmentNotification(payload);
                }
                else if (topic === 'user_settings_updated') {
                    console.log('[KAFKA] Syncing user settings');
                    await syncUserSettings(payload);
                }
            } catch (e) {
                console.error('[KAFKA ERROR]', e);
            }
        },
    });
    console.log('[KAFKA] Notification Consumer Started');
}

// Sync user settings from auth-service via Kafka
async function syncUserSettings(payload) {
    try {
        const { user_id, email, notification_settings } = payload;

        if (!user_id || !email || !notification_settings) {
            console.warn('[SYNC] Invalid payload:', payload);
            return;
        }

        const { prediction_symbols = [], investment_enabled = false } = notification_settings;

        await pool.query(`
            INSERT INTO user_notification_settings (user_id, email, prediction_symbols, investment_enabled, updated_at)
            VALUES ($1, $2, $3, $4, NOW())
            ON CONFLICT (user_id) DO UPDATE 
            SET email = $2, prediction_symbols = $3, investment_enabled = $4, updated_at = NOW()
        `, [user_id, email, prediction_symbols, investment_enabled]);

    } catch (e) {
        console.error('[SYNC ERROR]', e);
    }
}

async function processPredictionNotification(pred) {
    if (!pred.should_alert) {
        console.log(`[SAFE-ALERT] Skipping ${pred.symbol}: alert gate is closed`);
        return;
    }
    // Extract valid data from AI payload
    const forecast = pred.forecast?.next_24h || pred.forecast?.next_1h || {};
    const confidence = forecast.confidence || 0;

    // Check confidence (threshold 60%)
    if (confidence < 50) {
        console.log(`[PREDICTION] Skipping ${pred.symbol} due to low confidence: ${confidence}%`);
        return;
    }
    try {
        // SAFE-Alert is a security-style product event: deliver to every verified,
        // active account. UI preferences still control optional insight emails.
        const res = await authPool.query(
            `SELECT id AS user_id, email FROM users
             WHERE COALESCE(email_verified, true) = true
               AND lower(COALESCE(status, 'active')) = 'active'
               AND email IS NOT NULL`
        );

        if (res.rows.length === 0) {
            console.log(`[PREDICTION] No subscribers for ${pred.symbol}`);
            return;
        }

        console.log(`[PREDICTION] Sending to ${res.rows.length} users for ${pred.symbol} (Confidence: ${confidence}%)`);

        // Calculate change percent if missing
        let changePercent = 0;
        if (forecast.expected_price && pred.current_price) {
            changePercent = ((forecast.expected_price - pred.current_price) / pred.current_price) * 100;
        } else if (pred.change_percent) {
            changePercent = Number(pred.change_percent);
        }

        const direction = forecast.direction || pred.direction || 'UNKNOWN';

        // Extract advice/reason
        const adviceRaw = pred.causal_analysis?.explanation_vi
            || pred.causal_analysis?.summary
            || pred.reason
            || `Dự báo ${direction} với độ tin cậy ${confidence}%`;

        const subject = `🔥 ${pred.symbol}: Dự báo ${direction} (${changePercent.toFixed(2)}%)`;

        // Prepare template data
        const templateData = {
            ...pred,
            direction,
            change_percent: changePercent,
            confidence,
            current_price: pred.current_price,
            target_price: forecast.expected_price
        };

        const html = getPredictionTemplate(pred.symbol, templateData, adviceRaw);

        for (const row of res.rows) {
            if (row.email && row.email.includes('@')) {
                await sendEmail(row.email, subject, html);
            }
        }
    } catch (e) {
        console.error('Failed to process prediction notification:', e);
    }
}

async function processInvestmentNotification(payload) {
    try {
        const userId = payload.user_id;

        const pref = await authPool.query(
            `SELECT email FROM users
             WHERE id = $1 AND COALESCE(email_verified, true) = true
               AND lower(COALESCE(status, 'active')) = 'active'`,
            [userId]
        );

        if (pref.rows.length === 0) {
            console.log(`[INVESTMENT] No active verified account found for ${userId}`);
            return;
        }

        const email = pref.rows[0].email;
        const subject = `Kết quả đầu tư ${payload.symbol}: ${payload.actual_profit_usdt >= 0 ? 'LÃI' : 'LỖ'} ${parseFloat(payload.actual_profit_usdt).toFixed(2)}$`;

        const html = getInvestmentResultTemplate({
            symbol: payload.symbol,
            buy_price: payload.buy_price,
            sell_price: payload.sell_price,
            predicted_profit_usdt: payload.predicted_profit_usdt
        }, payload);

        if (!email.includes('@')) {
            console.warn('[EMAIL SKIP] Invalid email for user:', userId, email);
            return;
        }

        await sendEmail(email, subject, html);

    } catch (e) {
        console.error('Failed to process investment notification:', e);
    }
}


// --- API ---
app.get('/health', (req, res) => res.json({ status: 'ok' }));

// Get settings (return format compatible with old FE)
app.get('/v1/notifications/settings', async (req, res) => {
    const { user_id } = req.query;
    if (!user_id) return res.status(400).send('Missing user_id');
    try {
        const result = await pool.query('SELECT * FROM user_preferences WHERE user_id = $1', [user_id]);
        if (result.rows.length === 0) {
            return res.json([]);
        }

        const pref = result.rows[0];
        const settings = [];

        // Convert prediction_symbols array to individual objects for FE compatibility
        if (pref.prediction_symbols && pref.prediction_symbols.length > 0) {
            for (const symbol of pref.prediction_symbols) {
                settings.push({
                    type: 'PREDICTION',
                    symbol: symbol,
                    enabled: true
                });
            }
        }

        // Add investment setting
        if (pref.investment_enabled) {
            settings.push({
                type: 'INVESTMENT',
                symbol: 'ALL',
                enabled: true
            });
        }

        res.json(settings);
    } catch (e) {
        console.error('[API ERROR]', e);
        res.status(500).send(e.message);
    }
});

// Update setting
app.post('/v1/notifications/settings', async (req, res) => {
    const { user_id, email, type, symbol, enabled } = req.body;

    if (!user_id || !email || !type) return res.status(400).send('Missing fields');
    if (!email.includes('@')) return res.status(400).send('Invalid email format');

    try {
        if (type === 'PREDICTION') {
            // Update prediction_symbols array
            if (enabled) {
                // Add symbol to array if not exists
                await pool.query(`
                    INSERT INTO user_preferences (user_id, email, prediction_symbols, updated_at)
                    VALUES ($1, $2, ARRAY[$3]::TEXT[], NOW())
                    ON CONFLICT (user_id) DO UPDATE 
                    SET prediction_symbols = array_append(
                        CASE WHEN $3 = ANY(user_preferences.prediction_symbols) 
                        THEN user_preferences.prediction_symbols 
                        ELSE user_preferences.prediction_symbols 
                        END, $3
                    ), email = $2, updated_at = NOW()
                `, [user_id, email, symbol]);
            } else {
                // Remove symbol from array
                await pool.query(`
                    UPDATE user_preferences 
                    SET prediction_symbols = array_remove(prediction_symbols, $2), updated_at = NOW()
                    WHERE user_id = $1
                `, [user_id, symbol]);
            }
        } else if (type === 'INVESTMENT') {
            // Update investment_enabled boolean
            await pool.query(`
                INSERT INTO user_preferences (user_id, email, investment_enabled, updated_at)
                VALUES ($1, $2, $3, NOW())
                ON CONFLICT (user_id) DO UPDATE 
                SET investment_enabled = $3, email = $2, updated_at = NOW()
            `, [user_id, email, enabled]);
        }

        res.json({ success: true });
    } catch (e) {
        console.error('[API ERROR]', e);
        res.status(500).send(e.message);
    }
});

app.listen(3000, async () => {
    await initDB();
    await startConsumer();
    console.log('Notification Service running on port 3000');
});

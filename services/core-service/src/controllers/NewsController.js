const axios = require('axios');
const db = require('../config/db');

const fetchCrawlerFallback = async (limit = 100) => {
  try {
    const lim = Math.min(Math.max(parseInt(limit || '100', 10) || 100, 1), 500);
    const res = await axios.get(`http://crawler-service:8001/news/latest?limit=${lim}`, { timeout: 5000 });
    const data = res.data || {};
    return {
      count: data.count || 0,
      rows: Array.isArray(data.rows) ? data.rows : [],
      total: data.total || 0,
      source: 'crawler_fallback'
    };
  } catch (e) {
    return {
      count: 0,
      rows: [],
      total: 0,
      source: 'crawler_fallback_error'
    };
  }
};

/**
 * GET /api/v1/news
 * Query params:
 *  - start: ISO timestamp or epoch ms
 *  - end: ISO timestamp or epoch ms
 *  - source: string
 *  - limit: integer
 */
const getNews = async (req, res) => {
  try {
    const { start, end, source, limit, offset } = req.query;

    const clauses = [];
    const params = [];
    let idx = 1;

    if (start) {
      clauses.push(`time >= $${idx++}`);
      params.push(new Date(start));
    }
    if (end) {
      clauses.push(`time <= $${idx++}`);
      params.push(new Date(end));
    }
    if (source) {
      clauses.push(`source = $${idx++}`);
      params.push(source);
    }

    const lim = Math.min(parseInt(limit || '100', 10) || 100, 1000);
    const off = parseInt(offset || '0', 10) || 0;

    const where = clauses.length ? `WHERE ${clauses.join(' AND ')}` : '';
    const q = `SELECT time, url, source, title, sentiment_score, raw_score FROM news_sentiment ${where} ORDER BY time DESC LIMIT ${lim} OFFSET ${off};`;

    const { rows } = await db.query(q, params);

    // If analyzed-news table is empty, fall back to raw crawled news for UI continuity.
    if (!rows || rows.length === 0) {
      const fallback = await fetchCrawlerFallback(lim);
      return res.json(fallback);
    }

    // Get total count for UI pagination
    const countQ = `SELECT COUNT(*) as total FROM news_sentiment ${where}`;
    const countRes = await db.query(countQ, params);

    res.json({
      count: rows.length,
      rows,
      total: parseInt(countRes.rows[0].total || '0', 10),
      source: 'timescaledb'
    });
  } catch (err) {
    console.error('Error in getNews:', err);
    const fallback = await fetchCrawlerFallback(req.query.limit || '100');
    res.json(fallback);
  }
};

module.exports = {
  getNews,
};

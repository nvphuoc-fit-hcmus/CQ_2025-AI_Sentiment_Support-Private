const axios = require('axios');
const db = require('../config/db');

const fetchCrawlerFallback = async (limit = 100) => {
  try {
    const lim = Math.min(Math.max(parseInt(limit || '100', 10) || 100, 1), 10000);
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
    const { start, end, source, limit, offset, all, search, sentiment } = req.query;
    const includeAll = String(all || '').toLowerCase() === 'true';

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
    const pagination = includeAll ? '' : `LIMIT ${lim} OFFSET ${off}`;
    const q = `SELECT time, url, source, title, sentiment_score, raw_score FROM news_sentiment ${where} ORDER BY time DESC ${pagination};`;

    const { rows } = await db.query(q, params);

    if (includeAll) {
      const crawler = await fetchCrawlerFallback(10000);
      const combined = [...rows, ...(crawler.rows || [])];
      const unique = new Map();
      for (const row of combined) {
        const key = String(row.url || '').trim()
          || `${String(row.title || '').trim().toLowerCase()}|${new Date(row.time).getTime()}`;
        if (!unique.has(key)) unique.set(key, row);
      }
      const mergedRows = [...unique.values()].sort(
        (a, b) => new Date(b.time).getTime() - new Date(a.time).getTime()
      );
      const availableSources = [...new Set(mergedRows.map(row => row.source).filter(Boolean))]
        .sort((a, b) => String(a).localeCompare(String(b)));
      const normalizedSearch = String(search || '').trim().toLowerCase();
      const normalizedSentiment = String(sentiment || '').trim().toLowerCase();
      const startMs = start ? new Date(start).getTime() : null;
      const endMs = end ? new Date(end).getTime() : null;
      const filteredRows = mergedRows.filter(row => {
        const rowTime = new Date(row.time).getTime();
        if (Number.isFinite(startMs) && (!Number.isFinite(rowTime) || rowTime < startMs)) return false;
        if (Number.isFinite(endMs) && (!Number.isFinite(rowTime) || rowTime > endMs)) return false;
        if (source && row.source !== source) return false;
        if (normalizedSearch && !`${row.title || ''} ${row.source || ''}`.toLowerCase().includes(normalizedSearch)) return false;
        const score = Number(row.sentiment_score || 0);
        const sentimentKey = score > 0.1 ? 'positive' : score < -0.1 ? 'negative' : 'neutral';
        if (normalizedSentiment && normalizedSentiment !== 'all' && sentimentKey !== normalizedSentiment) return false;
        return true;
      });
      const stats = filteredRows.reduce((summary, row) => {
        const score = Number(row.sentiment_score || 0);
        summary[score > 0.1 ? 'positive' : score < -0.1 ? 'negative' : 'neutral'] += 1;
        return summary;
      }, { positive: 0, neutral: 0, negative: 0 });
      const requestedLimit = Math.min(Math.max(parseInt(limit || '50', 10) || 50, 1), 100);
      const requestedOffset = Math.max(parseInt(offset || '0', 10) || 0, 0);
      const pageRows = filteredRows.slice(requestedOffset, requestedOffset + requestedLimit);
      return res.json({
        count: pageRows.length,
        rows: pageRows,
        total: filteredRows.length,
        stats,
        sources: availableSources,
        range: filteredRows.length ? {
          newest: filteredRows[0].time,
          oldest: filteredRows[filteredRows.length - 1].time,
        } : null,
        source: 'timescaledb+crawler',
      });
    }

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

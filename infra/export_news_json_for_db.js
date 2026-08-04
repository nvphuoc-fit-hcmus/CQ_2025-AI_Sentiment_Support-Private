const fs = require('fs');
const crypto = require('crypto');

const input = process.argv[2];
const output = process.argv[3];
const cutoff = Date.parse('2025-08-27T00:00:00Z');
const now = Date.now();

if (!input || !output) {
  throw new Error('Usage: node export_news_json_for_db.js input.json output.csv');
}

const csv = (value) => {
  const text = value == null ? '' : String(value);
  return `"${text.replace(/"/g, '""')}"`;
};

const canonicalUrl = (raw, item) => {
  try {
    const url = new URL(String(raw || ''));
    for (const key of [...url.searchParams.keys()]) {
      if (key.toLowerCase().startsWith('utm_')) url.searchParams.delete(key);
    }
    url.hash = '';
    return url.toString();
  } catch {
    const identity = `${item.timestamp}|${item.source}|${item.title}`;
    return `dataset://new-json/${crypto.createHash('sha256').update(identity).digest('hex')}`;
  }
};

const rows = JSON.parse(fs.readFileSync(input, 'utf8'));
const selected = rows
  .filter((item) => {
    const timestamp = Date.parse(item.timestamp);
    return Number.isFinite(timestamp) && timestamp >= cutoff && timestamp <= now;
  })
  .map((item) => {
    const score = Number(item.sentimentScore);
    const raw = JSON.stringify({
      id: item.id || null,
      content: item.summary || item.snippet || '',
      snippet: item.snippet || '',
      sentiment: item.sentiment || null,
      coins: item.coin || [],
      tag: item.tag || null,
      origin: 'new.json',
    });
    return [
      new Date(item.timestamp).toISOString(),
      canonicalUrl(item.url, item),
      item.source || 'unknown',
      item.title || 'Không có tiêu đề',
      Number.isFinite(score) ? Math.max(-1, Math.min(1, score)) : 0,
      raw,
    ];
  });

const lines = [
  ['time', 'url', 'source', 'title', 'sentiment_score', 'raw_score'].map(csv).join(','),
  ...selected.map((row) => row.map(csv).join(',')),
];
fs.writeFileSync(output, lines.join('\n'), 'utf8');
console.log(`exported=${selected.length} output=${output}`);

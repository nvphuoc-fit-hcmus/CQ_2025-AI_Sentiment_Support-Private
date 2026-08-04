CREATE TEMP TABLE articles_max_import (
  time timestamptz,
  url text,
  source text,
  title text,
  sentiment_score double precision,
  raw_score jsonb
);

COPY articles_max_import (time, url, source, title, sentiment_score, raw_score)
FROM '/tmp/articles_db_import.csv'
WITH (FORMAT csv, HEADER true);

INSERT INTO news_sentiment (time, url, source, title, sentiment_score, raw_score)
SELECT DISTINCT ON (time, url)
  time, url, source, title, sentiment_score, raw_score
FROM articles_max_import
WHERE time IS NOT NULL AND url IS NOT NULL
ORDER BY time, url
ON CONFLICT (time, url) DO NOTHING;

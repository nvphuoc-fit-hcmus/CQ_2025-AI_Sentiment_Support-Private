import React, { useCallback, useEffect, useState } from 'react';
import { ExternalLink, Newspaper, RefreshCw, Search, SlidersHorizontal, TrendingDown, TrendingUp, Minus } from 'lucide-react';
import useStore from '../store';

const TIME_OPTIONS = [
    { value: '24h', label: '24 giờ qua', hours: 24 },
    { value: '7d', label: '7 ngày qua', hours: 24 * 7 },
    { value: '30d', label: '30 ngày qua', hours: 24 * 30 },
    { value: 'all', label: 'Tất cả', hours: null },
];

const sentimentInfo = (score) => {
    const value = Number(score || 0);
    if (value > 0.1) return { key: 'positive', label: 'Tích cực', icon: TrendingUp };
    if (value < -0.1) return { key: 'negative', label: 'Tiêu cực', icon: TrendingDown };
    return { key: 'neutral', label: 'Trung lập', icon: Minus };
};

const formatTime = (value) => {
    const date = new Date(value);
    if (Number.isNaN(date.getTime())) return 'Không rõ thời gian';
    return date.toLocaleString('vi-VN', {
        hour: '2-digit', minute: '2-digit', day: '2-digit', month: '2-digit', year: 'numeric'
    });
};

const getArticleLink = (article) => {
    const rawUrl = String(article.url || '').trim();
    if (/^https?:\/\//i.test(rawUrl)) {
        return { url: rawUrl, label: 'Đọc bài gốc', isSearch: false };
    }
    const searchQuery = [`"${article.title || ''}"`, article.source || '', 'crypto']
        .filter(Boolean)
        .join(' ');
    return {
        url: `https://www.google.com/search?q=${encodeURIComponent(searchQuery)}`,
        label: 'Tìm bài gốc',
        isSearch: true,
    };
};

export default function NewsDashboard() {
    const { authFetch } = useStore();
    const [articles, setArticles] = useState([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState('');
    const [query, setQuery] = useState('');
    const [debouncedQuery, setDebouncedQuery] = useState('');
    const [source, setSource] = useState('all');
    const [timeRange, setTimeRange] = useState('all');
    const [sentiment, setSentiment] = useState('all');
    const [lastUpdated, setLastUpdated] = useState(null);
    const [page, setPage] = useState(1);
    const [pageSize, setPageSize] = useState(50);
    const [totalArticles, setTotalArticles] = useState(0);
    const [serverStats, setServerStats] = useState({ positive: 0, neutral: 0, negative: 0 });
    const [serverSources, setServerSources] = useState([]);
    const [dataRange, setDataRange] = useState(null);

    const loadNews = useCallback(async () => {
        setLoading(true);
        setError('');
        try {
            // authFetch automatically prefixes regular endpoints with /api.
            const offset = (page - 1) * pageSize;
            const selectedTime = TIME_OPTIONS.find(item => item.value === timeRange);
            const params = new URLSearchParams({
                all: 'true',
                limit: String(pageSize),
                offset: String(offset),
            });
            if (selectedTime?.hours) {
                params.set('start', new Date(Date.now() - selectedTime.hours * 3600000).toISOString());
            }
            if (source !== 'all') params.set('source', source);
            if (sentiment !== 'all') params.set('sentiment', sentiment);
            if (debouncedQuery) params.set('search', debouncedQuery);
            // The API merges and de-duplicates TimescaleDB with current
            // crawler data, but returns only the requested page.
            const response = await authFetch(`/v1/news?${params.toString()}`);
            if (!response.ok) throw new Error(`Không thể tải tin tức (${response.status})`);
            const data = await response.json();
            setArticles(Array.isArray(data.rows) ? data.rows : []);
            setTotalArticles(Number(data.total || 0));
            setServerStats(data.stats || { positive: 0, neutral: 0, negative: 0 });
            setServerSources(Array.isArray(data.sources) ? data.sources : []);
            setDataRange(data.range ? {
                newest: new Date(data.range.newest),
                oldest: new Date(data.range.oldest),
            } : null);
            setLastUpdated(new Date());
        } catch (requestError) {
            setError(requestError.message || 'Không thể tải tin tức');
        } finally {
            setLoading(false);
        }
    }, [authFetch, page, pageSize, debouncedQuery, source, timeRange, sentiment]);

    useEffect(() => {
        loadNews();
        const timer = setInterval(loadNews, 60000);
        return () => clearInterval(timer);
    }, [loadNews]);

    useEffect(() => {
        setPage(1);
    }, [query, source, timeRange, sentiment, pageSize]);

    useEffect(() => {
        const timer = window.setTimeout(() => setDebouncedQuery(query.trim()), 350);
        return () => window.clearTimeout(timer);
    }, [query]);

    const totalPages = Math.max(1, Math.ceil(totalArticles / pageSize));

    const sources = serverSources;
    const filtered = articles;
    const stats = serverStats;

    return (
        <main className="news-dashboard">
            <section className="news-hero">
                <div>
                    <span className="news-eyebrow"><Newspaper size={13} /> TRUNG TÂM TIN TỨC</span>
                    <h1>Dòng tin thị trường</h1>
                    <p>Tổng hợp nguồn tin crypto và đánh giá sắc thái nội dung theo thời gian thực.</p>
                </div>
                <button className="news-refresh" onClick={loadNews} disabled={loading}>
                    <RefreshCw size={15} className={loading ? 'spin' : ''} />
                    {loading ? 'Đang cập nhật' : 'Cập nhật'}
                </button>
            </section>

            <section className="news-summary">
                <div className="news-summary-total">
                    <span>TIN PHÙ HỢP</span>
                    <strong>{totalArticles}</strong>
                    <small>
                        {dataRange
                            ? `Dữ liệu DB đến ${dataRange.newest.toLocaleDateString('vi-VN')}`
                            : lastUpdated
                                ? `Cập nhật lúc ${lastUpdated.toLocaleTimeString('vi-VN', { hour: '2-digit', minute: '2-digit' })}`
                                : 'Đang đồng bộ dữ liệu'}
                    </small>
                </div>
                <div className="news-stat positive"><TrendingUp size={17} /><span>Tích cực</span><strong>{stats.positive}</strong></div>
                <div className="news-stat neutral"><Minus size={17} /><span>Trung lập</span><strong>{stats.neutral}</strong></div>
                <div className="news-stat negative"><TrendingDown size={17} /><span>Tiêu cực</span><strong>{stats.negative}</strong></div>
            </section>

            <section className="news-filter-bar">
                <div className="news-search">
                    <Search size={15} />
                    <input value={query} onChange={event => setQuery(event.target.value)}
                        placeholder="Tìm theo tiêu đề hoặc nguồn..." />
                </div>
                <div className="news-filter-label"><SlidersHorizontal size={14} /> Bộ lọc</div>
                <select value={source} onChange={event => setSource(event.target.value)}>
                    <option value="all">Tất cả nguồn</option>
                    {sources.map(item => <option value={item} key={item}>{item}</option>)}
                </select>
                <select value={timeRange} onChange={event => setTimeRange(event.target.value)}>
                    {TIME_OPTIONS.map(item => <option value={item.value} key={item.value}>{item.label}</option>)}
                </select>
                <select value={sentiment} onChange={event => setSentiment(event.target.value)}>
                    <option value="all">Mọi sentiment</option>
                    <option value="positive">Tích cực</option>
                    <option value="neutral">Trung lập</option>
                    <option value="negative">Tiêu cực</option>
                </select>
            </section>

            {error && <div className="news-error">{error}</div>}

            <section className="news-feed">
                {loading && articles.length === 0 ? (
                    Array.from({ length: 7 }).map((_, index) => <div className="news-card news-skeleton" key={index} />)
                ) : filtered.length === 0 ? (
                    <div className="news-empty">
                        <Newspaper size={30} />
                        <strong>Không có tin phù hợp</strong>
                        <span>Hãy thử thay đổi nguồn, thời gian hoặc sentiment.</span>
                    </div>
                ) : filtered.map((article, index) => {
                    const info = sentimentInfo(article.sentiment_score);
                    const Icon = info.icon;
                    const score = Number(article.sentiment_score || 0);
                    const articleLink = getArticleLink(article);
                    return (
                        <article className="news-card" key={`${article.url || article.title}-${index}`}>
                            <div className={`news-sentiment-rail ${info.key}`} />
                            <div className="news-card-main">
                                <div className="news-card-meta">
                                    <span className="news-source">{article.source || 'Nguồn tổng hợp'}</span>
                                    <span className="news-dot">•</span>
                                    <time>{formatTime(article.time)}</time>
                                </div>
                                <h2>{article.title || 'Tin tức thị trường'}</h2>
                                <div className="news-card-footer">
                                    <span className={`news-sentiment ${info.key}`}>
                                        <Icon size={13} /> {info.label}
                                        <b>{score > 0 ? '+' : ''}{(score * 100).toFixed(0)}%</b>
                                    </span>
                                    <a href={articleLink.url} target="_blank" rel="noreferrer"
                                        title={articleLink.isSearch ? 'Dữ liệu lịch sử không lưu URL; tìm theo tiêu đề và nguồn' : 'Mở bài viết gốc'}>
                                        {articleLink.label} <ExternalLink size={13} />
                                    </a>
                                </div>
                            </div>
                        </article>
                    );
                })}
            </section>

            {!loading && totalArticles > 0 && (
                <nav className="news-pagination" aria-label="Phân trang tin tức">
                    <div className="news-page-size">
                        <span>Hiển thị</span>
                        <select value={pageSize} onChange={event => setPageSize(Number(event.target.value))}>
                            <option value={50}>50 tin</option>
                            <option value={100}>100 tin</option>
                        </select>
                        <span>mỗi trang</span>
                    </div>
                    <div className="news-page-controls">
                        <button type="button" disabled={page <= 1 || loading} onClick={() => setPage(value => Math.max(1, value - 1))}>
                            ← Trước
                        </button>
                        <strong>Trang {page} / {totalPages}</strong>
                        <button type="button" disabled={page >= totalPages || loading} onClick={() => setPage(value => Math.min(totalPages, value + 1))}>
                            Sau →
                        </button>
                    </div>
                    <span className="news-page-total">{totalArticles.toLocaleString('vi-VN')} tin</span>
                </nav>
            )}
        </main>
    );
}

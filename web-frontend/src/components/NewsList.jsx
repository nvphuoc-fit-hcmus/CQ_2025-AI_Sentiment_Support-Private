import React, { useEffect, useState } from 'react';
import useStore from '../store';
import { ExternalLink, Clock, RefreshCw } from 'lucide-react';
import { NewsListSkeleton } from './LoadingSpinner';
import { useTheme } from './ThemeProvider';

const parseRawNews = (raw) => {
    if (raw == null) return null;
    if (typeof raw === 'string') {
        try {
            return JSON.parse(raw);
        } catch {
            const numeric = Number(raw);
            return Number.isFinite(numeric) ? numeric : null;
        }
    }
    return raw;
};

const normalizeNewsItem = (item = {}) => {
    const raw = parseRawNews(item.raw_score);
    const rawObject = raw && typeof raw === 'object' && !Array.isArray(raw) ? raw : {};
    const nestedSentiment = rawObject.sentiment_score
        ?? rawObject.sentiment
        ?? rawObject.score;
    const rawNumeric = typeof raw === 'number' ? raw : Number(nestedSentiment);
    const storedScore = Number(item.sentiment_score);
    // Older rows were persisted as zero even though raw_score retained the
    // analyzed value. Prefer that recoverable value only for those legacy rows.
    const sentimentScore = Number.isFinite(rawNumeric) && storedScore === 0 && rawNumeric !== 0
        ? rawNumeric
        : (Number.isFinite(storedScore) ? storedScore : (Number.isFinite(rawNumeric) ? rawNumeric : 0));
    const sentimentLabel = sentimentScore > 0.1
        ? 'Tích cực'
        : sentimentScore < -0.1
            ? 'Tiêu cực'
            : 'Trung lập';

    return {
        ...rawObject,
        ...item,
        title: item.title || rawObject.title || 'Không có tiêu đề',
        source: item.source || rawObject.source || 'Không rõ nguồn',
        url: item.url || rawObject.url || rawObject.link || null,
        content: item.content || rawObject.content || rawObject.summary || rawObject.description || '',
        published_at: item.published_at || item.time || rawObject.published_at || rawObject.time,
        relevance_score: item.relevance_score ?? rawObject.relevance_score,
        sentiment_score: Math.max(-1, Math.min(1, sentimentScore)),
        sentiment_label: sentimentLabel,
    };
};

export default function NewsList() {
    const { authFetch, currentSymbol } = useStore();
    const { isDark } = useTheme();
    const [news, setNews] = useState([]);
    const [loading, setLoading] = useState(false);
    const [page, setPage] = useState(1);
    const [total, setTotal] = useState(0);
    const [selectedNews, setSelectedNews] = useState(null);
    const [modalOpen, setModalOpen] = useState(false);
    const LIMIT = 10;

    useEffect(() => {
        setPage(1); // Reset page on symbol change
        loadNews(1);
    }, [currentSymbol]);

    useEffect(() => {
        loadNews(page);
    }, [page]); // Reload when page changes

    // Auto refresh current page every minute
    useEffect(() => {
        const interval = setInterval(() => loadNews(page), 60000);
        return () => clearInterval(interval);
    }, [page, currentSymbol]);

    const loadNews = async (pageNum) => {
        setLoading(true);
        try {
            const offset = (pageNum - 1) * LIMIT;
            const res = await authFetch(`/v1/news?limit=${LIMIT}&offset=${offset}`);
            if (res.ok) {
                const data = await res.json();
                setNews(data.rows || []);
                if (data.total) setTotal(data.total);
            }
        } catch (e) {
            console.error(e);
        } finally {
            setLoading(false);
        }
    };

    const totalPages = Math.ceil(total / LIMIT);

    const handlePrev = () => {
        if (page > 1) setPage(p => p - 1);
    };

    const handleNext = () => {
        if (page < totalPages) setPage(p => p + 1);
    };

    const openNewsModal = (newsItem) => {
        setSelectedNews(newsItem);
        setModalOpen(true);
    };

    const formatNewsTime = (dateStr) => {
        if (!dateStr) return 'N/A';
        try {
            const date = new Date(dateStr);
            if (isNaN(date.getTime())) return 'N/A';
            return date.toLocaleTimeString('vi-VN', { hour: '2-digit', minute: '2-digit', hour12: false });
        } catch (e) {
            return 'N/A';
        }
    };

    // Show skeleton when loading initially (page 1) or when explicitly loading new page data
    if (loading && news.length === 0) {
        return (
            <div className="news-list" style={{ padding: '12px' }}>
                <NewsListSkeleton count={LIMIT} />
            </div>
        );
    }

    return (
        <div className="news-list" style={{ position: 'relative', display: 'flex', flexDirection: 'column', height: '100%' }}>
            {/* Loading overlay when refreshing in background */}
            {loading && news.length > 0 && (
                <div style={{
                    position: 'absolute',
                    top: 8,
                    right: 8,
                    zIndex: 10
                }}>
                    <RefreshCw size={14} className="spinning" style={{ color: 'var(--accent-blue)' }} />
                </div>
            )}

            <div style={{ flex: 1, overflowY: 'auto' }}>
                {loading ? (
                    <div style={{ padding: '12px' }}><NewsListSkeleton count={LIMIT} /></div>
                ) : news.length > 0 ? (
                    news.map((item, idx) => {
                        const newsDetail = normalizeNewsItem(item);
                        const sentimentScore = newsDetail.sentiment_score;
                        const sentimentLabel = newsDetail.sentiment_label;
                        const relevanceScore = Number(newsDetail.relevance_score || 0);
                        const content = newsDetail.content || '';
                        const snippet = content.substring(0, 150) + (content.length > 150 ? '...' : '');

                        return (
                            <div
                                key={idx}
                                className="news-item"
                                style={{
                                    animation: `fadeIn 0.3s ease-out ${idx * 0.05}s both`,
                                    cursor: 'pointer',
                                    transition: 'all 0.2s'
                                }}
                                onClick={() => openNewsModal(item)}
                                onMouseEnter={(e) => {
                                    e.currentTarget.style.backgroundColor = isDark ? 'rgba(255,255,255,0.05)' : 'rgba(0,0,0,0.03)';
                                }}
                                onMouseLeave={(e) => {
                                    e.currentTarget.style.backgroundColor = 'transparent';
                                }}
                            >
                                <div className="news-header">
                                    <span className="source-tag">
                                        {item.source || (item.url ? new URL(item.url).hostname.replace('www.', '') : 'Unknown')}
                                    </span>
                                    <span className="time-tag">
                                        <Clock size={10} />
                                        {formatNewsTime(newsDetail.published_at || item.time)}
                                    </span>
                                </div>

                                <div className="news-title" style={{
                                    color: isDark ? '#FFC107' : '#F57C00',
                                    fontWeight: 'bold',
                                    marginBottom: '8px'
                                }}>
                                        {newsDetail.title}
                                </div>

                                {/* Content Snippet */}
                                {snippet && (
                                    <div style={{
                                        fontSize: '12px',
                                        color: isDark ? 'rgba(255,255,255,0.6)' : 'rgba(0,0,0,0.6)',
                                        marginBottom: '8px',
                                        lineHeight: '1.4'
                                    }}>
                                        {snippet}
                                    </div>
                                )}

                                {/* Sentiment & Relevance Badges */}
                                <div style={{ display: 'flex', gap: '6px', flexWrap: 'wrap', marginTop: '8px' }}>
                                    {/* Sentiment Badge */}
                                    <span style={{
                                        display: 'inline-block',
                                        padding: '3px 8px',
                                        borderRadius: '12px',
                                        fontSize: '10px',
                                        fontWeight: 'bold',
                                        backgroundColor: sentimentScore > 0.3
                                            ? 'rgba(76, 175, 80, 0.2)'
                                            : (sentimentScore < -0.3 ? 'rgba(244, 67, 54, 0.2)' : 'rgba(33, 150, 243, 0.2)'),
                                        color: sentimentScore > 0.3
                                            ? '#4CAF50'
                                            : (sentimentScore < -0.3 ? '#F44336' : '#2196F3')
                                    }}>
                                        {sentimentLabel} ({(sentimentScore * 100).toFixed(0)}%)
                                    </span>

                                    {/* Relevance Badge */}
                                    {relevanceScore > 0 && (
                                        <span style={{
                                            display: 'inline-block',
                                            padding: '3px 8px',
                                            borderRadius: '12px',
                                            fontSize: '10px',
                                            fontWeight: 'bold',
                                            backgroundColor: isDark ? 'rgba(156, 39, 176, 0.2)' : 'rgba(156, 39, 176, 0.3)',
                                            color: '#9C27B0'
                                        }}>
                                            🎯 {(relevanceScore * 100).toFixed(0)}%
                                        </span>
                                    )}

                                    {(newsDetail.symbol || newsDetail.symbols?.[0]) && (
                                        <span className="symbol-tag">{newsDetail.symbol || newsDetail.symbols[0]}</span>
                                    )}
                                </div>
                            </div>
                        );
                    })
                ) : (
                    <div style={{ padding: '20px', textAlign: 'center', color: 'var(--text-secondary)', fontSize: '13px' }}>
                        Không có tin tức nào ở trang này.
                    </div>
                )}
            </div>

            {/* Pagination Controls */}
            <div className="pagination-controls" style={{
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center',
                padding: '10px 12px',
                borderTop: '1px solid var(--border-color)',
                marginTop: 'auto'
            }}>
                <button
                    onClick={handlePrev}
                    disabled={page === 1 || loading}
                    className="pagination-btn"
                    style={{
                        background: 'transparent',
                        border: '1px solid var(--border-color)',
                        color: page === 1 ? 'var(--text-tertiary)' : 'var(--text-primary)',
                        padding: '4px 10px',
                        borderRadius: '4px',
                        cursor: page === 1 || loading ? 'not-allowed' : 'pointer',
                        fontSize: '12px',
                        display: 'flex',
                        alignItems: 'center',
                        gap: '4px',
                        opacity: page === 1 ? 0.5 : 1
                    }}
                >
                    &lt; Trước
                </button>

                <span style={{ fontSize: '12px', color: 'var(--text-secondary)' }}>
                    Trang {page} / {totalPages || '...'}
                </span>

                <button
                    onClick={handleNext}
                    disabled={loading || page >= totalPages || news.length === 0}
                    className="pagination-btn"
                    style={{
                        background: 'transparent',
                        border: '1px solid var(--border-color)',
                        color: (loading || page >= totalPages) ? 'var(--text-tertiary)' : 'var(--text-primary)',
                        padding: '4px 10px',
                        borderRadius: '4px',
                        cursor: (loading || page >= totalPages) ? 'not-allowed' : 'pointer',
                        fontSize: '12px',
                        display: 'flex',
                        alignItems: 'center',
                        gap: '4px',
                        opacity: (loading || page >= totalPages) ? 0.5 : 1
                    }}
                >
                    Sau &gt;
                </button>
            </div>

            {/* News Detail Modal - Reuse same modal from MultiTimeframeChart */}
            {modalOpen && selectedNews && (
                <div
                    style={{
                        position: 'fixed',
                        top: 0,
                        left: 0,
                        right: 0,
                        bottom: 0,
                        backgroundColor: 'rgba(0, 0, 0, 0.8)',
                        backdropFilter: 'blur(4px)',
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        zIndex: 1000,
                        padding: '20px'
                    }}
                    onClick={() => setModalOpen(false)}
                >
                    <div
                        style={{
                            backgroundColor: isDark ? '#1a1e2e' : '#ffffff',
                            borderRadius: '12px',
                            maxWidth: '700px',
                            width: '100%',
                            maxHeight: '80vh',
                            overflow: 'auto',
                            boxShadow: '0 20px 60px rgba(0,0,0,0.5)',
                            position: 'relative'
                        }}
                        onClick={(e) => e.stopPropagation()}
                    >
                        {/* Close Button */}
                        <button
                            onClick={() => setModalOpen(false)}
                            style={{
                                position: 'absolute',
                                top: '16px',
                                right: '16px',
                                background: isDark ? 'rgba(255,255,255,0.1)' : 'rgba(0,0,0,0.1)',
                                border: 'none',
                                borderRadius: '50%',
                                width: '36px',
                                height: '36px',
                                cursor: 'pointer',
                                display: 'flex',
                                alignItems: 'center',
                                justifyContent: 'center',
                                fontSize: '20px',
                                color: isDark ? '#fff' : '#333',
                                transition: 'all 0.2s',
                                zIndex: 1
                            }}
                            onMouseEnter={(e) => {
                                e.target.style.background = isDark ? 'rgba(255,255,255,0.2)' : 'rgba(0,0,0,0.2)';
                            }}
                            onMouseLeave={(e) => {
                                e.target.style.background = isDark ? 'rgba(255,255,255,0.1)' : 'rgba(0,0,0,0.1)';
                            }}
                        >
                            ✕
                        </button>

                        {/* Modal Content */}
                        <div style={{ padding: '32px' }}>
                            {(() => {
                                const newsDetail = normalizeNewsItem(selectedNews);
                                const sentimentScore = newsDetail.sentiment_score;

                                return (
                                    <>
                                        {/* Sentiment Badge */}
                                        <div style={{ marginBottom: '16px', display: 'flex', gap: '8px', flexWrap: 'wrap' }}>
                                            <span style={{
                                                display: 'inline-block',
                                                padding: '6px 12px',
                                                borderRadius: '20px',
                                                fontSize: '12px',
                                                fontWeight: 'bold',
                                                backgroundColor: sentimentScore > 0.3
                                                    ? 'rgba(76, 175, 80, 0.2)'
                                                    : (sentimentScore < -0.3 ? 'rgba(244, 67, 54, 0.2)' : 'rgba(33, 150, 243, 0.2)'),
                                                color: sentimentScore > 0.3
                                                    ? '#4CAF50'
                                                    : (sentimentScore < -0.3 ? '#F44336' : '#2196F3')
                                            }}>
                                                {sentimentScore > 0.3 ? '📈 Tích cực' : (sentimentScore < -0.3 ? '📉 Tiêu cực' : '➖ Trung lập')}
                                                {' '}
                                                ({(sentimentScore * 100).toFixed(1)}%)
                                            </span>

                                            {newsDetail.category && (
                                                <span style={{
                                                    display: 'inline-block',
                                                    padding: '6px 12px',
                                                    borderRadius: '20px',
                                                    fontSize: '12px',
                                                    fontWeight: 'bold',
                                                    backgroundColor: isDark ? 'rgba(255, 193, 7, 0.2)' : 'rgba(255, 193, 7, 0.3)',
                                                    color: isDark ? '#FFC107' : '#F57C00'
                                                }}>
                                                    📂 {newsDetail.category}
                                                </span>
                                            )}

                                            {newsDetail.relevance_score !== undefined && (
                                                <span style={{
                                                    display: 'inline-block',
                                                    padding: '6px 12px',
                                                    borderRadius: '20px',
                                                    fontSize: '12px',
                                                    fontWeight: 'bold',
                                                    backgroundColor: isDark ? 'rgba(156, 39, 176, 0.2)' : 'rgba(156, 39, 176, 0.3)',
                                                    color: '#9C27B0'
                                                }}>
                                                    🎯 Relevance: {(newsDetail.relevance_score * 100).toFixed(0)}%
                                                </span>
                                            )}
                                        </div>

                                        {/* Title */}
                                        <h2 style={{
                                            margin: '0 0 16px 0',
                                            fontSize: '24px',
                                            fontWeight: 'bold',
                                            color: isDark ? '#FFC107' : '#F57C00',
                                            lineHeight: '1.4',
                                            paddingRight: '40px'
                                        }}>
                                            {newsDetail.title || 'Không có tiêu đề'}
                                        </h2>

                                        {/* Meta Info */}
                                        <div style={{
                                            display: 'flex',
                                            gap: '16px',
                                            marginBottom: '16px',
                                            fontSize: '14px',
                                            color: isDark ? 'rgba(255,255,255,0.7)' : 'rgba(0,0,0,0.6)',
                                            flexWrap: 'wrap'
                                        }}>
                                            <div>
                                                <strong>📍 Nguồn:</strong> {newsDetail.source || 'Unknown'}
                                            </div>
                                            <div>
                                                <strong>🕒 Thời gian:</strong> {(() => { const d = new Date(newsDetail.published_at || newsDetail.time || selectedNews.time); return isNaN(d.getTime()) ? 'N/A' : d.toLocaleString('vi-VN'); })()}
                                            </div>
                                        </div>

                                        {/* Symbols */}
                                        {newsDetail.symbols && newsDetail.symbols.length > 0 && (
                                            <div style={{
                                                marginBottom: '16px',
                                                fontSize: '13px',
                                                color: isDark ? 'rgba(255,255,255,0.8)' : 'rgba(0,0,0,0.7)'
                                            }}>
                                                <strong>💱 Symbols:</strong> {newsDetail.symbols.join(', ')}
                                            </div>
                                        )}

                                        {/* Divider */}
                                        <div style={{
                                            height: '1px',
                                            background: isDark ? 'rgba(255,255,255,0.1)' : 'rgba(0,0,0,0.1)',
                                            margin: '24px 0'
                                        }} />

                                        {/* Content */}
                                        {newsDetail.content && (
                                            <div style={{
                                                fontSize: '15px',
                                                lineHeight: '1.8',
                                                color: isDark ? 'rgba(255,255,255,0.9)' : 'rgba(0,0,0,0.8)',
                                                marginBottom: '24px',
                                                maxHeight: '400px',
                                                overflowY: 'auto',
                                                paddingRight: '8px'
                                            }}>
                                                {newsDetail.content}
                                            </div>
                                        )}

                                        {/* Link to Original Article */}
                                        {newsDetail.url && (
                                            <a
                                                href={newsDetail.url}
                                                target="_blank"
                                                rel="noopener noreferrer"
                                                style={{
                                                    display: 'inline-block',
                                                    padding: '12px 24px',
                                                    backgroundColor: isDark ? '#2196F3' : '#1976D2',
                                                    color: '#fff',
                                                    textDecoration: 'none',
                                                    borderRadius: '6px',
                                                    fontWeight: 'bold',
                                                    fontSize: '14px',
                                                    transition: 'all 0.2s',
                                                    boxShadow: '0 2px 8px rgba(33, 150, 243, 0.3)'
                                                }}
                                                onMouseEnter={(e) => {
                                                    e.target.style.backgroundColor = isDark ? '#1976D2' : '#1565C0';
                                                    e.target.style.transform = 'translateY(-2px)';
                                                    e.target.style.boxShadow = '0 4px 12px rgba(33, 150, 243, 0.4)';
                                                }}
                                                onMouseLeave={(e) => {
                                                    e.target.style.backgroundColor = isDark ? '#2196F3' : '#1976D2';
                                                    e.target.style.transform = 'translateY(0)';
                                                    e.target.style.boxShadow = '0 2px 8px rgba(33, 150, 243, 0.3)';
                                                }}
                                            >
                                                🔗 Đọc bài viết đầy đủ
                                            </a>
                                        )}
                                    </>
                                );
                            })()}
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
}

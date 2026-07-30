import React, { useEffect, useMemo, useState } from 'react';
import { BrainCircuit, CheckCircle2, CircleHelp, ExternalLink, Gauge, Newspaper, X } from 'lucide-react';
import { createPortal } from 'react-dom';
import safeAlertService from '../services/safeAlertService';
import { FACTOR_DETAILS } from './FactorHeatmap';

const FACTOR_LABELS = {
    institutional_inflow: 'Dòng vốn tổ chức',
    etf_flow: 'Dòng vốn ETF',
    regulatory_easing: 'Nới lỏng quy định',
    regulatory_tightening: 'Siết chặt quy định',
    exchange_risk: 'Rủi ro sàn giao dịch',
    liquidity_squeeze: 'Áp lực thanh khoản',
    whale_accumulation: 'Cá voi tích lũy',
    macro_uncertainty: 'Bất ổn vĩ mô',
    protocol_upgrade: 'Nâng cấp giao thức',
    network_outage: 'Sự cố mạng lưới',
};

const getDirection = (direction) => {
    const normalized = direction?.toUpperCase();
    if (normalized === 'BUY' || normalized === 'UP') {
        return { label: 'TĂNG', className: 'buy', summary: 'Thiên hướng tăng giá' };
    }
    if (normalized === 'SELL' || normalized === 'DOWN') {
        return { label: 'GIẢM', className: 'sell', summary: 'Thiên hướng giảm giá' };
    }
    return { label: 'GIỮ VỮNG', className: 'hold', summary: 'Xu hướng chưa rõ ràng' };
};

export default function DetailedAnalysisModal({
    open,
    onClose,
    symbol,
    horizon,
    signal,
    onSelectEvidence,
}) {
    const [generatedText, setGeneratedText] = useState('');
    const [isGenerating, setIsGenerating] = useState(false);
    const direction = useMemo(
        () => getDirection(signal?.direction || signal?.signal),
        [signal?.direction, signal?.signal],
    );

    useEffect(() => {
        if (!open) return undefined;
        const previousOverflow = document.body.style.overflow;
        const handleKeyDown = (event) => {
            if (event.key === 'Escape') onClose();
        };
        document.body.style.overflow = 'hidden';
        window.addEventListener('keydown', handleKeyDown);
        return () => {
            document.body.style.overflow = previousOverflow;
            window.removeEventListener('keydown', handleKeyDown);
        };
    }, [open, onClose]);

    useEffect(() => {
        if (!open || !signal) return undefined;
        let cancelled = false;
        setGeneratedText('');
        setIsGenerating(true);
        safeAlertService.rewriteExplanation({
            symbol,
            horizon,
            direction: signal.direction || signal.signal,
            confidence: signal.confidence || 0,
            probabilities: signal.probs || {},
            factors: signal.topFactors || [],
            article_titles: (signal.articles || []).map((article) => article.title).filter(Boolean),
            article_evidence: (signal.articles || []).map((article) => ({
                title: article.title,
                source: article.source || 'SAFE-Alert',
                summary: article.content || article.summary || '',
                relevance: article.relevanceScore,
            })).filter((article) => article.title),
        })
            .then((result) => {
                if (!cancelled && result?.text) setGeneratedText(result.text);
            })
            .catch(() => {
                if (!cancelled) setGeneratedText('');
            })
            .finally(() => {
                if (!cancelled) setIsGenerating(false);
            });
        return () => { cancelled = true; };
    }, [open, signal, symbol, horizon]);

    if (!open || !signal) return null;

    const confidence = Math.round((signal.confidence || 0) * 100);
    const factors = (signal.topFactors || []).map((factor) => ({
        key: factor,
        label: FACTOR_LABELS[factor] || factor.replaceAll('_', ' '),
        detail: FACTOR_DETAILS[factor] || {
            description: 'Yếu tố có khả năng ảnh hưởng đến tâm lý và diễn biến của thị trường.',
            example: 'Mức độ ảnh hưởng cần được đối chiếu với các tin tức được chọn trong từng thời điểm.',
        },
    }));
    const articles = signal.articles || [];
    const probabilities = signal.probs || {};
    const factorNames = factors.map((factor) => factor.label);
    const fallbackExplanation = `${direction.summary} đang được ghi nhận trong khung ${horizon.toUpperCase()}, với độ tin cậy ${confidence}%. `
        + (articles.length > 0
            ? `Để hình thành nhận định này, mô hình đã đối chiếu biến động thị trường với ${articles.length} bài viết có mức liên quan cao nhất trong cửa sổ phân tích. `
            : 'Tuy nhiên, hiện chưa có bài viết phù hợp vượt qua ngưỡng chọn lọc để củng cố thêm cho kết luận. ')
        + (factorNames.length > 0
            ? `Các thông tin được chọn cho thấy những chủ đề cần chú ý gồm ${factorNames.join(', ')}; trong đó ${factorNames[0]} đang có vai trò nổi bật hơn các yếu tố còn lại. `
            : '')
        + `Mức tin cậy ${confidence}% cho thấy đây chưa phải là một kết luận chắc chắn và tín hiệu vẫn có thể thay đổi khi xuất hiện dữ liệu mới. `
        + 'Vì vậy, kết quả nên được xem như một góc nhìn tham khảo, đồng thời cần đối chiếu thêm với diễn biến giá, khối lượng giao dịch và các thông tin mới trước khi đưa ra quyết định.';

    return createPortal(
        <div className="analysis-expanded-overlay" onMouseDown={onClose}>
            <section
                className="analysis-expanded-modal"
                role="dialog"
                aria-modal="true"
                aria-labelledby="analysis-expanded-title"
                onMouseDown={(event) => event.stopPropagation()}
            >
                <header className="analysis-expanded-header">
                    <div>
                        <div className="analysis-expanded-tags">
                            <span className="primary">Báo cáo thị trường</span>
                            <span>{symbol?.replace('USDT', '') || 'BTC'}</span>
                            <span>{horizon.toUpperCase()}</span>
                        </div>
                        <h2 id="analysis-expanded-title">Nhận định thị trường {symbol?.replace('USDT', '') || 'BTC'}</h2>
                        <p>Tổng hợp diễn biến, yếu tố ảnh hưởng và các nguồn tin đáng chú ý.</p>
                    </div>

                    <div className="analysis-expanded-outcome">
                        <span className={`analysis-expanded-signal ${direction.className}`}>
                            {direction.label}
                        </span>
                        <div
                            className={`analysis-expanded-confidence ${direction.className}`}
                            style={{ '--confidence': `${confidence * 3.6}deg` }}
                            aria-label={`Độ tin cậy ${confidence}%`}
                        >
                            <div>
                                <strong>{confidence}%</strong>
                                <span>Tin cậy</span>
                            </div>
                        </div>
                        <button type="button" className="analysis-expanded-close" onClick={onClose} aria-label="Đóng">
                            <X size={20} />
                        </button>
                    </div>
                </header>

                <div className="analysis-expanded-body">
                    <div className="analysis-expanded-grid">
                        <article className="analysis-expanded-card overview">
                            <div className="analysis-expanded-section-title">
                                <Gauge size={15} />
                                Tổng quan
                            </div>
                            <div className="analysis-expanded-overview-heading">
                                <span>Khung {horizon.toUpperCase()}</span>
                                <h3>{direction.summary}</h3>
                            </div>
                            <p>
                                Xác suất tăng <strong className="up">{Math.round((probabilities.UP || 0) * 100)}%</strong>,
                                trung tính <strong>{Math.round((probabilities.NEUTRAL || 0) * 100)}%</strong> và
                                giảm <strong className="down">{Math.round((probabilities.DOWN || 0) * 100)}%</strong>.
                                Kết luận được đưa ra từ dữ liệu thị trường cùng {articles.length} tin tức có mức liên quan cao.
                            </p>
                            <div className="analysis-expanded-probabilities">
                                <span className="up" style={{ width: `${(probabilities.UP || 0) * 100}%` }} />
                                <span className="neutral" style={{ width: `${(probabilities.NEUTRAL || 0) * 100}%` }} />
                                <span className="down" style={{ width: `${(probabilities.DOWN || 0) * 100}%` }} />
                            </div>
                            <div className="analysis-expanded-prob-labels">
                                <span>Tăng {Math.round((probabilities.UP || 0) * 100)}%</span>
                                <span>Trung tính {Math.round((probabilities.NEUTRAL || 0) * 100)}%</span>
                                <span>Giảm {Math.round((probabilities.DOWN || 0) * 100)}%</span>
                            </div>
                        </article>

                        <article className="analysis-expanded-card factors">
                            <div className="analysis-expanded-section-title">
                                <BrainCircuit size={15} />
                                Yếu tố ảnh hưởng
                            </div>
                            <div className="analysis-expanded-factor-list">
                                {factors.length > 0 ? factors.map((factor, index) => (
                                    <div
                                        className="analysis-expanded-factor"
                                        key={factor.key}
                                        tabIndex={0}
                                        aria-describedby={`expanded-factor-${factor.key}`}
                                    >
                                        <span className="rank">{index + 1}</span>
                                        <span>{factor.label}</span>
                                        {index === 0 && <strong>Nổi bật nhất</strong>}
                                        <CircleHelp size={13} className="help" aria-hidden="true" />
                                        <div
                                            className="analysis-expanded-factor-tooltip"
                                            id={`expanded-factor-${factor.key}`}
                                            role="tooltip"
                                        >
                                            <b>{factor.label}</b>
                                            <p>{factor.detail.description}</p>
                                            <span>{factor.detail.example}</span>
                                        </div>
                                    </div>
                                )) : <p>Chưa xác định được yếu tố chi phối đủ rõ ràng.</p>}
                            </div>
                        </article>
                    </div>

                    <article className="analysis-expanded-reasoning">
                        <div className="analysis-expanded-section-title">
                            <BrainCircuit size={15} />
                            Nhận định thị trường
                        </div>
                        {isGenerating ? (
                            <div className="analysis-expanded-writing">
                                <span />
                                Đang tổng hợp nhận định bằng tiếng Việt…
                            </div>
                        ) : (
                            <div className="analysis-expanded-reasoning-text">
                                {generatedText || fallbackExplanation}
                            </div>
                        )}
                        <div className="analysis-expanded-grounded">
                            <CheckCircle2 size={14} />
                            Nhận định được đối chiếu với các nguồn tin hiển thị bên dưới.
                        </div>
                    </article>

                    <article className="analysis-expanded-evidence">
                        <div className="analysis-expanded-section-title">
                            <Newspaper size={15} />
                            Tin tức tham khảo
                            <span className="count">{articles.length}</span>
                        </div>
                        {articles.length > 0 ? (
                            <div className="analysis-expanded-news-list">
                                {articles.map((article, index) => (
                                    <div
                                        className="analysis-expanded-news"
                                        key={`${article.title}-${index}`}
                                        role="button"
                                        tabIndex={0}
                                        onClick={() => onSelectEvidence?.(article)}
                                        onKeyDown={(event) => {
                                            if (event.key === 'Enter' || event.key === ' ') {
                                                event.preventDefault();
                                                onSelectEvidence?.(article);
                                            }
                                        }}
                                        aria-label={`Xem trước bài viết: ${article.title}`}
                                    >
                                        <div className="analysis-expanded-news-head">
                                            <span className="rank">{index + 1}</span>
                                            <div>
                                                <h4>{article.title}</h4>
                                                <p>
                                                    {article.source || 'SAFE-Alert'}
                                                    {article.relevanceScore != null
                                                        ? ` · Liên quan ${Math.round(article.relevanceScore * 100)}%`
                                                        : ''}
                                                </p>
                                            </div>
                                            {article.url && (
                                                <a
                                                    href={article.url}
                                                    target="_blank"
                                                    rel="noopener noreferrer"
                                                    aria-label="Mở bài viết gốc"
                                                    onClick={(event) => event.stopPropagation()}
                                                >
                                                    <ExternalLink size={15} />
                                                </a>
                                            )}
                                        </div>
                                        {article.content && <p className="analysis-expanded-news-summary">{article.content}</p>}
                                    </div>
                                ))}
                            </div>
                        ) : (
                            <div className="analysis-expanded-empty">
                                Chưa có bài viết nào vượt qua ngưỡng chọn lọc ở khung {horizon.toUpperCase()}.
                            </div>
                        )}
                    </article>
                </div>
            </section>
        </div>,
        document.body,
    );
}

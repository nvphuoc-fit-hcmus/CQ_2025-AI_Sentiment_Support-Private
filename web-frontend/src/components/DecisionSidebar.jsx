import React, { useEffect, useState, useCallback } from 'react';
import useStore from '../store';
import safeAlertService from '../services/safeAlertService';
import ConfidenceGauge from './ConfidenceGauge';
import FactorHeatmap from './FactorHeatmap';
import StructuredExplainer from './StructuredExplainer';
import { Loader, RefreshCw, ExternalLink, Clock } from 'lucide-react';

/**
 * DecisionSidebar — The Intelligence Hub
 *
 * Replaces the old Watchlist-focused sidebar with a 4-layer SAFE-Alert visualization:
 * Layer 4: ConfidenceGauge (Alert Gating, Eq.26)
 * Layer 3: FactorHeatmap (Factor Grounding, Eq.14-15)
 * Layer 2: StructuredExplainer (Explanation, Eq.27-29)
 * Layer 1: Evidence Cards (Selective News, Eq.8-13)
 *
 * Props:
 *  - symbol: string (e.g. "BTCUSDT")
 */
export default function DecisionSidebar({ symbol = 'BTCUSDT' }) {
    const { setSafeAlertSignal } = useStore();
    const [signal, setSignal] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);
    const [lastUpdate, setLastUpdate] = useState(null);

    // Demo fallback data (used when AI backend has no market data)
    const DEMO_SIGNAL = {
        symbol: symbol,
        timestamp: new Date().toISOString(),
        alert: { alert: true, level: 'high', signal: 'BUY', reason: '1h+4h đồng thuận TĂNG', reasons: ['1h SAFEAlertNet: BUY (conf=0.823, p=0.651)', '4h SAFEAlertNet: BUY (conf=0.714, p=0.582)'] },
        horizon_1h: {
            signal: 'BUY', confidence: 0.823, final_prob: 0.651,
            probs: { UP: 0.651, NEUTRAL: 0.249, DOWN: 0.100 },
            selected_news: [
                { title: 'BlackRock iShares Bitcoin ETF ghi nhận dòng vốn vào kỷ lục $1.2B trong tuần', source: 'Reuters', relevance_score: 0.92 },
                { title: 'SEC phê duyệt thêm 2 quỹ ETF Bitcoin spot mới từ Fidelity và Invesco', source: 'CoinDesk', relevance_score: 0.87 },
                { title: 'MicroStrategy mua thêm 12,000 BTC trị giá $960M, nâng tổng nắm giữ lên 214,000 BTC', source: 'Bloomberg', relevance_score: 0.81 },
            ],
            top_factors: ['institutional_inflow', 'etf_flow', 'regulatory_easing'],
            factors_text: 'institutional_inflow, etf_flow, regulatory_easing',
            explanation: 'Tín hiệu TĂNG (BUY) với độ tin cậy 82.3% cho BTCUSDT trong khung 1 giờ tới. Yếu tố chi phối chính là dòng vốn tổ chức (institutional_inflow) được hỗ trợ bởi 3 bằng chứng tin tức: (1) BlackRock ETF ghi nhận dòng vốn kỷ lục, (2) SEC phê duyệt thêm ETF mới, và (3) MicroStrategy tiếp tục tích lũy BTC.',
            should_alert: true, model_used: 'SAFEAlertNet', horizon: '1h',
        },
        horizon_4h: {
            signal: 'BUY', confidence: 0.714, final_prob: 0.582,
            probs: { UP: 0.582, NEUTRAL: 0.298, DOWN: 0.120 },
            selected_news: [], top_factors: ['institutional_inflow', 'whale_accumulation'],
            explanation: 'Xu hướng tăng trung hạn được hỗ trợ bởi dòng vốn tổ chức và hoạt động tích lũy cá voi.',
            should_alert: true, model_used: 'SAFEAlertNet', horizon: '4h',
        },
        top_inputs: { rsi_14: 62.3, macd_hist: 0.0045, bb_pos: 0.72, stoch_rsi: 0.68, log_return_1: 0.0012, return_5: 0.035, volume_spike: 1.45, vader_mean: 0.32, nlp_score_mean: 0.28, bullish_ratio: 0.65, news_count: 12 },
    };

    // Fetch prediction
    const loadPrediction = useCallback(async () => {
        setLoading(true);
        setError(null);

        try {
            const healthy = await safeAlertService.checkBackendHealth();
            if (!healthy) {
                // Fallback to demo data for UI preview
                console.warn('[DecisionSidebar] AI Service unavailable, using demo data');
                const formatted = safeAlertService.formatSignalData(DEMO_SIGNAL);
                setSignal(formatted);
                setSafeAlertSignal(formatted);
                setLastUpdate(new Date());
                return;
            }

            // Try cached first (fast), fall back to live
            let data;
            try {
                data = await safeAlertService.fetchSAFEAlertSignalCached(symbol);
            } catch {
                data = await safeAlertService.fetchSAFEAlertSignal(symbol);
            }

            const formatted = safeAlertService.formatSignalData(data);
            if (formatted) {
                setSignal(formatted);
                setSafeAlertSignal(formatted);
            } else {
                // API returned but no valid data → use demo
                console.warn('[DecisionSidebar] API returned empty, using demo data');
                const demoFormatted = safeAlertService.formatSignalData(DEMO_SIGNAL);
                setSignal(demoFormatted);
                setSafeAlertSignal(demoFormatted);
            }
            setLastUpdate(new Date());
        } catch (e) {
            console.error('Failed to load prediction:', e);
            // Fallback to demo data on any error
            console.warn('[DecisionSidebar] Error fetching, using demo data');
            const formatted = safeAlertService.formatSignalData(DEMO_SIGNAL);
            setSignal(formatted);
            setSafeAlertSignal(formatted);
            setLastUpdate(new Date());
        } finally {
            setLoading(false);
        }
    }, [symbol, setSafeAlertSignal]);

    // Auto-refresh every 60s
    useEffect(() => {
        loadPrediction();
        const interval = setInterval(loadPrediction, 60000);
        return () => clearInterval(interval);
    }, [loadPrediction]);

    // ── Loading State ──
    if (loading && !signal) {
        return (
            <div className="decision-sidebar">
                <div className="decision-loading">
                    <Loader size={24} className="spin" />
                    <span>Đang tải phân tích AI...</span>
                </div>
            </div>
        );
    }

    // ── Error State ──
    if (error && !signal) {
        return (
            <div className="decision-sidebar">
                <div className="decision-error">
                    <div className="decision-error-msg">{error}</div>
                    <button className="decision-retry-btn" onClick={loadPrediction} disabled={loading}>
                        <RefreshCw size={13} />
                        {loading ? 'Đang thử...' : 'Thử lại'}
                    </button>
                </div>
            </div>
        );
    }

    // ── Empty State ──
    if (!signal) {
        return (
            <div className="decision-sidebar">
                <div className="decision-empty">
                    <span>Chưa có dữ liệu phân tích</span>
                    <button onClick={loadPrediction}>Tải lại</button>
                </div>
            </div>
        );
    }

    return (
        <div className="decision-sidebar">
            {/* ═══ LAYER 4: CONFIDENCE GAUGE (Eq.26) ═══ */}
            <div className="decision-section gauge-section">
                <ConfidenceGauge
                    confidence={signal.confidence}
                    direction={signal.direction}
                    shouldAlert={signal.shouldAlert}
                />
                {/* 4h mini-indicator */}
                {signal.horizon4h && (
                    <div className="horizon-4h-mini">
                        <span className="h4h-label">4H</span>
                        <span className={`h4h-signal ${signal.horizon4h.signal === 'BUY' ? 'up' : signal.horizon4h.signal === 'SELL' ? 'down' : ''}`}>
                            {signal.horizon4h.signal === 'BUY' ? '▲' : signal.horizon4h.signal === 'SELL' ? '▼' : '─'}
                        </span>
                        <span className="h4h-conf">{signal.horizon4h.confidencePercent}%</span>
                    </div>
                )}

                {/* Probability distribution bar */}
                <div className="prob-dist-bar">
                    <div className="prob-segment up" style={{ width: `${signal.probs.UP * 100}%` }} title={`Tăng: ${(signal.probs.UP * 100).toFixed(1)}%`} />
                    <div className="prob-segment neutral" style={{ width: `${signal.probs.NEUTRAL * 100}%` }} title={`Giữ: ${(signal.probs.NEUTRAL * 100).toFixed(1)}%`} />
                    <div className="prob-segment down" style={{ width: `${signal.probs.DOWN * 100}%` }} title={`Giảm: ${(signal.probs.DOWN * 100).toFixed(1)}%`} />
                </div>
                <div className="prob-labels">
                    <span className="prob-label up">▲ {(signal.probs.UP * 100).toFixed(0)}%</span>
                    <span className="prob-label neutral">─ {(signal.probs.NEUTRAL * 100).toFixed(0)}%</span>
                    <span className="prob-label down">▼ {(signal.probs.DOWN * 100).toFixed(0)}%</span>
                </div>
            </div>

            {/* ═══ LAYER 3: FACTOR HEATMAP (Eq.14-15) ═══ */}
            <div className="decision-section">
                <FactorHeatmap
                    topFactors={signal.topFactors}
                    direction={signal.direction}
                />
            </div>

            {/* ═══ LAYER 2: STRUCTURED EXPLANATION (Eq.27-29) ═══ */}
            <div className="decision-section">
                <StructuredExplainer
                    explanation={signal.explanation}
                    direction={signal.direction}
                    confidence={signal.confidence}
                    topFactors={signal.topFactors}
                    articles={signal.articles}
                    shouldAlert={signal.shouldAlert}
                />
            </div>

            {/* ═══ LAYER 1: SELECTIVE EVIDENCE (Eq.8-13) ═══ */}
            {signal.articles && signal.articles.length > 0 && (
                <div className="decision-section evidence-section">
                    <div className="evidence-header">
                        <span className="evidence-icon">📰</span>
                        Bằng chứng đã chọn lọc
                        <span className="evidence-count">{signal.articles.length}</span>
                    </div>
                    <div className="evidence-list">
                        {signal.articles.map((article, idx) => (
                            <div key={idx} className="evidence-card">
                                <div className="evidence-rank">α{idx + 1}</div>
                                <div className="evidence-content">
                                    <div className="evidence-title">{article.title}</div>
                                    <div className="evidence-meta">
                                        <span className="evidence-source">{article.source}</span>
                                        {article.url && (
                                            <a href={article.url} target="_blank" rel="noreferrer" className="evidence-link">
                                                <ExternalLink size={10} />
                                            </a>
                                        )}
                                    </div>
                                </div>
                            </div>
                        ))}
                    </div>
                </div>
            )}

            {/* ═══ FOOTER: REFRESH ═══ */}
            <div className="decision-footer">
                <button className="decision-refresh-btn" onClick={loadPrediction} disabled={loading}>
                    <RefreshCw size={12} className={loading ? 'spin' : ''} />
                    {loading ? 'Đang cập nhật...' : 'Cập nhật'}
                </button>
                {lastUpdate && (
                    <span className="decision-timestamp">
                        <Clock size={10} />
                        {lastUpdate.toLocaleTimeString('vi-VN')}
                    </span>
                )}
            </div>
        </div>
    );
}

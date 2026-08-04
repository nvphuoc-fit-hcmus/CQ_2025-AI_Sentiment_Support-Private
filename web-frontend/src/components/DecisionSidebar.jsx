import React, { useEffect, useRef, useState, useCallback } from 'react';
import useStore from '../store';
import safeAlertService from '../services/safeAlertService';
import ConfidenceGauge from './ConfidenceGauge';
import FactorHeatmap from './FactorHeatmap';
import StructuredExplainer from './StructuredExplainer';
import DetailedAnalysisModal from './DetailedAnalysisModal';
import { Loader, RefreshCw, ExternalLink, Clock, X, Maximize2 } from 'lucide-react';
import { pushNotification } from '../utils/notificationCenter';

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
    const [analyzing, setAnalyzing] = useState(false);
    const [error, setError] = useState(null);
    const [lastUpdate, setLastUpdate] = useState(null);
    const [signalTimestamp, setSignalTimestamp] = useState(null);
    const [currentTime, setCurrentTime] = useState(() => new Date());
    const [activeHorizon, setActiveHorizon] = useState('1h');
    const [selectedEvidence, setSelectedEvidence] = useState(null);
    const [isExpanded, setIsExpanded] = useState(false);
    const activeSymbolRef = useRef(symbol);

    useEffect(() => {
        activeSymbolRef.current = symbol;
        // Never leave the previous coin's prediction visible while the newly
        // selected coin is loading or being analyzed.
        setSignal(null);
        setSignalTimestamp(null);
        setError(null);
    }, [symbol]);

    const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

    // Fetch prediction
    const loadPrediction = useCallback(async () => {
        setLoading(true);
        setError(null);

        try {
            const healthy = await safeAlertService.checkBackendHealth();
            if (!healthy) {
                throw new Error('AI service đang tạm thời không phản hồi.');
            }

            // Only render a completed model result. A cache miss must never
            // silently become fixed demo confidence or start a duplicate run.
            const data = await safeAlertService.fetchSAFEAlertSignalCached(symbol);

            if (activeSymbolRef.current !== symbol) return;

            const formatted = safeAlertService.formatSignalData(data);
            if (formatted) {
                setSignal(formatted);
                setSafeAlertSignal(formatted);
                if (formatted.shouldAlert || formatted.alertInfo?.alert) {
                    const directionLabel = formatted.direction === 'UP' || formatted.direction === 'BUY'
                        ? 'TĂNG'
                        : formatted.direction === 'DOWN' || formatted.direction === 'SELL' ? 'GIẢM' : 'GIỮ';
                    pushNotification({
                        type: 'safe-alert',
                        title: `SAFE-Alert ${formatted.symbol}: ${directionLabel}`,
                        message: `Khung 1H · Độ tin cậy ${formatted.confidencePercent}%${formatted.topFactors?.[0] ? ` · Yếu tố chính: ${formatted.topFactors[0]}` : ''}.`,
                        dedupeKey: `safe-alert-${formatted.symbol}-${formatted.timestamp?.toISOString?.() || formatted.timestamp}`,
                    });
                }
            } else {
                throw new Error(`Chưa có kết quả mô hình cho ${symbol}.`);
            }
            setSignalTimestamp(data?.timestamp ? new Date(data.timestamp) : null);
            setLastUpdate(new Date());
        } catch (e) {
            if (activeSymbolRef.current !== symbol) return;
            console.error('Failed to load prediction:', e);
            setSignal(null);
            setSafeAlertSignal(null);
            setError(
                e?.message?.includes('404')
                    ? `Chưa có kết quả mô hình cho ${symbol}. Hãy bấm “Phân tích ngay”.`
                    : (e?.message || 'Không thể tải kết quả mô hình lúc này.')
            );
        } finally {
            setLoading(false);
        }
    }, [symbol, setSafeAlertSignal]);

    // A manual refresh starts a real model run. The endpoint returns
    // immediately, then we poll until the newly generated signal is cached.
    const analyzeNow = useCallback(async () => {
        if (analyzing) return;
        setAnalyzing(true);
        setError(null);

        const previousTimestamp = signalTimestamp?.getTime?.() || signal?.timestamp?.getTime?.() || 0;
        try {
            await safeAlertService.triggerSAFEAlertRefresh(symbol);
            for (let attempt = 0; attempt < 60; attempt += 1) {
                await sleep(5000);
                if (activeSymbolRef.current !== symbol) return;
                let data;
                try {
                    data = await safeAlertService.fetchSAFEAlertSignalCached(symbol);
                } catch (pollError) {
                    // A new demo coin legitimately has no cache until its first
                    // model run finishes. Keep waiting instead of showing fake data.
                    if (pollError?.message?.includes('404')) continue;
                    throw pollError;
                }
                const nextTimestamp = data?.timestamp ? new Date(data.timestamp).getTime() : 0;
                if (nextTimestamp > previousTimestamp) {
                    if (activeSymbolRef.current !== symbol) return;
                    const formatted = safeAlertService.formatSignalData(data);
                    if (formatted) {
                        setSignal(formatted);
                        setSafeAlertSignal(formatted);
                    }
                    setSignalTimestamp(data?.timestamp ? new Date(data.timestamp) : null);
                    setLastUpdate(new Date());
                    return;
                }
            }
            throw new Error('Quá trình phân tích mất nhiều thời gian hơn dự kiến.');
        } catch (e) {
            console.error('Failed to run fresh SAFE-Alert analysis:', e);
            setError(e?.message || 'Không thể phân tích lại dữ liệu lúc này.');
        } finally {
            setAnalyzing(false);
        }
    }, [analyzing, signalTimestamp, signal, symbol, setSafeAlertSignal]);

    // Auto-refresh every 60s
    useEffect(() => {
        loadPrediction();
        const interval = setInterval(loadPrediction, 60000);
        return () => clearInterval(interval);
    }, [loadPrediction]);

    // The footer clock is the user's current local time. Keep it separate
    // from lastUpdate, which is the timestamp of the latest AI prediction.
    useEffect(() => {
        const timer = setInterval(() => setCurrentTime(new Date()), 1000);
        return () => clearInterval(timer);
    }, []);

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
                    <button className="decision-retry-btn" onClick={analyzeNow} disabled={loading || analyzing}>
                        <RefreshCw size={13} />
                        {analyzing ? 'Đang phân tích...' : 'Phân tích ngay'}
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

    const activeSignal = activeHorizon === '4h' && signal.horizon4h
        ? signal.horizon4h
        : signal;

    const nextScheduledUpdate = new Date(currentTime);
    nextScheduledUpdate.setSeconds(0, 0);
    if (nextScheduledUpdate.getMinutes() >= 2) {
        nextScheduledUpdate.setHours(nextScheduledUpdate.getHours() + 1);
    }
    nextScheduledUpdate.setMinutes(2);

    return (
        <div className="decision-sidebar">
            <div className="decision-top-controls">
                <div className="decision-horizon-switch" role="group" aria-label="Khung phân tích AI">
                    {['1h', '4h'].map((horizon) => (
                        <button
                            key={horizon}
                            type="button"
                            className={`decision-horizon-btn ${activeHorizon === horizon ? 'active' : ''}`}
                            onClick={() => setActiveHorizon(horizon)}
                            aria-pressed={activeHorizon === horizon}
                        >
                            {horizon.toUpperCase()}
                        </button>
                    ))}
                </div>
                <button
                    type="button"
                    className="decision-expand-btn"
                    onClick={() => setIsExpanded(true)}
                    title="Mở phân tích chi tiết"
                >
                    <Maximize2 size={13} />
                    Mở rộng
                </button>
            </div>
            {/* ═══ LAYER 4: CONFIDENCE GAUGE (Eq.26) ═══ */}
            <div className="decision-section gauge-section">
                <ConfidenceGauge
                    confidence={activeSignal.confidence}
                    direction={activeSignal.direction || activeSignal.signal}
                    shouldAlert={activeSignal.shouldAlert}
                    horizon={activeHorizon}
                />
                {/* Probability distribution bar */}
                <div className="prob-dist-bar">
                    <div className="prob-segment up" style={{ width: `${activeSignal.probs.UP * 100}%` }} title={`Tăng: ${(activeSignal.probs.UP * 100).toFixed(1)}%`} />
                    <div className="prob-segment neutral" style={{ width: `${activeSignal.probs.NEUTRAL * 100}%` }} title={`Giữ: ${(activeSignal.probs.NEUTRAL * 100).toFixed(1)}%`} />
                    <div className="prob-segment down" style={{ width: `${activeSignal.probs.DOWN * 100}%` }} title={`Giảm: ${(activeSignal.probs.DOWN * 100).toFixed(1)}%`} />
                </div>
                <div className="prob-labels">
                    <span className="prob-label up">▲ {(activeSignal.probs.UP * 100).toFixed(0)}%</span>
                    <span className="prob-label neutral">─ {(activeSignal.probs.NEUTRAL * 100).toFixed(0)}%</span>
                    <span className="prob-label down">▼ {(activeSignal.probs.DOWN * 100).toFixed(0)}%</span>
                </div>
            </div>

            {/* ═══ LAYER 3: FACTOR HEATMAP (Eq.14-15) ═══ */}
            <div className="decision-section">
                <FactorHeatmap
                    topFactors={activeSignal.topFactors}
                    direction={activeSignal.direction || activeSignal.signal}
                />
            </div>

            {/* ═══ LAYER 2: STRUCTURED EXPLANATION (Eq.27-29) ═══ */}
            <div className="decision-section">
                <StructuredExplainer
                    symbol={signal.symbol}
                    horizon={activeHorizon}
                    explanation={activeSignal.explanation}
                    probabilities={activeSignal.probs}
                    direction={activeSignal.direction || activeSignal.signal}
                    confidence={activeSignal.confidence}
                    topFactors={activeSignal.topFactors}
                    articles={activeSignal.articles}
                    shouldAlert={activeSignal.shouldAlert}
                />
            </div>

            {/* ═══ LAYER 1: SELECTIVE EVIDENCE (Eq.8-13) ═══ */}
            {activeSignal.articles && activeSignal.articles.length > 0 && (
                <div className="decision-section evidence-section">
                    <div className="evidence-header">
                        <span className="evidence-icon">📰</span>
                        Bằng chứng đã chọn lọc
                        <span className="evidence-count">{activeSignal.articles.length}</span>
                    </div>
                    <div className="evidence-list">
                        {activeSignal.articles.map((article, idx) => (
                            <button
                                key={idx}
                                type="button"
                                className="evidence-card"
                                onClick={() => setSelectedEvidence(article)}
                                aria-label={`Xem chi tiết: ${article.title}`}
                            >
                                <div className="evidence-rank">α{idx + 1}</div>
                                <div className="evidence-content">
                                    <div className="evidence-title">{article.title}</div>
                                    <div className="evidence-meta">
                                        <span className="evidence-source">{article.source}</span>
                                        {article.url && (
                                            <a
                                                href={article.url}
                                                target="_blank"
                                                rel="noreferrer"
                                                className="evidence-link"
                                                onClick={(event) => event.stopPropagation()}
                                            >
                                                <ExternalLink size={10} />
                                            </a>
                                        )}
                                    </div>
                                </div>
                            </button>
                        ))}
                    </div>
                </div>
            )}

            {/* ═══ FOOTER: REFRESH ═══ */}
            <div className="decision-footer">
                <button className="decision-refresh-btn" onClick={analyzeNow} disabled={loading || analyzing}>
                    <RefreshCw size={12} className={analyzing ? 'spin' : ''} />
                    {analyzing ? 'Đang phân tích...' : 'Cập nhật'}
                </button>
                <span className="decision-timestamp" title={signalTimestamp
                    ? `Kết quả AI hiện tại được tạo lúc ${signalTimestamp.toLocaleString('vi-VN')}`
                    : 'Lịch phân tích tự động'}>
                    Cập nhật tiếp theo lúc {nextScheduledUpdate.toLocaleTimeString('vi-VN', {
                        hour: '2-digit', minute: '2-digit', hour12: false,
                    })}
                </span>
                <span
                    className="decision-live-clock"
                    title={lastUpdate
                        ? `Tải tín hiệu lúc ${lastUpdate.toLocaleTimeString('vi-VN')}`
                        : 'Giờ hiện tại'}
                >
                    <Clock size={10} />
                    {currentTime.toLocaleTimeString('vi-VN', {
                        hour: '2-digit',
                        minute: '2-digit',
                        second: '2-digit',
                        hour12: false,
                    })}
                </span>
            </div>

            {selectedEvidence && (
                <div className="evidence-modal-overlay" onClick={() => setSelectedEvidence(null)}>
                    <article className="evidence-modal" onClick={(event) => event.stopPropagation()}>
                        <button
                            type="button"
                            className="evidence-modal-close"
                            onClick={() => setSelectedEvidence(null)}
                            aria-label="Đóng"
                        >
                            <X size={18} />
                        </button>

                        <div className="evidence-modal-eyebrow">Bằng chứng được mô hình chọn lọc</div>
                        <h2>{selectedEvidence.title}</h2>

                        <div className="evidence-modal-badges">
                            <span>{selectedEvidence.source || 'SAFE-Alert'}</span>
                            {selectedEvidence.relevanceScore != null && (
                                <span>Liên quan {Math.round(selectedEvidence.relevanceScore * 100)}%</span>
                            )}
                            {selectedEvidence.category && <span>{selectedEvidence.category}</span>}
                        </div>

                        {selectedEvidence.publishedAt && (
                            <div className="evidence-modal-time">
                                <Clock size={13} />
                                {new Date(selectedEvidence.publishedAt).toLocaleString('vi-VN')}
                            </div>
                        )}

                        <div className="evidence-modal-divider" />

                        {selectedEvidence.content ? (
                            <p className="evidence-modal-content">{selectedEvidence.content}</p>
                        ) : (
                            <p className="evidence-modal-empty">
                                Dữ liệu SAFE-Alert hiện chỉ lưu tiêu đề của bài viết này. Bạn có thể mở bài gốc để xem toàn bộ nội dung.
                            </p>
                        )}

                        {selectedEvidence.url && (
                            <a
                                href={selectedEvidence.url}
                                target="_blank"
                                rel="noopener noreferrer"
                                className="evidence-modal-link"
                            >
                                Đọc bài viết gốc
                                <ExternalLink size={14} />
                            </a>
                        )}
                    </article>
                </div>
            )}

            <DetailedAnalysisModal
                open={isExpanded}
                onClose={() => setIsExpanded(false)}
                symbol={signal.symbol}
                horizon={activeHorizon}
                signal={activeSignal}
                onSelectEvidence={setSelectedEvidence}
            />
        </div>
    );
}

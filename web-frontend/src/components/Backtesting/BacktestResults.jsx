import React, { useEffect, useRef, useState } from 'react';
import { createChart } from 'lightweight-charts';
import { TrendingUp, TrendingDown, DollarSign, Activity, Percent, Clock } from 'lucide-react';
import './Backtest.css';

export default function BacktestResults({ results }) {
    const chartContainerRef = useRef(null);
    const tooltipHideTimerRef = useRef(null);
    const tooltipHeldRef = useRef(false);
    const [eventTooltip, setEventTooltip] = useState(null);

    const cancelTooltipHide = () => {
        if (tooltipHideTimerRef.current) {
            clearTimeout(tooltipHideTimerRef.current);
            tooltipHideTimerRef.current = null;
        }
    };

    const hideTooltipSoon = () => {
        cancelTooltipHide();
        tooltipHideTimerRef.current = setTimeout(() => {
            if (!tooltipHeldRef.current) setEventTooltip(null);
        }, 260);
    };

    // Formatters
    const formatUSD = (val) => new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD' }).format(Number(val) || 0);
    const formatPct = (val) => `${(Number(val) || 0).toFixed(2)}%`;

    // Draw Chart
    useEffect(() => {
        if (!results || !results.equity_curve || !chartContainerRef.current) return;

        // Get theme colors from CSS variables
        const styles = getComputedStyle(document.documentElement);
        const textColor = styles.getPropertyValue('--text-secondary').trim();
        const gridColor = styles.getPropertyValue('--border-color').trim();
        const accentBlue = '#3861fb'; // Fallback or parse var
        const topColor = 'rgba(56, 97, 251, 0.4)';
        const bottomColor = 'rgba(56, 97, 251, 0.0)';

        const chart = createChart(chartContainerRef.current, {
            layout: {
                background: { color: 'transparent' },
                textColor: textColor || '#848e9c',
            },
            grid: {
                vertLines: { color: gridColor || '#252930', style: 3 },
                horzLines: { color: gridColor || '#252930', style: 3 }
            },
            width: chartContainerRef.current.clientWidth,
            height: 320,
            timeScale: {
                timeVisible: true,
                secondsVisible: false,
                borderColor: gridColor,
            },
            rightPriceScale: {
                borderColor: gridColor,
            },
        });

        const areaSeries = chart.addAreaSeries({
            lineColor: accentBlue,
            topColor: topColor,
            bottomColor: bottomColor,
            lineWidth: 2,
        });

        // Data mapping
        const mappedData = results.equity_curve
            .filter(pt => pt.time && Number.isFinite(Number(pt.value)))
            .map(pt => ({
                time: Math.floor(new Date(pt.time).getTime() / 1000),
                value: Number(pt.value)
            }))
            .filter(pt => Number.isFinite(pt.time) && Number.isFinite(pt.value));

        // lightweight-charts requires strictly increasing, unique timestamps.
        // Keep the latest equity value when a forced close updates the same bar.
        const data = Array.from(
            new Map(mappedData.map(point => [point.time, point])).values()
        ).sort((a, b) => a.time - b.time);

        if (data.length > 0) {
            try {
                areaSeries.setData(data);
                chart.timeScale().fitContent();
            } catch (error) {
                console.error('[BacktestResults] Invalid equity curve:', error);
            }
        }

        const tradesWithNews = (results.trades || []).filter(trade =>
            Array.isArray(trade.news_context) && trade.news_context.length > 0
        );
        const maxVisibleMarkers = 28;
        const markerStep = Math.max(1, Math.ceil(tradesWithNews.length / maxVisibleMarkers));
        const selectedEvents = tradesWithNews
            .filter((_, index) => index % markerStep === 0)
            .slice(0, maxVisibleMarkers);
        const eventMap = new Map();
        const eventByMarkerId = new Map();
        const equityByTime = new Map(data.map(point => [point.time, point.value]));
        const dataTimes = data.map(point => point.time);
        const snapToEquityTime = rawTime => {
            if (!dataTimes.length) return rawTime;
            let nearestTime = dataTimes[0];
            let nearestDistance = Math.abs(nearestTime - rawTime);
            for (let index = 1; index < dataTimes.length; index += 1) {
                const distance = Math.abs(dataTimes[index] - rawTime);
                if (distance < nearestDistance) {
                    nearestTime = dataTimes[index];
                    nearestDistance = distance;
                }
            }
            return nearestTime;
        };
        const eventMarkers = selectedEvents.map((trade, markerIndex) => {
            const rawTime = Math.floor(new Date(trade.entry_time).getTime() / 1000);
            const time = snapToEquityTime(rawTime);
            const averageSentiment = trade.news_context.reduce(
                (sum, news) => sum + Number(news.sentiment_score || 0), 0
            ) / trade.news_context.length;
            const markerId = `backtest-news-${markerIndex}-${time}`;
            const event = { trade, averageSentiment, equityValue: equityByTime.get(time) };
            eventMap.set(time, event);
            eventByMarkerId.set(markerId, event);
            return {
                id: markerId,
                time,
                position: trade.side === 'long' ? 'belowBar' : 'aboveBar',
                color: averageSentiment > 0.1 ? '#18c99a' : averageSentiment < -0.1 ? '#f05e76' : '#f3b84b',
                shape: 'circle',
                text: `N${trade.news_context.length}`,
                size: 1.7,
            };
        }).sort((a, b) => a.time - b.time);

        if (eventMarkers.length > 0) {
            areaSeries.setMarkers(eventMarkers);
        }

        const timeTolerance = data.length > 1
            ? Math.max(60, Math.abs(data[1].time - data[0].time) / 2)
            : 3600;
        chart.subscribeCrosshairMove(param => {
            if (!param.point || eventMap.size === 0) {
                hideTooltipSoon();
                return;
            }

            const directlyHoveredEvent = param.hoveredObjectId
                ? eventByMarkerId.get(String(param.hoveredObjectId))
                : null;
            if (directlyHoveredEvent) {
                cancelTooltipHide();
                setEventTooltip({
                    ...directlyHoveredEvent,
                    x: Math.max(12, Math.min(param.point.x + 14, chartContainerRef.current.clientWidth - 330)),
                    y: Math.max(10, Math.min(param.point.y - 55, 150)),
                });
                return;
            }

            if (!param.time) {
                hideTooltipSoon();
                return;
            }
            const hoveredTime = typeof param.time === 'number' ? param.time : Number(param.time);
            let nearest = null;
            let nearestDistance = Infinity;
            let nearestMarkerX = null;
            eventMap.forEach((event, eventTime) => {
                const distance = Math.abs(eventTime - hoveredTime);
                if (distance < nearestDistance) {
                    nearest = event;
                    nearestDistance = distance;
                    nearestMarkerX = chart.timeScale().timeToCoordinate(eventTime);
                }
            });
            const horizontalDistance = nearestMarkerX == null
                ? Infinity
                : Math.abs(param.point.x - nearestMarkerX);
            // Marker hover is handled by hoveredObjectId above. This wider
            // horizontal fallback also works on lightweight-charts builds
            // that do not expose marker IDs in crosshair events.
            if (!nearest || nearestDistance > timeTolerance || horizontalDistance > 22) {
                hideTooltipSoon();
                return;
            }
            cancelTooltipHide();
            setEventTooltip({
                ...nearest,
                x: Math.max(12, Math.min(param.point.x + 14, chartContainerRef.current.clientWidth - 330)),
                y: Math.max(10, Math.min(param.point.y - 55, 150)),
            });
        });

        const handleResize = () => {
            chart.applyOptions({ width: chartContainerRef.current.clientWidth });
        };

        window.addEventListener('resize', handleResize);
        return () => {
            cancelTooltipHide();
            window.removeEventListener('resize', handleResize);
            chart.remove();
        };
    }, [results]);

    if (!results) return null;

    return (
        <div className="results-card">
            <div style={{ paddingBottom: 16, borderBottom: '1px solid var(--border-color)', marginBottom: 16 }}>
                <h2 className="strategy-section-title" style={{ fontSize: 20, border: 'none', margin: 0 }}>
                    Performance Report
                    <span style={{ fontWeight: 'normal', color: 'var(--text-secondary)', marginLeft: 8, fontSize: 14 }}>
                        Strategy: {results.strategy_name || 'Untitled'}
                    </span>
                </h2>
            </div>

            {/* Metrics Grid */}
            <div className="metrics-grid">
                <StatCard
                    label="Net Profit"
                    value={formatUSD(results.net_profit)}
                    sub={formatPct(results.net_profit_percent)}
                    isPositive={results.net_profit >= 0}
                    icon={<DollarSign size={16} />}
                />
                <StatCard
                    label="Win Rate"
                    value={formatPct(results.win_rate)}
                    sub={`${results.winning_trades ?? 0}W / ${results.losing_trades ?? 0}L`}
                    isPositive={results.win_rate > 50}
                    icon={<Percent size={16} />}
                />
                <StatCard
                    label="Max Drawdown"
                    value={formatPct(results.max_drawdown)}
                    sub="Risk"
                    isPositive={false}
                    color="text-orange"
                    icon={<TrendingDown size={16} />}
                />
                <StatCard
                    label="Sharpe Ratio"
                    value={(Number(results.sharpe_ratio) || 0).toFixed(2)}
                    sub="Risk Adjusted"
                    isPositive={results.sharpe_ratio > 1}
                    icon={<Activity size={16} />}
                />
            </div>

            {/* Equity Curve Chart */}
            <div className="chart-wrapper" style={{ marginBottom: 24 }}>
                <div className="chart-title-bar">
                    <span className="chart-label">Equity Curve</span>
                    <span className={`badge ${results.net_profit >= 0 ? 'badge-long' : 'badge-short'}`}>
                        PnL: {formatUSD(results.net_profit)}
                    </span>
                </div>
                <div style={{ height: 320, position: 'relative' }}>
                    <div ref={chartContainerRef} style={{ width: '100%', height: '100%', position: 'absolute' }} />
                    {eventTooltip && (
                        <BacktestEventTooltip
                            event={eventTooltip}
                            formatUSD={formatUSD}
                            onMouseEnter={() => {
                                tooltipHeldRef.current = true;
                                cancelTooltipHide();
                            }}
                            onMouseLeave={() => {
                                tooltipHeldRef.current = false;
                                hideTooltipSoon();
                            }}
                        />
                    )}
                </div>
                <div className="equity-chart-legend">
                    <span><i className="positive" /> Tin tích cực</span>
                    <span><i className="neutral" /> Tin trung lập</span>
                    <span><i className="negative" /> Tin tiêu cực</span>
                    <small>Chấm N là tin mô hình đã thấy khi mở lệnh</small>
                </div>
            </div>

            {/* Recent Trades Table */}
            <div>
                <h3 className="strategy-section-title">
                    <Clock size={16} /> Recent Trades
                    <span style={{ marginLeft: 8, color: 'var(--text-secondary)', fontSize: 12, fontWeight: 'normal' }}>
                        ({results.trades.length} total)
                    </span>
                </h3>

                <div className="table-container">
                    <table className="trade-table">
                        <thead>
                            <tr>
                                <th>Time</th>
                                <th>Type</th>
                                <th style={{ textAlign: 'right' }}>Entry Price</th>
                                <th style={{ textAlign: 'right' }}>Exit Price</th>
                                <th style={{ textAlign: 'right' }}>Profit ($)</th>
                                <th style={{ textAlign: 'right' }}>Return %</th>
                            </tr>
                        </thead>
                        <tbody>
                            {results.trades.slice().reverse().slice(0, 50).map((trade, idx) => (
                                <tr key={idx}>
                                    <td style={{ color: 'var(--text-secondary)' }}>{new Date(trade.entry_time).toLocaleString()}</td>
                                    <td>
                                        <span className={`badge ${trade.side === 'long' ? 'badge-long' : 'badge-short'}`}>
                                            {trade.side.toUpperCase()}
                                        </span>
                                    </td>
                                    <td style={{ textAlign: 'right', fontFamily: 'monospace' }}>${Number(trade.entry_price || 0).toFixed(2)}</td>
                                    <td style={{ textAlign: 'right', fontFamily: 'monospace' }}>${trade.exit_price ? Number(trade.exit_price).toFixed(2) : '-'}</td>
                                    <td style={{ textAlign: 'right', fontFamily: 'monospace' }} className={(trade.profit || 0) >= 0 ? 'text-green' : 'text-red'}>
                                        {(trade.profit || 0) >= 0 ? '+' : ''}{formatUSD(Number(trade.profit || 0))}
                                    </td>
                                    <td style={{ textAlign: 'right', fontFamily: 'monospace' }} className={(trade.return_percent || 0) >= 0 ? 'text-green' : 'text-red'}>
                                        {Number(trade.return_percent || 0).toFixed(2)}%
                                    </td>
                                </tr>
                            ))}
                            {results.trades.length === 0 && (
                                <tr>
                                    <td colSpan={6} style={{ textAlign: 'center', padding: 32, color: 'var(--text-secondary)' }}>
                                        No trades executed.
                                    </td>
                                </tr>
                            )}
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
    );
}

function BacktestEventTooltip({ event, formatUSD, onMouseEnter, onMouseLeave }) {
    const { trade, averageSentiment, x, y } = event;
    const profit = Number(trade.profit || 0);
    const sentimentLabel = averageSentiment > 0.1 ? 'Tích cực' : averageSentiment < -0.1 ? 'Tiêu cực' : 'Trung lập';
    return (
        <div
            className="backtest-event-tooltip"
            style={{ left: x, top: y }}
            onMouseEnter={onMouseEnter}
            onMouseLeave={onMouseLeave}
        >
            <div className="backtest-event-head">
                <span className={averageSentiment > 0.1 ? 'positive' : averageSentiment < -0.1 ? 'negative' : 'neutral'}>
                    {sentimentLabel} · {(averageSentiment * 100).toFixed(1)}%
                </span>
                <b>{String(trade.side || '').toUpperCase()}</b>
            </div>
            <div className="backtest-event-trade">
                <span>Vào <strong>${Number(trade.entry_price || 0).toFixed(2)}</strong></span>
                <span>Ra <strong>${Number(trade.exit_price || 0).toFixed(2)}</strong></span>
                <span className={profit >= 0 ? 'profit' : 'loss'}>
                    {profit >= 0 ? '+' : ''}{formatUSD(profit)} ({Number(trade.return_percent || 0).toFixed(2)}%)
                </span>
            </div>
            <div className="backtest-event-reason">Kết thúc: {trade.reason || 'Theo tín hiệu'}</div>
            <div className="backtest-event-news">
                {(trade.news_context || []).slice(0, 3).map((news, index) => (
                    <div key={`${news.title}-${index}`}>
                        <i />
                        <span>{news.title || 'Tin thị trường'}</span>
                        <b>{Number(news.sentiment_score || 0) > 0 ? '+' : ''}{(Number(news.sentiment_score || 0) * 100).toFixed(0)}%</b>
                    </div>
                ))}
            </div>
        </div>
    );
}

function StatCard({ label, value, sub, isPositive, color, icon }) {
    const valueColorClass = color ? color : (isPositive ? 'text-green' : 'text-red');

    return (
        <div className="stat-card">
            <div className="stat-label">
                {label}
                <span className={valueColorClass}>{icon}</span>
            </div>
            <div className={`stat-value ${valueColorClass}`}>{value}</div>
            <div className="stat-sub">
                {isPositive ? <TrendingUp size={12} className="text-green" /> : <TrendingDown size={12} className={color ? color : "text-red"} />}
                {sub}
            </div>
        </div>
    );
}

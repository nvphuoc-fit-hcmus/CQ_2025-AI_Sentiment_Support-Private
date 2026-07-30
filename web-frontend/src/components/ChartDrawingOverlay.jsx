import React, { useCallback, useEffect, useMemo, useState } from 'react';

const TOOL_HINTS = {
    trend: 'Chọn hai điểm để vẽ đường xu hướng',
    hline: 'Chọn một mức giá để vẽ đường ngang',
    text: 'Chọn vị trí để thêm chú thích',
    patterns: 'Chọn hai góc để đánh dấu vùng giá',
    measure: 'Chọn hai điểm để đo biến động',
};

const toSeconds = (time) => {
    if (typeof time === 'number') return time;
    if (time && typeof time === 'object' && time.year) {
        return Date.UTC(time.year, time.month - 1, time.day) / 1000;
    }
    return 0;
};

const formatDuration = (seconds) => {
    const value = Math.abs(seconds);
    if (value >= 86400) return `${(value / 86400).toFixed(1)} ngày`;
    if (value >= 3600) return `${(value / 3600).toFixed(1)} giờ`;
    if (value >= 60) return `${Math.round(value / 60)} phút`;
    return `${Math.round(value)} giây`;
};

export default function ChartDrawingOverlay({
    activeTool = 'crosshair',
    chartRef,
    priceSeriesRef,
    containerRef,
    chartReadyVersion,
    candleData = [],
}) {
    const [drawings, setDrawings] = useState([]);
    const [startPoint, setStartPoint] = useState(null);
    const [hoverPoint, setHoverPoint] = useState(null);
    const [, forceRender] = useState(0);

    const toDataPoint = useCallback((event) => {
        const chart = chartRef.current;
        const series = priceSeriesRef.current;
        const container = containerRef.current;
        if (!chart || !series || !container) return null;
        const rect = container.getBoundingClientRect();
        const x = event.clientX - rect.left;
        const y = event.clientY - rect.top;
        const time = chart.timeScale().coordinateToTime(x);
        const rawPrice = series.coordinateToPrice(y);
        if (time == null || rawPrice == null) return null;

        if (!candleData.length) return { time, price: rawPrice };

        const targetTime = toSeconds(time);
        let low = 0;
        let high = candleData.length - 1;
        while (low < high) {
            const middle = Math.floor((low + high) / 2);
            if (toSeconds(candleData[middle].time) < targetTime) low = middle + 1;
            else high = middle;
        }
        const candidates = [candleData[low], candleData[Math.max(0, low - 1)]].filter(Boolean);
        const candle = candidates.reduce((nearest, item) => (
            Math.abs(toSeconds(item.time) - targetTime) < Math.abs(toSeconds(nearest.time) - targetTime)
                ? item
                : nearest
        ), candidates[0]);

        const levels = [
            ['O', Number(candle.open)],
            ['H', Number(candle.high)],
            ['L', Number(candle.low)],
            ['C', Number(candle.close)],
        ].filter(([, value]) => Number.isFinite(value));
        const [snapLabel, snapPrice] = levels.reduce((nearest, level) => (
            Math.abs(level[1] - rawPrice) < Math.abs(nearest[1] - rawPrice) ? level : nearest
        ), levels[0]);

        return {
            time: candle.time,
            price: snapPrice,
            snapLabel,
        };
    }, [chartRef, priceSeriesRef, containerRef, candleData]);

    const toScreenPoint = useCallback((point) => {
        if (!point || !chartRef.current || !priceSeriesRef.current) return null;
        const x = chartRef.current.timeScale().timeToCoordinate(point.time);
        const y = priceSeriesRef.current.priceToCoordinate(point.price);
        return x == null || y == null ? null : { x, y };
    }, [chartRef, priceSeriesRef]);

    useEffect(() => {
        setStartPoint(null);
        setHoverPoint(null);
    }, [activeTool]);

    useEffect(() => {
        const chart = chartRef.current;
        const container = containerRef.current;
        if (!chart || !container) return undefined;
        const redraw = () => forceRender((value) => value + 1);
        chart.timeScale().subscribeVisibleLogicalRangeChange(redraw);
        const observer = new ResizeObserver(redraw);
        observer.observe(container);
        return () => {
            chart.timeScale().unsubscribeVisibleLogicalRangeChange(redraw);
            observer.disconnect();
        };
    }, [chartRef, containerRef, chartReadyVersion]);

    const handlePointerDown = (event) => {
        if (activeTool === 'crosshair' || event.button !== 0) return;
        const point = toDataPoint(event);
        if (!point) return;

        if (activeTool === 'hline') {
            setDrawings((items) => [...items, { id: crypto.randomUUID(), type: 'hline', point }]);
            return;
        }

        if (activeTool === 'text') {
            const text = window.prompt('Nhập nội dung chú thích:');
            if (text?.trim()) {
                setDrawings((items) => [...items, {
                    id: crypto.randomUUID(),
                    type: 'text',
                    point,
                    text: text.trim().slice(0, 120),
                }]);
            }
            return;
        }

        if (!startPoint) {
            setStartPoint(point);
            setHoverPoint(point);
            return;
        }

        setDrawings((items) => [...items, {
            id: crypto.randomUUID(),
            type: activeTool,
            start: startPoint,
            end: point,
        }]);
        setStartPoint(null);
        setHoverPoint(null);
    };

    const handlePointerMove = (event) => {
        if (!startPoint) return;
        setHoverPoint(toDataPoint(event));
    };

    const handleContextMenu = (event) => {
        if (activeTool === 'crosshair') return;
        event.preventDefault();
        if (startPoint) {
            setStartPoint(null);
            setHoverPoint(null);
        } else {
            setDrawings((items) => items.slice(0, -1));
        }
    };

    const visibleDrawings = useMemo(() => {
        const items = [...drawings];
        if (startPoint && hoverPoint) {
            items.push({ id: 'preview', type: activeTool, start: startPoint, end: hoverPoint, preview: true });
        }
        return items;
    }, [drawings, startPoint, hoverPoint, activeTool]);

    return (
        <div
            className={`chart-drawing-overlay ${activeTool !== 'crosshair' ? 'enabled' : ''}`}
            onPointerDown={handlePointerDown}
            onPointerMove={handlePointerMove}
            onContextMenu={handleContextMenu}
        >
            {activeTool !== 'crosshair' && (
                <div className="chart-drawing-hint">
                    {startPoint
                        ? 'Chọn điểm kết thúc · Điểm sẽ hút vào O/H/L/C · Chuột phải để hủy'
                        : `${TOOL_HINTS[activeTool]} · Tự hút vào O/H/L/C · Chuột phải để hoàn tác`}
                </div>
            )}
            <svg width="100%" height="100%" aria-hidden="true">
                {visibleDrawings.map((drawing) => {
                    if (drawing.type === 'hline') {
                        const point = toScreenPoint(drawing.point);
                        if (!point) return null;
                        return (
                            <g key={drawing.id}>
                                <line className="drawing-hline" x1="0" y1={point.y} x2="100%" y2={point.y} />
                                <text className="drawing-price-label" x="8" y={point.y - 6}>
                                    {drawing.point.snapLabel} · {drawing.point.price.toLocaleString('vi-VN', { maximumFractionDigits: 2 })}
                                </text>
                            </g>
                        );
                    }

                    if (drawing.type === 'text') {
                        const point = toScreenPoint(drawing.point);
                        if (!point) return null;
                        return (
                            <g key={drawing.id}>
                                <circle className="drawing-text-dot" cx={point.x} cy={point.y} r="3" />
                                <text className="drawing-text" x={point.x + 7} y={point.y - 7}>{drawing.text}</text>
                            </g>
                        );
                    }

                    const start = toScreenPoint(drawing.start);
                    const end = toScreenPoint(drawing.end);
                    if (!start || !end) return null;

                    if (drawing.type === 'patterns') {
                        return (
                            <g key={drawing.id} className={drawing.preview ? 'drawing-preview' : ''}>
                                <rect
                                    className="drawing-pattern"
                                    x={Math.min(start.x, end.x)}
                                    y={Math.min(start.y, end.y)}
                                    width={Math.abs(end.x - start.x)}
                                    height={Math.abs(end.y - start.y)}
                                />
                                <text className="drawing-text" x={Math.min(start.x, end.x) + 7} y={Math.min(start.y, end.y) + 16}>
                                    Vùng theo dõi
                                </text>
                            </g>
                        );
                    }

                    if (drawing.type === 'measure') {
                        const change = ((drawing.end.price - drawing.start.price) / drawing.start.price) * 100;
                        const duration = formatDuration(toSeconds(drawing.end.time) - toSeconds(drawing.start.time));
                        return (
                            <g key={drawing.id} className={drawing.preview ? 'drawing-preview' : ''}>
                                <line className="drawing-measure" x1={start.x} y1={start.y} x2={end.x} y2={end.y} />
                                <circle className="drawing-handle" cx={start.x} cy={start.y} r="3" />
                                <circle className="drawing-handle" cx={end.x} cy={end.y} r="3" />
                                <rect
                                    className="drawing-measure-label-bg"
                                    x={(start.x + end.x) / 2 - 57}
                                    y={(start.y + end.y) / 2 - 21}
                                    width="114"
                                    height="24"
                                    rx="4"
                                />
                                <text className="drawing-measure-label" x={(start.x + end.x) / 2} y={(start.y + end.y) / 2 - 6}>
                                    {change >= 0 ? '+' : ''}{change.toFixed(2)}% · {duration}
                                </text>
                            </g>
                        );
                    }

                    return (
                        <g key={drawing.id} className={drawing.preview ? 'drawing-preview' : ''}>
                            <line className="drawing-trend" x1={start.x} y1={start.y} x2={end.x} y2={end.y} />
                            <circle className="drawing-handle" cx={start.x} cy={start.y} r="3" />
                            <circle className="drawing-handle" cx={end.x} cy={end.y} r="3" />
                            <text className="drawing-snap-label" x={start.x + 6} y={start.y + 14}>{drawing.start.snapLabel}</text>
                            <text className="drawing-snap-label" x={end.x + 6} y={end.y + 14}>{drawing.end.snapLabel}</text>
                        </g>
                    );
                })}
            </svg>
        </div>
    );
}

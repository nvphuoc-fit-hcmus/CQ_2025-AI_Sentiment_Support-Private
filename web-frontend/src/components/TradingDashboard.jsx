import React, { useState, useRef, useCallback } from 'react';
import MultiTimeframeChart from './MultiTimeframeChart';
import useStore from '../store';

const TIMEFRAMES = [
    { label: '1m', value: '1m', api: '1m' },
    { label: '5m', value: '5m', api: '5m' },
    { label: '15m', value: '15m', api: '15m' },
    { label: '1h', value: '1h', api: '1h' },
    { label: '4h', value: '4h', api: '4h' },
    { label: '1D', value: '1d', api: '1d' },
    { label: '1W', value: '1w', api: '1w' },
    { label: '1M', value: '1M', api: '1M' },
];

export default function TradingDashboard() {
    const { currentSymbol } = useStore();

    // State cho 4 biểu đồ - mỗi biểu đồ có timeframe riêng
    const [chart1TF, setChart1TF] = useState('1m');
    const [chart2TF, setChart2TF] = useState('1h');
    const [chart3TF, setChart3TF] = useState('1d');
    const [chart4TF, setChart4TF] = useState('1w');

    // Cross-hair sync: shared logical time across all 4 charts
    const [syncTime, setSyncTime] = useState(null);
    const syncSourceRef = useRef(null); // Which chart initiated the sync

    // Cross-hair sync callback — called by each chart on crosshair move
    const handleCrosshairSync = useCallback((chartId, time) => {
        // Prevent infinite loops: only sync if this chart is the initiator
        if (syncSourceRef.current && syncSourceRef.current !== chartId) return;
        syncSourceRef.current = chartId;
        setSyncTime(time);
        // Clear source after a short delay to allow other charts to respond
        requestAnimationFrame(() => {
            syncSourceRef.current = null;
        });
    }, []);

    const charts = [
        { id: 'chart1', tf: chart1TF, setTF: setChart1TF },
        { id: 'chart2', tf: chart2TF, setTF: setChart2TF },
        { id: 'chart3', tf: chart3TF, setTF: setChart3TF },
        { id: 'chart4', tf: chart4TF, setTF: setChart4TF },
    ];

    return (
        <div className="trading-dashboard">
            {/* 4 Charts Grid */}
            <div className="charts-grid">
                {charts.map((chart) => (
                    <div key={chart.id} className="chart-container">
                        <div className="chart-header">
                            <div className="chart-header-left">
                                <span className="chart-symbol">{currentSymbol.replace('USDT', '')}</span>
                                <span className="chart-pair">/USDT</span>
                            </div>
                            <div className="chart-timeframes">
                                {TIMEFRAMES.map(tf => (
                                    <button
                                        key={tf.value}
                                        className={`tf-btn ${chart.tf === tf.value ? 'active' : ''}`}
                                        onClick={() => chart.setTF(tf.value)}
                                    >
                                        {tf.label}
                                    </button>
                                ))}
                            </div>
                        </div>
                        <MultiTimeframeChart
                            symbol={currentSymbol}
                            timeframe={chart.tf}
                            chartId={chart.id}
                            syncTime={syncTime}
                            onCrosshairSync={handleCrosshairSync}
                        />
                    </div>
                ))}
            </div>
        </div>
    );
}

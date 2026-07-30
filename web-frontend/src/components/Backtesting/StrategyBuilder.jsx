import React, { useState } from 'react';
import { Plus, Trash2, Play, Settings } from 'lucide-react';
import './Backtest.css';

const INDICATORS = ['RSI', 'MACD', 'EMA20', 'SMA50', 'BollingerBands'];
const AI_PREDICTIONS = ['direction_1h', 'confidence_1h', 'direction_4h', 'confidence_4h', 'volatility'];

const SYMBOLS = [
    'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'XRPUSDT',
    'DOGEUSDT', 'ADAUSDT', 'AVAXUSDT', 'DOTUSDT', 'POLUSDT'
];

const TIMEFRAMES = [
    { label: '15 Minutes (15m)', value: '15m' },
    { label: '30 Minutes (30m)', value: '30m' },
    { label: '1 Hour (1H)', value: '1h' },
    { label: '4 Hours (4H)', value: '4h' },
    { label: '12 Hours (12H)', value: '12h' },
    { label: '1 Day (1D)', value: '1d' },
    { label: '1 Week (1W)', value: '1w' }
];

export default function StrategyBuilder({ onRunBacktest, isLoading }) {
    // Generate default name with timestamp
    const getDefaultStrategyName = () => {
        const now = new Date();
        const date = now.toLocaleDateString('vi-VN', { day: '2-digit', month: '2-digit', year: 'numeric' }).replace(/\//g, '-');
        const time = now.toLocaleTimeString('vi-VN', { hour: '2-digit', minute: '2-digit' }).replace(/:/g, 'h');
        return `My AI Strategy ${date} ${time}`;
    };

    const [strategyName, setStrategyName] = useState(getDefaultStrategyName());
    const [conditions, setConditions] = useState([
        { type: 'indicator', name: 'RSI', operator: '<', value: 30 }
    ]);
    const [logic, setLogic] = useState('AND');
    const [action, setAction] = useState('BUY');
    const [timeframe, setTimeframe] = useState('1h');

    // Params
    const [symbol, setSymbol] = useState('BTCUSDT');
    const [startDate, setStartDate] = useState(new Date(Date.now() - 30 * 24 * 60 * 60 * 1000).toISOString().split('T')[0]);
    const [endDate, setEndDate] = useState(new Date().toISOString().split('T')[0]);
    const [initialCapital, setInitialCapital] = useState(10000);
    const [takeProfit, setTakeProfit] = useState(5.0);
    const [stopLoss, setStopLoss] = useState(2.0);

    const addCondition = () => {
        setConditions([...conditions, { type: 'indicator', name: 'RSI', operator: '<', value: 30 }]);
    };

    const applySafeAlertPreset = () => {
        setStrategyName(`SAFE-Alert 1H + 4H + News ${getDefaultStrategyName().replace('My AI Strategy ', '')}`);
        setLogic('AND');
        setAction('BUY');
        setTimeframe('1h');
        setConditions([
            { type: 'ai', field: 'direction_1h', operator: '=', value: 'UP' },
            { type: 'ai', field: 'confidence_1h', operator: '>=', value: 0.52 },
            { type: 'ai', field: 'direction_4h', operator: '=', value: 'UP' },
            { type: 'news', field: 'sentiment_score', operator: '>', value: 0 },
        ]);
        setStartDate('2024-08-29');
        setEndDate('2025-08-26');
        setTakeProfit(5);
        setStopLoss(2);
    };

    const removeCondition = (index) => {
        setConditions(conditions.filter((_, i) => i !== index));
    };

    const updateCondition = (index, key, value) => {
        const updated = [...conditions];
        updated[index][key] = value;

        // Auto-update fields based on type change
        if (key === 'type') {
            if (value === 'indicator') {
                updated[index].name = 'RSI';
                updated[index].operator = '<';
                updated[index].value = 30;
            }
            if (value === 'ai') {
                updated[index].field = 'direction_1h';
                updated[index].operator = '=';
                updated[index].value = 'UP';
            }
            if (value === 'price') {
                updated[index].name = 'Close';
                updated[index].operator = '>';
                updated[index].value = 0;
            }
            if (value === 'news') {
                updated[index].field = 'sentiment_score';
                updated[index].operator = '>';
                updated[index].value = 0.5;
            }
        }

        // Auto-update defaults based on specific Indicator Name
        if (key === 'name' && updated[index].type === 'indicator') {
            if (value === 'RSI') {
                updated[index].operator = '<';
                updated[index].value = 30;
            } else if (value === 'MACD') {
                updated[index].operator = '>';
                updated[index].value = 0; // Histogram > 0 implies uptrend
            } else {
                updated[index].operator = '>';
                updated[index].value = 0;
            }
        }

        // Auto-update defaults based on AI Field
        if (key === 'field' && updated[index].type === 'ai') {
            if (value === 'direction_1h' || value === 'direction_4h') {
                updated[index].operator = '=';
                updated[index].value = 'UP';
            } else if (value === 'confidence_1h') {
                updated[index].operator = '>';
                updated[index].value = 0.52;
            } else if (value === 'confidence_4h') {
                updated[index].operator = '>';
                // The trained 4H policy uses tau≈0.366; 0.52 suppresses every
                // historical 4H prediction in the current checkpoint.
                updated[index].value = 0.37;
            } else if (value === 'volatility') {
                updated[index].operator = '=';
                updated[index].value = 'HIGH'; // Or LOW depending on strategy
            }
        }

        setConditions(updated);
    };

    const handleRun = () => {
        const strategy = {
            name: strategyName,
            conditions,
            logic,
            action,
            timeframe,
            take_profit: parseFloat(takeProfit),
            stop_loss: parseFloat(stopLoss)
        };

        onRunBacktest({
            strategy,
            symbol,
            start_date: startDate + 'T00:00:00Z',
            end_date: endDate + 'T23:59:59Z',
            initial_capital: parseFloat(initialCapital)
        });
    };

    return (
        <div className="strategy-card">
            <div className="strategy-section-title">
                <Settings size={18} className="text-blue" /> Strategy Configuration
            </div>
            <button type="button" className="btn-secondary" onClick={applySafeAlertPreset}
                style={{ width: '100%', marginBottom: 14 }}>
                Dùng mẫu SAFE-Alert 1H + 4H + News
            </button>

            {/* Basic Settings */}
            <div className="form-group">
                <label className="form-label">Strategy Name</label>
                <input
                    type="text"
                    value={strategyName}
                    onChange={(e) => setStrategyName(e.target.value)}
                    className="form-input"
                    placeholder="Enter strategy name..."
                />
            </div>

            <div className="form-row">
                <div className="form-group">
                    <label className="form-label">Symbol</label>
                    <div style={{ display: 'flex', gap: 8 }}>
                        <select
                            value={symbol}
                            onChange={(e) => setSymbol(e.target.value)}
                            className="form-select"
                            style={{ flex: 2 }}
                        >
                            {SYMBOLS.map(s => (
                                <option key={s} value={s}>{s}</option>
                            ))}
                        </select>
                        <select
                            value={timeframe}
                            onChange={(e) => setTimeframe(e.target.value)}
                            className="form-select"
                            style={{ flex: 1, minWidth: '80px' }}
                        >
                            {TIMEFRAMES.map(t => (
                                <option key={t.value} value={t.value}>{t.label.split(' ')[2].replace('(', '').replace(')', '')}</option>
                            ))}
                        </select>
                    </div>
                </div>
                <div className="form-group">
                    <label className="form-label">Initial Capital ($)</label>
                    <input
                        type="number"
                        value={initialCapital}
                        onChange={(e) => setInitialCapital(e.target.value)}
                        style={{ paddingRight: 0 }}
                        className="form-input"
                    />
                </div>
            </div>

            <div className="form-row">
                <div className="form-group">
                    <label className="form-label">Start Date</label>
                    <input
                        type="date"
                        value={startDate}
                        onChange={(e) => setStartDate(e.target.value)}
                        className="form-input"
                    />
                </div>
                <div className="form-group">
                    <label className="form-label">End Date</label>
                    <input
                        type="date"
                        value={endDate}
                        onChange={(e) => setEndDate(e.target.value)}
                        className="form-input"
                    />
                </div>
            </div>

            {/* Risk Management */}
            <h3 className="strategy-section-title" style={{ marginTop: '16px' }}>Risk Management (Exit)</h3>
            <div className="form-row">
                <div className="form-group">
                    <label className="form-label">Take Profit (%)</label>
                    <input
                        type="number"
                        step="0.5"
                        value={takeProfit}
                        onChange={(e) => setTakeProfit(e.target.value)}
                        className="form-input"
                        style={{ color: 'var(--accent-green)', fontWeight: 'bold' }}
                    />
                </div>
                <div className="form-group">
                    <label className="form-label">Stop Loss (%)</label>
                    <input
                        type="number"
                        step="0.5"
                        value={stopLoss}
                        onChange={(e) => setStopLoss(e.target.value)}
                        className="form-input"
                        style={{ color: 'var(--accent-red)', fontWeight: 'bold' }}
                    />
                </div>
            </div>

            {/* Conditions Section */}
            <div className="strategy-section-title">
                Entry Conditions
            </div>

            <div className="condition-list">
                {conditions.map((cond, idx) => (
                    <div key={idx} className="condition-item" style={{ flexWrap: 'nowrap', gap: 4, alignItems: 'center' }}>
                        <div className="condition-tag" style={{ width: 40, flexShrink: 0, textAlign: 'center', fontSize: 10, padding: '2px 0' }}>
                            {idx === 0 ? 'WHEN' : logic}
                        </div>

                        {/* Type */}
                        <select
                            value={cond.type}
                            onChange={(e) => updateCondition(idx, 'type', e.target.value)}
                            className="form-select"
                            style={{ width: '80px', flexShrink: 0, paddingRight: 2, fontSize: 12, textOverflow: 'ellipsis' }}
                        >
                            <option value="indicator">Ind</option>
                            <option value="ai">AI</option>
                            <option value="price">Price</option>
                            <option value="news">News</option>
                        </select>

                        {/* Dynamic Fields */}
                        {cond.type === 'indicator' && (
                            <select
                                value={cond.name}
                                onChange={(e) => updateCondition(idx, 'name', e.target.value)}
                                className="form-select"
                                style={{ flex: 1, minWidth: '55px', fontSize: 12, paddingRight: 2 }}
                            >
                                {INDICATORS.map(i => <option key={i} value={i}>{i}</option>)}
                            </select>
                        )}

                        {cond.type === 'ai' && (
                            <select
                                value={cond.field}
                                onChange={(e) => updateCondition(idx, 'field', e.target.value)}
                                className="form-select"
                                style={{ flex: 1, minWidth: '55px', fontSize: 12, paddingRight: 2 }}
                            >
                                <option value="direction_1h">Dir 1h</option>
                                <option value="confidence_1h">Conf 1h</option>
                                <option value="direction_4h">Dir 4h</option>
                                <option value="confidence_4h">Conf 4h</option>
                                <option value="volatility">Vol</option>
                            </select>
                        )}

                        {/* Operator */}
                        <select
                            value={cond.operator}
                            onChange={(e) => updateCondition(idx, 'operator', e.target.value)}
                            className="form-select"
                            style={{ width: '40px', textAlign: 'center', flexShrink: 0, padding: '4px 0', fontSize: 12 }}
                        >
                            <option value=">">&gt;</option>
                            <option value="<">&lt;</option>
                            <option value="=">=</option>
                            <option value=">=">&ge;</option>
                            <option value="<=">&le;</option>
                        </select>

                        {/* Value Input - Dynamic based on Field */}
                        {(cond.type === 'ai' && (cond.field === 'direction_1h' || cond.field === 'direction_4h')) ? (
                            <select
                                value={cond.value}
                                onChange={(e) => updateCondition(idx, 'value', e.target.value)}
                                className="form-select"
                                style={{ width: '60px', flexShrink: 0, fontSize: 12, paddingRight: 2 }}
                            >
                                <option value="UP">UP</option>
                                <option value="DOWN">DOWN</option>
                                <option value="SIDEWAYS">SIDE</option>
                            </select>
                        ) : (cond.type === 'ai' && cond.field === 'volatility') ? (
                            <select
                                value={cond.value}
                                onChange={(e) => updateCondition(idx, 'value', e.target.value)}
                                className="form-select"
                                style={{ width: '65px', flexShrink: 0, fontSize: 12, paddingRight: 2 }}
                            >
                                <option value="LOW">LOW</option>
                                <option value="MEDIUM">MED</option>
                                <option value="HIGH">HIGH</option>
                            </select>
                        ) : (
                            <input
                                type="text"
                                value={cond.value}
                                onChange={(e) => updateCondition(idx, 'value', e.target.value)}
                                className="form-input"
                                style={{ width: '55px', flexShrink: 0, minWidth: '40px', fontSize: 12, padding: '4px 6px' }}
                                placeholder="Val"
                            />
                        )}

                        <button
                            onClick={() => removeCondition(idx)}
                            className="btn-icon-danger"
                            title="Remove"
                            style={{ flexShrink: 0, width: 24, height: 24, display: 'flex', alignItems: 'center', justifyContent: 'center' }}
                        >
                            <Trash2 size={13} />
                        </button>
                    </div>
                ))}

                <button onClick={addCondition} className="btn-add">
                    <Plus size={16} /> Add Condition
                </button>
            </div>

            {/* Logic & Action */}
            <h3 className="strategy-section-title" style={{ marginTop: '24px' }}>Execution Rules</h3>
            <div className="logic-action-panel">
                <div style={{ flex: 1, minWidth: '110px' }}>
                    <label className="form-label">Logic Operator</label>
                    <select
                        value={logic}
                        onChange={(e) => setLogic(e.target.value)}
                        className="form-select"
                    >
                        <option value="AND">AND (All)</option>
                        <option value="OR">OR (Any)</option>
                    </select>
                </div>

                <span className="arrow-separator">→</span>

                <div style={{ flex: 1, minWidth: '110px' }}>
                    <label className="form-label">Action</label>
                    <select
                        value={action}
                        onChange={(e) => setAction(e.target.value)}
                        className="form-select"
                        style={{ fontWeight: 'bold', color: action === 'BUY' ? 'var(--accent-green)' : 'var(--accent-red)' }}
                    >
                        <option value="BUY">BUY / LONG</option>
                        <option value="SELL">SELL / SHORT</option>
                    </select>
                </div>
            </div>

            {/* Run Button */}
            <button
                onClick={handleRun}
                disabled={isLoading}
                className="btn-primary"
            >
                {isLoading ? (
                    <>
                        <div className="spinner" style={{ width: 16, height: 16, border: '2px solid white', borderTopColor: 'transparent', borderRadius: '50%', animation: 'spin 1s linear infinite' }}></div>
                        RUNNING SIMULATION...
                    </>
                ) : (
                    <><Play size={18} fill="currentColor" /> RUN BACKTEST</>
                )}
            </button>

        </div>
    );
}

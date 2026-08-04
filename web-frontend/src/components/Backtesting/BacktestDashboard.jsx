import React, { useState, useEffect } from 'react';
import StrategyBuilder from './StrategyBuilder';
import BacktestResults from './BacktestResults';
import backtestService from '../../services/backtestService';
import { useToast } from '../ToastProvider';
import { BarChart, History, Activity, AlertCircle, Lock } from 'lucide-react';
import './Backtest.css';

export default function BacktestDashboard() {
    const isVip = true;

    const [results, setResults] = useState(null);
    const [isLoading, setIsLoading] = useState(false);
    const [history, setHistory] = useState([]);
    const [showHistory, setShowHistory] = useState(false);
    const { showToast } = useToast();

    const handleRunBacktest = async (payload) => {
        setIsLoading(true);
        setResults(null);
        try {
            const data = await backtestService.runBacktest(payload);
            if (data.status === 'success') {
                const parsedResults = {
                    ...data.results,
                    strategy_name: payload.strategy.name,
                    trades: typeof data.results.trades === 'string' ? JSON.parse(data.results.trades) : data.results.trades,
                    equity_curve: typeof data.results.equity_curve === 'string' ? JSON.parse(data.results.equity_curve) : data.results.equity_curve,
                    news_timeline: typeof data.results.news_timeline === 'string' ? JSON.parse(data.results.news_timeline) : (data.results.news_timeline || [])
                };
                setResults(parsedResults);
                showToast('Backtest completed successfully!', 'success');
                fetchHistory(); // Refresh history
            } else {
                showToast('Backtest failed: ' + (data.error || 'Unknown error'), 'error');
            }
        } catch (err) {
            console.error(err);
            showToast('Error running backtest: ' + (err.response?.data?.error || err.message), 'error');
        } finally {
            setIsLoading(false);
        }
    };

    const fetchHistory = async () => {
        try {
            const hist = await backtestService.getHistory();
            setHistory(hist);
        } catch (err) {
            console.error('Failed to fetch history', err);
        }
    };

    useEffect(() => {
        fetchHistory();
    }, []);

    const loadHistoryItem = async (id) => {
        try {
            setIsLoading(true);
            const detail = await backtestService.getDetail(id);

            // Parse JSON fields
            const parsedResults = {
                ...detail,
                trades: typeof detail.trades === 'string' ? JSON.parse(detail.trades) : detail.trades,
                equity_curve: typeof detail.equity_curve === 'string' ? JSON.parse(detail.equity_curve) : detail.equity_curve,
                news_timeline: typeof detail.news_timeline === 'string' ? JSON.parse(detail.news_timeline) : (detail.news_timeline || [])
            };

            setResults(parsedResults);
            setShowHistory(false);
            window.scrollTo({ top: 0, behavior: 'smooth' });
        } catch (err) {
            showToast('Failed to load history', 'error');
        } finally {
            setIsLoading(false);
        }
    };

    return (
        <div className="backtest-container">
            <div className="backtest-header">
                <div>
                    <h1 className="backtest-title">
                        <BarChart className="text-blue" size={28} />
                        AI Backtest Nâng Cao
                    </h1>
                    <p className="backtest-subtitle">Mô phỏng hiệu suất chiến lược giao dịch với dữ liệu lịch sử và dự đoán AI.</p>
                </div>

                {isVip && (
                    <button
                        onClick={() => setShowHistory(!showHistory)}
                        className="btn-secondary"
                    >
                        <History size={16} /> {showHistory ? 'Ẩn Lịch sử' : 'Lịch sử Backtest'}
                    </button>
                )}
            </div>

            {showHistory && (
                <div className="history-panel">
                    <h3 className="strategy-section-title">Lịch sử Backtest</h3>
                    <div className="history-scroll">
                        {history.map(item => (
                            <div
                                key={item.id}
                                onClick={() => loadHistoryItem(item.id)}
                                className="history-item"
                            >
                                <div className="history-item-content">
                                    <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 4 }}>
                                        <span style={{ fontWeight: 'bold', color: 'var(--text-primary)' }}>{item.strategy_name || 'Untitled Strategy'}</span>
                                        <span className={`badge ${item.net_profit_percent >= 0 ? 'badge-long' : 'badge-short'}`}>
                                            {item.net_profit_percent?.toFixed(2)}%
                                        </span>
                                    </div>
                                    <div style={{ fontSize: 11, color: 'var(--text-secondary)' }}>
                                        {new Date(item.created_at).toLocaleString()} • {item.symbol}
                                    </div>
                                </div>
                            </div>
                        ))}
                        {history.length === 0 && <div className="loading-text">Chưa có dữ liệu lịch sử. Hãy chạy backtest đầu tiên!</div>}
                    </div>
                </div>
            )}

            {/* Main Content */}
            {/* Main Content */}
            {!isVip ? (
                <div className="vip-lock-container" style={{
                    display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center',
                    padding: '64px', margin: '32px auto', maxWidth: '600px',
                    backgroundColor: 'rgba(255, 215, 0, 0.05)', borderRadius: '16px', border: '1px solid rgba(255, 215, 0, 0.2)',
                    textAlign: 'center'
                }}>
                    <Lock size={64} style={{ color: '#FFD700', marginBottom: '24px', opacity: 1 }} />
                    <h2 style={{ fontSize: '24px', marginBottom: '16px', color: '#FFD700', fontWeight: 'bold' }}>Tính Năng Chỉ Dành Cho VIP</h2>
                    <p style={{ color: 'var(--text-secondary)', marginBottom: '32px', lineHeight: 1.6, maxWidth: '480px' }}>
                        Công cụ AI Backtesting nâng cao cho phép bạn kiểm thử chiến lược với dữ liệu quá khứ. <br />
                        Vui lòng nâng cấp tài khoản để mở khóa toàn bộ sức mạnh phân tích.
                    </p>
                    <button
                        className="btn-primary"
                        style={{ padding: '12px 32px', fontSize: '16px', background: 'linear-gradient(135deg, #FFD700 0%, #FFA500 100%)', color: '#000', fontWeight: 'bold', border: 'none' }}
                        onClick={() => window.dispatchEvent(new CustomEvent('showUpgradeModal'))}
                    >
                        💎 Nâng Cấp VIP Ngay
                    </button>
                </div>
            ) : (
                <div className="backtest-grid">

                    {/* Left Column: Input */}
                    <div className="input-column">
                        <StrategyBuilder onRunBacktest={handleRunBacktest} isLoading={isLoading} />
                    </div>

                    {/* Right Column: Results */}
                    <div className="results-column">
                        {results ? (
                            <BacktestResults results={results} />
                        ) : (
                            <div className="empty-state">
                                <div className="icon-placeholder">
                                    <BarChart size={40} />
                                </div>
                                <h3 style={{ fontSize: 20, marginBottom: 8, color: 'var(--text-primary)' }}>Ready to Simulate</h3>
                                <p style={{ maxWidth: 400, margin: '0 auto', lineHeight: 1.5 }}>
                                    Configure your strategy on the left and click "Run Backtest" to see performance results on historical data.
                                </p>

                                <div style={{ marginTop: 32, display: 'flex', gap: 16, justifyContent: 'center' }}>
                                    <div style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 12 }}>
                                        <span style={{ width: 8, height: 8, borderRadius: '50%', backgroundColor: 'var(--accent-green)' }}></span>
                                        Historical Data
                                    </div>
                                    <div style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 12 }}>
                                        <span style={{ width: 8, height: 8, borderRadius: '50%', backgroundColor: 'var(--accent-blue)' }}></span>
                                        AI Models
                                    </div>
                                </div>
                            </div>
                        )}
                    </div>

                </div>
            )}
        </div>
    );
}

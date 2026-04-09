import React, { useEffect, useState, useCallback } from 'react';
import safeAlertService from '../services/safeAlertService';
import './SAFEAlertPanel.css';

/**
 * SAFE-Alert Prediction Panel - FIXED
 *
 * Now correctly maps from backend response:
 * - horizon_1h.signal → Direction
 * - horizon_1h.confidence → Confidence %
 * - horizon_1h.selected_news → Articles
 * - top_inputs → Technical indicators
 */
export default function SAFEAlertPanel({ symbol = 'BTCUSDT' }) {
  const [signal, setSignal] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [backendOnline, setBackendOnline] = useState(true);

  // Fetch prediction
  const loadPrediction = useCallback(async () => {
    setLoading(true);
    setError(null);

    try {
      // Check backend health first
      const healthy = await safeAlertService.checkBackendHealth();
      setBackendOnline(healthy);

      if (!healthy) {
        setError('Backend API không khả dụng. Hãy chắc chắn server chạy trên port 8000');
        return;
      }

      // Fetch signal
      const data = await safeAlertService.fetchSAFEAlertSignal(symbol);
      const formatted = safeAlertService.formatSignalData(data);
      setSignal(formatted);
    } catch (e) {
      console.error('Failed to load prediction:', e);
      setError(e.message || 'Không thể tải dự đoán');
    } finally {
      setLoading(false);
    }
  }, [symbol]);

  // Auto-refresh every 60 seconds
  useEffect(() => {
    loadPrediction();
    const interval = setInterval(loadPrediction, 60000);
    return () => clearInterval(interval);
  }, [loadPrediction]);

  // Direction styling
  const getDirectionStyle = (direction) => {
    switch (direction?.toUpperCase()) {
      case 'BUY':
        return {
          color: '#10b981',
          bg: '#ecfdf5',
          icon: '🚀',
          label: 'ĐI LÊN (BUY)',
        };
      case 'SELL':
        return {
          color: '#ef4444',
          bg: '#fef2f2',
          icon: '📉',
          label: 'ĐI XUỐNG (SELL)',
        };
      default:
        return {
          color: '#6b7280',
          bg: '#f9fafb',
          icon: '➖',
          label: 'GIỮ VỮNG (HOLD)',
        };
    }
  };

  if (loading && !signal) {
    return (
      <div className="safe-alert-panel loading">
        <div className="loading-spinner">⏳ Đang tải dự đoán...</div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="safe-alert-panel error">
        <div className="error-header">❌ Lỗi</div>
        <div className="error-message">{error}</div>
        <button onClick={loadPrediction} className="retry-button">
          {loading ? 'Đang thử lại...' : 'Thử lại'}
        </button>
        {!backendOnline && (
          <div className="offline-note">
            💡 Hãy chạy backend: cd "/e/Khóa luận 1/SA/services/ai-service" && python -m uvicorn app.main:app --port 8000
          </div>
        )}
      </div>
    );
  }

  if (!signal) {
    return (
      <div className="safe-alert-panel empty">
        <div>📭 Chưa có dự đoán</div>
        <button onClick={loadPrediction}>Refresh</button>
      </div>
    );
  }

  const dirStyle = getDirectionStyle(signal.direction);
  const confidencePercent = parseFloat(signal.confidence);
  const confidenceBar = Math.min(confidencePercent, 100);

  return (
    <div className="safe-alert-panel">
      {/* Header */}
      <div className="panel-header">
        <h2>🤖 SAFE-Alert Dự đoán</h2>
        <div className="timestamp">
          {signal.timestamp.toLocaleTimeString('vi-VN')}
        </div>
      </div>

      {/* Main Prediction (1h) */}
      <div
        className="prediction-box"
        style={{ backgroundColor: dirStyle.bg, borderLeft: `4px solid ${dirStyle.color}` }}
      >
        <div className="prediction-row">
          <div className="prediction-icon">{dirStyle.icon}</div>
          <div className="prediction-content">
            <div className="prediction-label">Dự đoán Hướng (1h):</div>
            <div className="prediction-value" style={{ color: dirStyle.color }}>
              {dirStyle.label}
            </div>
          </div>
        </div>
      </div>

      {/* Confidence Score */}
      <div className="metric-section">
        <div className="metric-label">
          <strong>Độ Tin Cậy (1h):</strong> {signal.confidence}%
        </div>
        <div className="confidence-bar">
          <div
            className="confidence-fill"
            style={{
              width: `${confidenceBar}%`,
              backgroundColor:
                confidencePercent > 60
                  ? '#10b981'
                  : confidencePercent > 40
                    ? '#f59e0b'
                    : '#ef4444',
            }}
          />
        </div>
      </div>

      {/* 4h Prediction (Secondary) */}
      {signal.horizon4h && (
        <div className="secondary-box">
          <div className="secondary-label">Dự đoán 4h: <strong>{signal.horizon4h.signal}</strong></div>
          <div className="secondary-confidence">Độ tin cậy: {signal.horizon4h.confidence}%</div>
        </div>
      )}

      {/* Alert Signal */}
      <div
        className="alert-box"
        style={{
          backgroundColor: signal.alert === 'YES' ? '#fef2f2' : '#ecfdf5',
          borderLeft: `3px solid ${signal.alert === 'YES' ? '#ef4444' : '#10b981'}`,
        }}
      >
        <div className="alert-label">
          {signal.alert === 'YES' ? '🚨' : '✅'} Tín hiệu cảnh báo:
        </div>
        <div className="alert-value" style={{ color: signal.alert === 'YES' ? '#ef4444' : '#10b981' }}>
          {signal.alert === 'YES' ? `CÓ (${signal.alertLevel})` : 'KHÔNG'}
        </div>
      </div>

      {/* Selected Articles */}
      {signal.articles && signal.articles.length > 0 && (
        <div className="articles-section">
          <div className="section-title">📰 Bài Viết Được Chọn ({signal.articles.length})</div>
          <div className="articles-list">
            {signal.articles.map((article, idx) => (
              <div key={idx} className="article-item">
                <div className="article-number">{idx + 1}</div>
                <div className="article-content">
                  <div className="article-title">{article.title || 'Untitled'}</div>
                  {article.source && <div className="article-source">Nguồn: {article.source}</div>}
                  {article.url && (
                    <a href={article.url} target="_blank" rel="noreferrer" className="article-link">
                      Xem chi tiết →
                    </a>
                  )}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Top Technical Indicators */}
      <div className="metrics-section">
        <div className="section-title">📊 Chỉ Số Kỹ Thuật Hàng Đầu</div>
        <div className="metrics-grid">
          <div className="metric">
            <div className="metric-key">RSI (14)</div>
            <div className="metric-value">{signal.topInputs.rsi}</div>
            <div className="metric-desc">Chỉ số sức mạnh tương đối</div>
          </div>

          <div className="metric">
            <div className="metric-key">MACD Hist</div>
            <div className="metric-value">{signal.topInputs.macd}</div>
            <div className="metric-desc">Hội tụ-phân kỳ</div>
          </div>

          <div className="metric">
            <div className="metric-key">Bollinger %</div>
            <div className="metric-value">{signal.topInputs.bbPos}%</div>
            <div className="metric-desc">Vị trí Bollinger Bands</div>
          </div>

          <div className="metric">
            <div className="metric-key">Volume Spike</div>
            <div className="metric-value">{signal.topInputs.volume}</div>
            <div className="metric-desc">Độ tăng khối lượng</div>
          </div>

          <div className="metric">
            <div className="metric-key">Sentiment</div>
            <div className="metric-value">{signal.topInputs.vader}</div>
            <div className="metric-desc">Cảm xúc tin tức (VADER)</div>
          </div>

          <div className="metric">
            <div className="metric-key">Stoch RSI</div>
            <div className="metric-value">{signal.topInputs.stochRsi}</div>
            <div className="metric-desc">RSI Stochastic</div>
          </div>
        </div>
      </div>

      {/* Faithfulness Note */}
      <div className="faithfulness-note">
        <div className="note-icon">ℹ️</div>
        <div className="note-content">
          <strong>Ghi chú:</strong> Mô hình dựa chủ yếu trên dữ liệu thị trường.
          Tin tức có tác dụng hỗ trợ nhưng không phải yếu tố chính (Faithfulness: 0%).
        </div>
      </div>

      {/* Refresh Button */}
      <div className="panel-footer">
        <button onClick={loadPrediction} className="refresh-button" disabled={loading}>
          {loading ? '⏳ Đang cập nhật...' : '🔄 Cập nhật'}
        </button>
        <div className="refresh-note">Tự động cập nhật mỗi 60 giây</div>
      </div>
    </div>
  );
}

/**
 * SAFE-Alert API Service - FIXED
 *
 * Maps backend response ({horizon_1h, horizon_4h, alert, top_inputs})
 * to frontend expected format
 */

const BACKEND_URL = import.meta.env.VITE_BACKEND_URL || 'http://localhost:8000';

/**
 * Fetch SAFE-Alert signal for given symbol
 * @param {string} symbol - Crypto symbol (BTCUSDT, ETHUSDT, etc)
 * @returns {Promise<object>} Signal data with predictions
 */
export const fetchSAFEAlertSignal = async (symbol = 'BTCUSDT') => {
  try {
    const response = await fetch(`${BACKEND_URL}/v2/signal/${symbol}`);
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }
    const data = await response.json();
    return data;
  } catch (error) {
    console.error(`Failed to fetch SAFE-Alert signal for ${symbol}:`, error);
    throw error;
  }
};

/**
 * Fetch cached signal (doesn't retrain model)
 * @param {string} symbol - Crypto symbol
 * @returns {Promise<object>} Cached signal data
 */
export const fetchSAFEAlertSignalCached = async (symbol = 'BTCUSDT') => {
  try {
    const response = await fetch(`${BACKEND_URL}/v2/signal/${symbol}/cached`);
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }
    const data = await response.json();
    return data;
  } catch (error) {
    console.error(`Failed to fetch cached SAFE-Alert signal for ${symbol}:`, error);
    throw error;
  }
};

/**
 * Check backend health
 * @returns {Promise<boolean>} True if backend is healthy
 */
export const checkBackendHealth = async () => {
  try {
    const response = await fetch(`${BACKEND_URL}/health`);
    return response.ok;
  } catch (error) {
    console.error('Backend health check failed:', error);
    return false;
  }
};

/**
 * Format signal data for display
 * FIXED: Map from actual backend response structure
 *
 * Backend returns:
 * {
 *   horizon_1h: { signal, confidence, selected_news, ... },
 *   horizon_4h: { signal, confidence, selected_news, ... },
 *   alert: { alert, level, signal, reasons },
 *   top_inputs: { rsi_14, macd_hist, volume_spike, ... }
 * }
 *
 * @param {object} signal - Raw signal from backend API
 * @returns {object} Formatted signal for display
 */
export const formatSignalData = (signal) => {
  if (!signal || !signal.horizon_1h) return null;

  // Extract from horizon_1h (1-hour prediction)
  const h1 = signal.horizon_1h || {};
  const h4 = signal.horizon_4h || {};
  const alertInfo = signal.alert || {};
  const topInputs = signal.top_inputs || {};

  return {
    symbol: signal.symbol || 'BTCUSDT',
    // FIXED: Use horizon_1h.signal (UP/DOWN/NEUTRAL mapped to BUY/SELL/HOLD)
    direction: h1.signal || 'HOLD',
    // FIXED: Use horizon_1h.confidence (0-1 range)
    confidence: ((h1.confidence || 0) * 100).toFixed(1),
    // FIXED: Use alertInfo.alert (boolean)
    alert: alertInfo.alert ? 'YES' : 'NO',
    alertLevel: alertInfo.level || 'low',
    // FIXED: Map selected_news array (strings) to objects with title field
    articles: (h1.selected_news || []).map((item, idx) => {
      // Handle both string format and object format
      if (typeof item === 'string') {
        return {
          title: item,
          source: 'SAFE-Alert',
          url: null
        };
      }
      return {
        title: item.title || item,
        source: item.source || 'SAFE-Alert',
        url: item.url || null
      };
    }),
    // Backend doesn't provide these in live API (only in ablation study)
    metrics: {
      f1: 'N/A',
      sharpe: 'N/A',
      coverage: 'N/A',
      precision: 'N/A',
      faithfulness: '0.0%',
    },
    // FIXED: Add top technical indicators from backend
    topInputs: {
      rsi: topInputs.rsi_14?.toFixed(1) || 'N/A',
      macd: topInputs.macd_hist?.toFixed(4) || 'N/A',
      bbPos: topInputs.bb_pos?.toFixed(2) || 'N/A',
      stochRsi: topInputs.stoch_rsi?.toFixed(2) || 'N/A',
      volume: topInputs.volume_spike?.toFixed(2) || 'N/A',
      vader: topInputs.vader_mean?.toFixed(3) || 'N/A',
    },
    // Include 4h data as well
    horizon4h: {
      signal: h4.signal || 'HOLD',
      confidence: ((h4.confidence || 0) * 100).toFixed(1),
    },
    timestamp: new Date(signal.timestamp || Date.now()),
  };
};

export default {
  fetchSAFEAlertSignal,
  fetchSAFEAlertSignalCached,
  checkBackendHealth,
  formatSignalData,
};


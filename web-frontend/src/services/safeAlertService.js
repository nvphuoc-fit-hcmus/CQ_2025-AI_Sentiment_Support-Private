/**
 * SAFE-Alert API Service — Intelligence Hub Edition
 *
 * Full parsing of backend response including:
 * - Factor grounding (top_factors, factor_dist)
 * - Structured explanation (nl_explanation)
 * - Probability distribution (probs)
 * - Alert gating (should_alert, Eq.26)
 *
 * Backend returns:
 * {
 *   horizon_1h: { signal, confidence, probs, selected_news, top_factors, factors_text, explanation, should_alert },
 *   horizon_4h: { ... },
 *   alert: { alert, level, signal, reason, reasons },
 *   top_inputs: { rsi_14, macd_hist, bb_pos, volume_spike, ... }
 * }
 */

const BACKEND_URL = import.meta.env.VITE_BACKEND_URL || 'http://localhost:8000';

// Factor name translations (Vietnamese primary + English subtitle)
export const FACTOR_LABELS = {
  institutional_inflow:  { vi: 'Dòng vốn tổ chức',       en: 'Institutional Inflow' },
  etf_flow:              { vi: 'Dòng vốn ETF',            en: 'ETF Flow' },
  regulatory_easing:     { vi: 'Nới lỏng quy định',       en: 'Regulatory Easing' },
  regulatory_tightening: { vi: 'Siết chặt quy định',      en: 'Regulatory Tightening' },
  exchange_risk:         { vi: 'Rủi ro sàn giao dịch',    en: 'Exchange Risk' },
  liquidity_squeeze:     { vi: 'Siết thanh khoản',        en: 'Liquidity Squeeze' },
  whale_accumulation:    { vi: 'Cá voi tích lũy',         en: 'Whale Accumulation' },
  macro_uncertainty:     { vi: 'Bất định vĩ mô',          en: 'Macro Uncertainty' },
  protocol_upgrade:      { vi: 'Nâng cấp giao thức',      en: 'Protocol Upgrade' },
  network_outage:        { vi: 'Sự cố mạng lưới',         en: 'Network Outage' },
};

/**
 * Fetch SAFE-Alert signal for given symbol
 */
export const fetchSAFEAlertSignal = async (symbol = 'BTCUSDT') => {
  try {
    const response = await fetch(`${BACKEND_URL}/v2/signal/${symbol}`);
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    return await response.json();
  } catch (error) {
    console.error(`Failed to fetch SAFE-Alert signal for ${symbol}:`, error);
    throw error;
  }
};

/**
 * Fetch cached signal (doesn't retrain model)
 */
export const fetchSAFEAlertSignalCached = async (symbol = 'BTCUSDT') => {
  try {
    const response = await fetch(`${BACKEND_URL}/v2/signal/${symbol}/cached`);
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    return await response.json();
  } catch (error) {
    console.error(`Failed to fetch cached SAFE-Alert signal for ${symbol}:`, error);
    throw error;
  }
};

/**
 * Check backend health
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
 * Format signal data — FULL Intelligence Hub parsing
 *
 * Extracts all 4 layers of SAFE-Alert architecture:
 * Layer 1: Selective News (selected_news)
 * Layer 2: Factor Grounding (top_factors, factor_dist)
 * Layer 3: Prediction (signal, confidence, probs)
 * Layer 4: Alert Gating (should_alert, Eq.26)
 */
export const formatSignalData = (signal) => {
  if (!signal || !signal.horizon_1h) return null;

  const h1 = signal.horizon_1h || {};
  const h4 = signal.horizon_4h || {};
  const alertInfo = signal.alert || {};
  const topInputs = signal.top_inputs || {};

  // Parse probability distribution
  const probs1h = h1.probs || {};
  const probs4h = h4.probs || {};

  // Parse factor data
  const topFactors = h1.top_factors || [];
  const factorDist = h1.factor_dist || null; // Array of 10 floats if available

  return {
    symbol: signal.symbol || 'BTCUSDT',
    timestamp: new Date(signal.timestamp || Date.now()),

    // === Layer 4: Alert Gating (Eq.26) ===
    shouldAlert: h1.should_alert || false,
    alertInfo: {
      alert: alertInfo.alert || false,
      level: alertInfo.level || 'low',
      signal: alertInfo.signal || 'HOLD',
      reason: alertInfo.reason || '',
      reasons: alertInfo.reasons || [],
    },

    // === Layer 3: Prediction ===
    direction: h1.signal || 'HOLD',
    confidence: h1.confidence || 0,
    confidencePercent: ((h1.confidence || 0) * 100).toFixed(1),
    probs: {
      UP: probs1h.UP || 0.333,
      NEUTRAL: probs1h.NEUTRAL || 0.334,
      DOWN: probs1h.DOWN || 0.333,
    },

    // === Layer 2: Factor Grounding (Eq.14-15) ===
    topFactors: topFactors,
    factorsText: h1.factors_text || '',
    factorDist: factorDist, // Raw [C] weights from FactorModule

    // === Layer 1: Selective Evidence (Eq.8-13) ===
    articles: (h1.selected_news || []).map((item, idx) => {
      if (typeof item === 'string') {
        return { title: item, source: 'SAFE-Alert', url: null, relevanceScore: null };
      }
      return {
        title: item.title || item,
        source: item.source || 'SAFE-Alert',
        url: item.url || null,
        relevanceScore: item.relevance_score || null,
      };
    }),

    // === Structured Explanation (Eq.27-29) ===
    explanation: h1.explanation || h1.nl_explanation || '',

    // === 4h Horizon (secondary) ===
    horizon4h: {
      signal: h4.signal || 'HOLD',
      confidence: h4.confidence || 0,
      confidencePercent: ((h4.confidence || 0) * 100).toFixed(1),
      probs: {
        UP: probs4h.UP || 0.333,
        NEUTRAL: probs4h.NEUTRAL || 0.334,
        DOWN: probs4h.DOWN || 0.333,
      },
      shouldAlert: h4.should_alert || false,
      topFactors: h4.top_factors || [],
      explanation: h4.explanation || h4.nl_explanation || '',
    },

    // === Technical Indicators ===
    topInputs: {
      rsi: topInputs.rsi_14?.toFixed(1) || 'N/A',
      macd: topInputs.macd_hist?.toFixed(4) || 'N/A',
      bbPos: topInputs.bb_pos?.toFixed(2) || 'N/A',
      stochRsi: topInputs.stoch_rsi?.toFixed(2) || 'N/A',
      volume: topInputs.volume_spike?.toFixed(2) || 'N/A',
      vader: topInputs.vader_mean?.toFixed(3) || 'N/A',
    },
  };
};

export default {
  fetchSAFEAlertSignal,
  fetchSAFEAlertSignalCached,
  checkBackendHealth,
  formatSignalData,
  FACTOR_LABELS,
};

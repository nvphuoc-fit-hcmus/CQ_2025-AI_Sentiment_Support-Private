/**
 * Technical Indicators Calculation Utilities
 * Tính toán các chỉ báo kỹ thuật: SMA, EMA, RSI, MACD, Bollinger Bands
 */

/**
 * Simple Moving Average (SMA)
 * @param {Array} data - Array of candle data with 'close' property
 * @param {number} period - Period for SMA (e.g., 20, 50, 200)
 * @returns {Array} Array of {time, value} for line series
 */
export function calculateSMA(data, period) {
    if (!data || data.length < period) return [];

    const result = [];

    for (let i = period - 1; i < data.length; i++) {
        let sum = 0;
        for (let j = 0; j < period; j++) {
            sum += data[i - j].close;
        }
        const avg = sum / period;
        result.push({
            time: data[i].time,
            value: avg
        });
    }

    return result;
}

/**
 * Exponential Moving Average (EMA)
 * @param {Array} data - Array of candle data with 'close' property
 * @param {number} period - Period for EMA (e.g., 12, 26, 50)
 * @returns {Array} Array of {time, value} for line series
 */
export function calculateEMA(data, period) {
    if (!data || data.length < period) return [];

    const result = [];
    const multiplier = 2 / (period + 1);

    // Start with SMA for first value
    let sum = 0;
    for (let i = 0; i < period; i++) {
        sum += data[i].close;
    }
    let ema = sum / period;

    result.push({
        time: data[period - 1].time,
        value: ema
    });

    // Calculate EMA for remaining values
    for (let i = period; i < data.length; i++) {
        ema = (data[i].close - ema) * multiplier + ema;
        result.push({
            time: data[i].time,
            value: ema
        });
    }

    return result;
}

/**
 * Relative Strength Index (RSI)
 * @param {Array} data - Array of candle data with 'close' property
 * @param {number} period - Period for RSI (typically 14)
 * @returns {Array} Array of {time, value} for line series
 */
export function calculateRSI(data, period = 14) {
    if (!data || data.length < period + 1) return [];

    const result = [];
    let gains = 0;
    let losses = 0;

    // Calculate initial average gain and loss
    for (let i = 1; i <= period; i++) {
        const change = data[i].close - data[i - 1].close;
        if (change > 0) {
            gains += change;
        } else {
            losses -= change;
        }
    }

    let avgGain = gains / period;
    let avgLoss = losses / period;

    // Calculate RSI
    for (let i = period; i < data.length; i++) {
        const change = data[i].close - data[i - 1].close;
        const gain = change > 0 ? change : 0;
        const loss = change < 0 ? -change : 0;

        avgGain = (avgGain * (period - 1) + gain) / period;
        avgLoss = (avgLoss * (period - 1) + loss) / period;

        const rs = avgLoss === 0 ? 100 : avgGain / avgLoss;
        const rsi = 100 - (100 / (1 + rs));

        result.push({
            time: data[i].time,
            value: rsi
        });
    }

    return result;
}

/**
 * Bollinger Bands
 * @param {Array} data - Array of candle data with 'close' property
 * @param {number} period - Period for moving average (typically 20)
 * @param {number} stdDev - Number of standard deviations (typically 2)
 * @returns {Object} {upper: [], middle: [], lower: []}
 */
export function calculateBollingerBands(data, period = 20, stdDev = 2) {
    if (!data || data.length < period) return { upper: [], middle: [], lower: [] };

    const upper = [];
    const middle = [];
    const lower = [];

    for (let i = period - 1; i < data.length; i++) {
        // Calculate SMA
        let sum = 0;
        for (let j = 0; j < period; j++) {
            sum += data[i - j].close;
        }
        const sma = sum / period;

        // Calculate standard deviation
        let variance = 0;
        for (let j = 0; j < period; j++) {
            variance += Math.pow(data[i - j].close - sma, 2);
        }
        const std = Math.sqrt(variance / period);

        const time = data[i].time;
        middle.push({ time, value: sma });
        upper.push({ time, value: sma + stdDev * std });
        lower.push({ time, value: sma - stdDev * std });
    }

    return { upper, middle, lower };
}

/**
 * MACD (Moving Average Convergence Divergence)
 * @param {Array} data - Array of candle data with 'close' property
 * @param {number} fastPeriod - Fast EMA period (typically 12)
 * @param {number} slowPeriod - Slow EMA period (typically 26)
 * @param {number} signalPeriod - Signal line period (typically 9)
 * @returns {Object} {macd: [], signal: [], histogram: []}
 */
export function calculateMACD(data, fastPeriod = 12, slowPeriod = 26, signalPeriod = 9) {
    if (!data || data.length < slowPeriod) return { macd: [], signal: [], histogram: [] };

    const fastEMA = calculateEMA(data, fastPeriod);
    const slowEMA = calculateEMA(data, slowPeriod);

    // Calculate MACD line
    const macdLine = [];
    const startIndex = slowPeriod - fastPeriod;

    for (let i = 0; i < slowEMA.length; i++) {
        const time = slowEMA[i].time;
        const fastValue = fastEMA[i + startIndex].value;
        const slowValue = slowEMA[i].value;
        macdLine.push({
            time,
            value: fastValue - slowValue
        });
    }

    // Calculate signal line (EMA of MACD line)
    // We need to calculate EMA manually since macdLine has 'value' not 'close'
    const signalLine = [];
    if (macdLine.length < signalPeriod) {
        return { macd: macdLine, signal: [], histogram: [] };
    }

    const multiplier = 2 / (signalPeriod + 1);

    // Start with SMA for first signal value
    let sum = 0;
    for (let i = 0; i < signalPeriod; i++) {
        sum += macdLine[i].value;
    }
    let ema = sum / signalPeriod;

    signalLine.push({
        time: macdLine[signalPeriod - 1].time,
        value: ema
    });

    // Calculate EMA for remaining values
    for (let i = signalPeriod; i < macdLine.length; i++) {
        ema = (macdLine[i].value - ema) * multiplier + ema;
        signalLine.push({
            time: macdLine[i].time,
            value: ema
        });
    }

    // Calculate histogram
    const histogram = [];
    for (let i = 0; i < signalLine.length; i++) {
        const macdValue = macdLine[i + (macdLine.length - signalLine.length)].value;
        const signalValue = signalLine[i].value;
        histogram.push({
            time: signalLine[i].time,
            value: macdValue - signalValue,
            color: macdValue >= signalValue ? 'rgba(38, 166, 154, 0.5)' : 'rgba(239, 83, 80, 0.5)'
        });
    }

    return {
        macd: macdLine,
        signal: signalLine,
        histogram
    };
}

export function calculateVWAP(data) {
    let cumulativeValue = 0;
    let cumulativeVolume = 0;
    return data.map((item) => {
        const typicalPrice = (item.high + item.low + item.close) / 3;
        const volume = Number(item.value ?? item.volume ?? 0);
        cumulativeValue += typicalPrice * volume;
        cumulativeVolume += volume;
        return { time: item.time, value: cumulativeVolume ? cumulativeValue / cumulativeVolume : typicalPrice };
    });
}

export function calculateStochastic(data, period = 14, smooth = 3) {
    if (!data || data.length < period) return { k: [], d: [] };
    const k = [];
    for (let i = period - 1; i < data.length; i++) {
        const window = data.slice(i - period + 1, i + 1);
        const highest = Math.max(...window.map((item) => item.high));
        const lowest = Math.min(...window.map((item) => item.low));
        const value = highest === lowest ? 50 : ((data[i].close - lowest) / (highest - lowest)) * 100;
        k.push({ time: data[i].time, value });
    }
    const d = [];
    for (let i = smooth - 1; i < k.length; i++) {
        const value = k.slice(i - smooth + 1, i + 1).reduce((sum, item) => sum + item.value, 0) / smooth;
        d.push({ time: k[i].time, value });
    }
    return { k, d };
}

export function calculateATR(data, period = 14) {
    if (!data || data.length < period + 1) return [];
    const trueRanges = data.map((item, index) => {
        if (index === 0) return item.high - item.low;
        return Math.max(
            item.high - item.low,
            Math.abs(item.high - data[index - 1].close),
            Math.abs(item.low - data[index - 1].close),
        );
    });
    let atr = trueRanges.slice(1, period + 1).reduce((sum, value) => sum + value, 0) / period;
    const result = [{ time: data[period].time, value: atr }];
    for (let i = period + 1; i < data.length; i++) {
        atr = ((atr * (period - 1)) + trueRanges[i]) / period;
        result.push({ time: data[i].time, value: atr });
    }
    return result;
}

export function calculateADX(data, period = 14) {
    if (!data || data.length < period * 2 + 1) return [];
    const tr = [];
    const plusDM = [];
    const minusDM = [];
    for (let i = 1; i < data.length; i++) {
        const upMove = data[i].high - data[i - 1].high;
        const downMove = data[i - 1].low - data[i].low;
        tr.push(Math.max(
            data[i].high - data[i].low,
            Math.abs(data[i].high - data[i - 1].close),
            Math.abs(data[i].low - data[i - 1].close),
        ));
        plusDM.push(upMove > downMove && upMove > 0 ? upMove : 0);
        minusDM.push(downMove > upMove && downMove > 0 ? downMove : 0);
    }
    let smoothTR = tr.slice(0, period).reduce((a, b) => a + b, 0);
    let smoothPlus = plusDM.slice(0, period).reduce((a, b) => a + b, 0);
    let smoothMinus = minusDM.slice(0, period).reduce((a, b) => a + b, 0);
    const dx = [];
    for (let i = period; i < tr.length; i++) {
        smoothTR = smoothTR - smoothTR / period + tr[i];
        smoothPlus = smoothPlus - smoothPlus / period + plusDM[i];
        smoothMinus = smoothMinus - smoothMinus / period + minusDM[i];
        const plusDI = smoothTR ? (100 * smoothPlus) / smoothTR : 0;
        const minusDI = smoothTR ? (100 * smoothMinus) / smoothTR : 0;
        const value = plusDI + minusDI ? (100 * Math.abs(plusDI - minusDI)) / (plusDI + minusDI) : 0;
        dx.push({ time: data[i + 1].time, value });
    }
    let adx = dx.slice(0, period).reduce((sum, item) => sum + item.value, 0) / period;
    const result = [{ time: dx[period - 1].time, value: adx }];
    for (let i = period; i < dx.length; i++) {
        adx = ((adx * (period - 1)) + dx[i].value) / period;
        result.push({ time: dx[i].time, value: adx });
    }
    return result;
}

export function calculateOBV(data) {
    let obv = 0;
    return data.map((item, index) => {
        if (index > 0) {
            const volume = Number(item.value ?? item.volume ?? 0);
            if (item.close > data[index - 1].close) obv += volume;
            if (item.close < data[index - 1].close) obv -= volume;
        }
        return { time: item.time, value: obv };
    });
}

export function calculateSupertrend(data, period = 10, multiplier = 3) {
    const atr = calculateATR(data, period);
    if (!atr.length) return [];
    const atrByTime = new Map(atr.map((item) => [item.time, item.value]));
    const result = [];
    let finalUpper = 0;
    let finalLower = 0;
    let trendUp = true;
    data.forEach((item, index) => {
        const atrValue = atrByTime.get(item.time);
        if (atrValue == null) return;
        const middle = (item.high + item.low) / 2;
        const basicUpper = middle + multiplier * atrValue;
        const basicLower = middle - multiplier * atrValue;
        const previousClose = index > 0 ? data[index - 1].close : item.close;
        finalUpper = !finalUpper || basicUpper < finalUpper || previousClose > finalUpper ? basicUpper : finalUpper;
        finalLower = !finalLower || basicLower > finalLower || previousClose < finalLower ? basicLower : finalLower;
        if (item.close > finalUpper) trendUp = true;
        else if (item.close < finalLower) trendUp = false;
        result.push({
            time: item.time,
            value: trendUp ? finalLower : finalUpper,
            color: trendUp ? '#089981' : '#f23645',
        });
    });
    return result;
}

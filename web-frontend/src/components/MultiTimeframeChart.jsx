import React, { useEffect, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import { createChart } from 'lightweight-charts';
import useStore from '../store';
import { io } from 'socket.io-client';
import { LoadingSpinner } from './LoadingSpinner';
import { useTheme } from './ThemeProvider';
import {
    calculateADX, calculateATR, calculateBollingerBands, calculateEMA,
    calculateMACD, calculateOBV, calculateRSI, calculateSMA,
    calculateStochastic, calculateSupertrend, calculateVWAP,
} from '../utils/technicalIndicators';
import { ChartCandlestick, ChartLine, Check, SlidersHorizontal } from 'lucide-react';
import ChartDrawingOverlay from './ChartDrawingOverlay';

const LOWER_INDICATORS = ['rsi', 'macd', 'stochastic', 'atr', 'adx', 'obv'];

export default function MultiTimeframeChart({ symbol, timeframe, chartId, syncTime, onCrosshairSync, drawingTool = 'crosshair' }) {
    const chartContainerRef = useRef();
    const chartRef = useRef();
    const candleSeriesRef = useRef();
    const lineSeriesRef = useRef();
    const volumeSeriesRef = useRef();
    const socketRef = useRef(null);
    const loadingBoolRef = useRef(false);
    const oldestTimeRef = useRef(null);
    const latestTimeRef = useRef(null);

    // Technical Indicators Series
    const sma20SeriesRef = useRef();
    const ema12SeriesRef = useRef();
    const ema26SeriesRef = useRef();
    const ema20SeriesRef = useRef();
    const vwapSeriesRef = useRef();
    const supertrendSeriesRef = useRef();
    const bbUpperSeriesRef = useRef();
    const bbMiddleSeriesRef = useRef();
    const bbLowerSeriesRef = useRef();

    // RSI and MACD Series
    const rsiSeriesRef = useRef();
    const macdLineSeriesRef = useRef();
    const macdSignalSeriesRef = useRef();
    const macdHistogramSeriesRef = useRef();
    const stochasticKSeriesRef = useRef();
    const stochasticDSeriesRef = useRef();
    const atrSeriesRef = useRef();
    const adxSeriesRef = useRef();
    const obvSeriesRef = useRef();

    // News markers
    const newsMarkersRef = useRef([]);

    const { authFetch, safeAlertSignal } = useStore();
    const { isDark } = useTheme();

    // Alert marker series ref
    const alertMarkerSeriesRef = useRef(null);
    const [data, setData] = useState([]);
    const [newsData, setNewsData] = useState([]);
    const [isLoading, setIsLoading] = useState(false);

    // Indicator visibility toggles
    const [indicators, setIndicators] = useState({
        sma20: false,
        ema12: false,
        ema26: false,
        bb: false,
        rsi: false,
        macd: false,
        ema20: false,
        vwap: false,
        supertrend: false,
        stochastic: false,
        atr: false,
        adx: false,
        obv: false,
    });

    const [showNews, setShowNews] = useState(false);
    const [newsTooltip, setNewsTooltip] = useState(null);
    const [selectedNews, setSelectedNews] = useState(null);
    const [newsModalOpen, setNewsModalOpen] = useState(false);
    const [isHoveringNews, setIsHoveringNews] = useState(false);
    const [isExpanded, setIsExpanded] = useState(false); // For fullscreen chart modal
    const [showIndicatorMenu, setShowIndicatorMenu] = useState(false);
    const [indicatorMenuPosition, setIndicatorMenuPosition] = useState({});
    const [visibleNewsCount, setVisibleNewsCount] = useState(0);
    const [chartType, setChartType] = useState('candles');
    const [chartReadyVersion, setChartReadyVersion] = useState(0);
    const newsMapRef = useRef(new Map());
    const hoveredNewsRef = useRef(null);
    const indicatorTriggerRef = useRef(null);
    const indicatorMenuRef = useRef(null);

    useEffect(() => {
        if (!showIndicatorMenu) return undefined;
        const closeOnOutside = (event) => {
            if (
                !indicatorTriggerRef.current?.contains(event.target)
                && !indicatorMenuRef.current?.contains(event.target)
            ) {
                setShowIndicatorMenu(false);
            }
        };
        const closeOnResize = () => setShowIndicatorMenu(false);
        document.addEventListener('pointerdown', closeOnOutside);
        window.addEventListener('resize', closeOnResize);
        return () => {
            document.removeEventListener('pointerdown', closeOnOutside);
            window.removeEventListener('resize', closeOnResize);
        };
    }, [showIndicatorMenu]);

    // Initialize Chart - recreate when theme changes
    useEffect(() => {
        if (!chartContainerRef.current) return;

        chartContainerRef.current.innerHTML = '';

        // Theme-aware colors
        const chartColors = isDark ? {
            background: '#131722',
            textColor: '#d1d4dc',
            gridColor: 'rgba(42, 46, 57, 0.2)',
            borderColor: '#2B2B43'
        } : {
            background: '#ffffff',
            textColor: '#333333',
            gridColor: 'rgba(0, 0, 0, 0.1)',
            borderColor: '#e0e0e0'
        };

        const chart = createChart(chartContainerRef.current, {
            layout: {
                background: { type: 'solid', color: chartColors.background },
                textColor: chartColors.textColor,
            },
            grid: {
                vertLines: { color: chartColors.gridColor },
                horzLines: { color: chartColors.gridColor },
            },
            width: chartContainerRef.current.clientWidth,
            height: chartContainerRef.current.clientHeight,
            timeScale: {
                timeVisible: true,
                secondsVisible: false,
                borderColor: chartColors.borderColor,
            },
            rightPriceScale: {
                borderColor: chartColors.borderColor,
            },
        });

        const candlestickSeries = chart.addCandlestickSeries({
            upColor: '#089981',
            downColor: '#f23645',
            borderVisible: false,
            wickUpColor: '#089981',
            wickDownColor: '#f23645',
        });

        const lineSeries = chart.addAreaSeries({
            lineColor: '#2962FF',
            lineWidth: 2,
            topColor: 'rgba(41, 98, 255, 0.28)',
            bottomColor: 'rgba(41, 98, 255, 0.015)',
            title: 'Giá đóng cửa',
            visible: false,
            priceLineVisible: true,
            lastValueVisible: true,
            crosshairMarkerVisible: true,
            crosshairMarkerRadius: 4,
        });

        const volumeSeries = chart.addHistogramSeries({
            priceFormat: { type: 'volume' },
            priceScaleId: '',
        });

        chart.priceScale('').applyOptions({
            scaleMargins: { top: 0.8, bottom: 0 },
        });

        // Technical Indicators
        const sma20Series = chart.addLineSeries({
            color: '#2962FF',
            lineWidth: 2,
            title: 'SMA 20',
            visible: indicators.sma20
        });

        const ema12Series = chart.addLineSeries({
            color: '#FF6D00',
            lineWidth: 2,
            title: 'EMA 12',
            visible: indicators.ema12
        });

        const ema26Series = chart.addLineSeries({
            color: '#9C27B0',
            lineWidth: 2,
            title: 'EMA 26',
            visible: indicators.ema26
        });

        const ema20Series = chart.addLineSeries({
            color: '#E91E63', lineWidth: 2, title: 'EMA 20', visible: indicators.ema20,
        });
        const vwapSeries = chart.addLineSeries({
            color: '#FFD54F', lineWidth: 2, title: 'VWAP', visible: indicators.vwap,
        });
        const supertrendSeries = chart.addLineSeries({
            color: '#089981', lineWidth: 2, title: 'Supertrend', visible: indicators.supertrend,
        });

        // Bollinger Bands
        const bbUpperSeries = chart.addLineSeries({
            color: 'rgba(33, 150, 243, 0.5)',
            lineWidth: 1,
            title: 'BB Upper',
            visible: indicators.bb
        });

        const bbMiddleSeries = chart.addLineSeries({
            color: 'rgba(33, 150, 243, 0.8)',
            lineWidth: 1,
            lineStyle: 2, // Dashed
            title: 'BB Middle',
            visible: indicators.bb
        });

        const bbLowerSeries = chart.addLineSeries({
            color: 'rgba(33, 150, 243, 0.5)',
            lineWidth: 1,
            title: 'BB Lower',
            visible: indicators.bb
        });

        // RSI (Relative Strength Index) - separate scale
        const rsiSeries = chart.addLineSeries({
            color: '#FF9800',
            lineWidth: 2,
            title: 'RSI',
            visible: indicators.rsi,
            priceScaleId: 'rsi',
            priceFormat: {
                type: 'price',
                precision: 2,
                minMove: 0.01,
            }
        });

        // Configure RSI scale (0-100)
        chart.priceScale('rsi').applyOptions({
            scaleMargins: {
                top: 0.85,
                bottom: 0,
            },
            borderColor: chartColors.borderColor,
        });

        // MACD - separate scale (render histogram first, then lines on top)
        const macdHistogramSeries = chart.addHistogramSeries({
            title: 'MACD Histogram',
            visible: indicators.macd,
            priceScaleId: 'macd',
            priceFormat: {
                type: 'price',
                precision: 2,
            },
            lastValueVisible: false
        });

        const macdLineSeries = chart.addLineSeries({
            color: '#00BCD4',  // Cyan for MACD line - very distinct
            lineWidth: 3,      // Thicker line
            title: 'MACD',
            visible: indicators.macd,
            priceScaleId: 'macd',
            lastValueVisible: true,
            priceLineVisible: false
        });

        const macdSignalSeries = chart.addLineSeries({
            color: '#FF9800',  // Orange for Signal line - very distinct
            lineWidth: 3,      // Thicker line
            title: 'Signal',
            visible: indicators.macd,
            priceScaleId: 'macd',
            lastValueVisible: true,
            priceLineVisible: false
        });

        // Configure MACD scale
        chart.priceScale('macd').applyOptions({
            scaleMargins: {
                top: 0.9,
                bottom: 0,
            },
            borderColor: chartColors.borderColor,
        });

        const lowerSeriesOptions = (title, color) => ({
            color, lineWidth: 2, title, visible: false, priceScaleId: 'lower',
            priceLineVisible: false, lastValueVisible: true,
        });
        const stochasticKSeries = chart.addLineSeries(lowerSeriesOptions('Stochastic %K', '#7E57C2'));
        const stochasticDSeries = chart.addLineSeries(lowerSeriesOptions('Stochastic %D', '#FFB74D'));
        const atrSeries = chart.addLineSeries(lowerSeriesOptions('ATR 14', '#26C6DA'));
        const adxSeries = chart.addLineSeries(lowerSeriesOptions('ADX 14', '#AB47BC'));
        const obvSeries = chart.addLineSeries(lowerSeriesOptions('OBV', '#66BB6A'));
        chart.priceScale('lower').applyOptions({
            scaleMargins: { top: 0.76, bottom: 0 },
            borderColor: chartColors.borderColor,
        });

        chartRef.current = chart;
        candleSeriesRef.current = candlestickSeries;
        lineSeriesRef.current = lineSeries;
        setChartReadyVersion((version) => version + 1);
        volumeSeriesRef.current = volumeSeries;
        sma20SeriesRef.current = sma20Series;
        ema12SeriesRef.current = ema12Series;
        ema26SeriesRef.current = ema26Series;
        ema20SeriesRef.current = ema20Series;
        vwapSeriesRef.current = vwapSeries;
        supertrendSeriesRef.current = supertrendSeries;
        bbUpperSeriesRef.current = bbUpperSeries;
        bbMiddleSeriesRef.current = bbMiddleSeries;
        bbLowerSeriesRef.current = bbLowerSeries;
        rsiSeriesRef.current = rsiSeries;
        macdLineSeriesRef.current = macdLineSeries;
        macdSignalSeriesRef.current = macdSignalSeries;
        macdHistogramSeriesRef.current = macdHistogramSeries;
        stochasticKSeriesRef.current = stochasticKSeries;
        stochasticDSeriesRef.current = stochasticDSeries;
        atrSeriesRef.current = atrSeries;
        adxSeriesRef.current = adxSeries;
        obvSeriesRef.current = obvSeries;

        // Subscribe to visible range changes for infinite scroll
        chart.timeScale().subscribeVisibleLogicalRangeChange(range => {
            if (range && range.from < 0 && !loadingBoolRef.current && oldestTimeRef.current) {
                loadHistory(oldestTimeRef.current);
            }
        });

        // Subscribe to crosshair move for news marker tooltip + cross-hair sync
        chart.subscribeCrosshairMove(param => {
            if (!param.point || !param.time) {
                setNewsTooltip(null);
                hoveredNewsRef.current = null;
                setIsHoveringNews(false);
                return;
            }

            const time = typeof param.time === 'number' ? param.time : Math.floor(param.time);

            // Cross-hair sync: emit time to parent for other charts
            if (onCrosshairSync) {
                onCrosshairSync(chartId, time);
            }

            // Calculate tolerance based on timeframe
            let tolerance = 60;
            switch (timeframe) {
                case '1m': case '5m': case '15m': tolerance = 60; break;
                case '1h': tolerance = 300; break;
                case '4h': tolerance = 1800; break;
                case '1d': tolerance = 7200; break;
                case '1w': tolerance = 86400; break;
                case '1M': tolerance = 604800; break;
            }

            // Check if there's a news marker at this time (within tolerance)
            let foundNews = null;
            for (const [markerTime, news] of newsMapRef.current.entries()) {
                if (Math.abs(markerTime - time) <= tolerance) {
                    foundNews = news;
                    break;
                }
            }

            if (foundNews && param.point) {
                hoveredNewsRef.current = foundNews;
                setIsHoveringNews(true);
                setNewsTooltip({
                    x: param.point.x,
                    y: param.point.y,
                    news: foundNews
                });
            } else {
                hoveredNewsRef.current = null;
                setIsHoveringNews(false);
                setNewsTooltip(null);
            }
        });

        // Add click event listener for opening news modal
        const handleChartClick = (e) => {
            const hoveredNews = hoveredNewsRef.current;

            if (hoveredNews) {
                console.log('[Chart Click] Opening modal for news:', hoveredNews.title);
                setSelectedNews(hoveredNews);
                setNewsModalOpen(true);
            }
        };

        chartContainerRef.current.addEventListener('click', handleChartClick);

        const handleResize = () => {
            if (chartRef.current && chartContainerRef.current) {
                chartRef.current.applyOptions({
                    width: chartContainerRef.current.clientWidth,
                    height: chartContainerRef.current.clientHeight
                });
            }
        };

        window.addEventListener('resize', handleResize);

        return () => {
            window.removeEventListener('resize', handleResize);
            if (chartContainerRef.current) {
                chartContainerRef.current.removeEventListener('click', handleChartClick);
            }
            if (socketRef.current) {
                socketRef.current.disconnect();
            }
            chart.remove();
        };
    }, [chartId, isDark]); // Recreate chart when theme changes


    // Load Historical Data with infinite scroll support
    const loadHistory = async (endTimeUI = null, showLoading = false) => {
        if (loadingBoolRef.current) return;
        loadingBoolRef.current = true;
        if (showLoading) setIsLoading(true);

        try {
            const url = `/v1/klines?symbol=${symbol}&limit=1000&interval=${timeframe}${endTimeUI ? `&end=${endTimeUI}` : ''}`;
            const res = await authFetch(url);
            if (res.ok) {
                const raw = await res.json();
                if (raw.length === 0) {
                    loadingBoolRef.current = false;
                    setIsLoading(false);
                    return;
                }

                const formatted = raw
                    .map(k => {
                        let t;
                        if (k.time && typeof k.time === 'number') {
                            t = k.time;
                        } else {
                            const rawTime = k.open_time || k[0];
                            t = Math.floor(new Date(rawTime).getTime() / 1000);
                        }

                        return {
                            time: t,
                            open: parseFloat(k.open),
                            high: parseFloat(k.high),
                            low: parseFloat(k.low),
                            close: parseFloat(k.close),
                            value: parseFloat(k.value || k.volume || 0),
                            color: parseFloat(k.close) >= parseFloat(k.open)
                                ? 'rgba(8, 153, 129, 0.5)'
                                : 'rgba(242, 54, 69, 0.5)'
                        };
                    })
                    .filter(k => !isNaN(k.time))
                    .sort((a, b) => a.time - b.time);

                setData(prev => {
                    const combined = [...formatted, ...prev];
                    const unique = [];
                    const seen = new Set();
                    for (let c of combined) {
                        if (!seen.has(c.time)) {
                            seen.add(c.time);
                            unique.push(c);
                        }
                    }
                    return unique.sort((a, b) => a.time - b.time);
                });
            }
        } catch (e) {
            console.error(`[${chartId}] Fetch history failed`, e);
        } finally {
            loadingBoolRef.current = false;
            setIsLoading(false);
        }
    };

    // Load ALL news data from database (no time filtering)
    const loadNews = async () => {
        try {
            console.log(`[${chartId}] Loading ALL news from database for timeframe ${timeframe}`);

            // Fetch all news without time range filter
            // The chart will automatically display markers at correct positions based on their timestamps
            const url = `/v1/news?limit=5000`; // Increased limit to get more historical news
            const res = await authFetch(url);

            if (res.ok) {
                const newsResponse = await res.json();
                setNewsData(newsResponse.rows || []);
                console.log(`[${chartId}] Loaded ${newsResponse.rows?.length || 0} total news events for all timeframes`);
            } else {
                console.error(`[${chartId}] News API returned error:`, res.status);
            }
        } catch (e) {
            console.error(`[${chartId}] Fetch news failed`, e);
        }
    };

    // Load news once when component mounts or when news is enabled
    useEffect(() => {
        if (showNews) {
            loadNews();
        }
    }, [showNews, symbol]); // Only reload when showNews changes or symbol changes

    // Reload when symbol or timeframe changes
    useEffect(() => {
        setData([]);
        oldestTimeRef.current = null;
        latestTimeRef.current = null; // Reset latest time
        loadHistory(null, true); // Show loading spinner on symbol/timeframe change
    }, [symbol, timeframe]);

    // Update Chart Data - also run when theme changes to reapply data to new chart
    useEffect(() => {
        if (candleSeriesRef.current && volumeSeriesRef.current && data.length > 0) {
            candleSeriesRef.current.setData(data);
            lineSeriesRef.current?.setData(data.map((item) => ({
                time: item.time,
                value: item.close,
            })));
            const trendLookback = Math.min(24, data.length - 1);
            const trendIsUp = data[data.length - 1].close >= data[data.length - 1 - trendLookback].close;
            lineSeriesRef.current?.applyOptions(trendIsUp ? {
                lineColor: '#14b88a',
                topColor: 'rgba(20, 184, 138, 0.30)',
                bottomColor: 'rgba(20, 184, 138, 0.015)',
                priceLineColor: '#14b88a',
            } : {
                lineColor: '#f04468',
                topColor: 'rgba(240, 68, 104, 0.30)',
                bottomColor: 'rgba(240, 68, 104, 0.015)',
                priceLineColor: '#f04468',
            });
            volumeSeriesRef.current.setData(data.map(d => ({
                time: d.time,
                value: d.value,
                color: d.color
            })));

            // Calculate and set technical indicators
            if (indicators.sma20 && sma20SeriesRef.current) {
                const sma20Data = calculateSMA(data, 20);
                sma20SeriesRef.current.setData(sma20Data);
            }

            if (indicators.ema12 && ema12SeriesRef.current) {
                const ema12Data = calculateEMA(data, 12);
                ema12SeriesRef.current.setData(ema12Data);
            }

            if (indicators.ema26 && ema26SeriesRef.current) {
                const ema26Data = calculateEMA(data, 26);
                ema26SeriesRef.current.setData(ema26Data);
            }
            if (indicators.ema20 && ema20SeriesRef.current) {
                ema20SeriesRef.current.setData(calculateEMA(data, 20));
            }
            if (indicators.vwap && vwapSeriesRef.current) {
                vwapSeriesRef.current.setData(calculateVWAP(data));
            }
            if (indicators.supertrend && supertrendSeriesRef.current) {
                supertrendSeriesRef.current.setData(calculateSupertrend(data, 10, 3));
            }

            if (indicators.bb && bbUpperSeriesRef.current && bbMiddleSeriesRef.current && bbLowerSeriesRef.current) {
                const bbData = calculateBollingerBands(data, 20, 2);
                bbUpperSeriesRef.current.setData(bbData.upper);
                bbMiddleSeriesRef.current.setData(bbData.middle);
                bbLowerSeriesRef.current.setData(bbData.lower);
            }

            // Calculate and set RSI
            if (indicators.rsi && rsiSeriesRef.current) {
                const rsiData = calculateRSI(data, 14);
                rsiSeriesRef.current.setData(rsiData);
            }

            // Calculate and set MACD
            if (indicators.macd && macdLineSeriesRef.current && macdSignalSeriesRef.current && macdHistogramSeriesRef.current) {
                const macdData = calculateMACD(data, 12, 26, 9);
                console.log(`[${chartId}] MACD Data:`, {
                    macdPoints: macdData.macd.length,
                    signalPoints: macdData.signal.length,
                    histogramPoints: macdData.histogram.length,
                    lastMACD: macdData.macd[macdData.macd.length - 1],
                    lastSignal: macdData.signal[macdData.signal.length - 1]
                });
                macdLineSeriesRef.current.setData(macdData.macd);
                macdSignalSeriesRef.current.setData(macdData.signal);
                macdHistogramSeriesRef.current.setData(macdData.histogram);
            }
            if (indicators.stochastic && stochasticKSeriesRef.current && stochasticDSeriesRef.current) {
                const stochastic = calculateStochastic(data, 14, 3);
                stochasticKSeriesRef.current.setData(stochastic.k);
                stochasticDSeriesRef.current.setData(stochastic.d);
            }
            if (indicators.atr && atrSeriesRef.current) {
                atrSeriesRef.current.setData(calculateATR(data, 14));
            }
            if (indicators.adx && adxSeriesRef.current) {
                adxSeriesRef.current.setData(calculateADX(data, 14));
            }
            if (indicators.obv && obvSeriesRef.current) {
                obvSeriesRef.current.setData(calculateOBV(data));
            }

            // Update oldest time for infinite scroll
            if (oldestTimeRef.current === null || data[0].time < oldestTimeRef.current) {
                oldestTimeRef.current = data[0].time;
            }

            // Update latest time for realtime updates
            latestTimeRef.current = data[data.length - 1].time;
        }
    }, [data, isDark, indicators]);

    useEffect(() => {
        candleSeriesRef.current?.applyOptions({ visible: chartType === 'candles' });
        lineSeriesRef.current?.applyOptions({ visible: chartType === 'line' });
    }, [chartType, isDark]);

    // Update indicator visibility
    useEffect(() => {
        if (sma20SeriesRef.current) {
            sma20SeriesRef.current.applyOptions({ visible: indicators.sma20 });
        }
        if (ema12SeriesRef.current) {
            ema12SeriesRef.current.applyOptions({ visible: indicators.ema12 });
        }
        if (ema26SeriesRef.current) {
            ema26SeriesRef.current.applyOptions({ visible: indicators.ema26 });
        }
        ema20SeriesRef.current?.applyOptions({ visible: indicators.ema20 });
        vwapSeriesRef.current?.applyOptions({ visible: indicators.vwap });
        supertrendSeriesRef.current?.applyOptions({ visible: indicators.supertrend });
        if (bbUpperSeriesRef.current && bbMiddleSeriesRef.current && bbLowerSeriesRef.current) {
            bbUpperSeriesRef.current.applyOptions({ visible: indicators.bb });
            bbMiddleSeriesRef.current.applyOptions({ visible: indicators.bb });
            bbLowerSeriesRef.current.applyOptions({ visible: indicators.bb });
        }
        if (rsiSeriesRef.current) {
            rsiSeriesRef.current.applyOptions({ visible: indicators.rsi });
        }
        if (macdLineSeriesRef.current && macdSignalSeriesRef.current && macdHistogramSeriesRef.current) {
            macdLineSeriesRef.current.applyOptions({ visible: indicators.macd });
            macdSignalSeriesRef.current.applyOptions({ visible: indicators.macd });
            macdHistogramSeriesRef.current.applyOptions({ visible: indicators.macd });
        }
        stochasticKSeriesRef.current?.applyOptions({ visible: indicators.stochastic });
        stochasticDSeriesRef.current?.applyOptions({ visible: indicators.stochastic });
        atrSeriesRef.current?.applyOptions({ visible: indicators.atr });
        adxSeriesRef.current?.applyOptions({ visible: indicators.adx });
        obvSeriesRef.current?.applyOptions({ visible: indicators.obv });
    }, [indicators]);

    // Add news markers to chart
    useEffect(() => {
        if (candleSeriesRef.current && data.length > 0) {
            if (!showNews || newsData.length === 0) {
                // Clear markers if news is hidden or no data
                candleSeriesRef.current.setMarkers([]);
                newsMarkersRef.current = [];
                newsMapRef.current.clear();
                setVisibleNewsCount(0);
                setSelectedNews(null);
                return;
            }

            // Lightweight Charts requires marker timestamps to exist in the
            // candle series. Snap articles to their nearest loaded candle.
            const newsMap = new Map();
            const candleTimes = data.map(candle => candle.time);
            const firstTime = candleTimes[0];
            const lastTime = candleTimes[candleTimes.length - 1];
            const bucketSeconds = candleTimes.length > 1
                ? Math.max(60, candleTimes[1] - candleTimes[0])
                : 3600;
            const groupedNews = new Map();

            newsData.forEach(news => {
                const newsTime = Math.floor(new Date(news.time).getTime() / 1000);
                if (!Number.isFinite(newsTime) || newsTime < firstTime - bucketSeconds || newsTime > lastTime + bucketSeconds) return;

                let low = 0;
                let high = candleTimes.length - 1;
                while (low < high) {
                    const mid = Math.floor((low + high) / 2);
                    if (candleTimes[mid] < newsTime) low = mid + 1;
                    else high = mid;
                }
                const rightTime = candleTimes[low];
                const leftTime = low > 0 ? candleTimes[low - 1] : rightTime;
                const candleTime = Math.abs(newsTime - leftTime) <= Math.abs(rightTime - newsTime)
                    ? leftTime
                    : rightTime;
                const group = groupedNews.get(candleTime) || [];
                group.push(news);
                groupedNews.set(candleTime, group);
            });

            const markersWithNews = Array.from(groupedNews.entries())
                .map(([candleTime, articles]) => {
                    const news = [...articles].sort((a, b) =>
                        Math.abs(Number(b.sentiment_score || 0)) - Math.abs(Number(a.sentiment_score || 0))
                    )[0];

                    // Determine marker color based on sentiment
                    let color = '#2196F3'; // Blue for neutral
                    if (news.sentiment_score > 0.3) {
                        color = '#4CAF50'; // Green for positive
                    } else if (news.sentiment_score < -0.3) {
                        color = '#F44336'; // Red for negative
                    }

                    newsMap.set(candleTime, { ...news, relatedNews: articles, newsCount: articles.length });

                    return {
                        time: candleTime,
                        position: 'aboveBar',
                        color: color,
                        shape: 'circle',
                        text: articles.length > 1 ? `N${articles.length}` : 'N',
                        size: 1
                    };
                })
                .sort((a, b) => a.time - b.time); // Sort by time ascending (required by lightweight-charts)

            candleSeriesRef.current.setMarkers(markersWithNews);
            newsMarkersRef.current = markersWithNews;
            newsMapRef.current = newsMap;
            setVisibleNewsCount(markersWithNews.length);
            console.log(`[${chartId}] Added ${markersWithNews.length} news markers to chart (all news items)`);

            // Auto-select first news if none selected
            if (!selectedNews && newsMap.size > 0) {
                setSelectedNews(Array.from(newsMap.values())[0]);
            }
        }
    }, [newsData, data, showNews]);

    // SAFE-Alert signal markers on chart
    useEffect(() => {
        if (!candleSeriesRef.current || data.length === 0 || !safeAlertSignal) return;

        const lastCandle = data[data.length - 1];
        if (!lastCandle) return;

        const dir = safeAlertSignal.direction;
        const shouldAlert = safeAlertSignal.shouldAlert;

        // Create signal marker at the latest candle
        const signalMarker = {
            time: lastCandle.time,
            position: dir === 'BUY' || dir === 'UP' ? 'belowBar' : 'aboveBar',
            color: dir === 'BUY' || dir === 'UP'
                ? '#089981'
                : dir === 'SELL' || dir === 'DOWN'
                    ? '#f23645'
                    : '#787b86',
            shape: dir === 'BUY' || dir === 'UP'
                ? 'arrowUp'
                : dir === 'SELL' || dir === 'DOWN'
                    ? 'arrowDown'
                    : 'circle',
            text: shouldAlert
                ? `${dir === 'BUY' || dir === 'UP' ? '▲' : dir === 'SELL' || dir === 'DOWN' ? '▼' : '─'} ${(safeAlertSignal.confidence * 100).toFixed(0)}%`
                : '',
            size: shouldAlert ? 2 : 1,
        };

        // Merge with existing news markers (if any)
        const existingMarkers = newsMarkersRef.current || [];
        const allMarkers = [...existingMarkers, signalMarker].sort((a, b) => a.time - b.time);

        // Deduplicate by time (keep signal marker if conflict)
        const unique = [];
        const seen = new Set();
        for (const m of allMarkers) {
            if (!seen.has(m.time)) {
                seen.add(m.time);
                unique.push(m);
            }
        }

        candleSeriesRef.current.setMarkers(unique);
    }, [safeAlertSignal, data]);

    // Socket.IO for Realtime Updates (all timeframes)
    useEffect(() => {
        if (socketRef.current) {
            socketRef.current.disconnect();
        }

        // Get token from store
        const token = useStore.getState().token;
        if (!token) {
            console.warn(`[${chartId}] No token available, skipping Socket.IO connection`);
            return;
        }

        // Connect to Socket.IO gateway via Kong with JWT token
        const socket = io('http://localhost:8000', {
            path: '/stream-api/socket.io',
            transports: ['websocket'],
            reconnection: true,
            reconnectionDelay: 1000,
            reconnectionAttempts: 10,
            query: {
                token: token  // Send JWT token for Kong authentication
            }
        });

        socket.on('connect', () => {
            console.log(`[${chartId}] ✅ Socket.IO connected! Socket ID: ${socket.id}`);
            console.log(`[${chartId}] Current symbol: ${symbol}, timeframe: ${timeframe}`);

            // Subscribe to interval-specific room
            // IMPORTANT: Server uppercases the entire room name, so we need to match that
            const room = `${symbol}_${timeframe}`.toUpperCase();
            console.log(`[${chartId}] 📡 Emitting 'subscribe' event for room: ${room}`);
            socket.emit('subscribe', room);

            // Verify subscription after a short delay
            setTimeout(() => {
                console.log(`[${chartId}] ✓ Subscription should be complete for room: ${room}`);
            }, 500);
        });

        socket.on('price_event', (payload) => {
            console.log(`[${chartId}] Received price_event:`, payload);

            // payload format: { symbol: 'BTCUSDT', interval: '1m' or '1M', kline: {...} }
            // Normalize intervals to uppercase for comparison (1m vs 1M)
            const payloadInterval = (payload.interval || '').toUpperCase();
            const expectedInterval = timeframe.toUpperCase();

            if (payload.symbol === symbol && payloadInterval === expectedInterval && payload.kline) {
                const kline = payload.kline;
                const t = Math.floor(kline.openTime / 1000);

                // Only update if this is newer or equal to the latest data we have
                // This prevents "Cannot update oldest data" error from backfill data
                if (latestTimeRef.current !== null && t < latestTimeRef.current) {
                    console.log(`[${chartId}] Skipping old kline: ${t} < ${latestTimeRef.current} (backfill data)`);
                    return;
                }

                console.log(`[${chartId}] Updating chart with kline at time ${t}`);

                const candle = {
                    time: t,
                    open: parseFloat(kline.open),
                    high: parseFloat(kline.high),
                    low: parseFloat(kline.low),
                    close: parseFloat(kline.close),
                };

                const volume = {
                    time: t,
                    value: parseFloat(kline.volume),
                    color: parseFloat(kline.close) >= parseFloat(kline.open)
                        ? 'rgba(8, 153, 129, 0.5)'
                        : 'rgba(242, 54, 69, 0.5)'
                };

                if (candleSeriesRef.current && volumeSeriesRef.current) {
                    try {
                        candleSeriesRef.current.update(candle);
                        lineSeriesRef.current?.update({ time: candle.time, value: candle.close });
                        volumeSeriesRef.current.update(volume);
                    } catch (error) {
                        console.error(`[${chartId}] Error updating chart:`, error);
                    }
                }
            } else {
                console.log(`[${chartId}] Ignoring price_event - symbol: ${payload.symbol}, interval: ${payloadInterval}, expected: ${symbol}_${expectedInterval}`);
            }
        });

        socket.on('disconnect', () => {
            console.log(`[${chartId}] Socket.IO disconnected`);
        });

        socket.on('connect_error', (err) => {
            console.error(`[${chartId}] Socket.IO connection error:`, err);
        });

        socketRef.current = socket;

        return () => {
            if (socketRef.current) {
                socketRef.current.disconnect();
            }
        };
    }, [symbol, timeframe, chartId]);

    const toggleIndicator = (indicator) => {
        setIndicators((prev) => {
            const next = { ...prev };
            if (LOWER_INDICATORS.includes(indicator) && !prev[indicator]) {
                LOWER_INDICATORS.forEach((key) => { next[key] = false; });
            }
            next[indicator] = !prev[indicator];
            return next;
        });
    };

    const toggleIndicatorMenu = () => {
        if (!showIndicatorMenu && indicatorTriggerRef.current) {
            const rect = indicatorTriggerRef.current.getBoundingClientRect();
            const bodyZoom = Number.parseFloat(window.getComputedStyle(document.body).zoom) || 1;
            const openUpward = window.innerHeight - rect.bottom < 440 * bodyZoom;
            setIndicatorMenuPosition(openUpward
                ? {
                    right: (window.innerWidth - rect.right) / bodyZoom,
                    bottom: (window.innerHeight - rect.top) / bodyZoom + 8,
                }
                : {
                    right: (window.innerWidth - rect.right) / bodyZoom,
                    top: rect.bottom / bodyZoom + 8,
                });
        }
        setShowIndicatorMenu((open) => !open);
    };

    // Determine if system is in Abstain mode (Uncertain Zone)
    const isAbstain = safeAlertSignal && !safeAlertSignal.shouldAlert;

    return (
        <div style={{ width: '100%', height: '100%', position: 'relative' }}>
            {/* Uncertainty Zone overlay when system is in Abstain mode */}
            {isAbstain && (
                <div className="uncertainty-zone">
                    <span className="uncertainty-label">⚠ Uncertain Zone — Chờ tín hiệu tin cậy</span>
                </div>
            )}
            {/* Indicator Controls */}
            <div style={{
                position: 'absolute',
                top: 12,
                right: 12,
                zIndex: 20,
                backgroundColor: isDark ? 'rgba(19, 23, 34, 0.9)' : 'rgba(255, 255, 255, 0.9)',
                padding: '8px 12px',
                borderRadius: '6px',
                backdropFilter: 'blur(4px)',
                display: 'flex',
                gap: '8px',
                flexWrap: 'wrap',
                boxShadow: '0 2px 8px rgba(0,0,0,0.2)'
            }}>
                <div className="chart-type-switch" role="group" aria-label="Kiểu biểu đồ">
                    <button
                        type="button"
                        className={chartType === 'candles' ? 'active' : ''}
                        onClick={() => setChartType('candles')}
                        title="Biểu đồ nến"
                        aria-pressed={chartType === 'candles'}
                    >
                        <ChartCandlestick size={14} />
                    </button>
                    <button
                        type="button"
                        className={chartType === 'line' ? 'active' : ''}
                        onClick={() => setChartType('line')}
                        title="Biểu đồ đường"
                        aria-pressed={chartType === 'line'}
                    >
                        <ChartLine size={14} />
                    </button>
                </div>
                <div className="chart-indicator-control">
                    <button
                        ref={indicatorTriggerRef}
                        type="button"
                        className={`chart-indicator-trigger ${Object.values(indicators).some(Boolean) ? 'active' : ''}`}
                        onClick={toggleIndicatorMenu}
                    >
                        <SlidersHorizontal size={13} />
                        Chỉ báo
                        {Object.values(indicators).filter(Boolean).length > 0 && (
                            <span className="chart-indicator-count">
                                {Object.values(indicators).filter(Boolean).length}
                            </span>
                        )}
                    </button>

                    {showIndicatorMenu && createPortal((
                        <div ref={indicatorMenuRef} className="chart-indicator-menu" style={indicatorMenuPosition}>
                            <div className="chart-indicator-menu-title">Chỉ báo kỹ thuật</div>
                            <div className="chart-indicator-group-label">Xu hướng</div>
                            {[
                                ['sma20', 'SMA (20)', '#2962FF'],
                                ['ema12', 'EMA (12)', '#FF6D00'],
                                ['ema26', 'EMA (26)', '#9C27B0'],
                                ['ema20', 'EMA (20)', '#E91E63'],
                                ['vwap', 'VWAP', '#FFD54F'],
                                ['supertrend', 'Supertrend (10, 3)', '#089981'],
                                ['bb', 'Bollinger Bands (20, 2)', '#2196F3'],
                            ].map(([key, label, color]) => (
                                <button
                                    type="button"
                                    key={key}
                                    className={`chart-indicator-option ${indicators[key] ? 'selected' : ''}`}
                                    onClick={() => toggleIndicator(key)}
                                >
                                    <span className="chart-indicator-swatch" style={{ background: color }} />
                                    <span>{label}</span>
                                    {indicators[key] && <Check size={13} />}
                                </button>
                            ))}
                            <div className="chart-indicator-group-label">Động lượng</div>
                            {[
                                ['rsi', 'RSI (14)', '#FF9800'],
                                ['macd', 'MACD (12, 26, 9)', '#00BCD4'],
                                ['stochastic', 'Stochastic (14, 3)', '#7E57C2'],
                            ].map(([key, label, color]) => (
                                <button
                                    type="button"
                                    key={key}
                                    className={`chart-indicator-option ${indicators[key] ? 'selected' : ''}`}
                                    onClick={() => toggleIndicator(key)}
                                >
                                    <span className="chart-indicator-swatch" style={{ background: color }} />
                                    <span>{label}</span>
                                    {indicators[key] && <Check size={13} />}
                                </button>
                            ))}
                            <div className="chart-indicator-group-label">Biến động</div>
                            {[
                                ['atr', 'ATR (14)', '#26C6DA'],
                                ['adx', 'ADX (14)', '#AB47BC'],
                            ].map(([key, label, color]) => (
                                <button
                                    type="button"
                                    key={key}
                                    className={`chart-indicator-option ${indicators[key] ? 'selected' : ''}`}
                                    onClick={() => toggleIndicator(key)}
                                >
                                    <span className="chart-indicator-swatch" style={{ background: color }} />
                                    <span>{label}</span>
                                    {indicators[key] && <Check size={13} />}
                                </button>
                            ))}
                            <div className="chart-indicator-group-label">Khối lượng</div>
                            <button
                                type="button"
                                className={`chart-indicator-option ${indicators.obv ? 'selected' : ''}`}
                                onClick={() => toggleIndicator('obv')}
                            >
                                <span className="chart-indicator-swatch" style={{ background: '#66BB6A' }} />
                                <span>On Balance Volume</span>
                                {indicators.obv && <Check size={13} />}
                            </button>
                        </div>
                    ), document.body)}
                </div>
                <button
                    onClick={() => setShowNews(!showNews)}
                    style={{
                        padding: '4px 8px',
                        fontSize: '11px',
                        borderRadius: '4px',
                        border: 'none',
                        cursor: 'pointer',
                        backgroundColor: showNews ? '#FFC107' : (isDark ? 'rgba(255, 255, 255, 0.1)' : 'rgba(0, 0, 0, 0.1)'),
                        color: showNews ? '#000' : (isDark ? '#d1d4dc' : '#333'),
                        fontWeight: showNews ? 'bold' : 'normal',
                        transition: 'all 0.2s'
                    }}
                >
                    📰 News{showNews ? ` (${visibleNewsCount})` : ''}
                </button>

                {/* Only show expand button if not already expanded */}
                {!chartId.includes('-expanded') && (
                    <button
                        onClick={() => setIsExpanded(true)}
                        style={{
                            padding: '4px 8px',
                            fontSize: '11px',
                            borderRadius: '4px',
                            border: 'none',
                            cursor: 'pointer',
                            backgroundColor: isDark ? 'rgba(33, 150, 243, 0.2)' : 'rgba(33, 150, 243, 0.3)',
                            color: '#2196F3',
                            fontWeight: 'bold',
                            transition: 'all 0.2s'
                        }}
                        onMouseEnter={(e) => {
                            e.target.style.backgroundColor = isDark ? 'rgba(33, 150, 243, 0.3)' : 'rgba(33, 150, 243, 0.4)';
                        }}
                        onMouseLeave={(e) => {
                            e.target.style.backgroundColor = isDark ? 'rgba(33, 150, 243, 0.2)' : 'rgba(33, 150, 243, 0.3)';
                        }}
                    >
                        ⛶ Mở rộng
                    </button>
                )}
            </div>

            {/* News Legend */}
            {showNews && newsData.length > 0 && (
                <div style={{
                    position: 'absolute',
                    bottom: 45,
                    left: 12,
                    zIndex: 20,
                    backgroundColor: isDark ? 'rgba(19, 23, 34, 0.9)' : 'rgba(255, 255, 255, 0.9)',
                    padding: '6px 10px',
                    borderRadius: '4px',
                    backdropFilter: 'blur(4px)',
                    fontSize: '11px',
                    color: isDark ? '#d1d4dc' : '#333',
                    boxShadow: '0 2px 8px rgba(0,0,0,0.2)'
                }}>
                    {/* <div style={{ display: 'flex', gap: '12px', alignItems: 'center', marginBottom: selectedNews ? '8px' : '0' }}>
                        <span style={{ fontWeight: 'bold' }}>Tin tức:</span>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
                            <div style={{ width: '8px', height: '8px', borderRadius: '50%', backgroundColor: '#4CAF50' }}></div>
                            <span>Tích cực</span>
                        </div>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
                            <div style={{ width: '8px', height: '8px', borderRadius: '50%', backgroundColor: '#2196F3' }}></div>
                            <span>Trung lập</span>
                        </div>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
                            <div style={{ width: '8px', height: '8px', borderRadius: '50%', backgroundColor: '#F44336' }}></div>
                            <span>Tiêu cực</span>
                        </div>
                        <span style={{ marginLeft: '8px', opacity: 0.7 }}>({newsMarkersRef.current.length} sự kiện)</span>
                    </div> */}
                </div>
            )}

            {showNews && newsData.length > 0 && visibleNewsCount === 0 && (
                <div className="chart-news-empty">
                    Không có tin DB trùng với vùng nến đang tải
                </div>
            )}

            {/* News Tooltip */}
            {showNews && newsTooltip && (
                <div style={{
                    position: 'absolute',
                    left: newsTooltip.x + 15,
                    top: newsTooltip.y - 40,
                    zIndex: 30,
                    backgroundColor: isDark ? 'rgba(19, 23, 34, 0.95)' : 'rgba(255, 255, 255, 0.95)',
                    padding: '8px 12px',
                    borderRadius: '6px',
                    boxShadow: '0 4px 12px rgba(0,0,0,0.3)',
                    maxWidth: '300px',
                    pointerEvents: 'none',
                    border: `2px solid ${newsTooltip.news.sentiment_score > 0.3 ? '#4CAF50' : (newsTooltip.news.sentiment_score < -0.3 ? '#F44336' : '#2196F3')}`
                }}>
                    <div style={{ fontSize: '12px', fontWeight: 'bold', color: isDark ? '#FFC107' : '#F57C00', marginBottom: '4px' }}>
                        {newsTooltip.news.title || 'No title'}
                    </div>
                    <div style={{ fontSize: '10px', opacity: 0.8, color: isDark ? '#d1d4dc' : '#333', marginBottom: '4px' }}>
                        {newsTooltip.news.source || 'Unknown'} • {new Date(newsTooltip.news.time).toLocaleTimeString('vi-VN')}
                    </div>
                    <div style={{ fontSize: '9px', opacity: 0.6, fontStyle: 'italic', marginTop: '4px', borderTop: `1px solid ${isDark ? 'rgba(255,255,255,0.1)' : 'rgba(0,0,0,0.1)'}`, paddingTop: '4px' }}>
                        💡 Click để xem chi tiết
                    </div>
                </div>
            )}

            <div
                ref={chartContainerRef}
                style={{
                    width: '100%',
                    height: '100%',
                    cursor: isHoveringNews ? 'pointer' : 'default'
                }}
            />
            <ChartDrawingOverlay
                activeTool={drawingTool}
                chartRef={chartRef}
                priceSeriesRef={chartType === 'line' ? lineSeriesRef : candleSeriesRef}
                containerRef={chartContainerRef}
                chartReadyVersion={chartReadyVersion}
                candleData={data}
            />
            {/* Loading Overlay */}
            {isLoading && (
                <div style={{
                    position: 'absolute',
                    top: 0,
                    left: 0,
                    right: 0,
                    bottom: 0,
                    background: 'rgba(0, 0, 0, 0.6)',
                    backdropFilter: 'blur(2px)',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    zIndex: 10,
                    animation: 'fadeIn 0.2s ease-out'
                }}>
                    <LoadingSpinner size="md" text="Đang tải..." />
                </div>
            )}

            {/* News Detail Modal */}
            {newsModalOpen && selectedNews && (
                <div
                    style={{
                        position: 'fixed',
                        top: 0,
                        left: 0,
                        right: 0,
                        bottom: 0,
                        backgroundColor: 'rgba(0, 0, 0, 0.7)',
                        backdropFilter: 'blur(4px)',
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        zIndex: 1000,
                        padding: '20px'
                    }}
                    onClick={() => setNewsModalOpen(false)}
                >
                    <div
                        style={{
                            backgroundColor: isDark ? '#1a1e2e' : '#ffffff',
                            borderRadius: '12px',
                            maxWidth: '700px',
                            width: '100%',
                            maxHeight: '80vh',
                            overflow: 'auto',
                            boxShadow: '0 20px 60px rgba(0,0,0,0.5)',
                            position: 'relative'
                        }}
                        onClick={(e) => e.stopPropagation()}
                    >
                        {/* Close Button */}
                        <button
                            onClick={() => setNewsModalOpen(false)}
                            style={{
                                position: 'absolute',
                                top: '16px',
                                right: '16px',
                                background: isDark ? 'rgba(255,255,255,0.1)' : 'rgba(0,0,0,0.1)',
                                border: 'none',
                                borderRadius: '50%',
                                width: '36px',
                                height: '36px',
                                cursor: 'pointer',
                                display: 'flex',
                                alignItems: 'center',
                                justifyContent: 'center',
                                fontSize: '20px',
                                color: isDark ? '#fff' : '#333',
                                transition: 'all 0.2s',
                                zIndex: 1
                            }}
                            onMouseEnter={(e) => {
                                e.target.style.background = isDark ? 'rgba(255,255,255,0.2)' : 'rgba(0,0,0,0.2)';
                            }}
                            onMouseLeave={(e) => {
                                e.target.style.background = isDark ? 'rgba(255,255,255,0.1)' : 'rgba(0,0,0,0.1)';
                            }}
                        >
                            ✕
                        </button>

                        {/* Modal Content */}
                        <div style={{ padding: '32px' }}>
                            {(() => {
                                // Use raw_score if available, otherwise use root level
                                const newsDetail = selectedNews.raw_score || selectedNews;
                                const sentimentScore = newsDetail.sentiment_score || 0;

                                return (
                                    <>
                                        {/* Sentiment Badge */}
                                        <div style={{ marginBottom: '16px', display: 'flex', gap: '8px', flexWrap: 'wrap' }}>
                                            <span style={{
                                                display: 'inline-block',
                                                padding: '6px 12px',
                                                borderRadius: '20px',
                                                fontSize: '12px',
                                                fontWeight: 'bold',
                                                backgroundColor: sentimentScore > 0.3
                                                    ? 'rgba(76, 175, 80, 0.2)'
                                                    : (sentimentScore < -0.3 ? 'rgba(244, 67, 54, 0.2)' : 'rgba(33, 150, 243, 0.2)'),
                                                color: sentimentScore > 0.3
                                                    ? '#4CAF50'
                                                    : (sentimentScore < -0.3 ? '#F44336' : '#2196F3')
                                            }}>
                                                {sentimentScore > 0.3 ? '📈 Tích cực' : (sentimentScore < -0.3 ? '📉 Tiêu cực' : '➖ Trung lập')}
                                                {' '}
                                                ({(sentimentScore * 100).toFixed(1)}%)
                                            </span>

                                            {newsDetail.category && (
                                                <span style={{
                                                    display: 'inline-block',
                                                    padding: '6px 12px',
                                                    borderRadius: '20px',
                                                    fontSize: '12px',
                                                    fontWeight: 'bold',
                                                    backgroundColor: isDark ? 'rgba(255, 193, 7, 0.2)' : 'rgba(255, 193, 7, 0.3)',
                                                    color: isDark ? '#FFC107' : '#F57C00'
                                                }}>
                                                    📂 {newsDetail.category}
                                                </span>
                                            )}

                                            {newsDetail.relevance_score !== undefined && (
                                                <span style={{
                                                    display: 'inline-block',
                                                    padding: '6px 12px',
                                                    borderRadius: '20px',
                                                    fontSize: '12px',
                                                    fontWeight: 'bold',
                                                    backgroundColor: isDark ? 'rgba(156, 39, 176, 0.2)' : 'rgba(156, 39, 176, 0.3)',
                                                    color: '#9C27B0'
                                                }}>
                                                    🎯 Relevance: {(newsDetail.relevance_score * 100).toFixed(0)}%
                                                </span>
                                            )}
                                        </div>

                                        {/* Title */}
                                        <h2 style={{
                                            margin: '0 0 16px 0',
                                            fontSize: '24px',
                                            fontWeight: 'bold',
                                            color: isDark ? '#FFC107' : '#F57C00',
                                            lineHeight: '1.4',
                                            paddingRight: '40px'
                                        }}>
                                            {newsDetail.title || 'Không có tiêu đề'}
                                        </h2>

                                        {/* Meta Info */}
                                        <div style={{
                                            display: 'flex',
                                            gap: '16px',
                                            marginBottom: '16px',
                                            fontSize: '14px',
                                            color: isDark ? 'rgba(255,255,255,0.7)' : 'rgba(0,0,0,0.6)',
                                            flexWrap: 'wrap'
                                        }}>
                                            <div>
                                                <strong>📍 Nguồn:</strong> {newsDetail.source || 'Unknown'}
                                            </div>
                                            <div>
                                                <strong>🕒 Thời gian:</strong> {new Date(newsDetail.published_at || newsDetail.time || selectedNews.time).toLocaleString('vi-VN')}
                                            </div>
                                        </div>

                                        {/* Symbols */}
                                        {newsDetail.symbols && newsDetail.symbols.length > 0 && (
                                            <div style={{
                                                marginBottom: '16px',
                                                fontSize: '13px',
                                                color: isDark ? 'rgba(255,255,255,0.8)' : 'rgba(0,0,0,0.7)'
                                            }}>
                                                <strong>💱 Symbols:</strong> {newsDetail.symbols.join(', ')}
                                            </div>
                                        )}

                                        {/* Divider */}
                                        <div style={{
                                            height: '1px',
                                            background: isDark ? 'rgba(255,255,255,0.1)' : 'rgba(0,0,0,0.1)',
                                            margin: '24px 0'
                                        }} />

                                        {/* Content */}
                                        {newsDetail.content && (
                                            <div style={{
                                                fontSize: '15px',
                                                lineHeight: '1.8',
                                                color: isDark ? 'rgba(255,255,255,0.9)' : 'rgba(0,0,0,0.8)',
                                                marginBottom: '24px',
                                                maxHeight: '400px',
                                                overflowY: 'auto',
                                                paddingRight: '8px'
                                            }}>
                                                {newsDetail.content}
                                            </div>
                                        )}

                                        {/* Link to Original Article */}
                                        {newsDetail.url && (
                                            <a
                                                href={newsDetail.url}
                                                target="_blank"
                                                rel="noopener noreferrer"
                                                style={{
                                                    display: 'inline-block',
                                                    padding: '12px 24px',
                                                    backgroundColor: isDark ? '#2196F3' : '#1976D2',
                                                    color: '#fff',
                                                    textDecoration: 'none',
                                                    borderRadius: '6px',
                                                    fontWeight: 'bold',
                                                    fontSize: '14px',
                                                    transition: 'all 0.2s',
                                                    boxShadow: '0 2px 8px rgba(33, 150, 243, 0.3)'
                                                }}
                                                onMouseEnter={(e) => {
                                                    e.target.style.backgroundColor = isDark ? '#1976D2' : '#1565C0';
                                                    e.target.style.transform = 'translateY(-2px)';
                                                    e.target.style.boxShadow = '0 4px 12px rgba(33, 150, 243, 0.4)';
                                                }}
                                                onMouseLeave={(e) => {
                                                    e.target.style.backgroundColor = isDark ? '#2196F3' : '#1976D2';
                                                    e.target.style.transform = 'translateY(0)';
                                                    e.target.style.boxShadow = '0 2px 8px rgba(33, 150, 243, 0.3)';
                                                }}
                                            >
                                                🔗 Đọc bài viết đầy đủ
                                            </a>
                                        )}
                                    </>
                                );
                            })()}
                        </div>
                    </div>
                </div>
            )}

            {/* Expanded Chart Modal */}
            {isExpanded && (
                <div
                    style={{
                        position: 'fixed',
                        top: 0,
                        left: 0,
                        right: 0,
                        bottom: 0,
                        backgroundColor: 'rgba(0, 0, 0, 0.9)',
                        backdropFilter: 'blur(4px)',
                        display: 'flex',
                        flexDirection: 'column',
                        zIndex: 2000,
                        padding: '20px'
                    }}
                    onClick={() => setIsExpanded(false)}
                >
                    <div
                        style={{
                            backgroundColor: isDark ? '#1a1e2e' : '#ffffff',
                            borderRadius: '12px',
                            width: '100%',
                            height: '100%',
                            display: 'flex',
                            flexDirection: 'column',
                            boxShadow: '0 20px 60px rgba(0,0,0,0.5)',
                            position: 'relative',
                            overflow: 'hidden'
                        }}
                        onClick={(e) => e.stopPropagation()}
                    >
                        {/* Header */}
                        <div style={{
                            padding: '16px 24px',
                            borderBottom: `1px solid ${isDark ? 'rgba(255,255,255,0.1)' : 'rgba(0,0,0,0.1)'}`,
                            display: 'flex',
                            justifyContent: 'space-between',
                            alignItems: 'center'
                        }}>
                            <h2 style={{
                                margin: 0,
                                fontSize: '20px',
                                fontWeight: 'bold',
                                color: isDark ? '#FFC107' : '#F57C00'
                            }}>
                                {symbol} - {timeframe.toUpperCase()}
                            </h2>

                            <button
                                onClick={() => setIsExpanded(false)}
                                style={{
                                    background: isDark ? 'rgba(255,255,255,0.1)' : 'rgba(0,0,0,0.1)',
                                    border: 'none',
                                    borderRadius: '50%',
                                    width: '40px',
                                    height: '40px',
                                    cursor: 'pointer',
                                    display: 'flex',
                                    alignItems: 'center',
                                    justifyContent: 'center',
                                    fontSize: '24px',
                                    color: isDark ? '#fff' : '#333',
                                    transition: 'all 0.2s'
                                }}
                                onMouseEnter={(e) => {
                                    e.target.style.background = isDark ? 'rgba(255,255,255,0.2)' : 'rgba(0,0,0,0.2)';
                                }}
                                onMouseLeave={(e) => {
                                    e.target.style.background = isDark ? 'rgba(255,255,255,0.1)' : 'rgba(0,0,0,0.1)';
                                }}
                            >
                                ✕
                            </button>
                        </div>

                        {/* Chart Info */}
                        <div style={{
                            padding: '12px 24px',
                            fontSize: '14px',
                            color: isDark ? 'rgba(255,255,255,0.7)' : 'rgba(0,0,0,0.6)',
                            borderBottom: `1px solid ${isDark ? 'rgba(255,255,255,0.1)' : 'rgba(0,0,0,0.1)'}`
                        }}>
                            💡 Biểu đồ mở rộng - Xem rõ hơn với màn hình lớn, có thể scroll để xem tất cả news markers
                        </div>

                        {/* Expanded Chart Content - Full chart with all features */}
                        <div style={{
                            flex: 1,
                            padding: '20px',
                            position: 'relative',
                            minHeight: 0,
                            overflow: 'auto' // Allow scrolling if too many news
                        }}>
                            {/* Render the same chart component recursively */}
                            <MultiTimeframeChart
                                symbol={symbol}
                                timeframe={timeframe}
                                chartId={`${chartId}-expanded`}
                                drawingTool={drawingTool}
                            />
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
}


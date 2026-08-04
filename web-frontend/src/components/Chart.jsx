import React, { useEffect, useRef, useState } from 'react';
import { createChart } from 'lightweight-charts';
import useStore from '../store';

const chartTimestampToDate = (time) => typeof time === 'number'
  ? new Date(time * 1000)
  : new Date(Date.UTC(time.year, (time.month || 1) - 1, time.day || 1));

const formatVietnamChartTime = (time, tickMarkType = 3) => {
  const date = chartTimestampToDate(time);
  if (Number.isNaN(date.getTime())) return '';
  return new Intl.DateTimeFormat('vi-VN', tickMarkType <= 2
    ? { timeZone: 'Asia/Ho_Chi_Minh', day: '2-digit', month: '2-digit', year: tickMarkType === 0 ? 'numeric' : undefined }
    : { timeZone: 'Asia/Ho_Chi_Minh', hour: '2-digit', minute: '2-digit', hour12: false }
  ).format(date);
};

export default function Chart() {
  const chartContainerRef = useRef();
  const legendRef = useRef();
  const chartRef = useRef();
  const candleSeriesRef = useRef();
  const volumeSeriesRef = useRef();

  const { currentSymbol, price, authFetch } = useStore();
  const [data, setData] = useState([]); // Stores UI-Time data (potentially shifted)
  const [timeOffset, setTimeOffset] = useState(0); // Offset to shift History to match Realtime

  const loadingRef = useRef(false);
  const oldestTimeRef = useRef(null);
  const timeOffsetRef = useRef(0); // Ref for sync access in callbacks

  // Update Ref when state changes
  useEffect(() => {
    timeOffsetRef.current = timeOffset;
  }, [timeOffset]);

  // Initialize Chart
  useEffect(() => {
    if (!chartContainerRef.current) return;

    chartContainerRef.current.innerHTML = '';

    const chart = createChart(chartContainerRef.current, {
      layout: {
        background: { type: 'solid', color: '#131722' },
        textColor: '#d1d4dc',
      },
      grid: {
        vertLines: { color: 'rgba(42, 46, 57, 0.2)', style: 1, visible: true },
        horzLines: { color: 'rgba(42, 46, 57, 0.2)', style: 1, visible: true },
      },
      crosshair: {
        mode: 1,
      },
      width: chartContainerRef.current.clientWidth,
      height: chartContainerRef.current.clientHeight,
      timeScale: {
        timeVisible: true,
        secondsVisible: false,
        barSpacing: 10,
        minBarSpacing: 3,
        borderColor: '#2B2B43',
        tickMarkFormatter: formatVietnamChartTime,
      },
      localization: {
        locale: 'vi-VN',
        timeFormatter: (time) => new Intl.DateTimeFormat('vi-VN', {
          timeZone: 'Asia/Ho_Chi_Minh', day: '2-digit', month: '2-digit', year: 'numeric',
          hour: '2-digit', minute: '2-digit', hour12: false,
        }).format(chartTimestampToDate(time)),
      },
      rightPriceScale: {
        borderColor: '#2B2B43',
      },
    });

    const candlestickSeries = chart.addCandlestickSeries({
      upColor: '#089981',
      downColor: '#f23645',
      borderVisible: false,
      wickUpColor: '#089981',
      wickDownColor: '#f23645',
    });

    const volumeSeries = chart.addHistogramSeries({
      priceFormat: {
        type: 'volume',
      },
      priceScaleId: '',
    });

    chart.priceScale('').applyOptions({
      scaleMargins: {
        top: 0.8,
        bottom: 0,
      },
    });

    chartRef.current = chart;
    candleSeriesRef.current = candlestickSeries;
    volumeSeriesRef.current = volumeSeries;

    chart.subscribeCrosshairMove(param => {
      if (!legendRef.current) return;
      const candleData = param.seriesData.get(candlestickSeries);
      const volumeData = param.seriesData.get(volumeSeries);

      if (candleData) {
        const { open, high, low, close } = candleData;
        const color = close >= open ? '#089981' : '#f23645';
        const vol = volumeData ? volumeData.value : 0;

        legendRef.current.innerHTML = `
          <div style="font-size: 16px; font-weight: bold; margin-bottom: 4px;">${currentSymbol}</div>
          <div style="display: flex; gap: 12px; font-size: 12px; font-family: monospace;">
            <span style="color: ${color}">O: ${open.toFixed(2)}</span>
            <span style="color: ${color}">H: ${high.toFixed(2)}</span>
            <span style="color: ${color}">L: ${low.toFixed(2)}</span>
            <span style="color: ${color}">C: ${close.toFixed(2)}</span>
            <span style="color: #787b86">V: ${vol.toFixed(2)}</span>
          </div>
        `;
      }
    });

    chart.timeScale().subscribeVisibleLogicalRangeChange(range => {
      if (range && range.from < 0 && !loadingRef.current && oldestTimeRef.current) {
        loadHistory(oldestTimeRef.current);
      }
    });

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
      chart.remove();
    };
  }, [currentSymbol]);

  const loadHistory = async (endTimeUI = null) => {
    if (loadingRef.current) return;
    loadingRef.current = true;

    try {
      const url = `/v1/klines?symbol=${currentSymbol}&limit=1000&interval=1m${endTimeUI ? `&end=${endTimeUI}` : ''}`;
      const res = await authFetch(url);
      if (res.ok) {
        const raw = await res.json();
        if (raw.length === 0) {
          loadingRef.current = false;
          return;
        }

        const formatted = raw
          .map(k => {
            // API returns 'time' in seconds.
            // WebSocket returns 'kline' object with potential ms or seconds, but store.js converts to 't' (seconds).
            // Robust parsing:
            let t;
            if (k.time && typeof k.time === 'number') {
              // Already seconds (from our API)
              t = k.time;
            } else {
              // Fallback for raw formats or strings
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
              color: parseFloat(k.close) >= parseFloat(k.open) ? 'rgba(8, 153, 129, 0.5)' : 'rgba(242, 54, 69, 0.5)'
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
      console.error("Fetch history failed", e);
    } finally {
      loadingRef.current = false;
    }
  };

  // Initial Load
  useEffect(() => {
    setData([]);
    setTimeOffset(0);
    oldestTimeRef.current = null;
    loadHistory(null);
  }, [currentSymbol]);

  // Sync Data to Chart
  useEffect(() => {
    if (candleSeriesRef.current && volumeSeriesRef.current && data.length > 0) {
      candleSeriesRef.current.setData(data.map(d => ({ ...d }))); // O,H,L,C
      volumeSeriesRef.current.setData(data.map(d => ({
        time: d.time,
        value: d.value,
        color: d.color
      })));

      if (oldestTimeRef.current === null || data[0].time < oldestTimeRef.current) {
        oldestTimeRef.current = data[0].time;
      }
    }
  }, [data]);

  // Realtime Updates & Time Sync
  useEffect(() => {
    if (price && candleSeriesRef.current && volumeSeriesRef.current) {
      // Validate and update
      if (!isNaN(price.t)) {
        candleSeriesRef.current.update({
          time: price.t,
          open: price.o,
          high: price.h,
          low: price.l,
          close: price.c
        });

        volumeSeriesRef.current.update({
          time: price.t,
          value: price.value || 0,
          color: price.color
        });
      }
    }
  }, [price]);

  return (
    <div style={{ width: '100%', height: '100%', position: 'relative' }}>
      {/* Legend */}
      <div ref={legendRef}
        style={{
          position: 'absolute',
          top: 12,
          left: 12,
          zIndex: 20,
          color: '#d1d4dc',
          pointerEvents: 'none',
          backgroundColor: 'rgba(19, 23, 34, 0.6)',
          padding: '6px 10px',
          borderRadius: '4px',
          backdropFilter: 'blur(4px)'
        }}
      >
        <div style={{ fontSize: '16px', fontWeight: 'bold' }}>{currentSymbol}</div>
      </div>

      <div ref={chartContainerRef} style={{ width: '100%', height: '100%', position: 'relative', zIndex: 10 }} />
    </div>
  );
}

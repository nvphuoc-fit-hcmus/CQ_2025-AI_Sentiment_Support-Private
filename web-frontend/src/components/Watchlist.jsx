import { useState, useEffect } from 'react';
import useStore from '../store';
import { Search } from 'lucide-react';
import { io } from 'socket.io-client';

const TIMEFRAMES = [
  { label: '1 phút', value: '1m', seconds: 60 },
  { label: '5 phút', value: '5m', seconds: 300 },
  { label: '15 phút', value: '15m', seconds: 900 },
  { label: '1 giờ', value: '1h', seconds: 3600 },
  { label: '4 giờ', value: '4h', seconds: 14400 },
  { label: '1 ngày', value: '1d', seconds: 86400 },
  { label: '1 tuần', value: '1w', seconds: 604800 },
];

const styles = {
  container: {
    display: 'flex',
    flexDirection: 'column',
    height: '100%',
  },
  searchHeader: {
    padding: '12px',
    borderBottom: '1px solid var(--border-color)',
    display: 'flex',
    alignItems: 'center',
    gap: '8px',
    backgroundColor: 'var(--bg-panel)',
  },
  searchIcon: {
    color: 'var(--text-secondary)',
    flexShrink: 0,
  },
  searchInput: {
    background: 'transparent',
    border: 'none',
    outline: 'none',
    color: 'var(--text-primary)',
    width: '100%',
    fontSize: '13px',
    fontFamily: 'var(--font-family)',
  },
  timeframeSelector: {
    padding: '10px 12px',
    borderBottom: '1px solid var(--border-color)',
    display: 'flex',
    alignItems: 'center',
    gap: '8px',
    backgroundColor: 'var(--bg-panel)',
  },
  timeframeLabel: {
    fontSize: '11px',
    color: 'var(--text-secondary)',
    whiteSpace: 'nowrap',
    fontWeight: '500',
  },
  timeframeSelect: {
    background: 'var(--bg-dark)',
    border: '1px solid var(--border-color)',
    borderRadius: '6px',
    color: 'var(--text-primary)',
    fontSize: '11px',
    padding: '6px 28px 6px 10px',
    cursor: 'pointer',
    outline: 'none',
    flex: 1,
    appearance: 'none',
    backgroundImage:
      "url(\"data:image/svg+xml;charset=UTF-8,%3csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%23848e9c' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3e%3cpolyline points='6 9 12 15 18 9'%3e%3c/polyline%3e%3c/svg%3e\")",
    backgroundRepeat: 'no-repeat',
    backgroundPosition: 'right 6px center',
    backgroundSize: '16px',
    transition: 'all 250ms cubic-bezier(0.4, 0, 0.2, 1)',
    fontWeight: '500',
  },
  listContainer: {
    flex: 1,
    overflowY: 'auto',
  },
  item: (isSelected) => ({
    padding: '10px 12px',
    borderBottom: '1px solid var(--border-color)',
    cursor: 'pointer',
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    backgroundColor: isSelected ? 'var(--bg-hover)' : 'transparent',
    borderLeft: isSelected ? '3px solid var(--accent-blue)' : 'none',
    paddingLeft: isSelected ? '9px' : '12px',
    transition: 'all 250ms cubic-bezier(0.4, 0, 0.2, 1)',
    position: 'relative',
  }),
  symbolInfo: {
    display: 'flex',
    flexDirection: 'column',
    gap: '2px',
  },
  symbolName: {
    fontSize: '13px',
    fontWeight: '600',
    color: 'var(--text-primary)',
    letterSpacing: '0.3px',
  },
  symbolPair: {
    fontSize: '10px',
    color: 'var(--text-tertiary)',
    fontWeight: '500',
  },
  priceInfo: {
    textAlign: 'right',
    display: 'flex',
    flexDirection: 'column',
    gap: '2px',
  },
  price: {
    fontSize: '13px',
    fontFamily: 'var(--font-mono)',
    color: 'var(--text-primary)',
    fontWeight: '500',
  },
  change: (isUp) => ({
    fontSize: '11px',
    fontWeight: '600',
    color: isUp ? 'var(--accent-green)' : 'var(--accent-red)',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'flex-end',
    gap: '3px',
    fontFamily: 'var(--font-mono)',
  }),
  changeNeutral: {
    fontSize: '11px',
    color: 'var(--text-tertiary)',
  },
  loading: {
    fontSize: '11px',
    color: 'var(--text-tertiary)',
  },
};

export default function Watchlist() {
  const { token, supportedSymbols, currentSymbol, setSymbol, authFetch } =
    useStore();
  const [search, setSearch] = useState('');
  const [selectedTimeframe, setSelectedTimeframe] = useState('1m');
  const [oldPrices, setOldPrices] = useState({}); // Giá X thời gian trước
  const [currentPrices, setCurrentPrices] = useState({});

  // Fetch old prices when timeframe changes
  useEffect(() => {
    if (!token || supportedSymbols.length === 0) return;

    const fetchOldPrices = async () => {
      const timeframe = TIMEFRAMES.find((tf) => tf.value === selectedTimeframe);
      if (!timeframe) return;

      // Choose appropriate interval based on timeframe to ensure data exists (backfill limitations)
      let fetchInterval = '1m';
      if (['1d', '1w'].includes(selectedTimeframe)) fetchInterval = '1h';
      else if (['4h', '1h'].includes(selectedTimeframe)) fetchInterval = '5m';

      const now = Math.floor(Date.now() / 1000);
      const oldTime = now - timeframe.seconds;

      const prices = {};

      // Fetch old price for each symbol
      for (const symbol of supportedSymbols) {
        try {
          const url = `/v1/klines?symbol=${symbol}&limit=1&interval=${fetchInterval}&end=${oldTime}`;
          const res = await authFetch(url);
          if (res.ok) {
            const data = await res.json();
            if (data && data.length > 0) {
              // Use close price of the candle nearest to oldTime
              prices[symbol] = parseFloat(data[0].close);
            }
          }
        } catch (e) {
          console.error(
            `[Watchlist] Failed to fetch old price for ${symbol}:`,
            e,
          );
        }
      }

      setOldPrices(prices);
      console.log(
        `[Watchlist] Fetched old prices for ${selectedTimeframe} (int: ${fetchInterval}):`,
        prices,
      );
    };

    fetchOldPrices();

    // Refresh old prices every minute
    const interval = setInterval(fetchOldPrices, 60000);

    return () => clearInterval(interval);
  }, [selectedTimeframe, token, supportedSymbols, authFetch]);

  // WebSocket connection for real-time prices
  useEffect(() => {
    if (!token) return;

    const wsBase = import.meta.env.VITE_WS_URL || window.location.origin;
    const socketUrl = `${wsBase}?token=${encodeURIComponent(token)}`;

    const socket = io(socketUrl, {
      path: '/stream-api/socket.io',
      transports: ['websocket'],
      reconnectionDelay: 1000,
      reconnectionAttempts: 10,
    });

    socket.on('connect', () => {
      console.log('[Watchlist] WebSocket connected');
      // Subscribe to all symbols
      supportedSymbols.forEach((symbol) => {
        socket.emit('subscribe', symbol);
      });
    });

    socket.on('price_event', (msg) => {
      if (msg && msg.symbol && msg.kline) {
        const symbol = msg.symbol.toUpperCase();
        const price = parseFloat(msg.kline.c || msg.kline.close);

        // Update current price
        setCurrentPrices((prev) => ({
          ...prev,
          [symbol]: price,
        }));
      }
    });

    socket.on('disconnect', () => {
      console.log('[Watchlist] WebSocket disconnected');
    });

    return () => {
      supportedSymbols.forEach((symbol) => {
        socket.emit('unsubscribe', symbol);
      });
      socket.disconnect();
    };
  }, [token, supportedSymbols]);

  const filtered = supportedSymbols.filter((s) =>
    s.includes(search.toUpperCase()),
  );

  return (
    <div style={styles.container}>
      {/* Search Header */}
      <div style={styles.searchHeader}>
        <Search size={14} style={styles.searchIcon} />
        <input
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          placeholder='Tìm kiếm...'
          style={styles.searchInput}
        />
      </div>

      {/* Timeframe Selector */}
      <div style={styles.timeframeSelector}>
        <span style={styles.timeframeLabel}>% Thay đổi:</span>
        <select
          value={selectedTimeframe}
          onChange={(e) => setSelectedTimeframe(e.target.value)}
          style={styles.timeframeSelect}
        >
          {TIMEFRAMES.map((tf) => (
            <option key={tf.value} value={tf.value}>
              {tf.label}
            </option>
          ))}
        </select>
      </div>

      {/* List */}
      <div style={styles.listContainer}>
        {filtered.map((symbol) => {
          const currentPrice = currentPrices[symbol];
          const oldPrice = oldPrices[symbol];

          // Calculate % change
          let change = null;
          if (currentPrice && oldPrice && oldPrice !== currentPrice) {
            change = ((currentPrice - oldPrice) / oldPrice) * 100;
          }

          const isUp = change !== null && change >= 0;
          const isSelected = symbol === currentSymbol;

          return (
            <div
              key={symbol}
              onClick={() => setSymbol(symbol)}
              style={styles.item(isSelected)}
              className='watchlist-item'
              onMouseEnter={(e) => {
                if (!isSelected) {
                  e.currentTarget.style.backgroundColor = 'var(--bg-hover)';
                  e.currentTarget.style.paddingLeft = '16px';
                }
              }}
              onMouseLeave={(e) => {
                if (!isSelected) {
                  e.currentTarget.style.backgroundColor = 'transparent';
                  e.currentTarget.style.paddingLeft = '12px';
                }
              }}
            >
              <div style={styles.symbolInfo}>
                <span style={styles.symbolName}>
                  {symbol.replace('USDT', '')}
                </span>
                <span style={styles.symbolPair}>USDT</span>
              </div>
              {currentPrice ? (
                <div style={styles.priceInfo}>
                  <div style={styles.price}>
                    {currentPrice.toFixed(currentPrice < 10 ? 4 : 2)}
                  </div>
                  {change !== null ? (
                    <div style={styles.change(isUp)}>
                      {isUp ? '▲' : '▼'}
                      {Math.abs(change).toFixed(2)}%
                    </div>
                  ) : (
                    <div style={styles.changeNeutral}>--</div>
                  )}
                </div>
              ) : (
                <span style={styles.loading}>...</span>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}

import React, { useState, useEffect, useRef } from 'react';
import useStore from '../store';
import { useTheme } from './ThemeProvider';
import {
    Activity, LogOut, CheckCircle, BarChart3, TrendingUp,
    Sun, Moon, History, Search, X, Settings, ChevronDown, Bell,
    ShieldAlert, CircleDollarSign, KeyRound, Trash2, Newspaper
} from 'lucide-react';
import {
    getNotifications, markAllNotificationsRead, clearNotifications, pushNotification
} from '../utils/notificationCenter';

const SYMBOLS = [
    { symbol: 'BTCUSDT', name: 'Bitcoin', short: 'BTC', glyph: '₿', color: '#f7931a' },
    { symbol: 'ETHUSDT', name: 'Ethereum', short: 'ETH', glyph: '◆', color: '#627eea' },
    { symbol: 'BNBUSDT', name: 'BNB', short: 'BNB', glyph: '◆', color: '#f3ba2f' },
    { symbol: 'SOLUSDT', name: 'Solana', short: 'SOL', glyph: '≋', color: '#7c5cff' },
    { symbol: 'XRPUSDT', name: 'XRP', short: 'XRP', glyph: '×', color: '#23292f' },
    { symbol: 'DOGEUSDT', name: 'Dogecoin', short: 'DOGE', glyph: 'Ð', color: '#c2a633' },
    { symbol: 'ADAUSDT', name: 'Cardano', short: 'ADA', glyph: 'A', color: '#3468d4' },
    { symbol: 'AVAXUSDT', name: 'Avalanche', short: 'AVAX', glyph: 'A', color: '#e84142' },
    { symbol: 'DOTUSDT', name: 'Polkadot', short: 'DOT', glyph: '●', color: '#e6007a' },
    { symbol: 'POLUSDT', name: 'Polygon', short: 'POL', glyph: 'P', color: '#8247e5' },
];

function CoinIcon({ coin }) {
    return (
        <span
            className="coin-symbol-icon"
            style={{ '--coin-color': coin?.color || '#64748b' }}
            aria-hidden="true"
        >
            {coin?.glyph || '?'}
        </span>
    );
}

export default function Navbar({ currentPage, onNavigate }) {
    const { logout, currentSymbol, setSymbol, authFetch, user } = useStore();
    const { isDark, toggleTheme } = useTheme();
    const [searchOpen, setSearchOpen] = useState(false);
    const [searchQuery, setSearchQuery] = useState('');
    const [notificationOpen, setNotificationOpen] = useState(false);
    const [notifications, setNotifications] = useState(() => getNotifications());
    const searchRef = useRef(null);
    const inputRef = useRef(null);
    const notificationRef = useRef(null);

    useEffect(() => {
        const handleNotification = () => setNotifications(getNotifications());
        window.addEventListener('aegis-notification', handleNotification);
        return () => window.removeEventListener('aegis-notification', handleNotification);
    }, []);

    // Reconcile recently closed investments independently of the investment
    // page SSE connection. This prevents the bell from missing an event when
    // the browser changes page or EventSource reconnects at close time.
    useEffect(() => {
        if (!user?.id) return undefined;
        let cancelled = false;

        const syncClosedInvestments = async () => {
            try {
                const response = await authFetch(`/v1/investments/${user.id}?page=1&limit=10`);
                if (!response.ok || cancelled) return;
                const data = await response.json();
                (data.investments || [])
                    .filter(item => item.status === 'closed')
                    .forEach(item => {
                        const profit = Number(item.actual_profit_usdt || 0);
                        const invested = Number(item.usdt_amount || 0);
                        const profitPercent = invested > 0 ? (profit / invested) * 100 : 0;
                        pushNotification({
                            type: 'investment',
                            title: `Khoản đầu tư ${item.symbol || ''} đã hoàn tất`,
                            message: `${profit >= 0 ? 'Lợi nhuận' : 'Thua lỗ'} ${Math.abs(profit).toFixed(2)} USDT (${profitPercent >= 0 ? '+' : ''}${profitPercent.toFixed(2)}%).`,
                            dedupeKey: `investment-closed-${item.id}`,
                        });
                    });
            } catch (error) {
                console.warn('[NOTIFICATION] Không thể đồng bộ kết quả đầu tư:', error);
            }
        };

        syncClosedInvestments();
        const timer = window.setInterval(syncClosedInvestments, 20000);
        return () => {
            cancelled = true;
            window.clearInterval(timer);
        };
    }, [authFetch, user?.id]);

    // Keyboard shortcut Ctrl+K to open search
    useEffect(() => {
        const handleKey = (e) => {
            if (currentPage === 'trading' && (e.ctrlKey || e.metaKey) && e.key === 'k') {
                e.preventDefault();
                setSearchOpen(true);
                setTimeout(() => inputRef.current?.focus(), 100);
            }
            if (e.key === 'Escape') setSearchOpen(false);
        };
        window.addEventListener('keydown', handleKey);
        return () => window.removeEventListener('keydown', handleKey);
    }, [currentPage]);

    useEffect(() => {
        if (currentPage !== 'trading') {
            setSearchOpen(false);
            setSearchQuery('');
        }
    }, [currentPage]);

    // Close search on click outside
    useEffect(() => {
        const handleClick = (e) => {
            if (searchRef.current && !searchRef.current.contains(e.target)) {
                setSearchOpen(false);
            }
            if (notificationRef.current && !notificationRef.current.contains(e.target)) {
                setNotificationOpen(false);
            }
        };
        document.addEventListener('mousedown', handleClick);
        return () => document.removeEventListener('mousedown', handleClick);
    }, []);

    const filteredSymbols = SYMBOLS.filter(s =>
        s.symbol.includes(searchQuery.toUpperCase()) ||
        s.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
        s.short.includes(searchQuery.toUpperCase())
    );

    const handleSelect = (symbol) => {
        setSymbol(symbol);
        setSearchOpen(false);
        setSearchQuery('');
    };

    const currentCoin = SYMBOLS.find(s => s.symbol === currentSymbol);
    const unreadCount = notifications.filter(item => !item.read).length;

    const openNotifications = () => {
        const nextOpen = !notificationOpen;
        setNotificationOpen(nextOpen);
        if (nextOpen && unreadCount) setNotifications(markAllNotificationsRead());
    };

    const relativeTime = (date) => {
        const minutes = Math.max(0, Math.floor((Date.now() - new Date(date).getTime()) / 60000));
        if (minutes < 1) return 'Vừa xong';
        if (minutes < 60) return `${minutes} phút trước`;
        if (minutes < 1440) return `${Math.floor(minutes / 60)} giờ trước`;
        return `${Math.floor(minutes / 1440)} ngày trước`;
    };

    const notificationIcon = (type) => {
        if (type === 'investment') return <CircleDollarSign size={16} />;
        if (type === 'safe-alert') return <ShieldAlert size={16} />;
        return <KeyRound size={16} />;
    };

    return (
        <div className="navbar">
            {/* Left: Brand + Tabs */}
            <div className="nav-left">
                <div className="brand" onClick={() => onNavigate('trading')}>
                    <Activity className="brand-icon" size={22} />
                    <span>Aegis</span>
                </div>

                <div className="nav-separator" />

                <div className="nav-tabs">
                    <button
                        className={`nav-tab ${currentPage === 'trading' ? 'active' : ''}`}
                        onClick={() => onNavigate('trading')}
                    >
                        <BarChart3 size={15} />
                        <span>Giao Dịch</span>
                    </button>
                    <button
                        className={`nav-tab ${currentPage === 'investment' ? 'active' : ''}`}
                        onClick={() => onNavigate('investment')}
                    >
                        <TrendingUp size={15} />
                        <span>Đầu Tư</span>
                    </button>
                    <button
                        className={`nav-tab ${currentPage === 'backtesting' ? 'active' : ''}`}
                        onClick={() => onNavigate('backtesting')}
                    >
                        <History size={15} />
                        <span>Backtest</span>
                    </button>
                    <button
                        className={`nav-tab ${currentPage === 'news' ? 'active' : ''}`}
                        onClick={() => onNavigate('news')}
                    >
                        <Newspaper size={15} />
                        <span>Tin tức</span>
                    </button>
                </div>
            </div>

            {/* Center: Search */}
            {currentPage === 'trading' && <div className="nav-search-area" ref={searchRef}>
                <button
                    className="nav-search-trigger"
                    onClick={() => {
                        setSearchOpen(true);
                        setTimeout(() => inputRef.current?.focus(), 100);
                    }}
                >
                    <CoinIcon coin={currentCoin} />
                    <span className="nav-selected-market">
                        <strong>{currentCoin?.short || 'Tìm coin'}</strong>
                        <small>/ USDT</small>
                    </span>
                    <kbd className="nav-search-kbd">Ctrl+K</kbd>
                    <ChevronDown size={13} className={`nav-search-chevron ${searchOpen ? 'open' : ''}`} />
                </button>

                {searchOpen && (
                    <div className="nav-search-dropdown">
                        <div className="nav-search-input-wrap">
                            <Search size={14} className="nav-search-icon" />
                            <input
                                ref={inputRef}
                                type="text"
                                value={searchQuery}
                                onChange={(e) => setSearchQuery(e.target.value)}
                                placeholder="Tìm kiếm coin..."
                                className="nav-search-input"
                                autoFocus
                            />
                            <button className="nav-search-close" onClick={() => setSearchOpen(false)}>
                                <X size={14} />
                            </button>
                        </div>
                        <div className="nav-search-list">
                            {filteredSymbols.map(s => (
                                <button
                                    key={s.symbol}
                                    className={`nav-search-item ${currentSymbol === s.symbol ? 'active' : ''}`}
                                    onClick={() => handleSelect(s.symbol)}
                                >
                                    <CoinIcon coin={s} />
                                    <div className="nav-search-item-info">
                                        <span className="nav-search-item-short">{s.short}<small>/USDT</small></span>
                                        <span className="nav-search-item-name">{s.name}</span>
                                    </div>
                                    {currentSymbol === s.symbol && <CheckCircle size={14} className="nav-search-selected-check" />}
                                </button>
                            ))}
                            {filteredSymbols.length === 0 && (
                                <div className="nav-search-empty">Không tìm thấy coin phù hợp</div>
                            )}
                        </div>
                    </div>
                )}
            </div>}

            {/* Right: User info + actions */}
            <div className="nav-right">
                <button
                    className="nav-icon-btn"
                    onClick={toggleTheme}
                    title={isDark ? 'Chế độ sáng' : 'Chế độ tối'}
                >
                    {isDark ? <Sun size={16} /> : <Moon size={16} />}
                </button>

                <div className="notification-center" ref={notificationRef}>
                    <button className={`nav-icon-btn notification-bell ${notificationOpen ? 'active' : ''}`}
                        onClick={openNotifications} title="Thông báo" aria-label="Thông báo">
                        <Bell size={16} />
                        {unreadCount > 0 && (
                            <span className="notification-badge">{unreadCount > 9 ? '9+' : unreadCount}</span>
                        )}
                    </button>

                    {notificationOpen && (
                        <div className="notification-dropdown">
                            <div className="notification-header">
                                <div>
                                    <strong>Thông báo</strong>
                                    <span>{notifications.length ? `${notifications.length} cập nhật gần đây` : 'Chưa có cập nhật'}</span>
                                </div>
                                {notifications.length > 0 && (
                                    <button onClick={() => { clearNotifications(); setNotifications([]); }}
                                        title="Xóa tất cả"><Trash2 size={14} /></button>
                                )}
                            </div>
                            <div className="notification-list">
                                {notifications.length === 0 ? (
                                    <div className="notification-empty">
                                        <Bell size={24} />
                                        <strong>Bạn đã xem hết</strong>
                                        <span>Các cập nhật mới sẽ xuất hiện tại đây.</span>
                                    </div>
                                ) : notifications.map(item => (
                                    <div className={`notification-item type-${item.type}`} key={item.id}>
                                        <div className="notification-type-icon">{notificationIcon(item.type)}</div>
                                        <div className="notification-copy">
                                            <strong>{item.title}</strong>
                                            <p>{item.message}</p>
                                            <span>{relativeTime(item.createdAt)}</span>
                                        </div>
                                    </div>
                                ))}
                            </div>
                        </div>
                    )}
                </div>

                <button
                    className="nav-icon-btn"
                    onClick={() => window.dispatchEvent(new CustomEvent('showSettings'))}
                    title="Cài đặt"
                >
                    <Settings size={16} />
                </button>

                <button className="nav-icon-btn logout" onClick={logout} title="Đăng xuất">
                    <LogOut size={16} />
                </button>
            </div>
        </div>
    );
}

import React, { useState, useEffect, useRef } from 'react';
import useStore from '../store';
import { useToast } from './ToastProvider';
import { useTheme } from './ThemeProvider';
import {
    Activity, LogOut, CheckCircle, BarChart3, TrendingUp,
    Sparkles, Sun, Moon, History, Search, X, Settings
} from 'lucide-react';

const SYMBOLS = [
    { symbol: 'BTCUSDT', name: 'Bitcoin', short: 'BTC' },
    { symbol: 'ETHUSDT', name: 'Ethereum', short: 'ETH' },
    { symbol: 'BNBUSDT', name: 'BNB', short: 'BNB' },
    { symbol: 'SOLUSDT', name: 'Solana', short: 'SOL' },
    { symbol: 'XRPUSDT', name: 'Ripple', short: 'XRP' },
    { symbol: 'DOGEUSDT', name: 'Dogecoin', short: 'DOGE' },
    { symbol: 'ADAUSDT', name: 'Cardano', short: 'ADA' },
    { symbol: 'AVAXUSDT', name: 'Avalanche', short: 'AVAX' },
    { symbol: 'DOTUSDT', name: 'Polkadot', short: 'DOT' },
    { symbol: 'POLUSDT', name: 'Polygon', short: 'POL' },
];

export default function Navbar({ currentPage, onNavigate }) {
    const { logout, isVip, currentSymbol, setSymbol } = useStore();
    const { showToast } = useToast();
    const { isDark, toggleTheme } = useTheme();
    const [searchOpen, setSearchOpen] = useState(false);
    const [searchQuery, setSearchQuery] = useState('');
    const searchRef = useRef(null);
    const inputRef = useRef(null);

    // Listen for VIP upgrade event
    useEffect(() => {
        const handleUpgrade = () => {
            showToast('Chúc mừng! Tài khoản đã được nâng cấp VIP!', 'success');
        };
        window.addEventListener('vip_upgraded', handleUpgrade);
        return () => window.removeEventListener('vip_upgraded', handleUpgrade);
    }, [showToast]);

    // Keyboard shortcut Ctrl+K to open search
    useEffect(() => {
        const handleKey = (e) => {
            if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
                e.preventDefault();
                setSearchOpen(true);
                setTimeout(() => inputRef.current?.focus(), 100);
            }
            if (e.key === 'Escape') setSearchOpen(false);
        };
        window.addEventListener('keydown', handleKey);
        return () => window.removeEventListener('keydown', handleKey);
    }, []);

    // Close search on click outside
    useEffect(() => {
        const handleClick = (e) => {
            if (searchRef.current && !searchRef.current.contains(e.target)) {
                setSearchOpen(false);
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

    return (
        <div className="navbar">
            {/* Left: Brand + Tabs */}
            <div className="nav-left">
                <div className="brand" onClick={() => onNavigate('trading')}>
                    <Activity className="brand-icon" size={22} />
                    <span>TradeAI</span>
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
                        {!isVip && <span className="vip-badge-mini">VIP</span>}
                    </button>
                    <button
                        className={`nav-tab ${currentPage === 'backtesting' ? 'active' : ''}`}
                        onClick={() => onNavigate('backtesting')}
                    >
                        <History size={15} />
                        <span>Backtest</span>
                        {!isVip && <span className="vip-badge-mini">VIP</span>}
                    </button>
                </div>
            </div>

            {/* Center: Search */}
            <div className="nav-search-area" ref={searchRef}>
                <button
                    className="nav-search-trigger"
                    onClick={() => {
                        setSearchOpen(true);
                        setTimeout(() => inputRef.current?.focus(), 100);
                    }}
                >
                    <Search size={14} />
                    <span>{currentCoin ? `${currentCoin.short}/USDT` : 'Tìm coin...'}</span>
                    <kbd className="nav-search-kbd">Ctrl+K</kbd>
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
                                    <div className="nav-search-item-info">
                                        <span className="nav-search-item-short">{s.short}</span>
                                        <span className="nav-search-item-name">{s.name}</span>
                                    </div>
                                    <span className="nav-search-item-pair">/ USDT</span>
                                </button>
                            ))}
                            {filteredSymbols.length === 0 && (
                                <div className="nav-search-empty">Không tìm thấy coin phù hợp</div>
                            )}
                        </div>
                    </div>
                )}
            </div>

            {/* Right: User info + actions */}
            <div className="nav-right">
                <div className={`user-badge ${isVip ? 'vip-gold' : 'basic-gray'}`}>
                    {isVip && <Sparkles size={11} style={{ marginRight: 3 }} />}
                    {isVip ? 'VIP' : 'FREE'}
                    {isVip && <CheckCircle size={10} />}
                </div>

                <button
                    className="nav-icon-btn"
                    onClick={toggleTheme}
                    title={isDark ? 'Chế độ sáng' : 'Chế độ tối'}
                >
                    {isDark ? <Sun size={16} /> : <Moon size={16} />}
                </button>

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

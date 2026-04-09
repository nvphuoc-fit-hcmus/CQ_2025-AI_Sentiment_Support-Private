import React, { useState, useEffect } from 'react';
import useStore from '../store';
import { useRefreshVIPStatus } from '../hooks/useRefreshVIPStatus';
import NewsList from './NewsList';
import SAFEAlertPanel from './SAFEAlertPanel';  // FIXED: Import SAFEAlertPanel
import Watchlist from './Watchlist';
import { Newspaper, BrainCircuit, Lock, List } from 'lucide-react';


export default function Sidebar() {
    const { isVip } = useStore();
    const [activeTab, setActiveTab] = useState('watchlist');

    // Refresh VIP status on mount
    useRefreshVIPStatus();

    const triggerUpgrade = () => {
        window.dispatchEvent(new CustomEvent('showUpgradeModal'));
    };

    return (
        <div className="sidebar-container">
            {/* Tabs */}
            <div className="tabs">
                <button
                    onClick={() => setActiveTab('watchlist')}
                    className={`tab-btn ${activeTab === 'watchlist' ? 'active' : ''}`}
                    title="Danh sách theo dõi"
                >
                    <List size={16} />
                </button>
                <button
                    onClick={() => setActiveTab('news')}
                    className={`tab-btn ${activeTab === 'news' ? 'active' : ''}`}
                    title="Tin tức"
                >
                    <Newspaper size={16} />
                </button>
                <button
                    onClick={() => setActiveTab('insights')}
                    className={`tab-btn vip ${activeTab === 'insights' ? 'active' : ''}`}
                    title="Phân tích AI"
                >
                    <BrainCircuit size={16} />
                    {!isVip && <span className="vip-badge-mini" style={{ marginLeft: 4 }}>VIP</span>}
                </button>
            </div>

            {/* Content */}
            <div className="sidebar-content">
                {activeTab === 'watchlist' && <Watchlist />}
                {activeTab === 'news' && <NewsList />}
                {activeTab === 'insights' && (
                    <SAFEAlertPanel symbol={'BTCUSDT'} />
                )}
            </div>
        </div>
    );
}

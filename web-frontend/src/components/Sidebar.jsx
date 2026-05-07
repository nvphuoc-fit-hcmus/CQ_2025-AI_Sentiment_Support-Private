import React, { useState } from 'react';
import useStore from '../store';
import { useRefreshVIPStatus } from '../hooks/useRefreshVIPStatus';
import NewsList from './NewsList';
import DecisionSidebar from './DecisionSidebar';
import Watchlist from './Watchlist';
import { BrainCircuit, List, Newspaper, ChevronLeft, ChevronRight } from 'lucide-react';

export default function Sidebar() {
    const { isVip, currentSymbol } = useStore();
    const [activeTab, setActiveTab] = useState('decision');
    const [collapsed, setCollapsed] = useState(false);

    useRefreshVIPStatus();

    const tabs = [
        { id: 'decision', icon: BrainCircuit, title: 'Phân tích AI' },
        { id: 'market', icon: List, title: 'Thị trường' },
        { id: 'news', icon: Newspaper, title: 'Tin tức' },
    ];

    return (
        <div className={`sidebar-container ${collapsed ? 'collapsed' : ''}`}>
            {/* Collapse toggle */}
            <button
                className="sidebar-collapse-btn"
                onClick={() => setCollapsed(!collapsed)}
                title={collapsed ? 'Mở sidebar' : 'Đóng sidebar'}
            >
                {collapsed ? <ChevronLeft size={14} /> : <ChevronRight size={14} />}
            </button>

            {!collapsed && (
                <>
                    {/* Tabs */}
                    <div className="tabs">
                        {tabs.map(tab => (
                            <button
                                key={tab.id}
                                onClick={() => setActiveTab(tab.id)}
                                className={`tab-btn ${activeTab === tab.id ? 'active' : ''}`}
                                title={tab.title}
                            >
                                <tab.icon size={15} />
                            </button>
                        ))}
                    </div>

                    {/* Content */}
                    <div className="sidebar-content">
                        {activeTab === 'decision' && (
                            <DecisionSidebar symbol={currentSymbol || 'BTCUSDT'} />
                        )}
                        {activeTab === 'market' && <Watchlist />}
                        {activeTab === 'news' && <NewsList />}
                    </div>
                </>
            )}
        </div>
    );
}

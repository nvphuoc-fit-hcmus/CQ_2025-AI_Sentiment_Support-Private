import React from 'react';
import { MousePointer2, TrendingUp, Brush, Type, Grid, Smile, Ruler, Database } from 'lucide-react';

export default function LeftToolbar({ currentPage, onNavigate }) {
    const [activeTool, setActiveTool] = React.useState('crosshair');

    const handleNavigate = (page, tool) => {
        setActiveTool(tool);
        if (typeof onNavigate === 'function') {
            onNavigate(page);
        }
    };

    return (
        <div className="left-toolbar">
            <button
                className={`toolbar-btn ${currentPage === 'trading' || activeTool === 'crosshair' ? 'active' : ''}`}
                title="Giao Dịch"
                onClick={() => handleNavigate('trading', 'crosshair')}
            >
                <MousePointer2 size={18} />
            </button>
            <button
                className={`toolbar-btn ${currentPage === 'backtesting' || activeTool === 'trend' ? 'active' : ''}`}
                title="Backtest"
                onClick={() => handleNavigate('backtesting', 'trend')}
            >
                <TrendingUp size={18} />
            </button>
            <button
                className={`toolbar-btn ${activeTool === 'brush' ? 'active' : ''}`}
                title="Brush"
                onClick={() => setActiveTool('brush')}
            >
                <Brush size={18} />
            </button>
            <button
                className={`toolbar-btn ${activeTool === 'text' ? 'active' : ''}`}
                title="Text"
                onClick={() => setActiveTool('text')}
            >
                <Type size={18} />
            </button>
            <button
                className={`toolbar-btn ${activeTool === 'patterns' ? 'active' : ''}`}
                title="Patterns"
                onClick={() => setActiveTool('patterns')}
            >
                <Grid size={18} />
            </button>
            <button
                className={`toolbar-btn ${currentPage === 'investment' || activeTool === 'prediction' ? 'active' : ''}`}
                title="Đầu Tư"
                onClick={() => handleNavigate('investment', 'prediction')}
            >
                <Database size={18} />
            </button>
            <button
                className={`toolbar-btn ${activeTool === 'icons' ? 'active' : ''}`}
                title="Icons"
                onClick={() => setActiveTool('icons')}
            >
                <Smile size={18} />
            </button>
            <button
                className={`toolbar-btn ${activeTool === 'measure' ? 'active' : ''}`}
                title="Measure"
                onClick={() => setActiveTool('measure')}
            >
                <Ruler size={18} />
            </button>
        </div>
    );
}

import React from 'react';
import { MousePointer2, TrendingUp, Minus, Type, Grid, Ruler } from 'lucide-react';

export default function LeftToolbar({ activeTool = 'crosshair', onToolChange }) {
    const tools = [
        { id: 'crosshair', icon: MousePointer2, title: 'Con trỏ' },
        { id: 'trend', icon: TrendingUp, title: 'Đường xu hướng' },
        { id: 'hline', icon: Minus, title: 'Đường ngang' },
        { id: 'text', icon: Type, title: 'Chú thích' },
        { id: 'patterns', icon: Grid, title: 'Mẫu hình' },
        { id: 'measure', icon: Ruler, title: 'Đo lường' },
    ];

    return (
        <div className="left-toolbar">
            {tools.map((tool, index) => (
                <React.Fragment key={tool.id}>
                    {index === 3 && <div className="toolbar-separator" />}
                    <button
                        className={`toolbar-btn ${activeTool === tool.id ? 'active' : ''}`}
                        title={tool.title}
                        onClick={() => onToolChange?.(tool.id)}
                        aria-label={tool.title}
                        aria-pressed={activeTool === tool.id}
                    >
                        <tool.icon size={16} />
                    </button>
                </React.Fragment>
            ))}
        </div>
    );
}

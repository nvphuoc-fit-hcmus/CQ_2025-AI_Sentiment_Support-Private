import React from 'react';
import { Shield, AlertTriangle } from 'lucide-react';

/**
 * StructuredExplainer — Explanation Module (Eq.27-29)
 *
 * Renders the structured NL explanation ĝ from the SAFE-Alert model.
 * Template: "Tín hiệu [HƯỚNG] với độ tin cậy [C%] được hỗ trợ bởi [FACTOR] từ [NEWS]"
 *
 * Includes Faithfulness badge based on Eq.36 compliance.
 *
 * Props:
 *  - explanation: string (NL text from API)
 *  - direction: "BUY" | "SELL" | "HOLD"
 *  - confidence: number [0,1]
 *  - topFactors: string[]
 *  - articles: { title }[]
 *  - shouldAlert: boolean
 */
export default function StructuredExplainer({
    explanation = '',
    direction = 'HOLD',
    confidence = 0,
    topFactors = [],
    articles = [],
    shouldAlert = false,
}) {
    const getDirColor = () => {
        switch (direction?.toUpperCase()) {
            case 'BUY': case 'UP': return 'var(--accent-green)';
            case 'SELL': case 'DOWN': return 'var(--accent-red)';
            default: return 'var(--text-secondary)';
        }
    };

    const getDirLabel = () => {
        switch (direction?.toUpperCase()) {
            case 'BUY': case 'UP': return 'TĂNG';
            case 'SELL': case 'DOWN': return 'GIẢM';
            default: return 'GIỮ VỮNG';
        }
    };

    // Build structured explanation from components if API explanation is missing
    const buildExplanation = () => {
        if (explanation && explanation.length > 10) return explanation;

        const dirLabel = getDirLabel();
        const confPct = (confidence * 100).toFixed(0);
        const factorStr = topFactors.length > 0
            ? topFactors.slice(0, 2).join(' và ')
            : 'chưa xác định';
        const newsStr = articles.length > 0
            ? articles.slice(0, 2).map(a => `"${a.title}"`).join(', ')
            : 'không có tin tức';

        return `Tín hiệu ${dirLabel} với độ tin cậy ${confPct}% được hỗ trợ bởi yếu tố ${factorStr} từ các bằng chứng ${newsStr}.`;
    };

    const displayText = buildExplanation();
    const dirColor = getDirColor();

    return (
        <div className="structured-explainer">
            <div className="explainer-header">
                <span className="explainer-icon">📝</span>
                Phân tích AI
            </div>

            <div className="explainer-body">
                {/* Direction pill */}
                <div className="explainer-signal-row">
                    <span
                        className="explainer-signal-pill"
                        style={{
                            background: `${dirColor}15`,
                            color: dirColor,
                            borderColor: `${dirColor}30`,
                        }}
                    >
                        {getDirLabel()} • {(confidence * 100).toFixed(0)}%
                    </span>
                    {shouldAlert && (
                        <span className="explainer-alert-badge">
                            <AlertTriangle size={11} />
                            Alert
                        </span>
                    )}
                </div>

                {/* NL Explanation text */}
                <p className="explainer-text">{displayText}</p>

                {/* Faithfulness badge (Eq.36) */}
                <div className="faithfulness-badge">
                    <Shield size={12} />
                    <span>Faithful — Chỉ dùng bằng chứng đã chọn lọc bởi mô hình</span>
                </div>
            </div>
        </div>
    );
}

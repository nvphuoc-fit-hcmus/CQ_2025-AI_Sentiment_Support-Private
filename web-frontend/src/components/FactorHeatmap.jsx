import React from 'react';
import { FACTOR_LABELS } from '../services/safeAlertService';

/**
 * FactorHeatmap — Factor-grounded Reasoning (Eq.14-15)
 *
 * Displays top_factors as horizontal bars with Vietnamese labels + English subtitles.
 * Bar width proportional to factor rank (1st=100%, 2nd=75%, 3rd=55%, etc.).
 * Click interaction: filters evidence tab (future).
 *
 * Props:
 *  - topFactors: string[] (e.g. ["institutional_inflow", "etf_flow", "regulatory_easing"])
 *  - direction: "BUY" | "SELL" | "HOLD" (for bar color)
 *  - onFactorClick: (factorName) => void
 */
export default function FactorHeatmap({ topFactors = [], direction = 'HOLD', onFactorClick }) {
    if (!topFactors || topFactors.length === 0) {
        return (
            <div className="factor-heatmap empty">
                <div className="factor-heatmap-title">
                    <span className="factor-icon">◆</span>
                    Yếu tố chi phối
                </div>
                <div className="factor-empty-msg">Chưa có dữ liệu yếu tố</div>
            </div>
        );
    }

    // Bar width percentages for ranked factors
    const barWidths = [100, 75, 55, 40, 30, 25, 20, 18, 15, 12];

    // Color based on direction
    const getBarColor = () => {
        switch (direction?.toUpperCase()) {
            case 'BUY': case 'UP': return 'var(--accent-green)';
            case 'SELL': case 'DOWN': return 'var(--accent-red)';
            default: return 'var(--accent-blue)';
        }
    };

    const barColor = getBarColor();

    return (
        <div className="factor-heatmap">
            <div className="factor-heatmap-title">
                <span className="factor-icon">◆</span>
                Yếu tố chi phối
                <span className="factor-count">{topFactors.length}</span>
            </div>
            <div className="factor-list">
                {topFactors.slice(0, 5).map((factor, idx) => {
                    const label = FACTOR_LABELS[factor] || { vi: factor, en: factor };
                    const width = barWidths[idx] || 10;

                    return (
                        <div
                            key={factor}
                            className="factor-item"
                            onClick={() => onFactorClick?.(factor)}
                            role="button"
                            tabIndex={0}
                        >
                            <div className="factor-label">
                                <span className="factor-name-vi">{label.vi}</span>
                                <span className="factor-name-en">{label.en}</span>
                            </div>
                            <div className="factor-bar-track">
                                <div
                                    className="factor-bar-fill"
                                    style={{
                                        width: `${width}%`,
                                        background: `linear-gradient(90deg, ${barColor}, transparent)`,
                                        opacity: 1 - idx * 0.15,
                                    }}
                                />
                            </div>
                            <div className="factor-rank">#{idx + 1}</div>
                        </div>
                    );
                })}
            </div>
        </div>
    );
}

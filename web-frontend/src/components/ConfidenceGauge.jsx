import React from 'react';

/**
 * ConfidenceGauge — SVG Radial Arc (Eq.26 Alert Gating)
 *
 * Displays confidence ĉ ∈ [0,1] as a 270° arc gauge.
 * Colors shift: Gray (Abstain) → Yellow (Uncertain) → Green/Red (Confident).
 * Pulse animation when should_alert === true.
 *
 * Props:
 *  - confidence: number [0,1]
 *  - direction: "BUY" | "SELL" | "HOLD"
 *  - shouldAlert: boolean (Eq.26 gating result)
 */
export default function ConfidenceGauge({ confidence = 0, direction = 'HOLD', shouldAlert = false }) {
    const size = 160;
    const strokeWidth = 12;
    const radius = (size - strokeWidth) / 2;
    const center = size / 2;

    // Arc: 270 degrees (from 135° to 405°)
    const startAngle = 135;
    const maxSweep = 270;
    const sweep = maxSweep * Math.min(Math.max(confidence, 0), 1);

    // SVG arc path
    const polarToCartesian = (cx, cy, r, angleDeg) => {
        const rad = ((angleDeg - 90) * Math.PI) / 180;
        return { x: cx + r * Math.cos(rad), y: cy + r * Math.sin(rad) };
    };

    const describeArc = (cx, cy, r, start, end) => {
        const s = polarToCartesian(cx, cy, r, start);
        const e = polarToCartesian(cx, cy, r, end);
        const largeArc = end - start > 180 ? 1 : 0;
        return `M ${s.x} ${s.y} A ${r} ${r} 0 ${largeArc} 1 ${e.x} ${e.y}`;
    };

    const bgPath = describeArc(center, center, radius, startAngle, startAngle + maxSweep);
    const fgPath = sweep > 0.5
        ? describeArc(center, center, radius, startAngle, startAngle + sweep)
        : '';

    // Color logic: Gray (Abstain) → Yellow (Uncertain) → Direction-aware
    const getColor = () => {
        if (!shouldAlert || confidence < 0.4) return 'var(--text-tertiary)'; // Gray - Abstain
        if (confidence < 0.6) return 'var(--accent-yellow)'; // Yellow - Uncertain

        // High confidence: direction-aware
        if (direction === 'BUY' || direction === 'UP') return 'var(--accent-green)';
        if (direction === 'SELL' || direction === 'DOWN') return 'var(--accent-red)';
        return 'var(--accent-blue)'; // HOLD/NEUTRAL
    };

    const getDirectionLabel = () => {
        switch (direction?.toUpperCase()) {
            case 'BUY': case 'UP': return { text: 'TĂNG', icon: '▲' };
            case 'SELL': case 'DOWN': return { text: 'GIẢM', icon: '▼' };
            default: return { text: 'GIỮ', icon: '─' };
        }
    };

    const color = getColor();
    const { text: dirText, icon: dirIcon } = getDirectionLabel();
    const pct = (confidence * 100).toFixed(0);

    return (
        <div className={`confidence-gauge ${shouldAlert ? 'pulse-active' : ''}`}>
            <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`}>
                {/* Background arc (track) */}
                <path
                    d={bgPath}
                    fill="none"
                    stroke="var(--border-color)"
                    strokeWidth={strokeWidth}
                    strokeLinecap="round"
                />
                {/* Foreground arc (value) */}
                {fgPath && (
                    <path
                        d={fgPath}
                        fill="none"
                        stroke={color}
                        strokeWidth={strokeWidth}
                        strokeLinecap="round"
                        className="gauge-arc"
                        style={{
                            filter: shouldAlert ? `drop-shadow(0 0 6px ${color})` : 'none',
                        }}
                    />
                )}

                {/* Outer pulse ring (when alerting) */}
                {shouldAlert && (
                    <circle
                        cx={center}
                        cy={center}
                        r={radius + strokeWidth / 2 + 4}
                        fill="none"
                        stroke={color}
                        strokeWidth={1.5}
                        opacity={0.4}
                        className="gauge-pulse-ring"
                    />
                )}

                {/* Center content */}
                <text
                    x={center}
                    y={center - 12}
                    textAnchor="middle"
                    className="gauge-percentage"
                    fill={color}
                >
                    {pct}%
                </text>
                <text
                    x={center}
                    y={center + 10}
                    textAnchor="middle"
                    className="gauge-direction"
                    fill="var(--text-primary)"
                >
                    {dirIcon} {dirText}
                </text>
                <text
                    x={center}
                    y={center + 28}
                    textAnchor="middle"
                    className="gauge-sublabel"
                    fill="var(--text-secondary)"
                >
                    {shouldAlert ? 'Tín hiệu tin cậy' : 'Chờ tín hiệu'}
                </text>
            </svg>

            {/* Horizon badges */}
            <div className="gauge-horizon-tag">1H</div>
        </div>
    );
}

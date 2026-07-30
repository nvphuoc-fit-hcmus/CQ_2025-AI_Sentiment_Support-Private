import React, { useEffect, useMemo, useState } from 'react';
import { AlertTriangle, BrainCircuit, CheckCircle2 } from 'lucide-react';
import safeAlertService from '../services/safeAlertService';

const FACTOR_LABELS_VI = {
    institutional_inflow: 'Dòng vốn tổ chức',
    etf_flow: 'Dòng vốn ETF',
    regulatory_easing: 'Nới lỏng quy định',
    regulatory_tightening: 'Siết chặt quy định',
    exchange_risk: 'Rủi ro sàn giao dịch',
    liquidity_squeeze: 'Áp lực thanh khoản',
    whale_accumulation: 'Cá voi tích lũy',
    macro_uncertainty: 'Bất ổn vĩ mô',
    protocol_upgrade: 'Nâng cấp giao thức',
    network_outage: 'Sự cố mạng lưới',
};

export default function StructuredExplainer({
    symbol = 'BTCUSDT',
    horizon = '1h',
    direction = 'HOLD',
    confidence = 0,
    probabilities = {},
    topFactors = [],
    articles = [],
    shouldAlert = false,
}) {
    const normalizedDirection = direction?.toUpperCase();
    const confidencePercent = Math.round(confidence * 100);

    const directionInfo = (() => {
        if (normalizedDirection === 'BUY' || normalizedDirection === 'UP') {
            return {
                label: 'TĂNG',
                action: 'Thiên hướng tăng',
                detail: 'Ưu tiên quan sát cơ hội tăng giá',
                color: 'var(--accent-green)',
            };
        }
        if (normalizedDirection === 'SELL' || normalizedDirection === 'DOWN') {
            return {
                label: 'GIẢM',
                action: 'Thiên hướng giảm',
                detail: 'Thận trọng với rủi ro giảm giá',
                color: 'var(--accent-red)',
            };
        }
        return {
            label: 'GIỮ VỮNG',
            action: 'Xu hướng chưa rõ',
            detail: 'Thị trường chưa hình thành xu hướng rõ ràng',
            color: 'var(--text-secondary)',
        };
    })();

    const factorLabels = topFactors
        .slice(0, 3)
        .map((factor) => FACTOR_LABELS_VI[factor] || factor.replaceAll('_', ' '));
    const primaryFactor = factorLabels[0] || 'Chưa xác định';

    const fallbackText = useMemo(() => {
        const evidence = articles.length > 0
            ? `Mô hình đã chọn ${articles.length} bài báo có mức liên quan cao để làm cơ sở cho dự báo.`
            : 'Hiện chưa có bài báo phù hợp để củng cố dự báo.';
        const factors = factorLabels.length > 0
            ? `Các yếu tố đáng chú ý gồm ${factorLabels.join(', ')}.`
            : 'Chưa xác định được yếu tố chi phối đủ rõ ràng.';
        return `${directionInfo.detail} trong khung ${horizon.toUpperCase()}, với độ tin cậy ${confidencePercent}%. ${evidence} ${factors} Kết quả này chỉ mang tính tham khảo; nhà đầu tư nên tiếp tục theo dõi biến động giá, khối lượng và các thông tin mới trước khi đưa ra quyết định.`;
    }, [articles.length, confidencePercent, directionInfo.detail, factorLabels, horizon]);

    const [generatedText, setGeneratedText] = useState('');
    const [generating, setGenerating] = useState(false);

    useEffect(() => {
        let cancelled = false;
        setGeneratedText('');
        setGenerating(true);
        safeAlertService.rewriteExplanation({
            symbol,
            horizon,
            direction,
            confidence,
            probabilities,
            factors: topFactors,
            article_titles: articles.map((article) => article.title).filter(Boolean),
        })
            .then((result) => {
                if (!cancelled && result?.text) setGeneratedText(result.text);
            })
            .catch(() => {
                if (!cancelled) setGeneratedText('');
            })
            .finally(() => {
                if (!cancelled) setGenerating(false);
            });
        return () => { cancelled = true; };
    }, [symbol, horizon, direction, confidence, probabilities, topFactors, articles]);

    return (
        <div className="structured-explainer">
            <div className="explainer-header">
                <span className="explainer-title">
                    <BrainCircuit size={15} />
                    Góc nhìn thị trường
                </span>
                {shouldAlert && (
                    <span className="explainer-alert-badge">
                        <AlertTriangle size={11} />
                        Cần chú ý
                    </span>
                )}
            </div>

            <div className="explainer-market-brief">
                <div className="explainer-verdict">
                    <div className="explainer-verdict-main">
                        <span className="explainer-kicker">Kết luận cho khung {horizon.toUpperCase()}</span>
                        <strong style={{ color: directionInfo.color }}>{directionInfo.action}</strong>
                    </div>
                    <span
                        className="explainer-signal-pill"
                        style={{
                            background: `color-mix(in srgb, ${directionInfo.color} 12%, transparent)`,
                            color: directionInfo.color,
                            borderColor: `color-mix(in srgb, ${directionInfo.color} 30%, transparent)`,
                        }}
                    >
                        {directionInfo.label}
                    </span>
                </div>

                <div className="explainer-metrics-row">
                    <div className="explainer-metric">
                        <span>Độ tin cậy</span>
                        <strong>{confidencePercent}%</strong>
                    </div>
                    <div className="explainer-metric">
                        <span>Tin được chọn</span>
                        <strong>{articles.length}</strong>
                    </div>
                    <div className="explainer-metric primary">
                        <span>Yếu tố chính</span>
                        <strong>{primaryFactor}</strong>
                    </div>
                </div>

                <div className="explainer-narrative">
                    <span className="explainer-detail-label">
                        {generating ? 'Đang tạo nhận định chuyên sâu…' : 'Nhận định chuyên sâu'}
                    </span>
                    <p className="explainer-generated-text">{generatedText || fallbackText}</p>
                </div>

            </div>

            <div className="faithfulness-badge">
                <CheckCircle2 size={13} />
                <span>Đã đối chiếu với các tin tức do mô hình chọn lọc.</span>
            </div>
        </div>
    );
}

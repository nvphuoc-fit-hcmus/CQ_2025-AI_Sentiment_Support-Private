import React from 'react';
import { FACTOR_LABELS } from '../services/safeAlertService';
import { CircleHelp } from 'lucide-react';

export const FACTOR_DETAILS = {
    institutional_inflow: {
        description: 'Dòng tiền từ quỹ đầu tư, doanh nghiệp hoặc tổ chức tài chính lớn đi vào thị trường.',
        example: 'Ví dụ minh họa: một quỹ công bố mua thêm Bitcoin hoặc tăng tỷ trọng tài sản số.',
    },
    etf_flow: {
        description: 'Lượng vốn ròng đi vào hoặc rút khỏi các quỹ ETF tiền mã hóa.',
        example: 'Ví dụ minh họa: ETF Bitcoin ghi nhận dòng vốn vào ròng trong nhiều phiên liên tiếp.',
    },
    regulatory_easing: {
        description: 'Các thay đổi pháp lý theo hướng tạo điều kiện thuận lợi hơn cho thị trường tiền mã hóa.',
        example: 'Ví dụ minh họa: cơ quan quản lý phê duyệt sản phẩm đầu tư hoặc ban hành khung pháp lý rõ ràng hơn.',
    },
    regulatory_tightening: {
        description: 'Các biện pháp quản lý làm tăng hạn chế, chi phí tuân thủ hoặc rủi ro pháp lý.',
        example: 'Ví dụ minh họa: sàn giao dịch bị điều tra hoặc một quốc gia siết quy định giao dịch tài sản số.',
    },
    exchange_risk: {
        description: 'Rủi ro liên quan đến thanh khoản, bảo mật, dự trữ hoặc khả năng vận hành của sàn giao dịch.',
        example: 'Ví dụ minh họa: sàn tạm dừng rút tiền, bị tấn công hoặc xuất hiện nghi vấn thiếu hụt dự trữ.',
    },
    liquidity_squeeze: {
        description: 'Tình trạng thanh khoản suy giảm khiến giao dịch lớn dễ làm giá biến động mạnh.',
        example: 'Ví dụ minh họa: độ sâu sổ lệnh giảm và chênh lệch giá mua–bán tăng nhanh.',
    },
    whale_accumulation: {
        description: 'Hoạt động gia tăng nắm giữ của các ví lớn hoặc nhà đầu tư có sức ảnh hưởng.',
        example: 'Ví dụ minh họa: nhiều Bitcoin được rút khỏi sàn và chuyển về các ví lớn để nắm giữ.',
    },
    macro_uncertainty: {
        description: 'Sự bất định từ lãi suất, lạm phát, chính sách tiền tệ hoặc tăng trưởng kinh tế.',
        example: 'Ví dụ minh họa: thị trường chờ quyết định lãi suất của Fed và phản ứng mạnh với dữ liệu lạm phát.',
    },
    protocol_upgrade: {
        description: 'Thay đổi kỹ thuật quan trọng nhằm cải thiện hiệu năng, bảo mật hoặc chức năng của mạng lưới.',
        example: 'Ví dụ minh họa: blockchain triển khai bản nâng cấp giúp giảm phí hoặc tăng tốc độ xử lý.',
    },
    network_outage: {
        description: 'Sự cố khiến blockchain hoặc dịch vụ cốt lõi bị gián đoạn hay hoạt động không ổn định.',
        example: 'Ví dụ minh họa: mạng ngừng tạo khối trong một khoảng thời gian hoặc giao dịch bị đình trệ.',
    },
};

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
                    const detail = FACTOR_DETAILS[factor] || {
                        description: 'Yếu tố này có thể ảnh hưởng đến tâm lý và diễn biến của thị trường.',
                        example: 'Ví dụ minh họa phụ thuộc vào bối cảnh của từng bài báo được chọn.',
                    };
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
                            <CircleHelp size={12} className="factor-help-icon" aria-hidden="true" />
                            <div className="factor-tooltip" role="tooltip">
                                <div className="factor-tooltip-title">{label.vi}</div>
                                <p>{detail.description}</p>
                                <div className="factor-tooltip-example">{detail.example}</div>
                            </div>
                        </div>
                    );
                })}
            </div>
        </div>
    );
}

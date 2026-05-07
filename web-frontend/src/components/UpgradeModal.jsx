import React, { useState, useEffect } from 'react';
import { X, Copy, Loader, CheckCircle, XCircle, Clock, Crown, Zap, Brain, BarChart3 } from 'lucide-react';
import useStore from '../store';
import { useToast } from './ToastProvider';

const MY_BANK = {
    BANK_ID: "MBBank",
    ACCOUNT_NO: "0398103087",
    ACCOUNT_NAME: "HUỲNH MẪN",
    TEMPLATE: "compact",
};

const PAYMENT_TIMEOUT = 10 * 60 * 1000;
const POLL_INTERVAL = 3000;

const features = [
    { icon: Brain, text: 'SAFE-Alert AI dự báo' },
    { icon: BarChart3, text: 'Mô phỏng đầu tư' },
    { icon: Zap, text: 'Backtesting chiến lược' },
];

export default function UpgradeModal({ onClose }) {
    const { token, isVip, setIsVip, authFetch } = useStore();
    const { showToast } = useToast();
    const [step, setStep] = useState(1);
    const [userId, setUserId] = useState(null);
    const [loadingCode, setLoadingCode] = useState(true);
    const [paymentStatus, setPaymentStatus] = useState('idle');
    const [timeRemaining, setTimeRemaining] = useState(PAYMENT_TIMEOUT);

    useEffect(() => {
        if (!token) return;
        try {
            const payload = JSON.parse(atob(token.split('.')[1]));
            setUserId(payload.sub);
            setLoadingCode(false);
        } catch (e) {
            console.error("Failed to decode token", e);
            setLoadingCode(false);
        }
    }, [token]);

    useEffect(() => {
        if (paymentStatus === 'waiting' && timeRemaining > 0) {
            const timer = setTimeout(() => {
                setTimeRemaining(prev => prev - 1000);
            }, 1000);
            return () => clearTimeout(timer);
        } else if (paymentStatus === 'waiting' && timeRemaining <= 0) {
            setPaymentStatus('expired');
        }
    }, [paymentStatus, timeRemaining]);

    useEffect(() => {
        if (isVip && paymentStatus !== 'success') {
            setPaymentStatus('success');
            setTimeout(() => onClose(), 3000);
        }
    }, [isVip, paymentStatus, onClose]);

    const checkVipStatus = async () => {
        try {
            const res = await authFetch('/auth/me');
            if (res.ok) {
                const data = await res.json();
                if (data.user && data.user.is_vip) {
                    setIsVip(true);
                }
            }
        } catch (error) {
            console.error('Failed to check VIP status:', error);
        }
    };

    const copyToClipboard = (text) => {
        navigator.clipboard.writeText(text);
        showToast('Đã sao chép!', 'success');
    };

    const formatTime = (ms) => {
        const minutes = Math.floor(ms / 60000);
        const seconds = Math.floor((ms % 60000) / 1000);
        return `${minutes}:${seconds.toString().padStart(2, '0')}`;
    };

    return (
        <div className="modal-overlay" onClick={onClose}>
            <div className="upgrade-modal" onClick={e => e.stopPropagation()}>
                <button className="modal-close" onClick={onClose}>
                    <X size={18} />
                </button>

                {/* Step 1: Plan Selection */}
                {step === 1 && (
                    <div className="upgrade-step">
                        <div className="upgrade-crown">
                            <Crown size={36} />
                        </div>
                        <h2 className="upgrade-title">Nâng cấp VIP</h2>
                        <p className="upgrade-desc">
                            Mở khóa toàn bộ tính năng AI chuyên sâu
                        </p>

                        <div className="upgrade-features">
                            {features.map((f, i) => (
                                <div key={i} className="upgrade-feature">
                                    <f.icon size={16} />
                                    <span>{f.text}</span>
                                </div>
                            ))}
                        </div>

                        <div className="upgrade-price-card">
                            <div className="upgrade-price-row">
                                <span>Gói dịch vụ</span>
                                <span className="upgrade-price-value">Trọn đời</span>
                            </div>
                            <div className="upgrade-price-row">
                                <span>Giá tiền</span>
                                <span className="upgrade-price-highlight">10,000 VNĐ</span>
                            </div>
                        </div>

                        <button className="upgrade-cta" onClick={() => setStep(2)}>
                            Tiếp tục thanh toán
                        </button>
                    </div>
                )}

                {/* Step 2: QR Payment */}
                {step === 2 && paymentStatus === 'idle' && (
                    <div className="upgrade-step">
                        {loadingCode ? (
                            <div style={{ textAlign: 'center', padding: 24 }}>
                                <Loader className="spin" size={32} />
                            </div>
                        ) : (
                            <>
                                <p className="upgrade-desc">Quét mã QR để thanh toán</p>

                                <div className="upgrade-qr-wrap">
                                    <img
                                        src={`https://img.vietqr.io/image/${MY_BANK.BANK_ID}-${MY_BANK.ACCOUNT_NO}-${MY_BANK.TEMPLATE}.png?amount=10000&addInfo=VIP%20${userId}&accountName=${encodeURIComponent(MY_BANK.ACCOUNT_NAME)}`}
                                        alt="QR Thanh Toán"
                                        className="upgrade-qr-img"
                                    />
                                </div>

                                <div className="upgrade-bank-info">
                                    <div className="upgrade-bank-row">
                                        <span>Ngân hàng</span>
                                        <span className="text-bold">MB Bank</span>
                                    </div>
                                    <div className="upgrade-bank-row">
                                        <span>Chủ TK</span>
                                        <span className="text-bold">{MY_BANK.ACCOUNT_NAME}</span>
                                    </div>
                                    <div className="upgrade-bank-row">
                                        <span>Số TK</span>
                                        <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                                            <span className="text-bold" style={{ fontFamily: 'var(--font-mono)' }}>{MY_BANK.ACCOUNT_NO}</span>
                                            <button className="upgrade-copy-btn" onClick={() => copyToClipboard(MY_BANK.ACCOUNT_NO)}>
                                                <Copy size={12} />
                                            </button>
                                        </div>
                                    </div>
                                    <div className="upgrade-bank-row">
                                        <span>Nội dung</span>
                                        <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                                            <span style={{ fontWeight: 700, color: 'var(--accent-red)' }}>VIP {userId}</span>
                                            <button className="upgrade-copy-btn" onClick={() => copyToClipboard(`VIP ${userId}`)}>
                                                <Copy size={12} />
                                            </button>
                                        </div>
                                    </div>
                                </div>

                                <button className="upgrade-cta green" onClick={() => { setPaymentStatus('waiting'); setTimeRemaining(PAYMENT_TIMEOUT); }}>
                                    Tôi đã chuyển khoản
                                </button>
                            </>
                        )}
                    </div>
                )}

                {/* Waiting */}
                {paymentStatus === 'waiting' && (
                    <div className="upgrade-step" style={{ textAlign: 'center' }}>
                        <Loader className="spin" size={48} style={{ color: 'var(--accent-green)', margin: '0 auto 16px' }} />
                        <h3>Đang kiểm tra thanh toán...</h3>
                        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 6, color: 'var(--text-secondary)', margin: '12px 0' }}>
                            <Clock size={14} />
                            <span>Còn lại: {formatTime(timeRemaining)}</span>
                        </div>
                        <p className="upgrade-desc">Vui lòng đợi xác nhận. Có thể mất vài phút.</p>
                        <button className="upgrade-check-btn" onClick={checkVipStatus}>Kiểm tra ngay</button>
                    </div>
                )}

                {/* Success */}
                {paymentStatus === 'success' && (
                    <div className="upgrade-step" style={{ textAlign: 'center' }}>
                        <CheckCircle size={56} style={{ color: 'var(--accent-green)', margin: '0 auto 16px' }} />
                        <h3 style={{ color: 'var(--accent-green)' }}>Thanh toán thành công!</h3>
                        <p className="upgrade-desc">Tài khoản VIP đã được kích hoạt. Đang tải lại...</p>
                    </div>
                )}

                {/* Failed / Expired */}
                {(paymentStatus === 'failed' || paymentStatus === 'expired') && (
                    <div className="upgrade-step" style={{ textAlign: 'center' }}>
                        <XCircle size={56} style={{ color: 'var(--accent-red)', margin: '0 auto 16px' }} />
                        <h3 style={{ color: 'var(--accent-red)' }}>
                            {paymentStatus === 'expired' ? 'Hết thời gian' : 'Thanh toán thất bại'}
                        </h3>
                        <p className="upgrade-desc">
                            {paymentStatus === 'expired' ? 'Vui lòng thử lại.' : 'Không thể xử lý. Vui lòng thử lại.'}
                        </p>
                        <button className="upgrade-cta" onClick={() => { setPaymentStatus('idle'); setStep(1); setTimeRemaining(PAYMENT_TIMEOUT); }}>
                            Thử lại
                        </button>
                    </div>
                )}
            </div>
        </div>
    );
}

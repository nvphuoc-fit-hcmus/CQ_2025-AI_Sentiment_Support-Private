import React, { useState } from 'react';
import useStore from '../store';
import { Activity, ArrowRight, Eye, EyeOff, Shield, Zap, BarChart3 } from 'lucide-react';

export default function Login() {
    const { login, register } = useStore();
    const [mode, setMode] = useState('login');
    const [email, setEmail] = useState('');
    const [password, setPassword] = useState('');
    const [confirmPassword, setConfirmPassword] = useState('');
    const [showPassword, setShowPassword] = useState(false);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);

    const validatePassword = (pwd) => {
        const regex = /^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[\W_]).{8,}$/;
        return regex.test(pwd);
    };

    const submit = async (e) => {
        e.preventDefault();
        setError(null);

        if (mode === 'register') {
            if (!validatePassword(password)) {
                setError('Mật khẩu phải có ít nhất 8 ký tự, bao gồm chữ hoa, chữ thường, số và ký tự đặc biệt.');
                return;
            }
            if (password !== confirmPassword) {
                setError('Mật khẩu xác nhận không khớp.');
                return;
            }
        }

        setLoading(true);
        try {
            if (mode === 'login') {
                await login(email.trim().toLowerCase(), password);
            } else {
                await register(email.trim().toLowerCase(), password);
            }

            const user = useStore.getState().user;
            const isAdmin = user && (user.role === 'admin' || user.role === 'Admin');
            if (isAdmin) {
                window.location.href = '/admin/dashboard';
            } else {
                window.location.href = '/';
            }
        } catch (err) {
            const errMsg = err.message || 'Thao tác thất bại';
            if (errMsg.includes('User already exists')) setError('Email này đã được đăng ký.');
            else if (errMsg.includes('Invalid credentials')) setError('Email hoặc mật khẩu không đúng.');
            else setError(errMsg);
        } finally {
            setLoading(false);
        }
    };

    const features = [
        { icon: BarChart3, title: 'Biểu đồ đa khung', desc: 'Phân tích 4 timeframe cùng lúc' },
        { icon: Zap, title: 'AI Dự báo', desc: 'SAFE-Alert với độ tin cậy cao' },
        { icon: Shield, title: 'Cảnh báo thông minh', desc: 'Chỉ phát tín hiệu khi đáng tin' },
    ];

    return (
        <div className="login-page-v2">
            {/* Animated background */}
            <div className="login-bg">
                <div className="login-bg-gradient" />
                <div className="login-bg-grid" />
                <div className="login-bg-glow login-bg-glow-1" />
                <div className="login-bg-glow login-bg-glow-2" />
            </div>

            <div className="login-container">
                {/* Left: Features */}
                <div className="login-features">
                    <div className="login-brand">
                        <Activity size={32} className="login-brand-icon" />
                        <span className="login-brand-text">TradeAI</span>
                    </div>
                    <h1 className="login-headline">
                        Hệ thống dự báo crypto<br />
                        <span className="login-headline-accent">AI thời gian thực</span>
                    </h1>
                    <p className="login-subheadline">
                        Kết hợp tin tức, phân tích kỹ thuật và AI để tạo tín hiệu giao dịch đáng tin cậy.
                    </p>
                    <div className="login-features-list">
                        {features.map((f, i) => (
                            <div key={i} className="login-feature-item">
                                <div className="login-feature-icon">
                                    <f.icon size={18} />
                                </div>
                                <div>
                                    <div className="login-feature-title">{f.title}</div>
                                    <div className="login-feature-desc">{f.desc}</div>
                                </div>
                            </div>
                        ))}
                    </div>
                </div>

                {/* Right: Form */}
                <div className="login-card-v2">
                    <h2 className="login-card-title">
                        {mode === 'login' ? 'Đăng nhập' : 'Tạo tài khoản'}
                    </h2>
                    <p className="login-card-subtitle">
                        {mode === 'login'
                            ? 'Đăng nhập để truy cập bảng điều khiển'
                            : 'Đăng ký miễn phí để bắt đầu'}
                    </p>

                    <form onSubmit={submit} className="login-form-v2">
                        <div className="login-field">
                            <label className="login-label">Email</label>
                            <input
                                type="email"
                                required
                                className="login-input"
                                value={email}
                                onChange={(e) => setEmail(e.target.value)}
                                placeholder="email@example.com"
                            />
                        </div>

                        <div className="login-field">
                            <label className="login-label">Mật khẩu</label>
                            <div className="login-input-wrap">
                                <input
                                    type={showPassword ? 'text' : 'password'}
                                    required
                                    className="login-input"
                                    value={password}
                                    onChange={(e) => setPassword(e.target.value)}
                                    placeholder="••••••••"
                                />
                                <button
                                    type="button"
                                    className="login-eye-btn"
                                    onClick={() => setShowPassword(!showPassword)}
                                    tabIndex={-1}
                                >
                                    {showPassword ? <EyeOff size={16} /> : <Eye size={16} />}
                                </button>
                            </div>
                        </div>

                        {mode === 'register' && (
                            <div className="login-field">
                                <label className="login-label">Xác nhận mật khẩu</label>
                                <input
                                    type="password"
                                    required
                                    className="login-input"
                                    value={confirmPassword}
                                    onChange={(e) => setConfirmPassword(e.target.value)}
                                    placeholder="••••••••"
                                />
                            </div>
                        )}

                        {error && <div className="login-error">{error}</div>}

                        <button type="submit" disabled={loading} className="login-submit-btn">
                            {loading ? (
                                <span className="login-loading">Đang xử lý...</span>
                            ) : (
                                <>
                                    {mode === 'login' ? 'Đăng Nhập' : 'Đăng Ký'}
                                    <ArrowRight size={16} />
                                </>
                            )}
                        </button>
                    </form>

                    <div className="login-switch">
                        {mode === 'login' ? 'Chưa có tài khoản? ' : 'Đã có tài khoản? '}
                        <button
                            className="login-switch-btn"
                            onClick={() => { setMode(mode === 'login' ? 'register' : 'login'); setError(null); }}
                        >
                            {mode === 'login' ? 'Đăng ký ngay' : 'Đăng nhập'}
                        </button>
                    </div>
                </div>
            </div>
        </div>
    );
}

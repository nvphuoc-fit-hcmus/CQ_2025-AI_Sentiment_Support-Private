import React, { useState } from 'react';
import useStore from '../store';
import { Activity, ArrowRight, Eye, EyeOff, Shield, Zap, BarChart3 } from 'lucide-react';

const apiBase = (() => {
    const configured = import.meta.env.VITE_API_URL;
    if (!configured) return 'http://localhost:8000';
    try {
        return new URL(configured).origin;
    } catch {
        return configured.replace(/\/+$/, '');
    }
})();

export default function Login() {
    const { login, register } = useStore();
    const [mode, setMode] = useState('login');
    const [email, setEmail] = useState('');
    const [verificationEmail, setVerificationEmail] = useState('');
    const [otp, setOtp] = useState('');
    const [password, setPassword] = useState('');
    const [confirmPassword, setConfirmPassword] = useState('');
    const [showPassword, setShowPassword] = useState(false);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);
    const [notice, setNotice] = useState(null);

    const validatePassword = (pwd) =>
        /^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[\W_]).{8,}$/.test(pwd);

    const submit = async (event) => {
        event.preventDefault();
        setError(null);
        setNotice(null);

        if (mode === 'register') {
            if (!validatePassword(password)) {
                setError('Mật khẩu phải có ít nhất 8 ký tự, gồm chữ hoa, chữ thường, số và ký tự đặc biệt.');
                return;
            }
            if (password !== confirmPassword) {
                setError('Mật khẩu xác nhận không khớp.');
                return;
            }
        }

        setLoading(true);
        try {
            if (mode === 'forgot') {
                const normalizedEmail = email.trim().toLowerCase();
                const response = await fetch(`${apiBase}/auth/forgot-password/request-otp`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ email: normalizedEmail }),
                });
                const data = await response.json();
                if (!response.ok) throw new Error(data.error || data.message);
                setVerificationEmail(normalizedEmail);
                setMode('reset');
                if (data.development_otp) {
                    setOtp(data.development_otp);
                    setNotice(`Chế độ phát triển: mã OTP là ${data.development_otp}.`);
                } else {
                    setNotice('Mã OTP 6 số đã được gửi đến email và có hiệu lực trong 10 phút.');
                }
                return;
            }

            if (mode === 'reset') {
                if (!validatePassword(password)) {
                    setError('Mật khẩu phải có ít nhất 8 ký tự, gồm chữ hoa, chữ thường, số và ký tự đặc biệt.');
                    return;
                }
                if (password !== confirmPassword) {
                    setError('Mật khẩu xác nhận không khớp.');
                    return;
                }
                const response = await fetch(`${apiBase}/auth/forgot-password/reset`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ email: verificationEmail, otp, new_password: password }),
                });
                const data = await response.json();
                if (!response.ok) throw new Error(data.error || data.message);
                setMode('login');
                setEmail(verificationEmail);
                setOtp('');
                setPassword('');
                setConfirmPassword('');
                setNotice('Đặt lại mật khẩu thành công. Bạn có thể đăng nhập ngay.');
                return;
            }

            if (mode === 'verify') {
                const response = await fetch(`${apiBase}/auth/verify-email`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ email: verificationEmail, otp }),
                });
                const data = await response.json();
                if (!response.ok) throw new Error(data.error || data.message);
                setMode('login');
                setEmail(verificationEmail);
                setOtp('');
                setNotice('Xác thực email thành công. Bạn có thể đăng nhập ngay.');
                return;
            }

            if (mode === 'register') {
                const normalizedEmail = email.trim().toLowerCase();
                const result = await register(normalizedEmail, password);
                setVerificationEmail(normalizedEmail);
                setMode('verify');
                setPassword('');
                setConfirmPassword('');
                if (result?.development_otp) {
                    setOtp(result.development_otp);
                    setNotice(`Chế độ phát triển: mã OTP là ${result.development_otp}.`);
                } else {
                    setNotice('Mã OTP 6 số đã được gửi đến email của bạn.');
                }
                return;
            }

            await login(email.trim().toLowerCase(), password);
            const user = useStore.getState().user;
            window.location.href = user && String(user.role).toLowerCase() === 'admin'
                ? '/admin/dashboard'
                : '/';
        } catch (err) {
            const message = err.message || 'Thao tác thất bại';
            if (message.includes('email_exists') || message.includes('User already exists')) {
                setError('Email này đã được đăng ký.');
            } else if (message.includes('invalid_credentials') || message.includes('Invalid credentials')) {
                setError('Email hoặc mật khẩu không đúng.');
            } else if (message.includes('verification_otp_invalid_or_expired')) {
                setError('Mã OTP không đúng hoặc đã hết hạn.');
            } else if (message.includes('password_reset_otp_invalid_or_expired')) {
                setError('Mã OTP không đúng hoặc đã hết hạn.');
            } else if (message.includes('weak_password')) {
                setError('Mật khẩu mới chưa đáp ứng yêu cầu bảo mật.');
            } else if (message.includes('recovery_email_not_found')) {
                setError('Email không tồn tại trong hệ thống, chưa xác thực hoặc tài khoản không hoạt động.');
            } else if (message.includes('email_not_verified')) {
                setVerificationEmail(email.trim().toLowerCase());
                setMode('verify');
                setError('Tài khoản chưa xác thực. Vui lòng nhập OTP đã gửi đến email.');
            } else {
                setError(message);
            }
        } finally {
            setLoading(false);
        }
    };

    const resendOtp = async () => {
        setLoading(true);
        setError(null);
        setNotice(null);
        try {
            const response = await fetch(`${apiBase}/auth/resend-verification`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ email: verificationEmail }),
            });
            const data = await response.json();
            if (!response.ok) throw new Error(data.error || data.message);
            if (data.development_otp) {
                setOtp(data.development_otp);
                setNotice(`Chế độ phát triển: mã OTP mới là ${data.development_otp}.`);
            } else {
                setNotice('Đã gửi lại mã OTP. Mã mới có hiệu lực trong 10 phút.');
            }
        } catch (err) {
            setError(err.message || 'Không thể gửi lại OTP.');
        } finally {
            setLoading(false);
        }
    };

    const switchMode = (nextMode) => {
        setMode(nextMode);
        setError(null);
        setNotice(null);
        setOtp('');
    };

    const features = [
        { icon: BarChart3, title: 'Biểu đồ đa khung', desc: 'Phân tích nhiều khung thời gian cùng lúc' },
        { icon: Zap, title: 'AI dự báo', desc: 'SAFE-Alert với độ tin cậy rõ ràng' },
        { icon: Shield, title: 'Cảnh báo thông minh', desc: 'Chỉ phát tín hiệu khi đủ bằng chứng' },
    ];

    const title = mode === 'login' ? 'Đăng nhập'
        : mode === 'register' ? 'Tạo tài khoản'
            : mode === 'verify' ? 'Xác thực email'
                : mode === 'forgot' ? 'Quên mật khẩu'
                    : 'Đặt lại mật khẩu';
    const subtitle = mode === 'login'
        ? 'Đăng nhập để truy cập bảng điều khiển'
        : mode === 'register'
            ? 'Đăng ký miễn phí để bắt đầu'
            : mode === 'verify'
                ? `Nhập mã 6 số đã gửi đến ${verificationEmail}`
                : mode === 'forgot'
                    ? 'Nhập email đã đăng ký để nhận mã OTP'
                    : `Nhập OTP đã gửi đến ${verificationEmail} và tạo mật khẩu mới`;

    return (
        <div className="login-page-v2">
            <div className="login-bg">
                <div className="login-bg-gradient" />
                <div className="login-bg-grid" />
                <div className="login-bg-glow login-bg-glow-1" />
                <div className="login-bg-glow login-bg-glow-2" />
            </div>

            <div className="login-container">
                <div className="login-features">
                    <div className="login-brand">
                        <Activity size={32} className="login-brand-icon" />
                        <span className="login-brand-text">Aegis</span>
                    </div>
                    <h1 className="login-headline">
                        Hệ thống dự báo crypto<br />
                        <span className="login-headline-accent">AI thời gian thực</span>
                    </h1>
                    <p className="login-subheadline">
                        Kết hợp tin tức, phân tích kỹ thuật và AI để tạo tín hiệu giao dịch đáng tin cậy.
                    </p>
                    <div className="login-features-list">
                        {features.map(({ icon: Icon, title: featureTitle, desc }) => (
                            <div key={featureTitle} className="login-feature-item">
                                <div className="login-feature-icon"><Icon size={18} /></div>
                                <div>
                                    <div className="login-feature-title">{featureTitle}</div>
                                    <div className="login-feature-desc">{desc}</div>
                                </div>
                            </div>
                        ))}
                    </div>
                </div>

                <div className="login-card-v2">
                    <h2 className="login-card-title">{title}</h2>
                    <p className="login-card-subtitle">{subtitle}</p>

                    <form onSubmit={submit} className="login-form-v2">
                        {mode !== 'verify' && mode !== 'reset' ? (
                            <>
                                <div className="login-field">
                                    <label className="login-label">Email</label>
                                    <input type="email" required className="login-input" value={email}
                                        onChange={(event) => setEmail(event.target.value)}
                                        placeholder="email@example.com" />
                                </div>
                                {mode !== 'forgot' && <div className="login-field">
                                    <label className="login-label">Mật khẩu</label>
                                    <div className="login-input-wrap">
                                        <input type={showPassword ? 'text' : 'password'} required className="login-input"
                                            value={password} onChange={(event) => setPassword(event.target.value)}
                                            placeholder="••••••••" />
                                        <button type="button" className="login-eye-btn"
                                            onClick={() => setShowPassword(!showPassword)} tabIndex={-1}>
                                            {showPassword ? <EyeOff size={16} /> : <Eye size={16} />}
                                        </button>
                                    </div>
                                </div>}
                                {mode === 'register' && (
                                    <div className="login-field">
                                        <label className="login-label">Xác nhận mật khẩu</label>
                                        <input type="password" required className="login-input" value={confirmPassword}
                                            onChange={(event) => setConfirmPassword(event.target.value)}
                                            placeholder="••••••••" />
                                    </div>
                                )}
                            </>
                        ) : (
                            <>
                            <div className="login-field">
                                <label className="login-label">Mã OTP</label>
                                <input type="text" required inputMode="numeric" autoComplete="one-time-code"
                                    maxLength={6} pattern="[0-9]{6}" className="login-input"
                                    value={otp} onChange={(event) => setOtp(event.target.value.replace(/\D/g, '').slice(0, 6))}
                                    placeholder="000000"
                                    style={{ textAlign: 'center', letterSpacing: '0.45em', fontSize: '1.25rem', fontWeight: 700 }} />
                            </div>
                            {mode === 'reset' && (
                                <>
                                    <div className="login-field">
                                        <label className="login-label">Mật khẩu mới</label>
                                        <input type="password" required className="login-input" value={password}
                                            onChange={(event) => setPassword(event.target.value)} placeholder="••••••••" />
                                    </div>
                                    <div className="login-field">
                                        <label className="login-label">Xác nhận mật khẩu mới</label>
                                        <input type="password" required className="login-input" value={confirmPassword}
                                            onChange={(event) => setConfirmPassword(event.target.value)} placeholder="••••••••" />
                                    </div>
                                </>
                            )}
                            </>
                        )}

                        {notice && <div className="login-error" style={{ borderColor: '#16c79a', color: '#16c79a' }}>{notice}</div>}
                        {error && <div className="login-error">{error}</div>}

                        <button type="submit" disabled={loading || ((mode === 'verify' || mode === 'reset') && otp.length !== 6)}
                            className="login-submit-btn">
                            {loading ? <span className="login-loading">Đang xử lý...</span> : (
                                <>
                                    {mode === 'login' ? 'Đăng nhập'
                                        : mode === 'register' ? 'Đăng ký'
                                            : mode === 'forgot' ? 'Gửi mã OTP'
                                                : mode === 'reset' ? 'Đặt lại mật khẩu'
                                                    : 'Xác nhận OTP'}
                                    <ArrowRight size={16} />
                                </>
                            )}
                        </button>
                    </form>

                    {mode === 'verify' ? (
                        <div className="login-switch">
                            Chưa nhận được mã?{' '}
                            <button type="button" className="login-switch-btn" onClick={resendOtp} disabled={loading}>
                                Gửi lại OTP
                            </button>
                            <span> · </span>
                            <button type="button" className="login-switch-btn" onClick={() => switchMode('login')}>
                                Quay lại đăng nhập
                            </button>
                        </div>
                    ) : mode === 'forgot' || mode === 'reset' ? (
                        <div className="login-switch">
                            <button type="button" className="login-switch-btn" onClick={() => switchMode('login')}>
                                Quay lại đăng nhập
                            </button>
                        </div>
                    ) : (
                        <div className="login-switch">
                            {mode === 'login' ? 'Chưa có tài khoản? ' : 'Đã có tài khoản? '}
                            <button type="button" className="login-switch-btn"
                                onClick={() => switchMode(mode === 'login' ? 'register' : 'login')}>
                                {mode === 'login' ? 'Đăng ký ngay' : 'Đăng nhập'}
                            </button>
                        </div>
                    )}
                    {mode === 'login' && (
                        <div className="login-switch" style={{ marginTop: 10 }}>
                            <button type="button" className="login-switch-btn" onClick={() => switchMode('forgot')}>
                                Quên mật khẩu?
                            </button>
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
}

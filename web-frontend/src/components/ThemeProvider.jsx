import React, { createContext, useContext, useState, useEffect } from 'react';
import useStore from '../store';
import { useToast } from './ToastProvider';
import { pushNotification } from '../utils/notificationCenter';

const ThemeContext = createContext();

export const THEMES = {
    DARK: 'dark',
    LIGHT: 'light'
};

/**
 * Theme Provider Component
 * Provides dark/light mode switching functionality
 */
export const ThemeProvider = ({ children }) => {
    // Get initial theme from localStorage or system preference
    const getInitialTheme = () => {
        const stored = localStorage.getItem('theme');
        if (stored) return stored;

        // Check system preference
        if (window.matchMedia && window.matchMedia('(prefers-color-scheme: light)').matches) {
            return THEMES.LIGHT;
        }
        return THEMES.DARK;
    };

    const [theme, setTheme] = useState(getInitialTheme);

    // Apply theme to document
    useEffect(() => {
        const root = document.documentElement;

        if (theme === THEMES.LIGHT) {
            root.setAttribute('data-theme', 'light');
            // Light mode — TradingView-inspired
            root.style.setProperty('--bg-primary', '#f0f3fa');
            root.style.setProperty('--bg-secondary', '#ffffff');
            root.style.setProperty('--bg-tertiary', '#f0f3fa');
            root.style.setProperty('--bg-hover', '#f0f3fa');
            root.style.setProperty('--text-primary', '#131722');
            root.style.setProperty('--text-secondary', '#787b86');
            root.style.setProperty('--text-muted', '#a3a6af');
            root.style.setProperty('--border-color', '#e0e3eb');
            root.style.setProperty('--card-bg', '#ffffff');
            root.style.setProperty('--card-border', '#e0e3eb');
            root.style.setProperty('--navbar-bg', '#ffffff');
            root.style.setProperty('--sidebar-bg', '#ffffff');
            root.style.setProperty('--input-bg', '#f0f3fa');
            root.style.setProperty('--input-border', '#e0e3eb');
            root.style.setProperty('--modal-overlay', 'rgba(0, 0, 0, 0.5)');
            root.style.setProperty('--shadow-color', 'rgba(0, 0, 0, 0.08)');
        } else {
            root.setAttribute('data-theme', 'dark');
            // Dark mode — TradingView-inspired
            root.style.setProperty('--bg-primary', '#131722');
            root.style.setProperty('--bg-secondary', '#1e222d');
            root.style.setProperty('--bg-tertiary', '#2a2e39');
            root.style.setProperty('--bg-hover', '#2a2e39');
            root.style.setProperty('--text-primary', '#d1d4dc');
            root.style.setProperty('--text-secondary', '#787b86');
            root.style.setProperty('--text-muted', '#555962');
            root.style.setProperty('--border-color', '#2a2e39');
            root.style.setProperty('--card-bg', '#1e222d');
            root.style.setProperty('--card-border', '#2a2e39');
            root.style.setProperty('--navbar-bg', '#131722');
            root.style.setProperty('--sidebar-bg', '#1e222d');
            root.style.setProperty('--input-bg', '#1a1e2d');
            root.style.setProperty('--input-border', '#2a2e39');
            root.style.setProperty('--modal-overlay', 'rgba(0, 0, 0, 0.75)');
            root.style.setProperty('--shadow-color', 'rgba(0, 0, 0, 0.5)');
        }

        // Save to localStorage
        localStorage.setItem('theme', theme);
    }, [theme]);

    // Listen for system theme changes
    useEffect(() => {
        const mediaQuery = window.matchMedia('(prefers-color-scheme: light)');
        const handleChange = (e) => {
            // Only auto-switch if user hasn't manually set a preference
            if (!localStorage.getItem('theme')) {
                setTheme(e.matches ? THEMES.LIGHT : THEMES.DARK);
            }
        };

        mediaQuery.addEventListener('change', handleChange);
        return () => mediaQuery.removeEventListener('change', handleChange);
    }, []);

    const toggleTheme = () => {
        setTheme(prev => prev === THEMES.DARK ? THEMES.LIGHT : THEMES.DARK);
    };

    const value = {
        theme,
        setTheme,
        toggleTheme,
        isDark: theme === THEMES.DARK,
        isLight: theme === THEMES.LIGHT
    };

    return (
        <ThemeContext.Provider value={value}>
            {children}
        </ThemeContext.Provider>
    );
};

/**
 * Custom hook to use theme context
 */
export const useTheme = () => {
    const context = useContext(ThemeContext);
    if (!context) {
        throw new Error('useTheme must be used within a ThemeProvider');
    }
    return context;
};

/**
 * Theme Toggle Button Component
 */
export const ThemeToggle = ({ className = '' }) => {
    const { theme, toggleTheme, isDark } = useTheme();

    return (
        <button
            className={`theme-toggle ${className}`}
            onClick={toggleTheme}
            title={isDark ? 'Chuyển sang chế độ sáng' : 'Chuyển sang chế độ tối'}
            aria-label="Toggle theme"
            style={{
                background: 'var(--bg-tertiary)',
                border: '1px solid var(--border-color)',
                borderRadius: '8px',
                padding: '8px 12px',
                cursor: 'pointer',
                display: 'flex',
                alignItems: 'center',
                gap: '8px',
                color: 'var(--text-primary)',
                transition: 'all 0.2s ease',
                fontSize: '14px'
            }}
        >
            {isDark ? (
                <>
                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                        <circle cx="12" cy="12" r="5" />
                        <path d="M12 1v2M12 21v2M4.22 4.22l1.42 1.42M18.36 18.36l1.42 1.42M1 12h2M21 12h2M4.22 19.78l1.42-1.42M18.36 5.64l1.42-1.42" />
                    </svg>
                    <span>Sáng</span>
                </>
            ) : (
                <>
                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                        <path d="M21 12.79A9 9 0 1111.21 3 7 7 0 0021 12.79z" />
                    </svg>
                    <span>Tối</span>
                </>
            )}
        </button>
    );
};

/**
 * Settings Panel with Theme Selection
 */
export const SettingsPanel = ({ isOpen, onClose }) => {
    const { theme, setTheme, isDark } = useTheme();
    const { authFetch, logout } = useStore();
    const { showToast } = useToast();
    const [profile, setProfile] = useState(null);
    const [displayName, setDisplayName] = useState('');
    const [currentPassword, setCurrentPassword] = useState('');
    const [newPassword, setNewPassword] = useState('');
    const [confirmPassword, setConfirmPassword] = useState('');
    const [passwordOtp, setPasswordOtp] = useState('');
    const [passwordOtpSent, setPasswordOtpSent] = useState(false);
    const [saving, setSaving] = useState(false);

    useEffect(() => {
        if (!isOpen) return;
        authFetch('/auth/me').then(async response => {
            if (!response.ok) throw new Error('Không tải được hồ sơ');
            const data = await response.json();
            setProfile(data.user);
            setDisplayName(data.user?.display_name || '');
            useStore.setState({ user: { ...useStore.getState().user, ...data.user } });
        }).catch(error => showToast(error.message, 'error'));
    }, [isOpen, authFetch, showToast]);

    const saveProfile = async () => {
        setSaving(true);
        try {
            const response = await authFetch('/auth/me', {
                method: 'PATCH',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ display_name: displayName })
            });
            const data = await response.json();
            if (!response.ok) throw new Error(data.message || data.error || 'Không thể cập nhật hồ sơ');
            setProfile(data.user);
            useStore.setState({ user: { ...useStore.getState().user, ...data.user } });
            showToast('Đã cập nhật hồ sơ', 'success');
        } catch (error) {
            showToast(error.message, 'error');
        } finally {
            setSaving(false);
        }
    };

    const requestPasswordOtp = async () => {
        if (newPassword.length < 8) return showToast('Mật khẩu mới phải có ít nhất 8 ký tự', 'error');
        if (newPassword !== confirmPassword) return showToast('Mật khẩu xác nhận không khớp', 'error');
        setSaving(true);
        try {
            const response = await authFetch('/auth/change-password/request-otp', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ current_password: currentPassword, new_password: newPassword })
            });
            const data = await response.json();
            if (!response.ok) throw new Error(data.message || data.error || 'Không thể gửi mã OTP');
            setPasswordOtpSent(true);
            if (data.development_otp) {
                setPasswordOtp(data.development_otp);
                showToast(`Chế độ phát triển: OTP là ${data.development_otp}`, 'success');
            } else {
                showToast('Mã OTP đã được gửi đến email của bạn', 'success');
            }
        } catch (error) {
            showToast(error.message, 'error');
        } finally {
            setSaving(false);
        }
    };

    const changePassword = async () => {
        if (passwordOtp.length !== 6) return showToast('Vui lòng nhập đủ 6 số OTP', 'error');
        setSaving(true);
        try {
            const response = await authFetch('/auth/change-password', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    current_password: currentPassword,
                    new_password: newPassword,
                    otp: passwordOtp
                })
            });
            const data = await response.json();
            if (!response.ok) {
                const message = data.error === 'password_change_otp_invalid_or_expired'
                    ? 'Mã OTP không đúng hoặc đã hết hạn'
                    : data.message || data.error || 'Không thể đổi mật khẩu';
                throw new Error(message);
            }
            pushNotification({
                type: 'security',
                title: 'Mật khẩu đã được thay đổi',
                message: 'Mật khẩu tài khoản Aegis vừa được cập nhật thành công.',
                dedupeKey: `password-changed-${Date.now()}`,
            });
            showToast('Đổi mật khẩu thành công. Vui lòng đăng nhập lại.', 'success');
            await logout();
        } catch (error) {
            showToast(error.message, 'error');
        } finally {
            setSaving(false);
        }
    };

    if (!isOpen) return null;

    return (
        <div
            className="settings-overlay"
            onClick={onClose}
            style={{
                position: 'fixed',
                top: 0,
                left: 0,
                right: 0,
                bottom: 0,
                background: 'var(--modal-overlay)',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                zIndex: 1000,
                animation: 'fadeIn 0.2s ease-out'
            }}
        >
            <div
                className="settings-panel"
                onClick={e => e.stopPropagation()}
                style={{
                    background: 'var(--card-bg)',
                    borderRadius: '16px',
                    padding: '24px',
                    width: '520px',
                    maxWidth: '90vw',
                    maxHeight: '88vh',
                    overflowY: 'auto',
                    boxShadow: '0 20px 60px var(--shadow-color)',
                    border: '1px solid var(--border-color)',
                    animation: 'slideUp 0.3s ease-out'
                }}
            >
                <div className="settings-modal-header">
                    <div>
                        <div className="settings-modal-kicker">TÀI KHOẢN</div>
                        <h2 style={{ margin: 0, color: 'var(--text-primary)', fontSize: '22px' }}>Cài đặt cá nhân</h2>
                        <p>Quản lý thông tin tài khoản và bảo mật.</p>
                    </div>
                    <button
                        className="settings-close-btn"
                        onClick={onClose}
                        style={{
                            background: 'none',
                            border: 'none',
                            color: 'var(--text-secondary)',
                            cursor: 'pointer',
                            fontSize: '24px',
                            padding: '4px'
                        }}
                    >
                        ×
                    </button>
                </div>

                <div className="settings-section" style={{ marginBottom: 24 }}>
                    <div className="settings-section-heading">
                        <div>
                            <h3>Hồ sơ cá nhân</h3>
                            <p>Thông tin hiển thị trên tài khoản Aegis.</p>
                        </div>
                    </div>
                    <label style={{ display: 'block', color: 'var(--text-secondary)', fontSize: 12, marginBottom: 6 }}>Tên hiển thị</label>
                    <input value={displayName} onChange={e => setDisplayName(e.target.value)}
                        placeholder="Nhập tên hiển thị" className="form-input" style={{ width: '100%', marginBottom: 12 }} />
                    <label style={{ display: 'block', color: 'var(--text-secondary)', fontSize: 12, marginBottom: 6 }}>Email</label>
                    <input value={profile?.email || ''} disabled className="form-input"
                        style={{ width: '100%', marginBottom: 8, opacity: .75 }} />
                    <div style={{ fontSize: 12, color: profile?.email_verified ? '#16c79a' : '#f59e0b', marginBottom: 12 }}>
                        {profile?.email_verified ? '✓ Email đã xác thực' : '⚠ Email chưa xác thực'}
                    </div>
                    <button className="btn-primary" onClick={saveProfile} disabled={saving}>Lưu hồ sơ</button>
                </div>

                <div className="settings-section settings-security-section">
                    <div className="settings-section-heading">
                        <div>
                            <h3>Đổi mật khẩu</h3>
                            <p>Xác nhận bằng OTP được gửi tới email của bạn.</p>
                        </div>
                    </div>
                    <input type="password" value={currentPassword} onChange={e => setCurrentPassword(e.target.value)}
                        placeholder="Mật khẩu hiện tại" className="form-input" style={{ width: '100%', marginBottom: 10 }} />
                    <input type="password" value={newPassword} onChange={e => setNewPassword(e.target.value)}
                        placeholder="Mật khẩu mới" className="form-input" style={{ width: '100%', marginBottom: 10 }} />
                    <input type="password" value={confirmPassword} onChange={e => setConfirmPassword(e.target.value)}
                        placeholder="Xác nhận mật khẩu mới" className="form-input" style={{ width: '100%', marginBottom: 12 }} />
                    {passwordOtpSent && (
                        <input type="text" inputMode="numeric" autoComplete="one-time-code" maxLength={6}
                            value={passwordOtp}
                            onChange={e => setPasswordOtp(e.target.value.replace(/\D/g, '').slice(0, 6))}
                            placeholder="Nhập OTP 6 số" className="form-input"
                            style={{ width: '100%', marginBottom: 12, textAlign: 'center', letterSpacing: '0.35em', fontWeight: 700 }} />
                    )}
                    <div style={{ display: 'flex', gap: 10 }}>
                        {!passwordOtpSent ? (
                            <button className="btn-secondary" onClick={requestPasswordOtp} disabled={saving}>
                                Gửi mã xác nhận
                            </button>
                        ) : (
                            <>
                                <button className="btn-primary" onClick={changePassword}
                                    disabled={saving || passwordOtp.length !== 6}>Xác nhận đổi mật khẩu</button>
                                <button className="btn-secondary" onClick={requestPasswordOtp} disabled={saving}>
                                    Gửi lại mã
                                </button>
                            </>
                        )}
                    </div>
                </div>

                <div style={{ marginTop: '24px', paddingTop: '16px', borderTop: '1px solid var(--border-color)' }}>
                    <p style={{ color: 'var(--text-muted)', fontSize: '12px', textAlign: 'center', margin: 0 }}>
                        Thêm cài đặt sẽ được cập nhật trong phiên bản tiếp theo
                    </p>
                </div>
            </div>
        </div>
    );
};

export default ThemeProvider;

import React, { createContext, useContext, useState, useEffect } from 'react';

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
                    width: '400px',
                    maxWidth: '90vw',
                    boxShadow: '0 20px 60px var(--shadow-color)',
                    border: '1px solid var(--border-color)',
                    animation: 'slideUp 0.3s ease-out'
                }}
            >
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '24px' }}>
                    <h2 style={{ margin: 0, color: 'var(--text-primary)', fontSize: '20px' }}>Cài đặt</h2>
                    <button
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

                {/* Theme Section */}
                <div className="settings-section">
                    <h3 style={{ color: 'var(--text-secondary)', fontSize: '14px', marginBottom: '12px', fontWeight: 500 }}>
                        Giao diện
                    </h3>

                    <div style={{ display: 'flex', gap: '12px' }}>
                        {/* Dark Mode Button */}
                        <button
                            onClick={() => setTheme(THEMES.DARK)}
                            style={{
                                flex: 1,
                                padding: '16px',
                                borderRadius: '12px',
                                border: `2px solid ${isDark ? 'var(--accent-color, #3498db)' : 'var(--border-color)'}`,
                                background: isDark ? 'rgba(52, 152, 219, 0.1)' : 'var(--bg-secondary)',
                                cursor: 'pointer',
                                display: 'flex',
                                flexDirection: 'column',
                                alignItems: 'center',
                                gap: '8px',
                                transition: 'all 0.2s ease'
                            }}
                        >
                            <div style={{
                                width: '48px',
                                height: '32px',
                                background: '#1a1a1a',
                                borderRadius: '6px',
                                border: '1px solid #333'
                            }} />
                            <span style={{ color: 'var(--text-primary)', fontSize: '14px' }}>Tối</span>
                            {isDark && (
                                <svg width="16" height="16" viewBox="0 0 24 24" fill="var(--accent-color, #3498db)">
                                    <path d="M9 16.17L4.83 12l-1.42 1.41L9 19 21 7l-1.41-1.41z" />
                                </svg>
                            )}
                        </button>

                        {/* Light Mode Button */}
                        <button
                            onClick={() => setTheme(THEMES.LIGHT)}
                            style={{
                                flex: 1,
                                padding: '16px',
                                borderRadius: '12px',
                                border: `2px solid ${!isDark ? 'var(--accent-color, #3498db)' : 'var(--border-color)'}`,
                                background: !isDark ? 'rgba(52, 152, 219, 0.1)' : 'var(--bg-secondary)',
                                cursor: 'pointer',
                                display: 'flex',
                                flexDirection: 'column',
                                alignItems: 'center',
                                gap: '8px',
                                transition: 'all 0.2s ease'
                            }}
                        >
                            <div style={{
                                width: '48px',
                                height: '32px',
                                background: '#ffffff',
                                borderRadius: '6px',
                                border: '1px solid #ddd'
                            }} />
                            <span style={{ color: 'var(--text-primary)', fontSize: '14px' }}>Sáng</span>
                            {!isDark && (
                                <svg width="16" height="16" viewBox="0 0 24 24" fill="var(--accent-color, #3498db)">
                                    <path d="M9 16.17L4.83 12l-1.42 1.41L9 19 21 7l-1.41-1.41z" />
                                </svg>
                            )}
                        </button>
                    </div>
                </div>

                {/* More settings can be added here */}
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

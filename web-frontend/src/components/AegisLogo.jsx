import React from 'react';

/**
 * Aegis "Signal Shield" mark.
 * Uses currentColor so every surface can inherit the application's accent
 * token and remain consistent in dark and light themes.
 */
export default function AegisLogo({ size = 24, className = '', title = 'Aegis' }) {
    return (
        <svg
            className={className}
            width={size}
            height={size}
            viewBox="0 0 32 32"
            role="img"
            aria-label={title}
            fill="none"
        >
            <path
                d="M16 2.8 27.2 7v9.2c0 6.2-4.8 10.7-11.2 12.7C9.6 26.9 4.8 22.4 4.8 16.2V7L16 2.8Z"
                stroke="currentColor"
                strokeWidth="2.2"
                strokeLinejoin="round"
            />
            <rect x="10.4" y="17.4" width="2.8" height="5.4" rx="1.2" fill="currentColor" />
            <rect x="14.6" y="13.8" width="2.8" height="9" rx="1.2" fill="currentColor" />
            <rect x="18.8" y="10.2" width="2.8" height="12.6" rx="1.2" fill="currentColor" />
        </svg>
    );
}

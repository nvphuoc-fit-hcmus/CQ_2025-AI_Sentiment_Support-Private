/**
 * GapRecovery - Utility for recovering missed data via API
 * 
 * Fetches missed klines from the Missed Data Recovery API
 * Handles recovery status and progress tracking
 */

import GapDetector from './GapDetector';

class GapRecovery {
    /**
     * Fetch missed data from API
     * @param {string} symbol - Trading pair symbol
     * @param {number} fromSeq - Starting sequence (exclusive)
     * @param {number} toSeq - Ending sequence (inclusive)
     * @param {string} token - JWT authentication token
     * @param {Function} onProgress - Progress callback (optional)
     * @returns {Promise<Array>} Array of recovered klines
     */
    static async fetchMissedData(symbol, fromSeq, toSeq, token, onProgress = null) {
        try {
            console.log(`[GapRecovery] Fetching missed data for ${symbol}: ${fromSeq} -> ${toSeq}`);

            if (onProgress) {
                onProgress({ status: 'fetching', symbol, fromSeq, toSeq });
            }

            // Determine API base URL
            const apiBase = import.meta.env.VITE_API_URL || window.location.origin;
            const url = `${apiBase}/api/v1/klines/missed?symbol=${symbol}&fromSeq=${fromSeq}&toSeq=${toSeq}`;

            const response = await fetch(url, {
                headers: {
                    'Authorization': `Bearer ${token}`,
                    'Content-Type': 'application/json'
                }
            });

            if (!response.ok) {
                const error = await response.json().catch(() => ({ error: 'Unknown error' }));
                throw new Error(`HTTP ${response.status}: ${error.error || response.statusText}`);
            }

            const result = await response.json();

            console.log(`[GapRecovery] Recovered ${result.count} messages for ${symbol}`);

            // Update gap statistics
            const gap = { fromSeq, toSeq, gapSize: toSeq - fromSeq };
            GapDetector.updateStats(symbol, gap, result.count);

            if (onProgress) {
                onProgress({
                    status: 'completed',
                    symbol,
                    fromSeq,
                    toSeq,
                    recovered: result.count
                });
            }

            return result.data || [];
        } catch (error) {
            console.error('[GapRecovery] Failed to fetch missed data:', error);

            if (onProgress) {
                onProgress({
                    status: 'error',
                    symbol,
                    fromSeq,
                    toSeq,
                    error: error.message
                });
            }

            throw error;
        }
    }

    /**
     * Recover gap automatically
     * @param {Object} gap - Gap information from GapDetector
     * @param {string} token - JWT authentication token
     * @param {Function} onMessage - Callback for each recovered message
     * @param {Function} onProgress - Progress callback (optional)
     * @returns {Promise<number>} Number of messages recovered
     */
    static async recoverGap(gap, token, onMessage, onProgress = null) {
        try {
            const messages = await this.fetchMissedData(
                gap.symbol,
                gap.fromSeq,
                gap.toSeq,
                token,
                onProgress
            );

            // Process each recovered message
            let processedCount = 0;
            for (const msg of messages) {
                if (onMessage) {
                    onMessage(msg, true); // true = recovered message
                    processedCount++;
                }
            }

            return processedCount;
        } catch (error) {
            console.error('[GapRecovery] Gap recovery failed:', error);
            return 0;
        }
    }

    /**
     * Batch recover multiple gaps
     * @param {Array} gaps - Array of gap objects
     * @param {string} token - JWT authentication token
     * @param {Function} onMessage - Callback for each recovered message
     * @param {Function} onProgress - Progress callback (optional)
     * @returns {Promise<Object>} Recovery summary
     */
    static async recoverMultipleGaps(gaps, token, onMessage, onProgress = null) {
        const results = {
            total: gaps.length,
            succeeded: 0,
            failed: 0,
            recovered: 0
        };

        for (const gap of gaps) {
            try {
                const count = await this.recoverGap(gap, token, onMessage, onProgress);
                results.succeeded++;
                results.recovered += count;
            } catch (error) {
                results.failed++;
                console.error(`[GapRecovery] Failed to recover gap for ${gap.symbol}:`, error);
            }
        }

        return results;
    }

    /**
     * Check if recovery is needed based on gap size
     * @param {Object} gap - Gap information
     * @param {number} threshold - Maximum gap size to ignore (default: 0)
     * @returns {boolean} True if recovery is needed
     */
    static shouldRecover(gap, threshold = 0) {
        return gap && gap.gapSize > threshold;
    }

    /**
     * Format gap for display
     * @param {Object} gap - Gap information
     * @returns {string} Formatted gap description
     */
    static formatGap(gap) {
        if (!gap) return 'No gap';

        return `${gap.symbol}: Missing ${gap.gapSize} messages (seq ${gap.fromSeq} to ${gap.toSeq})`;
    }

    /**
     * Get recovery status message
     * @param {Object} progress - Progress object from onProgress callback
     * @returns {string} Status message
     */
    static getStatusMessage(progress) {
        if (!progress) return '';

        switch (progress.status) {
            case 'fetching':
                return `Đang khôi phục dữ liệu ${progress.symbol}...`;
            case 'completed':
                return `Đã khôi phục ${progress.recovered} tin nhắn cho ${progress.symbol}`;
            case 'error':
                return `Lỗi khôi phục ${progress.symbol}: ${progress.error}`;
            default:
                return '';
        }
    }
}

export default GapRecovery;

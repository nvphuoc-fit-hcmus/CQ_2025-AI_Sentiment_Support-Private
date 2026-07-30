const TechnicalIndicators = require('./indicators');
const StrategyParser = require('./strategy-parser');

class BacktestEngine {
    /**
     * Initialize Backtest Engine
     * @param {Object} strategy - Strategy configuration (name, conditions, logic, etc.)
     * @param {Object} data - Historical data { candles, predictions, news }
     * @param {number} initialCapital - Start capital ($)
     */
    constructor(strategy, data, initialCapital = 10000) {
        this.strategy = strategy;
        this.candles = data.candles || [];
        this.predictions = data.predictions || [];
        this.news = data.news || [];
        this.initialCapital = initialCapital;

        // State
        this.currentCapital = initialCapital;
        this.positions = []; // Closed positions
        this.activePosition = null; // Currently open position
        this.equityInterim = []; // Equity curve over time
        this.logs = [];
    }

    /**
     * Run the backtest simulation
     * @returns {Object} Performance metrics and trade history
     */
    run() {
        if (!this.candles.length) {
            return { error: "No historical data provided" };
        }

        // Sort data by time just in case
        this.candles.sort((a, b) => new Date(a.time) - new Date(b.time));
        this.predictions.sort((a, b) => new Date(a.time) - new Date(b.time));
        this.news.sort((a, b) => new Date(a.time) - new Date(b.time));

        const startTime = Date.now();
        const minCandlesForIndicators = 50; // Need history for EMA/RSI
        const initialEquityCandle = this.candles[minCandlesForIndicators - 1];
        if (initialEquityCandle) {
            this.equityInterim.push({
                time: new Date(initialEquityCandle.time),
                value: this.initialCapital
            });
        }

        // MAIN LOOP
        for (let i = minCandlesForIndicators; i < this.candles.length; i++) {
            const candle = this.candles[i];
            const prevCandle = this.candles[i - 1];
            const currentTime = new Date(candle.time);

            // 1. Prepare Context (Data Snapshot at this moment)
            // Get recent candles for indicators
            const recentCandles = this.candles.slice(i - minCandlesForIndicators, i + 1);

            // Calculate indicators
            const indicators = TechnicalIndicators.calculateAll(recentCandles);

            // Find latest prediction available BEFORE or AT current time
            const prediction = this.getLatestPrediction(currentTime);

            // Get recent news (last 24h)
            const recentNews = this.getRecentNews(currentTime, 24);

            const context = {
                candle,
                currentPrice: parseFloat(candle.close),
                indicators,
                prediction,
                news: recentNews
            };

            // 2. Evaluate Strategy
            const action = StrategyParser.evaluateStrategy(this.strategy, context);

            // 3. Execute Trades
            this.executeLogic(action, candle, currentTime, context);

            // 4. Record Equity
            this.recordEquity(currentTime, parseFloat(candle.close));
        }

        // Force close active position at the end
        if (this.activePosition) {
            const lastCandle = this.candles[this.candles.length - 1];
            this.closePosition(lastCandle.close, lastCandle.time, 'END_OF_DATA');
            this.recordEquity(new Date(lastCandle.time), parseFloat(lastCandle.close));
        }

        const endTime = Date.now();

        return {
            ...this.calculatePerformance(),
            execution_time_ms: endTime - startTime,
            data_points_analyzed: this.candles.length
        };
    }

    /**
     * Execute trading logic based on signal
     */
    executeLogic(signal, candle, time, context) {
        const price = parseFloat(candle.close);

        // Stop Loss / Take Profit Check for Active Position
        if (this.activePosition) {
            if (this.checkStopLossTakeProfit(this.activePosition, price, candle.high, candle.low, time)) {
                return; // Position closed by SL/TP
            }
        }

        // Process Signal
        if (signal === 'BUY' && !this.activePosition) {
            // OPEN LONG
            const qty = (this.currentCapital * 0.99) / price; // Use 99% capital (fees buffer)
            const entryNotional = price * qty;
            const entryFee = entryNotional * 0.001;
            this.activePosition = {
                type: 'long',
                entry_price: price,
                entry_time: time,
                qty: qty,
                entry_notional: entryNotional,
                entry_fee: entryFee,
                news_context: (context.news || []).slice(-5),
                // Stop Loss / TP from strategy or default
                stop_loss: this.strategy.stop_loss ? price * (1 - this.strategy.stop_loss / 100) : null,
                take_profit: this.strategy.take_profit ? price * (1 + this.strategy.take_profit / 100) : null
            };

            // Fee (0.1%)
            this.currentCapital -= entryFee;

        } else if (signal === 'SELL') {
            // Close LONG if exists
            if (this.activePosition && this.activePosition.type === 'long') {
                this.closePosition(price, time, 'SIGNAL_SELL');
            }
            // Open SHORT if no position (and action logic allows it, implied by getting SELL signal)
            else if (!this.activePosition) {
                const qty = (this.currentCapital * 0.99) / price;
                const entryNotional = price * qty;
                const entryFee = entryNotional * 0.001;
                this.activePosition = {
                    type: 'short',
                    entry_price: price,
                    entry_time: time,
                    qty: qty,
                    entry_notional: entryNotional,
                    entry_fee: entryFee,
                    news_context: (context.news || []).slice(-5),
                    // Short: SL is Above, TP is Below
                    stop_loss: this.strategy.stop_loss ? price * (1 + this.strategy.stop_loss / 100) : null,
                    take_profit: this.strategy.take_profit ? price * (1 - this.strategy.take_profit / 100) : null
                };
                this.currentCapital -= entryFee;
            }
        }
    }

    /**
     * Check and execute Stop Loss or Take Profit
     */
    checkStopLossTakeProfit(pos, currentPrice, high, low, currentTime) {
        if (pos.type === 'long') {
            // Long Logic: SL below, TP above
            if (pos.stop_loss && low <= pos.stop_loss) {
                this.closePosition(pos.stop_loss, currentTime, 'STOP_LOSS');
                return true;
            }
            if (pos.take_profit && high >= pos.take_profit) {
                this.closePosition(pos.take_profit, currentTime, 'TAKE_PROFIT');
                return true;
            }
        } else if (pos.type === 'short') {
            // Short Logic: SL above, TP below
            if (pos.stop_loss && high >= pos.stop_loss) {
                this.closePosition(pos.stop_loss, currentTime, 'STOP_LOSS'); // Covered at higher price (Loss)
                return true;
            }
            if (pos.take_profit && low <= pos.take_profit) {
                this.closePosition(pos.take_profit, currentTime, 'TAKE_PROFIT'); // Covered at lower price (Profit)
                return true;
            }
        }

        return false;
    }

    /**
     * Close the active position
     */
    closePosition(price, time, reason) {
        if (!this.activePosition) return;

        const pos = this.activePosition;
        const exitNotional = pos.qty * price;
        const exitFee = exitNotional * 0.001;
        const entryNotional = pos.entry_notional ?? (pos.qty * pos.entry_price);
        const entryFee = pos.entry_fee ?? (entryNotional * 0.001);
        const grossProfit = pos.type === 'long'
            ? (price - pos.entry_price) * pos.qty
            : (pos.entry_price - price) * pos.qty;

        // The opening fee was deducted when the position was created. At close,
        // apply gross price PnL and the exit fee to account capital. Trade-level
        // profit includes both fees so trade sums reconcile with final equity.
        this.currentCapital += grossProfit - exitFee;
        const profit = grossProfit - entryFee - exitFee;
        const returnPct = entryNotional > 0 ? (profit / entryNotional) * 100 : 0;

        this.positions.push({
            symbol: this.strategy.symbol,
            entry_time: pos.entry_time,
            exit_time: time,
            entry_price: pos.entry_price,
            exit_price: price,
            qty: pos.qty,
            side: pos.type,
            gross_profit: grossProfit,
            entry_fee: entryFee,
            exit_fee: exitFee,
            total_fees: entryFee + exitFee,
            profit: profit,
            return_percent: returnPct,
            reason: reason,
            news_context: pos.news_context || [],
            cumulative_capital: this.currentCapital
        });

        this.activePosition = null;
    }

    /**
     * Record marked-to-market account equity.
     * currentCapital already represents the full account balance; the position
     * was opened notionally without removing its value from cash. Therefore we
     * add only unrealised PnL here, not the full position notional.
     */
    recordEquity(time, currentPrice) {
        let equity = this.currentCapital;

        if (this.activePosition) {
            const pos = this.activePosition;
            const unrealizedPnl = pos.type === 'long'
                ? (currentPrice - pos.entry_price) * pos.qty
                : (pos.entry_price - currentPrice) * pos.qty;
            const estimatedExitFee = pos.qty * currentPrice * 0.001;
            equity = this.currentCapital + unrealizedPnl - estimatedExitFee;
        }

        const normalizedTime = new Date(time);
        const lastPoint = this.equityInterim[this.equityInterim.length - 1];
        if (lastPoint && new Date(lastPoint.time).getTime() === normalizedTime.getTime()) {
            lastPoint.value = equity;
        } else {
            this.equityInterim.push({ time: normalizedTime, value: equity });
        }
    }

    getLatestPrediction(time) {
        // Find last prediction where pred.time <= time
        // Backward search since arrays are sorted asc
        for (let i = this.predictions.length - 1; i >= 0; i--) {
            if (new Date(this.predictions[i].time) <= time) {
                return this.predictions[i];
            }
        }
        return null;
    }

    getRecentNews(time, hoursLookback) {
        const lookbackTime = new Date(time.getTime() - (hoursLookback * 60 * 60 * 1000));
        return this.news.filter(n => {
            const t = new Date(n.time);
            return t <= time && t >= lookbackTime;
        });
    }

    calculatePerformance() {
        const totalTrades = this.positions.length;
        if (totalTrades === 0) {
            return {
                total_trades: 0,
                winning_trades: 0,
                losing_trades: 0,
                win_rate: 0,
                total_profit: 0,
                total_loss: 0,
                net_profit: 0,
                net_profit_percent: 0,
                final_equity: this.currentCapital,
                total_fees: 0,
                max_drawdown: 0,
                sharpe_ratio: 0,
                trades: [],
                equity_curve: this.equityInterim
            };
        }

        const winningTrades = this.positions.filter(p => p.profit > 0);
        const losingTrades = this.positions.filter(p => p.profit <= 0);

        const winRate = (winningTrades.length / totalTrades) * 100;

        const initialCap = this.initialCapital;
        const finalEquity = this.currentCapital;
        const totalProfit = winningTrades.reduce((sum, p) => sum + p.profit, 0);
        const totalLoss = Math.abs(losingTrades.reduce((sum, p) => sum + p.profit, 0));
        // Account balance is authoritative and includes every opening/closing
        // fee. This also guarantees the headline PnL matches the equity curve.
        const netProfit = finalEquity - initialCap;
        const netProfitPercent = (netProfit / initialCap) * 100;

        // Max Drawdown
        let maxDrawdown = 0;
        let peak = initialCap;

        this.equityInterim.forEach(point => {
            if (point.value > peak) peak = point.value;
            const drawdown = ((peak - point.value) / peak) * 100;
            if (drawdown > maxDrawdown) maxDrawdown = drawdown;
        });

        // Sharpe Ratio (Simplified Annualized)
        // R_p = mean return, sigma_p = std dev of return
        const returns = this.positions.map(p => p.return_percent);
        const avgReturn = returns.reduce((a, b) => a + b, 0) / returns.length;
        const stdDev = Math.sqrt(returns.map(x => Math.pow(x - avgReturn, 2)).reduce((a, b) => a + b, 0) / returns.length);

        // Sharpe = (Mean Return - RiskFree) / StdDev. Assume RiskFree=0 for crypto/short-term
        const sharpeRatio = stdDev > 0 ? (avgReturn / stdDev) : 0;

        return {
            total_trades: totalTrades,
            winning_trades: winningTrades.length,
            losing_trades: losingTrades.length,
            win_rate: winRate,
            total_profit: totalProfit,
            total_loss: totalLoss,
            net_profit: netProfit,
            net_profit_percent: netProfitPercent,
            final_equity: finalEquity,
            total_fees: this.positions.reduce((sum, p) => sum + (p.total_fees || 0), 0),
            max_drawdown: maxDrawdown,
            sharpe_ratio: sharpeRatio,

            trades: this.positions,
            equity_curve: this.equityInterim
            // Can add more stats: avg_win, avg_loss, profit_factor...
        };
    }
}

module.exports = BacktestEngine;

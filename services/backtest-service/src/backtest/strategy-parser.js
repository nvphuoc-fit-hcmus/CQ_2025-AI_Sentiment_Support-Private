/**
 * Strategy Parser
 * Parses JSON strategy definition and evaluates conditions
 */

const TechnicalIndicators = require('./indicators');

class StrategyParser {
    /**
     * Evaluate a strategy against current market context
     * @param {Object} strategy - Strategy definition
     * @param {Object} context - Market context (indicators, prediction, news, price)
     * @returns {string} Signal: 'BUY', 'SELL', or 'HOLD'
     */
    static evaluateStrategy(strategy, context) {
        const { conditions, logic, action } = strategy;

        if (!conditions || conditions.length === 0) {
            return 'HOLD';
        }

        // Evaluate each condition
        const results = conditions.map(condition => {
            return this.evaluateCondition(condition, context);
        });

        // Apply logic (AND/OR)
        let conditionMet = false;

        if (logic === 'AND') {
            conditionMet = results.every(r => r);
        } else if (logic === 'OR') {
            conditionMet = results.some(r => r);
        } else {
            // Default to AND
            conditionMet = results.every(r => r);
        }

        return conditionMet ? action : 'HOLD';
    }

    /**
     * Evaluate a single condition
     * @param {Object} condition - Condition definition
     * @param {Object} context - Market context
     * @returns {boolean} True if condition is met
     */
    static evaluateCondition(condition, context) {
        const { type, name, field, operator, value } = condition;

        let actualValue;

        // Get the actual value based on condition type
        if (type === 'indicator') {
            actualValue = context.indicators[name.toLowerCase()];
        } else if (type === 'ai') {
            if (!context.prediction) return false;
            const aliases = {
                direction_1h: ['direction_1h', 'forecast.next_1h.direction', 'direction'],
                confidence_1h: ['confidence_1h', 'forecast.next_1h.confidence', 'confidence'],
                direction_4h: ['direction_4h', 'forecast.next_4h.direction'],
                confidence_4h: ['confidence_4h', 'forecast.next_4h.confidence'],
            };
            const candidatePaths = aliases[field] || [field];
            for (const path of candidatePaths) {
                actualValue = this.getNestedValue(context.prediction, path);
                if (actualValue !== undefined && actualValue !== null) break;
            }
        } else if (type === 'price') {
            actualValue = context.currentPrice;
        } else if (type === 'news') {
            if (!context.news || context.news.length === 0) {
                if (process.env.BACKTEST_DEBUG === 'true') console.log('[STRATEGY] No news data available');
                return false;
            }
            // Aggregate news sentiment
            const avgSentiment = context.news.reduce((sum, n) => sum + (n.sentiment_score || 0), 0) / context.news.length;
            actualValue = avgSentiment;

            // Debug logging
            if (process.env.BACKTEST_DEBUG === 'true' && context.news.length > 0) {
                console.log(`[STRATEGY] News count: ${context.news.length}, Avg sentiment: ${avgSentiment.toFixed(4)}, Condition: ${operator} ${value}`);
            }
        } else {
            return false;
        }

        if (actualValue === undefined || actualValue === null) {
            return false;
        }

        // Evaluate operator
        const result = this.evaluateOperator(actualValue, operator, value);

        // Debug logging for news conditions
        if (process.env.BACKTEST_DEBUG === 'true' && type === 'news') {
            console.log(`[STRATEGY] News condition result: ${actualValue.toFixed(4)} ${operator} ${value} = ${result}`);
        }

        return result;
    }

    /**
     * Get nested value from object using dot notation
     * @param {Object} obj - Object
     * @param {string} path - Path (e.g., 'forecast.next_1h.confidence')
     * @returns {any} Value
     */
    static getNestedValue(obj, path) {
        return path.split('.').reduce((current, key) => current?.[key], obj);
    }

    /**
     * Evaluate comparison operator
     * @param {number} actual - Actual value
     * @param {string} operator - Operator (>, <, >=, <=, ==, !=)
     * @param {number} expected - Expected value
     * @returns {boolean} Result
     */
    static evaluateOperator(actual, operator, expected) {
        // Try numeric comparison first
        const numActual = Number(actual);
        const numExpected = Number(expected);

        // If both can be converted to valid numbers, use numeric comparison
        if (!isNaN(numActual) && !isNaN(numExpected)) {
            switch (operator) {
                case '>':
                    return numActual > numExpected;
                case '<':
                    return numActual < numExpected;
                case '>=':
                    return numActual >= numExpected;
                case '<=':
                    return numActual <= numExpected;
                case '==':
                case '=':
                    return Math.abs(numActual - numExpected) < 0.0001; // Float comparison
                case '!=':
                    return Math.abs(numActual - numExpected) >= 0.0001;
                case 'crosses_above':
                    // Requires historical context - simplified for now
                    return numActual > numExpected;
                case 'crosses_below':
                    return numActual < numExpected;
                default:
                    return false;
            }
        }

        // Fallback to string comparison for non-numeric values
        const act = String(actual).toUpperCase();
        const exp = String(expected).toUpperCase();

        switch (operator) {
            case '==':
            case '=':
                return act === exp;
            case '!=':
                return act !== exp;
            default:
                return false; // >, < not supported for strings
        }
    }

    /**
     * Validate strategy configuration
     * @param {Object} strategy - Strategy to validate
     * @returns {Object} { valid: boolean, errors: Array }
     */
    static validateStrategy(strategy) {
        const errors = [];

        if (!strategy.name || strategy.name.trim() === '') {
            errors.push('Strategy name is required');
        }

        if (!strategy.conditions || !Array.isArray(strategy.conditions)) {
            errors.push('Conditions must be an array');
        } else if (strategy.conditions.length === 0) {
            errors.push('At least one condition is required');
        } else {
            strategy.conditions.forEach((cond, idx) => {
                if (!cond.type) {
                    errors.push(`Condition ${idx + 1}: type is required`);
                }
                if (!cond.operator) {
                    errors.push(`Condition ${idx + 1}: operator is required`);
                }
                if (cond.value === undefined || cond.value === null) {
                    errors.push(`Condition ${idx + 1}: value is required`);
                }
            });
        }

        if (!['AND', 'OR'].includes(strategy.logic)) {
            errors.push('Logic must be AND or OR');
        }

        if (!['BUY', 'SELL'].includes(strategy.action)) {
            errors.push('Action must be BUY or SELL');
        }

        return {
            valid: errors.length === 0,
            errors
        };
    }
}

module.exports = StrategyParser;

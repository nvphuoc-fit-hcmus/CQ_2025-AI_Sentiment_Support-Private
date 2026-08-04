-- Backtest Service Database Schema
-- Create database: backtest_db

-- Backtest Results Table
CREATE TABLE IF NOT EXISTS backtest_results (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    
    -- Strategy Info
    strategy_name VARCHAR(255) NOT NULL,
    strategy_config JSONB NOT NULL,
    
    -- Test Parameters
    symbol VARCHAR(20) NOT NULL,
    start_date TIMESTAMPTZ NOT NULL,
    end_date TIMESTAMPTZ NOT NULL,
    initial_capital DOUBLE PRECISION DEFAULT 10000,
    
    -- Performance Metrics
    total_trades INTEGER DEFAULT 0,
    winning_trades INTEGER DEFAULT 0,
    losing_trades INTEGER DEFAULT 0,
    win_rate DOUBLE PRECISION,
    
    total_profit DOUBLE PRECISION DEFAULT 0,
    total_loss DOUBLE PRECISION DEFAULT 0,
    net_profit DOUBLE PRECISION DEFAULT 0,
    net_profit_percent DOUBLE PRECISION,
    
    max_drawdown DOUBLE PRECISION,
    max_drawdown_duration INTEGER,
    sharpe_ratio DOUBLE PRECISION,
    profit_factor DOUBLE PRECISION,
    
    avg_win DOUBLE PRECISION,
    avg_loss DOUBLE PRECISION,
    largest_win DOUBLE PRECISION,
    largest_loss DOUBLE PRECISION,
    avg_trade_duration INTEGER,
    
    -- Detailed Data
    trades JSONB,
    equity_curve JSONB,
    
    -- Execution Info
    execution_time_ms INTEGER,
    data_points_analyzed INTEGER,
    news_count INTEGER DEFAULT 0,
    news_timeline JSONB DEFAULT '[]'::jsonb
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_backtest_user_created ON backtest_results (user_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_backtest_symbol ON backtest_results (symbol);
CREATE INDEX IF NOT EXISTS idx_backtest_performance ON backtest_results (win_rate DESC, net_profit DESC);

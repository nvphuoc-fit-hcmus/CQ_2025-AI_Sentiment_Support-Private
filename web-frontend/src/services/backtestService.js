import axios from 'axios';

const API_BASE = import.meta.env.VITE_API_URL || window.location.origin;

const backtestService = {
    // Run backtest
    runBacktest: async (payload) => {
        // payload: { strategy, symbol, start_date, end_date, initial_capital }
        const token = localStorage.getItem('token');
        const response = await axios.post(`${API_BASE}/backtest-api/v1/backtest/run`, payload, {
            headers: {
                Authorization: `Bearer ${token}`
            }
        });
        return response.data;
    },

    // Get history
    getHistory: async () => {
        const token = localStorage.getItem('token');
        const response = await axios.get(`${API_BASE}/backtest-api/v1/backtest/history`, {
            headers: {
                Authorization: `Bearer ${token}`
            }
        });
        return response.data;
    },

    // Get detail
    getDetail: async (id) => {
        const token = localStorage.getItem('token');
        const response = await axios.get(`${API_BASE}/backtest-api/v1/backtest/${id}`, {
            headers: {
                Authorization: `Bearer ${token}`
            }
        });
        return response.data;
    }
};

export default backtestService;

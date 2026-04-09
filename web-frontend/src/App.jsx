import React, { useEffect } from 'react';
import { BrowserRouter, Routes, Route, Navigate, useNavigate } from 'react-router-dom';
import useStore from './store';
import Login from './components/Login.jsx';
import TradingDashboard from './components/TradingDashboard';
import Navbar from './components/Navbar';
import Sidebar from './components/Sidebar';
import LeftToolbar from './components/LeftToolbar';
import InvestmentSimulator from './components/InvestmentSimulator';
import BacktestDashboard from './components/Backtesting';
import { ToastProvider } from './components/ToastProvider';
import { ThemeProvider, SettingsPanel } from './components/ThemeProvider';
import './index.css';

import UpgradeModal from './components/UpgradeModal';
import AdminDashboard from './components/Admin/Dashboard';
import Forbidden from './components/Forbidden';

// Protected Route cho User thường
function ProtectedUserRoute({ children }) {
  const { token } = useStore();

  if (!token) {
    return <Navigate to="/login" replace />;
  }

  return children;
}

// Protected Route cho Admin
function ProtectedAdminRoute({ children }) {
  const { token, user } = useStore();

  if (!token) {
    return <Navigate to="/login" replace />;
  }

  const isAdmin = user && (user.role === 'admin' || user.role === 'Admin');

  if (!isAdmin) {
    return <Forbidden />;
  }

  return children;
}

// Component cho trang User (Trading + Investment)
function UserApp() {
  const [currentPage, setCurrentPage] = React.useState('trading');
  const [showUpgradeModal, setShowUpgradeModal] = React.useState(false);
  const [showSettings, setShowSettings] = React.useState(false);

  useEffect(() => {
    const handleNavigate = (e) => setCurrentPage(e.detail.page);
    window.addEventListener('navigate', handleNavigate);

    const handleShowUpgrade = () => setShowUpgradeModal(true);
    window.addEventListener('showUpgradeModal', handleShowUpgrade);

    const handleShowSettings = () => setShowSettings(true);
    window.addEventListener('showSettings', handleShowSettings);

    return () => {
      window.removeEventListener('navigate', handleNavigate);
      window.removeEventListener('showUpgradeModal', handleShowUpgrade);
      window.removeEventListener('showSettings', handleShowSettings);
    };
  }, []);

  return (
    <div className="app-container">
      <Navbar currentPage={currentPage} onNavigate={setCurrentPage} />

      {currentPage === 'trading' ? (
        <div className="main-content">
          <LeftToolbar currentPage={currentPage} onNavigate={setCurrentPage} />
          <div className="chart-area">
            <TradingDashboard />
          </div>
          <Sidebar />
        </div>
      ) : (
        <div className="full-page-content">
          {currentPage === 'investment' && <InvestmentSimulator />}
          {currentPage === 'backtesting' && <BacktestDashboard />}
        </div>
      )}

      {showUpgradeModal && <UpgradeModal onClose={() => setShowUpgradeModal(false)} />}
      <SettingsPanel isOpen={showSettings} onClose={() => setShowSettings(false)} />
    </div>
  );
}

// Component cho trang Admin
function AdminApp() {
  return (
    <div className="admin-app-container" style={{ minHeight: '100vh', background: '#121212' }}>
      <AdminDashboard />
    </div>
  );
}

// Component Login với auto-redirect
function LoginPage() {
  const { token, user } = useStore();
  const navigate = useNavigate();

  useEffect(() => {
    if (token && user) {
      const isAdmin = user.role === 'admin' || user.role === 'Admin';
      if (isAdmin) {
        navigate('/admin/dashboard', { replace: true });
      } else {
        navigate('/', { replace: true });
      }
    }
  }, [token, user, navigate]);

  return <Login />;
}

function App() {
  const { token, connectSocket, init } = useStore();

  useEffect(() => {
    init(); // Restore user from token
    if (token) connectSocket();
  }, [token, connectSocket, init]);

  return (
    <BrowserRouter>
      <ThemeProvider>
        <ToastProvider>
          <Routes>
            {/* Login Route */}
            <Route path="/login" element={<LoginPage />} />

            {/* Admin Routes */}
            <Route
              path="/admin/dashboard"
              element={
                <ProtectedAdminRoute>
                  <AdminApp />
                </ProtectedAdminRoute>
              }
            />

            {/* User Routes */}
            <Route
              path="/"
              element={
                <ProtectedUserRoute>
                  <UserApp />
                </ProtectedUserRoute>
              }
            />

            {/* Fallback */}
            <Route path="*" element={<Navigate to="/" replace />} />
          </Routes>
        </ToastProvider>
      </ThemeProvider>
    </BrowserRouter>
  );
}

export default App;

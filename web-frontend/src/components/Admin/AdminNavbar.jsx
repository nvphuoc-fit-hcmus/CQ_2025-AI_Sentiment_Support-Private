import React from 'react';
import useStore from '../../store';
import { Shield, LogOut } from 'lucide-react';
import './AdminNavbar.css';

export default function AdminNavbar() {
    const { logout } = useStore();

    const handleLogout = () => {
        logout();
        window.location.href = '/login';
    };

    return (
        <div className="admin-navbar">
            <div className="admin-nav-left">
                <div className="admin-brand">
                    <Shield className="admin-brand-icon" size={28} />
                    <span>Aegis Admin</span>
                </div>
            </div>

            <div className="admin-nav-right">
                <button className="admin-logout-btn" onClick={handleLogout} title="Đăng xuất">
                    <LogOut size={18} />
                    <span>Đăng xuất</span>
                </button>
            </div>
        </div>
    );
}

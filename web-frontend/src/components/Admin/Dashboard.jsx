import React, { useEffect, useState } from 'react';
import useStore from '../../store';
import './Dashboard.css';
import AdminNavbar from './AdminNavbar';
import ConfirmModal from './ConfirmModal';
import { useToast } from '../ToastProvider';
import { Shield, ShieldAlert, BadgeCheck, Lock, Unlock, ArrowUpCircle, ArrowDownCircle, ChevronLeft, ChevronRight } from 'lucide-react';

const AdminDashboard = () => {
    const { authFetch, user } = useStore();
    const { showToast } = useToast();
    const [users, setUsers] = useState([]);
    const [pagination, setPagination] = useState({ page: 1, limit: 10, totalPages: 1, total: 0 });
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState('');

    // Modal state
    const [confirmModal, setConfirmModal] = useState({
        isOpen: false,
        title: '',
        message: '',
        type: 'warning',
        onConfirm: () => { }
    });

    const fetchUsers = async (page = 1) => {
        try {
            setLoading(true);
            const res = await authFetch(`/admin/users?page=${page}&limit=${pagination.limit}`);
            if (res.ok) {
                const data = await res.json();
                setUsers(data.users || []);
                if (data.pagination) {
                    setPagination(prev => ({ ...prev, ...data.pagination }));
                }
            } else {
                setError('Không thể tải danh sách người dùng');
            }
        } catch (err) {
            setError(err.message);
        } finally {
            setLoading(false);
        }
    };

    useEffect(() => {
        fetchUsers(1);
    }, []);

    const handleAction = async (userId, action, confirmMsg, actionType = 'warning') => {
        setConfirmModal({
            isOpen: true,
            title: action === 'ban' ? 'Cảnh báo' : 'Xác nhận',
            message: confirmMsg,
            type: actionType,
            onConfirm: async () => {
                try {
                    let url = '';
                    let body = {};
                    let successMsg = '';

                    if (action === 'promote_vip') {
                        url = `/admin/users/${userId}/role`;
                        body = { role: 'VIP' };
                        successMsg = 'Đã nâng cấp VIP thành công!';
                    } else if (action === 'demote_regular') {
                        url = `/admin/users/${userId}/role`;
                        body = { role: 'user' };
                        successMsg = 'Đã hạ xuống tài khoản thường!';
                    } else if (action === 'ban') {
                        url = `/admin/users/${userId}/status`;
                        body = { status: 'Banned' };
                        successMsg = 'Đã khóa tài khoản!';
                    } else if (action === 'unlock') {
                        url = `/admin/users/${userId}/status`;
                        body = { status: 'Active' };
                        successMsg = 'Đã mở khóa tài khoản!';
                    }

                    const res = await authFetch(url, {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify(body)
                    });

                    if (res.ok) {
                        showToast(successMsg, 'success');
                        fetchUsers(pagination.page); // Refresh current page
                    } else {
                        const err = await res.json();
                        showToast(`Thao tác thất bại: ${err.message || err.error}`, 'error');
                    }
                } catch (err) {
                    console.error(err);
                    showToast('Thao tác thất bại! Vui lòng thử lại.', 'error');
                }
            }
        });
    };

    const handlePageChange = (newPage) => {
        if (newPage >= 1 && newPage <= pagination.totalPages) {
            fetchUsers(newPage);
        }
    };

    return (
        <>
            <AdminNavbar />
            <div className="admin-dashboard-container">
                <div className="admin-header">
                    <h2><Shield size={28} style={{ marginRight: 10, verticalAlign: 'middle' }} />Quản trị Người dùng</h2>
                    <div className="admin-stats">
                        Tổng số người dùng: <strong>{pagination.total}</strong>
                    </div>
                </div>

                {error && <div className="error-msg">{error}</div>}

                {loading ? (
                    <div className="loading">Đang tải dữ liệu...</div>
                ) : (
                    <>
                        <div className="table-wrapper">
                            <table className="user-table">
                                <thead>
                                    <tr>
                                        <th>ID</th>
                                        <th>Email</th>
                                        <th>Quyền (Role)</th>
                                        <th>Trạng thái</th>
                                        <th>Hành động</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {users.map(u => (
                                        <tr key={u.id}>
                                            <td className="monospace-font" title={u.id}>{u.id.substring(0, 8)}...</td>
                                            <td>{u.email}</td>
                                            <td>
                                                <span className={`badge role-${u.role.toLowerCase()}`}>{u.role}</span>
                                            </td>
                                            <td>
                                                <span className={`badge status-${u.status.toLowerCase()}`}>{u.status}</span>
                                            </td>
                                            <td>
                                                <div className="action-buttons">
                                                    {/* Prevent admin from modifying their own account */}
                                                    {user && u.id === user.id ? (
                                                        <span className="text-muted" style={{ fontSize: '0.85em', fontStyle: 'italic' }}>
                                                            (Tài khoản của bạn)
                                                        </span>
                                                    ) : (
                                                        <>
                                                            {u.status !== 'Banned' && u.status !== 'Locked' && (
                                                                <>
                                                                    {false && (
                                                                        <button
                                                                            className="btn-action btn-vip"
                                                                            onClick={() => handleAction(u.id, 'promote_vip', `Bạn có chắc chắn muốn nâng cấp VIP cho ${u.email}?`, 'success')}
                                                                            title="Nâng lên VIP"
                                                                        >
                                                                            <ArrowUpCircle size={14} /> VIP
                                                                        </button>
                                                                    )}
                                                                    {false && (
                                                                        <button
                                                                            className="btn-action btn-regular"
                                                                            onClick={() => handleAction(u.id, 'demote_regular', `Bạn có chắc chắn muốn hạ cấp ${u.email} xuống thường?`, 'info')}
                                                                            title="Hạ xuống thường"
                                                                        >
                                                                            <ArrowDownCircle size={14} /> Thường
                                                                        </button>
                                                                    )}
                                                                    {u.role !== 'admin' && u.role !== 'Admin' && (
                                                                        <button
                                                                            className="btn-action btn-ban"
                                                                            onClick={() => handleAction(u.id, 'ban', `CẢNH BÁO: Bạn có chắc chắn muốn KHÓA tài khoản ${u.email}?`, 'danger')}
                                                                            title="Khóa tài khoản"
                                                                        >
                                                                            <Lock size={14} /> Khóa
                                                                        </button>
                                                                    )}
                                                                </>
                                                            )}
                                                            {(u.status === 'Banned' || u.status === 'Locked') && (
                                                                <button
                                                                    className="btn-action btn-unlock"
                                                                    onClick={() => handleAction(u.id, 'unlock', `Mở khóa tài khoản cho ${u.email}?`, 'success')}
                                                                    title="Mở khóa tài khoản"
                                                                >
                                                                    <Unlock size={14} /> Mở khóa
                                                                </button>
                                                            )}
                                                        </>
                                                    )}
                                                </div>
                                            </td>
                                        </tr>
                                    ))}
                                </tbody>
                            </table>
                        </div>

                        <div className="pagination-controls">
                            <button
                                disabled={pagination.page === 1}
                                onClick={() => handlePageChange(pagination.page - 1)}
                                className="btn-page"
                            >
                                <ChevronLeft size={16} /> Trước
                            </button>
                            <span className="page-info">
                                Trang {pagination.page} / {pagination.totalPages}
                            </span>
                            <button
                                disabled={pagination.page === pagination.totalPages}
                                onClick={() => handlePageChange(pagination.page + 1)}
                                className="btn-page"
                            >
                                Sau <ChevronRight size={16} />
                            </button>
                        </div>
                    </>
                )}
            </div>

            <ConfirmModal
                isOpen={confirmModal.isOpen}
                onClose={() => setConfirmModal({ ...confirmModal, isOpen: false })}
                onConfirm={confirmModal.onConfirm}
                title={confirmModal.title}
                message={confirmModal.message}
                type={confirmModal.type}
            />
        </>
    );
};

export default AdminDashboard;

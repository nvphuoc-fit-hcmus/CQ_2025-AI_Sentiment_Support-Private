const express = require('express');
const { pool } = require('../db');
const authMiddleware = require('../middleware/auth');
const AuditLogger = require('../utils/AuditLogger');

const router = express.Router();

// Middleware to ensure user is admin
const requireAdmin = (req, res, next) => {
    if (!req.user || req.user.role !== 'admin' && req.user.role !== 'Admin') {
        return res.status(403).json({ error: 'forbidden', message: 'Admin access required' });
    }
    next();
};

router.use(authMiddleware);
router.use(requireAdmin);

// List all users with pagination
router.get('/users', async (req, res) => {
    try {
        const page = parseInt(req.query.page) || 1;
        const limit = parseInt(req.query.limit) || 20;
        const offset = (page - 1) * limit;

        const countRes = await pool.query('SELECT COUNT(*) FROM users');
        const total = parseInt(countRes.rows[0].count);

        const r = await pool.query(
            'SELECT id, email, role, status, is_vip, created_at FROM users ORDER BY created_at DESC LIMIT $1 OFFSET $2',
            [limit, offset]
        );

        res.json({
            users: r.rows,
            pagination: {
                total,
                page,
                limit,
                totalPages: Math.ceil(total / limit)
            }
        });
    } catch (err) {
        console.error(err);
        res.status(500).json({ error: 'db_error' });
    }
});

// Change user role
router.post('/users/:id/role', async (req, res) => {
    res.status(410).json({ error: 'vip_removed', message: 'Chức năng VIP đã được gỡ bỏ.' });
});

// Change user status (Active, Banned)
router.post('/users/:id/status', async (req, res) => {
    const { id } = req.params;
    const { status } = req.body; // 'Active', 'Banned', 'Locked'

    if (!['Active', 'Banned', 'Locked'].includes(status)) {
        return res.status(400).json({ error: 'invalid_status', message: 'Status must be Active, Banned, or Locked' });
    }

    try {
        await pool.query(
            'UPDATE users SET status = $1 WHERE id = $2',
            [status, id]
        );

        AuditLogger.logSecurityEvent('ADMIN_UPDATE_STATUS', 'INFO', {
            admin_id: req.user.sub,
            target_user_id: id,
            new_status: status
        });

        res.json({ success: true, message: `User status updated to ${status}` });
    } catch (err) {
        console.error(err);
        res.status(500).json({ error: 'db_error' });
    }
});

module.exports = router;

const express = require('express');
const bcrypt = require('bcryptjs');
const crypto = require('crypto');
const { pool } = require('../db');
const { signToken, getPublicKeyPem, verifyToken, getJWKS } = require('../utils/jwt');
const TokenBlacklist = require('../services/TokenBlacklist');
const RefreshToken = require('../services/RefreshToken');
const AuditLogger = require('../utils/AuditLogger');
const authMiddleware = require('../middleware/auth');
const { publishUserSettingsUpdate } = require('../utils/kafkaProducer');
const { sendVerificationEmail, sendPasswordChangeOtpEmail, sendPasswordChangedEmail } = require('../services/EmailService');

const router = express.Router();
router.use(express.json());

const hashToken = (token) => crypto.createHash('sha256').update(token).digest('hex');
const createOtp = () => String(crypto.randomInt(0, 1000000)).padStart(6, '0');

// Register using email + password. New accounts must verify their email.
router.post('/register', async (req, res) => {
  const { email, password, display_name } = req.body || {};
  if (!email || !password) return res.status(400).json({ error: 'email and password required' });

  // basic email normalization
  const normEmail = String(email).trim().toLowerCase();
  try {
    const hash = await bcrypt.hash(password, 12);
    const verificationOtp = createOtp();
    const verificationHash = hashToken(verificationOtp);
    const r = await pool.query(
      `INSERT INTO users(
         email, password, display_name, is_vip, role, status, email_verified,
         email_verification_token_hash, email_verification_expires_at
       ) VALUES($1, $2, $3, false, 'Regular', 'Active', false, $4, NOW() + INTERVAL '10 minutes')
       RETURNING id, email, display_name, created_at, role, status, email_verified`,
      [normEmail, hash, String(display_name || '').trim() || null, verificationHash]
    );
    const user = r.rows[0];
    const mail = await sendVerificationEmail(normEmail, verificationOtp);
    res.status(201).json({
      user,
      verification_required: true,
      message: 'Mã OTP 6 số đã được gửi đến email của bạn.',
      ...(process.env.NODE_ENV !== 'production' && !mail.delivered
        ? { development_otp: mail.developmentOtp }
        : {}),
    });
  } catch (err) {
    if (err.code === '23505') return res.status(409).json({ error: 'email_exists' });
    console.error(err);
    res.status(500).json({ error: 'db_error' });
  }
});

router.post('/verify-email', async (req, res) => {
  const token = String(req.body?.otp || req.body?.token || '').trim();
  const email = String(req.body?.email || '').trim().toLowerCase();
  if (!token) return res.status(400).json({ error: 'verification_otp_required' });
  const result = await pool.query(
    `UPDATE users
     SET email_verified = true,
         email_verification_token_hash = NULL,
         email_verification_expires_at = NULL
     WHERE email_verification_token_hash = $1
       AND email_verification_expires_at > NOW()
       AND ($2 = '' OR lower(email) = lower($2))
     RETURNING id, email`,
    [hashToken(token), email]
  );
  if (!result.rowCount) return res.status(400).json({ error: 'verification_otp_invalid_or_expired' });
  res.json({ verified: true, message: 'Email đã được xác thực. Bạn có thể đăng nhập.' });
});

router.post('/resend-verification', async (req, res) => {
  const normEmail = String(req.body?.email || '').trim().toLowerCase();
  const userResult = await pool.query(
    'SELECT id, email_verified FROM users WHERE lower(email) = lower($1) LIMIT 1',
    [normEmail]
  );
  const user = userResult.rows[0];
  // Deliberately return the same response to avoid account enumeration.
  if (!user || user.email_verified) return res.json({ message: 'Nếu tài khoản hợp lệ, email xác thực đã được gửi.' });
  const token = createOtp();
  await pool.query(
    `UPDATE users SET email_verification_token_hash=$1,
      email_verification_expires_at=NOW() + INTERVAL '10 minutes' WHERE id=$2`,
    [hashToken(token), user.id]
  );
  const mail = await sendVerificationEmail(normEmail, token);
  res.json({
    message: 'Email xác thực đã được gửi lại.',
    ...(process.env.NODE_ENV !== 'production' && !mail.delivered
      ? { development_otp: mail.developmentOtp }
      : {}),
  });
});

// Password recovery by email OTP.
router.post('/forgot-password/request-otp', async (req, res) => {
  const normEmail = String(req.body?.email || '').trim().toLowerCase();
  if (!normEmail) return res.status(400).json({ error: 'email_required' });
  const result = await pool.query(
    `SELECT id,email FROM users WHERE lower(email)=lower($1)
     AND COALESCE(email_verified,true)=true
     AND lower(COALESCE(status,'active'))='active' LIMIT 1`,
    [normEmail]
  );
  const user = result.rows[0];
  if (!user) {
    return res.status(404).json({
      error: 'recovery_email_not_found',
      message: 'Email không tồn tại trong hệ thống, chưa được xác thực hoặc tài khoản không hoạt động.',
    });
  }

  const otp = createOtp();
  await pool.query(
    `UPDATE users SET password_reset_otp_hash=$1,
     password_reset_otp_expires_at=NOW()+INTERVAL '10 minutes' WHERE id=$2`,
    [hashToken(otp), user.id]
  );
  const mail = await sendPasswordChangeOtpEmail(user.email, otp);
  res.json({
    otp_required: true,
    message: 'Mã OTP 6 số đã được gửi và có hiệu lực trong 10 phút.',
    ...(process.env.NODE_ENV !== 'production' && !mail.delivered
      ? { development_otp: mail.developmentOtp } : {}),
  });
});

router.post('/forgot-password/reset', async (req, res) => {
  const normEmail = String(req.body?.email || '').trim().toLowerCase();
  const otp = String(req.body?.otp || '').trim();
  const newPassword = String(req.body?.new_password || '');
  if (!normEmail || !otp || !newPassword) {
    return res.status(400).json({ error: 'email_otp_and_new_password_required' });
  }
  if (!/^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[\W_]).{8,}$/.test(newPassword)) {
    return res.status(400).json({ error: 'weak_password' });
  }
  const result = await pool.query(
    `SELECT id,email FROM users WHERE lower(email)=lower($1)
     AND password_reset_otp_hash=$2 AND password_reset_otp_expires_at>NOW()
     AND COALESCE(email_verified,true)=true
     AND lower(COALESCE(status,'active'))='active' LIMIT 1`,
    [normEmail, hashToken(otp)]
  );
  if (!result.rowCount) {
    return res.status(400).json({ error: 'password_reset_otp_invalid_or_expired' });
  }
  const passwordHash = await bcrypt.hash(newPassword, 12);
  await pool.query(
    `UPDATE users SET password=$1,password_reset_otp_hash=NULL,
     password_reset_otp_expires_at=NULL WHERE id=$2`,
    [passwordHash, result.rows[0].id]
  );
  await sendPasswordChangedEmail(result.rows[0].email);
  res.json({ changed: true, message: 'Mật khẩu đã được đặt lại.' });
});

// Login using email + password
router.post('/login', async (req, res) => {
  const { email, password } = req.body || {};
  if (!email || !password) return res.status(400).json({ error: 'email and password required' });

  const normEmail = String(email).trim().toLowerCase();
  const ipAddress = req.ip || req.connection.remoteAddress;
  const userAgent = req.headers['user-agent'] || 'unknown';

  try {
    const r = await pool.query('SELECT id, email, password, is_vip, role, status, COALESCE(email_verified, true) AS email_verified FROM users WHERE lower(email) = lower($1) LIMIT 1', [normEmail]);
    const row = r.rows[0];

    if (!row) {
      // Log failed authentication attempt
      AuditLogger.logAuthAttempt(null, normEmail, ipAddress, userAgent, false, 'user_not_found');
      return res.status(401).json({ error: 'invalid_credentials' });
    }

    // Check status
    if (row.status === 'Banned' || row.status === 'Locked') {
      AuditLogger.logAuthAttempt(row.id, normEmail, ipAddress, userAgent, false, 'account_locked_or_banned');
      return res.status(403).json({ error: 'account_locked_or_banned', message: 'Your account is locked or banned.' });
    }

    const ok = await bcrypt.compare(password, row.password);
    if (!ok) {
      // Log failed authentication attempt
      AuditLogger.logAuthAttempt(row.id, normEmail, ipAddress, userAgent, false, 'invalid_password');
      return res.status(401).json({ error: 'invalid_credentials' });
    }
    if (!row.email_verified) {
      return res.status(403).json({ error: 'email_not_verified', message: 'Vui lòng xác thực email trước khi đăng nhập.' });
    }

    // Include is_vip, role, status in the token
    const token = signToken({
      sub: row.id,
      email: row.email,
      is_vip: !!row.is_vip,
      role: row.role,
      status: row.status
    });

    // Phase 2: Create refresh token
    const refreshTokenData = await RefreshToken.createRefreshToken(
      row.id,
      ipAddress,
      userAgent
    );

    // Log successful authentication
    AuditLogger.logAuthAttempt(row.id, row.email, ipAddress, userAgent, true);

    // Set refresh token in httpOnly cookie
    const refreshTokenDays = parseInt(process.env.REFRESH_TOKEN_DAYS || '7', 10);
    res.cookie('refresh_token', refreshTokenData.token, {
      httpOnly: true,
      secure: process.env.NODE_ENV === 'production', // HTTPS only in production
      sameSite: 'lax', // Changed from 'strict' to 'lax' for better cross-origin support
      maxAge: refreshTokenDays * 24 * 60 * 60 * 1000, // Days to milliseconds
      path: '/' // Available for all paths
    });

    res.json({
      token,
      is_vip: !!row.is_vip,
      role: row.role,
      status: row.status,
      refresh_token_expires_at: refreshTokenData.expiresAt
    });
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: 'db_error' });
  }
});

// Refresh access token using refresh token
router.post('/refresh', async (req, res) => {
  try {
    // Get refresh token from cookie
    const refreshToken = req.cookies?.refresh_token;

    if (!refreshToken) {
      return res.status(401).json({
        error: {
          code: 'REFRESH_TOKEN_MISSING',
          message: 'Refresh token not provided',
          timestamp: new Date().toISOString()
        }
      });
    }

    // Validate refresh token
    const tokenRecord = await RefreshToken.validateRefreshToken(refreshToken);

    if (!tokenRecord) {
      return res.status(401).json({
        error: {
          code: 'REFRESH_TOKEN_INVALID',
          message: 'Refresh token is invalid or expired',
          timestamp: new Date().toISOString()
        }
      });
    }

    // Phase 2: Replay attack detection
    // Check if this token was already used and rotated
    const isReused = await RefreshToken.isTokenReused(refreshToken);

    if (isReused) {
      // SECURITY: Token reuse detected - revoke all user tokens
      await RefreshToken.revokeAllUserTokens(tokenRecord.user_id, 'replay_attack_detected');

      // Log critical security event
      AuditLogger.logSecurityEvent('REPLAY_ATTACK_DETECTED', 'CRITICAL', {
        user_id: tokenRecord.user_id,
        ip_address: req.ip || req.connection.remoteAddress,
        user_agent: req.headers['user-agent']
      });

      return res.status(401).json({
        error: {
          code: 'REPLAY_ATTACK_DETECTED',
          message: 'All tokens revoked due to security incident',
          details: 'Refresh token reuse detected',
          timestamp: new Date().toISOString()
        }
      });
    }

    // Get user data
    const userResult = await pool.query(
      'SELECT id, email, is_vip, role, status FROM users WHERE id = $1 LIMIT 1',
      [tokenRecord.user_id]
    );

    if (userResult.rows.length === 0) {
      return res.status(404).json({
        error: {
          code: 'USER_NOT_FOUND',
          message: 'User not found',
          timestamp: new Date().toISOString()
        }
      });
    }

    const user = userResult.rows[0];

    // Check status logic on refresh too
    if (user.status === 'Banned' || user.status === 'Locked') {
      return res.status(403).json({ error: 'account_locked_or_banned', message: 'Your account is locked or banned.' });
    }

    // Generate new access token
    const newAccessToken = signToken({
      sub: user.id,
      email: user.email,
      is_vip: !!user.is_vip,
      role: user.role,
      status: user.status
    });

    // Phase 2: Token rotation - revoke old refresh token and create new one
    const ipAddress = req.ip || req.connection.remoteAddress;
    const userAgent = req.headers['user-agent'] || 'unknown';

    // Revoke old refresh token
    await RefreshToken.revokeRefreshTokenByHash(tokenRecord.token_hash, 'token_rotation');

    // Create new refresh token
    const newRefreshTokenData = await RefreshToken.createRefreshToken(
      user.id,
      ipAddress,
      userAgent
    );

    // Update last_used timestamp for tracking
    await RefreshToken.updateLastUsed(tokenRecord.token_hash);

    // Set new refresh token in cookie
    const refreshTokenDays = parseInt(process.env.REFRESH_TOKEN_DAYS || '7', 10);
    res.cookie('refresh_token', newRefreshTokenData.token, {
      httpOnly: true,
      secure: process.env.NODE_ENV === 'production',
      sameSite: 'lax', // Changed from 'strict' to 'lax' for better cross-origin support
      maxAge: refreshTokenDays * 24 * 60 * 60 * 1000, // Days to milliseconds
      path: '/' // Available for all paths
    });

    // Log token refresh
    AuditLogger.logSecurityEvent('TOKEN_REFRESHED', 'INFO', {
      user_id: user.id,
      ip_address: ipAddress,
      user_agent: userAgent
    });

    res.json({
      token: newAccessToken,
      is_vip: !!user.is_vip,
      role: user.role,
      status: user.status,
      refresh_token_expires_at: newRefreshTokenData.expiresAt,
      message: 'Token refreshed successfully'
    });

  } catch (err) {
    console.error('Refresh token error:', err);
    res.status(500).json({
      error: {
        code: 'INTERNAL_ERROR',
        message: 'Failed to refresh token',
        timestamp: new Date().toISOString()
      }
    });
  }
});

// Logout - revoke current token and refresh token
router.post('/logout', authMiddleware, async (req, res) => {
  try {
    const user = req.user;

    if (!user.jti) {
      return res.status(400).json({
        error: {
          code: 'INVALID_TOKEN',
          message: 'Token does not contain JTI',
          timestamp: new Date().toISOString()
        }
      });
    }

    // Calculate TTL based on token expiration
    const now = Math.floor(Date.now() / 1000);
    const ttl = user.exp - now;

    if (ttl > 0) {
      // Add access token to blacklist
      await TokenBlacklist.addToBlacklist(user.jti, ttl);

      // Log token revocation
      AuditLogger.logTokenRevocation(user.jti, user.sub, 'user_logout');
    }

    // Phase 2: Revoke refresh token if present
    const refreshToken = req.cookies?.refresh_token;
    if (refreshToken) {
      await RefreshToken.revokeRefreshToken(refreshToken, 'user_logout');

      // Clear refresh token cookie
      res.clearCookie('refresh_token', { path: '/' });
    }

    res.json({
      message: 'Logged out successfully',
      timestamp: new Date().toISOString()
    });
  } catch (err) {
    console.error('Logout error:', err);
    res.status(500).json({
      error: {
        code: 'INTERNAL_ERROR',
        message: 'Failed to logout',
        timestamp: new Date().toISOString()
      }
    });
  }
});

router.get('/public-key', (req, res) => {
  res.type('text/plain').send(getPublicKeyPem());
});

// JWKS endpoint for Kong/external JWT validation (Phase 3 - Task 11.2)
// Follows OpenID Connect Discovery specification
// Kong can use this to automatically verify JWT signatures
router.get('/.well-known/jwks.json', (req, res) => {
  try {
    const jwks = getJWKS();
    res.json(jwks);
  } catch (error) {
    console.error('Error generating JWKS:', error);
    res.status(500).json({
      error: {
        code: 'JWKS_GENERATION_ERROR',
        message: 'Failed to generate JWKS',
        timestamp: new Date().toISOString()
      }
    });
  }
});

// Get current user info (requires JWT)
router.get('/me', authMiddleware, async (req, res) => {
  try {
    // authMiddleware already validated token and checked blacklist
    // req.user contains the decoded JWT payload
    const userId = req.user.sub;

    if (!userId) {
      return res.status(401).json({ error: 'invalid_token' });
    }

    const r = await pool.query('SELECT id, email, display_name, role, status, COALESCE(email_verified, true) AS email_verified, created_at FROM users WHERE id = $1 LIMIT 1', [userId]);
    const user = r.rows[0];

    if (!user) {
      return res.status(404).json({ error: 'user_not_found' });
    }

    // Issue a new token with updated claims
    const newToken = signToken({
      sub: user.id,
      email: user.email,
      role: user.role,
      status: user.status
    });

    res.json({
      user: { id: user.id, email: user.email, display_name: user.display_name, email_verified: user.email_verified, role: user.role, status: user.status, created_at: user.created_at },
      token: newToken // Return fresh token
    });
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: 'db_error' });
  }
});

router.patch('/me', authMiddleware, async (req, res) => {
  const displayName = String(req.body?.display_name || '').trim();
  if (displayName.length > 80) return res.status(400).json({ error: 'display_name_too_long' });
  const result = await pool.query(
    `UPDATE users SET display_name=$1 WHERE id=$2
     RETURNING id,email,display_name,role,status,COALESCE(email_verified,true) AS email_verified,created_at`,
    [displayName || null, req.user.sub]
  );
  if (!result.rowCount) return res.status(404).json({ error: 'user_not_found' });
  res.json({ user: result.rows[0] });
});

router.post('/change-password/request-otp', authMiddleware, async (req, res) => {
  const { current_password, new_password } = req.body || {};
  if (!current_password || !new_password) return res.status(400).json({ error: 'passwords_required' });
  if (String(new_password).length < 8) return res.status(400).json({ error: 'password_too_short' });
  const result = await pool.query('SELECT email,password FROM users WHERE id=$1', [req.user.sub]);
  if (!result.rowCount || !(await bcrypt.compare(current_password, result.rows[0].password))) {
    return res.status(400).json({ error: 'current_password_incorrect' });
  }

  const otp = createOtp();
  await pool.query(
    `UPDATE users SET password_change_otp_hash=$1,
      password_change_otp_expires_at=NOW() + INTERVAL '10 minutes' WHERE id=$2`,
    [hashToken(otp), req.user.sub]
  );
  const mail = await sendPasswordChangeOtpEmail(result.rows[0].email, otp);
  res.json({
    otp_required: true,
    message: 'Mã OTP xác nhận đã được gửi đến email của bạn.',
    ...(process.env.NODE_ENV !== 'production' && !mail.delivered
      ? { development_otp: mail.developmentOtp }
      : {}),
  });
});

router.post('/change-password', authMiddleware, async (req, res) => {
  const { current_password, new_password, otp } = req.body || {};
  if (!current_password || !new_password || !otp) return res.status(400).json({ error: 'passwords_and_otp_required' });
  if (String(new_password).length < 8) return res.status(400).json({ error: 'password_too_short' });
  const result = await pool.query(
    `SELECT email,password FROM users WHERE id=$1 AND password_change_otp_hash=$2
      AND password_change_otp_expires_at > NOW()`,
    [req.user.sub, hashToken(String(otp).trim())]
  );
  if (!result.rowCount) return res.status(400).json({ error: 'password_change_otp_invalid_or_expired' });
  if (!(await bcrypt.compare(current_password, result.rows[0].password))) {
    return res.status(400).json({ error: 'current_password_incorrect' });
  }
  const passwordHash = await bcrypt.hash(new_password, 12);
  await pool.query(
    `UPDATE users SET password=$1, password_change_otp_hash=NULL,
      password_change_otp_expires_at=NULL WHERE id=$2`,
    [passwordHash, req.user.sub]
  );
  await RefreshToken.revokeAllUserTokens(req.user.sub, 'password_changed');
  try {
    await sendPasswordChangedEmail(result.rows[0].email);
  } catch (emailError) {
    console.error('[EMAIL] Failed to send password changed notification:', emailError.message);
  }
  res.clearCookie('refresh_token', { path: '/' });
  res.json({ changed: true, message: 'Mật khẩu đã được đổi. Vui lòng đăng nhập lại.' });
});

const sseService = require('../services/sseService');

// SSE Endpoint for realtime user events
router.get('/events/sse', (req, res) => {
  const token = req.query.token;
  if (!token) return res.status(401).end();

  try {
    const payload = verifyToken(token);
    const userId = payload.sub;

    // Set headers for SSE
    res.writeHead(200, {
      'Content-Type': 'text/event-stream',
      'Cache-Control': 'no-cache',
      'Connection': 'keep-alive',
      'X-Accel-Buffering': 'no' // Disable Nginx/Kong buffering
    });

    res.write('retry: 10000\n\n');

    sseService.addClient(userId, res);
  } catch (e) {
    console.error('SSE Auth failed:', e);
    res.status(401).end();
  }
});
// Get notification settings
router.get('/notifications/settings', authMiddleware, async (req, res) => {
  try {
    const userId = req.user.sub;
    const r = await pool.query(
      'SELECT notification_settings FROM users WHERE id = $1 LIMIT 1',
      [userId]
    );

    if (r.rows.length === 0) {
      return res.status(404).json({ error: 'user_not_found' });
    }

    const settings = r.rows[0].notification_settings || {
      prediction_symbols: [],
      investment_enabled: false
    };

    res.json(settings);
  } catch (err) {
    console.error('[NOTIF SETTINGS GET]', err);
    res.status(500).json({ error: 'db_error' });
  }
});

// Update notification settings
router.post('/notifications/settings', authMiddleware, async (req, res) => {
  try {
    const userId = req.user.sub;
    const { type, symbol, enabled } = req.body;

    if (!type) {
      return res.status(400).json({ error: 'type required (PREDICTION or INVESTMENT)' });
    }

    // Get current settings and email
    const r = await pool.query(
      'SELECT email, notification_settings FROM users WHERE id = $1 LIMIT 1',
      [userId]
    );

    if (r.rows.length === 0) {
      return res.status(404).json({ error: 'user_not_found' });
    }

    const userEmail = r.rows[0].email;
    let settings = r.rows[0].notification_settings || {
      prediction_symbols: [],
      investment_enabled: false
    };

    if (type === 'PREDICTION') {
      if (!symbol) {
        return res.status(400).json({ error: 'symbol required for PREDICTION type' });
      }

      if (enabled) {
        // Add symbol if not exists
        if (!settings.prediction_symbols.includes(symbol)) {
          settings.prediction_symbols.push(symbol);
        }
      } else {
        // Remove symbol
        settings.prediction_symbols = settings.prediction_symbols.filter(s => s !== symbol);
      }
    } else if (type === 'INVESTMENT') {
      settings.investment_enabled = !!enabled;
    }

    // Update DB
    await pool.query(
      'UPDATE users SET notification_settings = $1 WHERE id = $2',
      [JSON.stringify(settings), userId]
    );

    // Publish to Kafka for notification-service to sync
    await publishUserSettingsUpdate(userId, userEmail, settings);

    res.json({ success: true, settings });
  } catch (err) {
    console.error('[NOTIF SETTINGS POST]', err);
    res.status(500).json({ error: 'db_error' });
  }
});

module.exports = router;

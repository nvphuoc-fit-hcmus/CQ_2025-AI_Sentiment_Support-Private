const nodemailer = require('nodemailer');

function createTransport() {
  if (!process.env.SMTP_EMAIL || !process.env.SMTP_PASSWORD) return null;
  return nodemailer.createTransport({
    service: process.env.SMTP_SERVICE || 'gmail',
    auth: {
      user: process.env.SMTP_EMAIL,
      pass: process.env.SMTP_PASSWORD,
    },
  });
}

async function sendVerificationEmail(email, otp) {
  const transport = createTransport();
  if (!transport) {
    console.warn(`[EMAIL] SMTP chưa cấu hình. OTP xác thực cho ${email}: ${otp}`);
    return { delivered: false, developmentOtp: otp };
  }

  await transport.sendMail({
    from: `"Aegis" <${process.env.SMTP_EMAIL}>`,
    to: email,
    subject: `${otp} là mã xác thực Aegis của bạn`,
    text: `Mã xác thực Aegis của bạn là ${otp}. Mã có hiệu lực trong 10 phút.`,
    html: `<div style="font-family:Arial,sans-serif;max-width:520px;margin:auto;padding:28px">
      <h2 style="margin:0 0 12px">Xác thực tài khoản Aegis</h2>
      <p>Nhập mã bên dưới để hoàn tất đăng ký:</p>
      <div style="font-size:32px;font-weight:700;letter-spacing:10px;padding:18px 20px;background:#f3f6fb;border-radius:10px;text-align:center">${otp}</div>
      <p style="color:#64748b">Mã có hiệu lực trong 10 phút. Không chia sẻ mã này với người khác.</p>
    </div>`,
  });
  return { delivered: true };
}

async function sendPasswordChangeOtpEmail(email, otp) {
  const transport = createTransport();
  if (!transport) {
    console.warn(`[EMAIL] SMTP chưa cấu hình. OTP đổi mật khẩu cho ${email}: ${otp}`);
    return { delivered: false, developmentOtp: otp };
  }

  await transport.sendMail({
    from: `"Aegis" <${process.env.SMTP_EMAIL}>`,
    to: email,
    subject: `${otp} là mã xác nhận đổi mật khẩu Aegis`,
    text: `Mã xác nhận đổi mật khẩu của bạn là ${otp}. Mã có hiệu lực trong 10 phút. Nếu không yêu cầu thay đổi này, hãy bỏ qua email.`,
    html: `<div style="font-family:Arial,sans-serif;max-width:520px;margin:auto;padding:28px">
      <h2 style="margin:0 0 12px">Xác nhận đổi mật khẩu</h2>
      <p>Nhập mã dưới đây để xác nhận thay đổi mật khẩu Aegis:</p>
      <div style="font-size:32px;font-weight:700;letter-spacing:10px;padding:18px 20px;background:#f3f6fb;border-radius:10px;text-align:center">${otp}</div>
      <p style="color:#64748b">Mã có hiệu lực trong 10 phút. Nếu bạn không yêu cầu đổi mật khẩu, hãy bỏ qua email này.</p>
    </div>`,
  });
  return { delivered: true };
}

async function sendPasswordChangedEmail(email) {
  const transport = createTransport();
  if (!transport) {
    console.warn(`[EMAIL] SMTP chưa cấu hình. Không thể gửi thông báo đổi mật khẩu cho ${email}`);
    return { delivered: false };
  }

  await transport.sendMail({
    from: `"Aegis" <${process.env.SMTP_EMAIL}>`,
    to: email,
    subject: 'Mật khẩu Aegis của bạn đã được thay đổi',
    text: 'Mật khẩu tài khoản Aegis của bạn vừa được thay đổi thành công. Nếu bạn không thực hiện thao tác này, hãy liên hệ quản trị viên ngay.',
    html: `<div style="font-family:Arial,sans-serif;max-width:520px;margin:auto;padding:28px">
      <h2 style="margin:0 0 12px">Mật khẩu đã được thay đổi</h2>
      <p>Mật khẩu tài khoản Aegis của bạn vừa được cập nhật thành công.</p>
      <div style="padding:14px 16px;background:#fff7ed;border-left:4px solid #f59e0b;border-radius:8px;color:#7c2d12">
        Nếu bạn không thực hiện thao tác này, hãy liên hệ quản trị viên ngay.
      </div>
      <p style="color:#64748b;font-size:13px;margin-top:20px">Đây là email bảo mật tự động, vui lòng không trả lời.</p>
    </div>`,
  });
  return { delivered: true };
}

module.exports = { sendVerificationEmail, sendPasswordChangeOtpEmail, sendPasswordChangedEmail };

import { create } from 'zustand';
import { io } from 'socket.io-client';
import GapDetector from './utils/GapDetector';
import GapRecovery from './utils/GapRecovery';

const getGatewayUrl = () => {
  if (import.meta.env.VITE_API_URL) {
    try {
      const u = new URL(import.meta.env.VITE_API_URL);
      return u.origin;
    } catch (e) { }
  }
  return 'http://localhost:8000';
};

const GATEWAY = getGatewayUrl();
const API_BASE = `${GATEWAY}/api`;
const AUTH_BASE = `${GATEWAY}/auth`;
const WS_BASE = import.meta.env.VITE_WS_URL || GATEWAY;

function normalizeIncoming(msg) {
  try {
    if (msg && msg.kline) {
      const k = msg.kline;
      // Use openTime (Start of Candle) to align with Historical Data and prevent "thin" separated candles
      const timeSec = Math.floor((k.openTime || k.t) / 1000);
      const close = Number(k.close);
      const open = Number(k.open);
      const volume = Number(k.volume || k.v);
      return {
        t: timeSec,
        o: open,
        h: Number(k.high),
        l: Number(k.low),
        c: close,
        value: volume,
        color: close >= open ? '#089981' : '#f23645'
      };
    }
  } catch (e) { /* ignore */ }
  return null;
}

const useStore = create((set, get) => ({
  token: localStorage.getItem('token') || null,
  isVip: localStorage.getItem('isVip') === 'true',
  user: null, // Add user state
  currentSymbol: 'BTCUSDT',

  // Gap detection state
  isRecovering: false,
  gapStats: {},
  recoveryProgress: null,

  // SAFE-Alert signal state (shared between DecisionSidebar ↔ Charts)
  safeAlertSignal: null,
  setSafeAlertSignal: (signal) => set({ safeAlertSignal: signal }),

  // Helper to decode token
  decodeUser: (token) => {
    try {
      if (!token) return null;
      const payload = JSON.parse(atob(token.split('.')[1]));
      return { id: payload.sub, email: payload.email, role: payload.role };
    } catch (e) {
      console.error('Failed to decode token', e);
      return null;
    }
  },

  // Init user from stored token
  init: () => {
    const token = localStorage.getItem('token');
    if (token) {
      const user = get().decodeUser(token);
      set({ user });
    }
  },

  price: null,
  socket: null,
  marketState: {},
  supportedSymbols: ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'XRPUSDT', 'DOGEUSDT', 'ADAUSDT', 'AVAXUSDT', 'DOTUSDT', 'POLUSDT'],

  setSymbol: (symbol) => {
    set({ currentSymbol: symbol, price: null });
  },

  setIsVip: (isVip) => {
    localStorage.setItem('isVip', String(isVip));
    set({ isVip: !!isVip });
  },

  // Process message with gap detection
  processMessageWithGapDetection: async (msg, onMessage, isRecovered = false) => {
    if (!msg || !msg.symbol || !msg.seq) {
      console.warn('[Store] Invalid message:', msg);
      return;
    }

    const symbol = msg.symbol.toUpperCase();

    // Detect gaps for live messages only
    if (!isRecovered) {
      const gap = GapDetector.detectGap(symbol, msg.seq);

      if (gap) {
        console.warn(`[Store] Gap detected for ${symbol}:`, gap);

        // Recover gap asynchronously
        set({ isRecovering: true });

        try {
          await GapRecovery.recoverGap(
            gap,
            get().token,
            (recoveredMsg) => {
              // Process recovered message
              if (onMessage) onMessage(recoveredMsg, true);
            },
            (progress) => {
              set({ recoveryProgress: progress });
            }
          );

          // Update gap stats
          const stats = GapDetector.getStats(symbol);
          set((state) => ({
            gapStats: { ...state.gapStats, [symbol]: stats }
          }));
        } catch (error) {
          console.error('[Store] Gap recovery failed:', error);
        } finally {
          set({ isRecovering: false, recoveryProgress: null });
        }
      }

      // Update last seen sequence
      GapDetector.setLastSeq(symbol, msg.seq);
    }

    // Process current message
    if (onMessage) onMessage(msg, isRecovered);
  },

  // Get gap statistics for a symbol
  getGapStats: (symbol) => {
    return GapDetector.getStats(symbol);
  },

  // Clear gap tracking for a symbol
  clearGapTracking: (symbol) => {
    GapDetector.clearSeq(symbol);
    GapDetector.clearStats(symbol);
    set((state) => {
      const newStats = { ...state.gapStats };
      delete newStats[symbol];
      return { gapStats: newStats };
    });
  },

  // call init on create
  // SSE Connection for real-time user events
  connectSSE: () => {
    // Prevent multiple connections
    if (get().sseSource) {
      get().sseSource.close();
    }

    const token = get().token;
    if (!token) return;

    // Decode user to get ID
    const user = get().decodeUser(token);
    if (!user) return;
    set({ user });

    // Construct SSE URL with token query param
    const sseUrl = `${AUTH_BASE}/events/sse?token=${encodeURIComponent(token)}`;

    console.log(`[Store] Connecting SSE: ${sseUrl}`);

    const eventSource = new EventSource(sseUrl);

    eventSource.onopen = () => {
      console.log('[Store] SSE Connected');
    };

    eventSource.addEventListener('vip_update', async (e) => {
      try {
        const data = JSON.parse(e.data);
        console.log('[Store] Received VIP Update via SSE:', data);

        if (data && data.isVip) {
          // IMPORTANT: Fetch fresh token with updated VIP claim
          // This ensures middleware can validate VIP status immediately
          try {
            const meResponse = await fetch(`${AUTH_BASE}/me`, {
              method: 'GET',
              headers: {
                'Authorization': `Bearer ${get().token}`
              },
              credentials: 'include'
            });

            if (meResponse.ok) {
              const meData = await meResponse.json();

              // Update token with fresh one containing is_vip claim
              if (meData.token) {
                localStorage.setItem('token', meData.token);
                const user = get().decodeUser(meData.token);
                set({ token: meData.token, isVip: true, user });
                console.log('[Store] Token refreshed with VIP claim');
              } else {
                // Fallback: just update isVip flag
                set({ isVip: true });
                localStorage.setItem('isVip', 'true');
              }
            } else {
              // Fallback: just update isVip flag
              set({ isVip: true });
              localStorage.setItem('isVip', 'true');
            }
          } catch (fetchErr) {
            console.error('[Store] Failed to refresh token after VIP upgrade:', fetchErr);
            // Fallback: just update isVip flag
            set({ isVip: true });
            localStorage.setItem('isVip', 'true');
          }

          // Dispatch event to notify UI
          window.dispatchEvent(new CustomEvent('vip_upgraded'));

          if (Notification.permission === 'granted') {
            new Notification('Account Upgraded!', {
              body: 'Your account has been upgraded to VIP successfully.'
            });
          }
        }
      } catch (err) {
        console.error('[Store] Error parsing SSE message:', err);
      }
    });

    eventSource.onerror = (err) => {
      console.error('[Store] SSE Error, retrying in 5s...', err);
      eventSource.close();
      set({ sseSource: null });
      setTimeout(() => {
        // Reconnect if still logged in
        if (get().token) get().connectSSE();
      }, 5000);
    };

    set({ sseSource: eventSource });
  },

  // Alias for backward compatibility if needed, or update call sites
  connectSocket: () => {
    get().connectSSE();
  },

  register: async (email, password, isVip) => {
    const res = await fetch(`${AUTH_BASE}/register`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ email, password, is_vip: isVip }),
      credentials: 'include' // Enable cookie sending/receiving
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.error || res.statusText || 'Registration failed');
    }
    await get().login(email, password);
  },

  login: async (email, password) => {
    const res = await fetch(`${AUTH_BASE}/login`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ email, password }),
      credentials: 'include' // Enable cookie sending/receiving
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.error || 'Login failed');
    }
    const data = await res.json();

    localStorage.setItem('token', data.token);
    localStorage.setItem('isVip', String(data.is_vip));

    const user = get().decodeUser(data.token);
    set({ token: data.token, isVip: !!data.is_vip, user });

    const { socket } = get();
    if (socket) socket.disconnect();
    get().connectSocket();
  },

  logout: async () => {
    const { token } = get();

    // Call backend logout to revoke tokens
    if (token) {
      try {
        await fetch(`${AUTH_BASE}/logout`, {
          method: 'POST',
          headers: { 'Authorization': `Bearer ${token}` },
          credentials: 'include' // Send httpOnly cookie
        });
      } catch (e) {
        console.error('Logout request failed:', e);
      }
    }

    localStorage.removeItem('token');
    localStorage.removeItem('isVip');
    const { socket } = get();
    if (socket) socket.disconnect();
    set({ token: null, isVip: false, socket: null, user: null });
  },

  // Silent refresh: Get new access token using refresh token cookie
  refreshAccessToken: async () => {
    try {
      const res = await fetch(`${AUTH_BASE}/refresh`, {
        method: 'POST',
        credentials: 'include' // Send httpOnly refresh token cookie
      });

      if (!res.ok) {
        // Refresh token expired or invalid
        return null;
      }

      const data = await res.json();
      if (data.token) {
        // Update stored token
        localStorage.setItem('token', data.token);
        const user = get().decodeUser(data.token);
        set({ token: data.token, user });
        return data.token;
      }

      return null;
    } catch (e) {
      console.error('Token refresh failed:', e);
      return null;
    }
  },

  authFetch: async (endpoint, options = {}) => {
    const { token } = get();
    let url = endpoint;
    if (!endpoint.startsWith('http')) {
      if (endpoint.startsWith('/auth') || endpoint.startsWith('/admin')) {
        url = `${GATEWAY}${endpoint}`;
      } else if (endpoint.startsWith('/v1/investments')) {
        url = `${GATEWAY}/invest-api${endpoint}`;
      } else {
        url = `${API_BASE}${endpoint}`;
      }
    }
    const headers = {
      'Content-Type': 'application/json',
      ...(options.headers || {}),
    };
    if (token) headers['Authorization'] = `Bearer ${token}`;

    // Make the initial request
    let response = await fetch(url, { ...options, headers });

    // If 401 and not already a refresh/logout request, try silent refresh
    if (response.status === 401 && !endpoint.includes('/refresh') && !endpoint.includes('/logout') && !endpoint.includes('/login')) {
      console.log('Access token expired, attempting silent refresh...');

      const newToken = await get().refreshAccessToken();

      if (newToken) {
        // Retry the original request with new token
        headers['Authorization'] = `Bearer ${newToken}`;
        response = await fetch(url, { ...options, headers });
        console.log('Request retried with refreshed token');
      } else {
        // Refresh failed, logout user
        console.log('Silent refresh failed, logging out...');
        await get().logout();
        throw new Error('Session expired, please login again');
      }
    }

    return response;
  }
}));

export default useStore;
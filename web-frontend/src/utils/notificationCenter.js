const STORAGE_KEY = 'aegis_notifications';
const MAX_ITEMS = 40;

export function getNotifications() {
    try {
        const value = JSON.parse(localStorage.getItem(STORAGE_KEY) || '[]');
        return Array.isArray(value) ? value : [];
    } catch {
        return [];
    }
}

export function saveNotifications(items) {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(items.slice(0, MAX_ITEMS)));
}

export function pushNotification({ type = 'system', title, message, dedupeKey }) {
    if (!title) return;
    const items = getNotifications();
    if (dedupeKey && items.some(item => item.dedupeKey === dedupeKey)) return;

    const notification = {
        id: `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
        type,
        title,
        message: message || '',
        createdAt: new Date().toISOString(),
        read: false,
        dedupeKey: dedupeKey || null,
    };
    const next = [notification, ...items].slice(0, MAX_ITEMS);
    saveNotifications(next);
    window.dispatchEvent(new CustomEvent('aegis-notification', { detail: notification }));
}

export function markAllNotificationsRead() {
    const next = getNotifications().map(item => ({ ...item, read: true }));
    saveNotifications(next);
    return next;
}

export function clearNotifications() {
    saveNotifications([]);
}

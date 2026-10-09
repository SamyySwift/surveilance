// Talks to the FastAPI backend (server.py). Change API_BASE if you're
// running the backend somewhere other than localhost:8000.
const API_BASE = 'http://localhost:8000';

// The backend prints (or reads from GATEKEEPER_API_TOKEN) a shared token that
// every request and stream must carry. It's kept in localStorage so a reload
// doesn't ask for it again.
const TOKEN_KEY = 'gatekeeper.apiToken';
// WebSocket close code the backend uses when the stream token is rejected.
const WS_POLICY_VIOLATION = 1008;

let token = readStoredToken();
let unauthorizedHandler = null;

function readStoredToken() {
  try {
    return localStorage.getItem(TOKEN_KEY);
  } catch {
    return null;
  }
}

export function getToken() {
  return token;
}

export function setToken(value) {
  token = value || null;
  try {
    if (token) localStorage.setItem(TOKEN_KEY, token);
    else localStorage.removeItem(TOKEN_KEY);
  } catch {
    // Storage unavailable (private window etc.): the token still works for this tab.
  }
}

// Registers the one callback that runs whenever the backend rejects the
// token. Returns an unsubscribe function, so it can be used as an effect.
export function onUnauthorized(handler) {
  unauthorizedHandler = handler;
  return () => {
    if (unauthorizedHandler === handler) unauthorizedHandler = null;
  };
}

function authedFetch(path, options = {}) {
  const headers = { ...options.headers, Authorization: `Bearer ${token ?? ''}` };
  return fetch(`${API_BASE}${path}`, { ...options, headers });
}

async function request(path, options = {}) {
  const res = await authedFetch(path, options);

  if (res.status === 401) unauthorizedHandler?.();

  if (!res.ok) {
    let body;
    try {
      body = await res.json();
    } catch {
      body = { detail: res.statusText };
    }
    const detail = body.detail;
    const message = typeof detail === 'object' ? detail.message : detail || 'Request failed';
    const errors = typeof detail === 'object' ? detail.errors : undefined;
    const err = new Error(message);
    err.errors = errors;
    err.status = res.status;
    throw err;
  }

  if (res.status === 204) return null;
  return res.json();
}

function postJson(path, body) {
  return request(path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
}

// Images behind the API token (enrollment photos, alarm snapshots) can't be
// a plain <img src>. Resolves to an object URL; the caller must
// URL.revokeObjectURL() it.
async function objectUrl(path) {
  const res = await authedFetch(path);
  if (res.status === 401) unauthorizedHandler?.();
  if (!res.ok) throw new Error(`Failed to load image (${res.status})`);
  return URL.createObjectURL(await res.blob());
}

export const api = {
  base: API_BASE,

  listStaff: () => request('/api/staff'),

  listZones: () => request('/api/zones'),

  enrollStaff: (name, employeeId, badgeId, zones, files) => {
    const form = new FormData();
    form.append('name', name);
    if (employeeId) form.append('employee_id', employeeId);
    if (badgeId) form.append('badge_id', badgeId);
    if (zones && zones.length > 0) form.append('zones', zones.join(','));
    for (const file of files) form.append('photos', file);
    return request('/api/staff', { method: 'POST', body: form });
  },

  // Captures enrollment photos at a door camera for a few seconds.
  enrollFromCamera: (staffId, zone, seconds = 6) =>
    postJson(`/api/staff/${staffId}/enroll-from-camera`, { zone, seconds }),

  addPhotos: (staffId, files) => {
    const form = new FormData();
    for (const file of files) form.append('photos', file);
    return request(`/api/staff/${staffId}/photos`, { method: 'POST', body: form });
  },

  listPhotos: (staffId) => request(`/api/staff/${staffId}/photos`),

  deletePhoto: (staffId, embeddingId) =>
    request(`/api/staff/${staffId}/photos/${embeddingId}`, { method: 'DELETE' }),

  updateStaff: (staffId, updates) =>
    request(`/api/staff/${staffId}`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(updates),
    }),

  deleteStaff: (staffId) => request(`/api/staff/${staffId}`, { method: 'DELETE' }),

  listAlarms: (limit = 100) => request(`/api/alarms?limit=${limit}`),

  ackAlarm: (alarmId, note) => postJson(`/api/alarms/${alarmId}/ack`, { note: note || null }),

  // Deletes alarms (and their snapshots) up to the newest one on screen, so
  // one that arrived since the last refresh isn't wiped unseen.
  clearAlarms: (upToId) => request(`/api/alarms?up_to_id=${upToId}`, { method: 'DELETE' }),

  // Emails the alarm status to these addresses. Resolves to { to, subject }.
  emailAlarms: (to) => postJson('/api/alarms/email', { to }),

  listAccessEvents: (limit = 50) => request(`/api/access-events?limit=${limit}`),

  // The same check a door's badge reader triggers. Takes up to ~5 s while
  // the backend waits for a clear view of the face.
  badge: (zone, badgeId) => postJson('/api/access/badge', { zone, badgeId }),

  status: () => request('/api/status'),

  // Browsers can't set headers on a WebSocket, so the token is sent as the
  // first message instead. The backend closes with 1008 if it's rejected.
  // With video off the backend sends status and who's in view, but no frames.
  openStream: (zone, { video = true } = {}) => {
    const ws = new WebSocket(`${API_BASE.replace(/^http/, 'ws')}/ws/stream/${encodeURIComponent(zone)}`);
    ws.addEventListener('open', () => {
      ws.send(token ?? '');
      if (!video) ws.send(JSON.stringify({ video: false }));
    });
    ws.addEventListener('close', (event) => {
      if (event.code === WS_POLICY_VIOLATION) unauthorizedHandler?.();
    });
    return ws;
  },

  setStreamVideo: (ws, video) => {
    if (ws?.readyState === WebSocket.OPEN) ws.send(JSON.stringify({ video }));
  },

  photoUrlPath: (staffId, embeddingId) => `/api/staff/${staffId}/photos/${embeddingId}/image`,

  snapshotPath: (name) => `/api/snapshots/${encodeURIComponent(name)}`,

  objectUrl,
};

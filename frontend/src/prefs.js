// Per-viewer preferences (which feeds have video hidden). Storage can be
// unavailable (private windows, blocked site data), so failures are ignored.
export function loadPref(key, fallback) {
  try {
    const raw = localStorage.getItem(key);
    return raw === null ? fallback : JSON.parse(raw);
  } catch {
    return fallback;
  }
}

export function savePref(key, value) {
  try {
    localStorage.setItem(key, JSON.stringify(value));
  } catch {
    // Not persisted; the setting still applies for this session.
  }
}

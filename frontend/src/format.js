// Time of day for today's events; date and time for anything older.
export function formatTime(unixSeconds) {
  const d = new Date(unixSeconds * 1000);
  const time = d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
  if (d.toDateString() === new Date().toDateString()) return time;
  return `${d.toLocaleDateString([], { month: 'short', day: 'numeric' })}, ${time}`;
}

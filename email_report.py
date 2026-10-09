"""
Alarm emails, sent through Resend (resend.com).

- Automatic alerts: every alarm except "info" (the same ones that sound and
  call the webhook) is emailed to config.ALARM_EMAIL_TO. Alarms raised close
  together share one email, and emails are at least
  config.ALARM_EMAIL_MIN_INTERVAL_SECONDS apart, so a burst of alarms can't
  flood anyone's inbox. Temporary failures are retried; the last failure is
  shown on the dashboard, so broken email is never silent.
- A status report on demand, from the dashboard's "Email" button: every open
  alarm, alarms acknowledged in the last 24 hours, and camera status.

Snapshots aren't attached: they show people's faces, and an email leaves a
copy in every mailbox it reaches.
"""

import hashlib
import html
import re
import threading
import time
from typing import Callable, Optional

import requests

import config
import database

RESEND_URL = "https://api.resend.com/emails"
ACKNOWLEDGED_WINDOW_SECONDS = 24 * 3600
GATHER_SECONDS = 5.0  # after an alarm, wait this long so others raised with it share the email
MAX_RECIPIENTS = 10
# Deliberately simple: one @, a dot in the domain, and no spaces or separators.
_ADDRESS = re.compile(r"[^@\s<>,;\"']+@[^@\s<>,;\"']+\.[^@\s<>,;\"']+")

_GREY = "#52606d"
_SEVERITY_COLOURS = {"high": "#d1002c", "medium": "#a87500", "info": _GREY}
_TD = "padding:6px 10px 6px 0;border-top:1px solid #e3e8ee;vertical-align:top;font-size:14px"


class EmailError(Exception):
    """Sending failed. `retryable` when it looks temporary (network, rate limit, Resend outage)."""

    def __init__(self, message: str, retryable: bool = False):
        super().__init__(message)
        self.retryable = retryable


def configured() -> bool:
    return bool(config.RESEND_API_KEY)


def automatic() -> bool:
    """Whether new alarms are emailed as they happen."""
    return configured() and config.ALARM_EMAIL_AUTOMATIC and bool(config.ALARM_EMAIL_TO)


def check_recipients(addresses: list[str]) -> list[str]:
    """Trimmed, de-duplicated addresses. Raises ValueError, naming any that aren't email addresses."""
    cleaned = list(dict.fromkeys(a.strip() for a in addresses if a.strip()))
    if not cleaned:
        raise ValueError("Enter at least one email address.")
    if len(cleaned) > MAX_RECIPIENTS:
        raise ValueError(f"Send to at most {MAX_RECIPIENTS} addresses at a time.")
    bad = [a for a in cleaned if not _ADDRESS.fullmatch(a)]
    if bad:
        raise ValueError(f"Not an email address: {', '.join(bad)}")
    return cleaned


# ---------- Camera status ----------

_camera_status: Optional[Callable[[], dict]] = None


def set_camera_status(provider: Callable[[], dict]):
    """The server registers how to read camera status (zone -> online), so emails can include it."""
    global _camera_status
    _camera_status = provider


def _cameras() -> Optional[dict]:
    try:
        return _camera_status() if _camera_status else None
    except Exception:
        return None


# ---------- Content ----------

def _when(t: float) -> str:
    return time.strftime("%d %b %H:%M", time.localtime(t))


def _plural(n: int, word: str) -> str:
    return f"{n} {word}{'' if n == 1 else 's'}"


def _alarm_text(r) -> str:
    line = f"  {_when(r['created_at'])}  {r['severity'].upper():<6}  {(r['zone'] or '-'):<10}  {r['message']}"
    if r["acknowledged_at"]:
        note = f": {r['acknowledged_note']}" if r["acknowledged_note"] else ""
        line += f"\n{'':16}acknowledged {_when(r['acknowledged_at'])}{note}"
    return line


def _alarms_html(rows) -> str:
    if not rows:
        return f'<p style="margin:4px 0 0;color:{_GREY}">None.</p>'
    out = []
    for r in rows:
        ack = ""
        if r["acknowledged_at"]:
            note = f": {html.escape(r['acknowledged_note'])}" if r["acknowledged_note"] else ""
            ack = f'<div style="color:{_GREY};font-size:12px">Acknowledged {_when(r["acknowledged_at"])}{note}</div>'
        colour = _SEVERITY_COLOURS.get(r["severity"], _GREY)
        out.append(
            "<tr>"
            f'<td style="{_TD};width:110px;white-space:nowrap;font-family:monospace;color:{_GREY}">{_when(r["created_at"])}</td>'
            f'<td style="{_TD};width:70px;font-family:monospace;font-weight:600;color:{colour}">{html.escape(r["severity"].upper())}</td>'
            f'<td style="{_TD};width:90px">{html.escape(r["zone"] or "-")}</td>'
            f'<td style="{_TD}">{html.escape(r["message"])}{ack}</td>'
            "</tr>"
        )
    return f'<table style="border-collapse:collapse;width:100%">{"".join(out)}</table>'


def _render(headline: str, sections: list, cameras: Optional[dict], now: float) -> tuple[str, str]:
    """(plain text, HTML): a headline, camera status (left out if None) and [(title, alarm rows)]."""
    text = [headline, f"Gatekeeper, {_when(now)}"]
    if cameras is not None:
        text += ["", "CAMERAS"]
        text += [f"  {zone:<10}  {'online' if online else 'OFFLINE'}" for zone, online in cameras.items()] or ["  none"]
    for title, rows in sections:
        text += ["", f"{title.upper()} ({len(rows)})", *([_alarm_text(r) for r in rows] or ["  none"])]
    text += ["", "Snapshots aren't attached. Open the dashboard to see them."]

    heading = "margin:24px 0 6px;font-size:13px;letter-spacing:.5px;text-transform:uppercase;color:#1f2933"
    parts = [
        f'<h1 style="margin:0;font-size:20px">{html.escape(headline)}</h1>',
        f'<p style="margin:4px 0 0;color:{_GREY};font-size:13px">Gatekeeper, {_when(now)}</p>',
    ]
    if cameras is not None:
        cells = "".join(
            f'<td style="padding:0 18px 0 0;font-size:14px">'
            f'<span style="color:{"#0a8754" if online else "#d1002c"}">&#9679;</span> '
            f'{html.escape(zone)} {"online" if online else "<b>offline</b>"}</td>'
            for zone, online in cameras.items()
        )
        parts += [f'<h2 style="{heading}">Cameras</h2>',
                  f'<table style="border-collapse:collapse"><tr>{cells or "<td>None.</td>"}</tr></table>']
    for title, rows in sections:
        parts += [f'<h2 style="{heading}">{html.escape(title)} ({len(rows)})</h2>', _alarms_html(rows)]
    parts.append(f'<p style="margin:24px 0 0;color:{_GREY};font-size:12px">'
                 "Snapshots aren't attached. Open the dashboard to see them.</p>")
    body = ('<div style="font-family:-apple-system,\'Segoe UI\',Helvetica,Arial,sans-serif;'
            f'color:#1f2933;max-width:680px">{"".join(parts)}</div>')
    return "\n".join(text), body


def build_report(now: Optional[float] = None) -> tuple[str, str, str]:
    """(subject, plain text, HTML) for the status report: open alarms, recently acknowledged ones, cameras."""
    now = time.time() if now is None else now
    open_alarms = database.list_alarms(500, unacknowledged_only=True)
    acknowledged = database.list_acknowledged_alarms(now - ACKNOWLEDGED_WINDOW_SECONDS)
    cameras = _cameras()
    offline = [zone for zone, online in (cameras or {}).items() if not online]

    headline = _plural(len(open_alarms), "open alarm") if open_alarms else "No open alarms"
    if offline:
        headline += f", {_plural(len(offline), 'camera')} offline"
    text, body = _render(headline, [("Open alarms", open_alarms),
                                    ("Acknowledged in the last 24 hours", acknowledged)], cameras, now)
    return f"Gatekeeper: {headline}", text, body


def build_alert(new_alarms: list, now: Optional[float] = None) -> tuple[str, str, str]:
    """(subject, plain text, HTML) for an automatic email about these newly raised alarms."""
    now = time.time() if now is None else now
    new_ids = {r["id"] for r in new_alarms}
    older = [r for r in database.list_alarms(500, unacknowledged_only=True) if r["id"] not in new_ids]
    if len(new_alarms) == 1:
        headline, subject = new_alarms[0]["message"], f"Gatekeeper alarm: {new_alarms[0]['message']}"
    else:
        headline = _plural(len(new_alarms), "new alarm")
        subject = f"Gatekeeper: {headline}"
    sections = [("New alarm" if len(new_alarms) == 1 else "New alarms", new_alarms)]
    if older:
        sections.append(("Still open from before", older))
    text, body = _render(headline, sections, _cameras(), now)
    return subject, text, body


# ---------- Sending ----------

def _deliver(to: list[str], subject: str, text: str, body: str, idempotency_key: Optional[str] = None):
    headers = {"Authorization": f"Bearer {config.RESEND_API_KEY}"}
    if idempotency_key:
        # A retried send that had actually gone through isn't delivered twice.
        headers["Idempotency-Key"] = idempotency_key
    try:
        r = requests.post(RESEND_URL, headers=headers, timeout=15, json={
            "from": config.ALARM_EMAIL_FROM, "to": to, "subject": subject, "text": text, "html": body,
        })
    except requests.RequestException as e:
        raise EmailError(f"Couldn't reach Resend: {e}", retryable=True) from e
    if r.ok:
        return
    try:
        detail = r.json()
    except ValueError:
        detail = {}
    message = detail.get("message") or r.text.strip() or r.reason
    if "own email address" in message:
        message += (" Resend's test sender only delivers to your Resend account's address. Verify a domain"
                    " in Resend and set ALARM_EMAIL_FROM to an address on it to email anyone else.")
    raise EmailError(f"Resend refused the email ({r.status_code}): {message}",
                     retryable=r.status_code >= 500 or detail.get("name") == "rate_limit_exceeded")


def send_report(recipients: list[str]) -> str:
    """Email the status report to `recipients` (already checked). Returns the subject. Raises EmailError."""
    subject, text, body = build_report()
    _deliver(recipients, subject, text, body)
    return subject


def send_alert(alarm_ids: list[int]):
    """Email these new alarms to config.ALARM_EMAIL_TO. Raises EmailError."""
    rows = database.get_alarms(alarm_ids)
    if not rows:
        return  # cleared from the dashboard before the email went out
    subject, text, body = build_alert(rows)
    # Stable across retries, unique per alarm (ids alone repeat if the database is recreated).
    key = hashlib.sha256(";".join(f"{r['id']}@{r['created_at']}" for r in rows).encode()).hexdigest()
    _deliver(config.ALARM_EMAIL_TO, subject, text, body, idempotency_key=f"gatekeeper-{key[:40]}")


class AlertMailer:
    """
    Sends automatic alarm emails from a background thread, so a slow or
    unreachable mail service never holds up a camera. Alarms added within
    `gather` seconds of each other share one email, and emails are at least
    `min_interval` seconds apart: anything raised in between waits for the
    next one. Temporary failures are retried at the next interval.
    """

    def __init__(self, send: Callable[[list[int]], None], gather: float = GATHER_SECONDS,
                 min_interval: Optional[float] = None):
        self._send = send
        self._gather = gather
        self._min_interval = config.ALARM_EMAIL_MIN_INTERVAL_SECONDS if min_interval is None else min_interval
        self._pending: list[int] = []
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._last_attempt = float("-inf")
        self.last_error: Optional[str] = None
        threading.Thread(target=self._run, daemon=True, name="alarm-email").start()

    def add(self, alarm_id: int):
        with self._lock:
            self._pending.append(alarm_id)
            self._wake.set()

    def _run(self):
        while True:
            self._wake.wait()
            time.sleep(self._gather)
            time.sleep(max(0.0, self._last_attempt + self._min_interval - time.time()))
            with self._lock:
                ids, self._pending = self._pending, []
                self._wake.clear()
            if not ids:
                continue
            self._last_attempt = time.time()
            try:
                self._send(ids)
                self.last_error = None
            except EmailError as e:
                self._report(str(e))
                if e.retryable:
                    with self._lock:
                        self._pending[:0] = ids
                        self._wake.set()
            except Exception as e:  # never let a bug stop all future alarm emails
                self._report(f"{type(e).__name__}: {e}")

    def _report(self, message: str):
        self.last_error = message
        print(f"[warn] Alarm email failed: {message}")


_mailer: Optional[AlertMailer] = None
_mailer_lock = threading.Lock()


def alarm_raised(alarm_id: int):
    """Queue a newly raised alarm for the next automatic email (if automatic emails are on)."""
    global _mailer
    if not automatic():
        return
    with _mailer_lock:
        if _mailer is None:
            _mailer = AlertMailer(send_alert)
    _mailer.add(alarm_id)


def delivery_problem() -> Optional[str]:
    """Why the last automatic email failed, or None if it worked (or none has been sent)."""
    return _mailer.last_error if _mailer else None

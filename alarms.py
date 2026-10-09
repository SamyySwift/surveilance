"""
Alarm outputs and the audit trail.

record_alarm() is the one place alarms go. It saves a snapshot of the frame
(if there is one), stores the alarm in the database for the dashboard and
later review, and notifies people for anything above "info": sound, the
optional webhook, and email (email_report.py). record_access_event() does the
same for badge decisions.

Snapshots show people's faces, so they live in config.SNAPSHOT_DIR with
owner-only permissions and are deleted after config.SNAPSHOT_RETENTION_DAYS.
"""

import os
import platform
import shutil
import subprocess
import threading
import time
import uuid
from typing import Optional

import cv2
import numpy as np
import requests

import config
import database
import email_report


def play_alarm_sound():
    """
    Fire-and-forget an audible alert as a background subprocess, so it never
    blocks the detection loop. Fails with a printed warning rather than
    crashing the whole system if no sound backend is available.
    """
    if not config.ALARM_SOUND_ENABLED:
        return

    system = platform.system()
    try:
        if system == "Darwin":
            sound = config.ALARM_SOUND_PATH or "/System/Library/Sounds/Sosumi.aiff"
            subprocess.Popen(["afplay", sound], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        elif system == "Linux":
            if not config.ALARM_SOUND_PATH:
                print("[warn] ALARM_SOUND_PATH must be set to play sound on Linux; skipping.")
                return
            player = shutil.which("paplay") or shutil.which("aplay")
            if not player:
                print("[warn] No 'paplay' or 'aplay' found on this system; skipping alarm sound.")
                return
            subprocess.Popen([player, config.ALARM_SOUND_PATH], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        elif system == "Windows":
            import winsound
            if config.ALARM_SOUND_PATH:
                winsound.PlaySound(config.ALARM_SOUND_PATH, winsound.SND_FILENAME | winsound.SND_ASYNC)
            else:
                winsound.MessageBeep(winsound.MB_ICONHAND)

        else:
            print(f"[warn] Unrecognized platform '{system}'; can't play an alarm sound.")

    except Exception as e:
        print(f"[warn] Failed to play alarm sound: {e}")


def _post_webhook(payload: dict):
    try:
        requests.post(config.ALARM_WEBHOOK_URL, json=payload, timeout=3)
    except requests.RequestException as e:
        print(f"  [warn] Failed to call alarm webhook: {e}")


def notify(reason: str, payload: Optional[dict] = None):
    """Tell people: console, sound, and the webhook if one is configured.
    Replace or extend this with your siren / security-desk integration."""
    print(f"[ALARM] {reason}")
    play_alarm_sound()
    if config.ALARM_WEBHOOK_URL:
        # Off the calling thread, so a slow webhook can't stall a camera.
        threading.Thread(target=_post_webhook, args=({"reason": reason, **(payload or {})},),
                         daemon=True).start()


# ---------- Snapshots ----------

_last_purge = 0.0
_purge_lock = threading.Lock()


def save_snapshot(frame: np.ndarray, bbox=None, label: str = "") -> Optional[str]:
    """Save a frame (with the face boxed) for review. Returns the file name."""
    if frame is None:
        return None
    os.makedirs(config.SNAPSHOT_DIR, mode=0o700, exist_ok=True)
    image = frame.copy()
    if bbox is not None:
        x1, y1, x2, y2 = (int(v) for v in bbox[:4])
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), max(2, image.shape[1] // 400))
    if image.shape[1] > 1280:
        scale = 1280 / image.shape[1]
        image = cv2.resize(image, (1280, int(image.shape[0] * scale)), interpolation=cv2.INTER_AREA)
    ok, jpeg = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    if not ok:
        return None
    safe_label = "".join(c if c.isalnum() or c in "-_" else "_" for c in label)[:40]
    name = f"{int(time.time() * 1000)}_{safe_label}_{uuid.uuid4().hex[:6]}.jpg"
    fd = os.open(os.path.join(config.SNAPSHOT_DIR, name), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as f:
        f.write(jpeg.tobytes())
    _maybe_purge()
    return name


def snapshot_path(name: str) -> Optional[str]:
    """Full path of a stored snapshot, or None. Rejects anything that isn't a plain file name."""
    if not name or os.path.basename(name) != name:
        return None
    path = os.path.join(config.SNAPSHOT_DIR, name)
    return path if os.path.isfile(path) else None


def purge_old_snapshots(retention_days: Optional[float] = None) -> int:
    """Delete snapshots older than the retention period. Returns how many."""
    days = config.SNAPSHOT_RETENTION_DAYS if retention_days is None else retention_days
    if not os.path.isdir(config.SNAPSHOT_DIR):
        return 0
    cutoff = time.time() - days * 86400
    removed = []
    for name in os.listdir(config.SNAPSHOT_DIR):
        path = os.path.join(config.SNAPSHOT_DIR, name)
        if os.path.isfile(path) and os.path.getmtime(path) < cutoff:
            os.remove(path)
            removed.append(name)
    database.clear_snapshot_references(removed)
    return len(removed)


def clear_alarms(up_to_id: int) -> int:
    """Delete alarms up to and including up_to_id, with their snapshots. Returns how many."""
    count, snapshots = database.delete_alarms(up_to_id)
    for name in snapshots:
        path = snapshot_path(name)
        if path:
            try:
                os.remove(path)
            except FileNotFoundError:
                pass  # purged meanwhile
    return count


def _maybe_purge():
    global _last_purge
    with _purge_lock:
        if time.time() - _last_purge < 3600:
            return
        _last_purge = time.time()
    purge_old_snapshots()


# ---------- Recording ----------

def record_alarm(kind: str, severity: str, message: str, zone: Optional[str] = None,
                 staff_id: Optional[int] = None, track_id: Optional[int] = None,
                 frame: Optional[np.ndarray] = None, bbox=None, details: Optional[dict] = None) -> int:
    snapshot = save_snapshot(frame, bbox, label=f"{zone or 'system'}_{kind}") if frame is not None else None
    alarm_id = database.add_alarm(kind, severity, message, zone=zone, staff_id=staff_id,
                                  track_id=track_id, snapshot=snapshot, details=details)
    if severity != "info":
        notify(message, {"alarm_id": alarm_id, "kind": kind, "severity": severity, "zone": zone,
                         "time": time.time()})
        email_report.alarm_raised(alarm_id)
    else:
        print(f"[info] {message}")
    return alarm_id


def record_access_event(zone: str, decision, badge_id: Optional[str], staff_name: Optional[str],
                        model_id: Optional[str], liveness_model_id: Optional[str],
                        fallback_frame: Optional[np.ndarray] = None) -> tuple[int, Optional[str]]:
    """
    Store a badge decision (pipeline.BadgeDecision) in the audit trail with a
    snapshot. Returns (event id, snapshot name).
    """
    frame = decision.frame if decision.frame is not None else fallback_frame
    snapshot = save_snapshot(frame, decision.bbox, label=f"{zone}_{decision.decision}")
    event_id = database.add_access_event(
        zone=zone, decision=decision.decision, reason=decision.reason, badge_id=badge_id,
        staff_id=decision.staff_id, staff_name=staff_name, match_score=decision.match_score,
        liveness_score=decision.liveness_score, planarity=decision.planarity,
        samples=decision.samples, model_id=model_id, liveness_model_id=liveness_model_id,
        snapshot=snapshot, details=decision.details,
    )
    return event_id, snapshot

"""
The alarm webhook: what it's told about an alarm, and which alarms call it.
Nothing is actually sent.
"""

import base64
import threading

import numpy as np

import alarms
import config
import database


def capture_webhook(monkeypatch, snapshot=False):
    sent, called = [], threading.Event()

    def fake_post(payload):
        sent.append(payload)
        called.set()

    monkeypatch.setattr(config, "ALARM_WEBHOOK_URL", "https://hooks.example.com/alarm")
    monkeypatch.setattr(config, "ALARM_WEBHOOK_SNAPSHOT", snapshot)
    monkeypatch.setattr(alarms, "_post_webhook", fake_post)
    return sent, called


def test_snapshot_is_attached_as_base64_jpeg_when_switched_on(monkeypatch):
    sent, called = capture_webhook(monkeypatch, snapshot=True)
    frame = np.full((720, 1280, 3), 128, np.uint8)
    alarms.record_alarm("unknown_person", "medium", "Unrecognized person at vault", zone="vault",
                        frame=frame, bbox=np.array([500, 200, 700, 450]))
    assert called.wait(2)
    p = sent[0]
    image = base64.b64decode(p["snapshot_base64"])
    assert image[:2] == b"\xff\xd8"  # JPEG
    with open(alarms.snapshot_path(p["snapshot_name"]), "rb") as f:
        assert f.read() == image
    assert p["snapshot_mime_type"] == "image/jpeg"


def test_snapshot_is_not_attached_unless_switched_on(monkeypatch):
    sent, called = capture_webhook(monkeypatch, snapshot=False)
    alarms.record_alarm("unknown_person", "medium", "Unrecognized person at vault", zone="vault",
                        frame=np.full((720, 1280, 3), 128, np.uint8))
    assert called.wait(2)
    assert (sent[0]["snapshot_name"], sent[0]["snapshot_mime_type"], sent[0]["snapshot_base64"]) == (None, None, None)


def test_webhook_gets_who_where_and_why(monkeypatch):
    sent, called = capture_webhook(monkeypatch)
    staff_id = database.add_staff("Jane Doe", zones=["entrance"])
    details = {"reason": "Jane Doe has no access to vault", "match_score": 0.71, "liveness": 0.98}
    alarm_id = alarms.record_alarm("unauthorized", "high", "Jane Doe is not authorized for vault",
                                   zone="vault", staff_id=staff_id, track_id=3, details=details)
    assert called.wait(2)
    p = sent[0]
    assert p["reason"] == "Jane Doe is not authorized for vault"
    assert (p["alarm_id"], p["kind"], p["title"], p["severity"], p["zone"]) == (
        alarm_id, "unauthorized", "Unauthorized person", "high", "vault")
    assert (p["staff_id"], p["staff_name"], p["track_id"]) == (staff_id, "Jane Doe", 3)
    assert p["details"] == details
    assert p["time_text"]


def test_webhook_for_a_camera_alarm_has_no_person_or_snapshot(monkeypatch):
    sent, called = capture_webhook(monkeypatch, snapshot=True)
    alarms.record_alarm("camera_offline", "high", "Camera at vault is offline", zone="vault")
    assert called.wait(2)
    assert sent[0]["title"] == "Camera offline"
    assert sent[0]["staff_name"] is None
    assert sent[0]["details"] == {}
    assert sent[0]["snapshot_base64"] is None


def test_info_events_dont_call_the_webhook(monkeypatch):
    sent, called = capture_webhook(monkeypatch)
    alarms.record_alarm("camera_restored", "info", "Cleared: camera at vault is offline", zone="vault")
    assert not called.wait(0.3)

"""
The alarm webhook: what it's told about an alarm, and which alarms call it.
Nothing is actually sent.
"""

import threading

import alarms
import config
import database


def capture_webhook(monkeypatch):
    sent, called = [], threading.Event()

    def fake_post(payload):
        sent.append(payload)
        called.set()

    monkeypatch.setattr(config, "ALARM_WEBHOOK_URL", "https://hooks.example.com/alarm")
    monkeypatch.setattr(alarms, "_post_webhook", fake_post)
    return sent, called


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


def test_webhook_for_a_camera_alarm_has_no_person(monkeypatch):
    sent, called = capture_webhook(monkeypatch)
    alarms.record_alarm("camera_offline", "high", "Camera at vault is offline", zone="vault")
    assert called.wait(2)
    assert sent[0]["title"] == "Camera offline"
    assert sent[0]["staff_name"] is None
    assert sent[0]["details"] == {}


def test_info_events_dont_call_the_webhook(monkeypatch):
    sent, called = capture_webhook(monkeypatch)
    alarms.record_alarm("camera_restored", "info", "Cleared: camera at vault is offline", zone="vault")
    assert not called.wait(0.3)

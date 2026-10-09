"""
Automatic alarm emails: grouping and spacing, retries, what's in them, and
which alarms send one. Resend itself is never contacted.
"""

import threading
import time

import pytest
import requests

import alarms
import config
import database
import email_report
from email_report import AlertMailer, EmailError


def wait_for(predicate, timeout=5.0):
    end = time.time() + timeout
    while time.time() < end:
        if predicate():
            return True
        time.sleep(0.02)
    return False


class Recorder:
    """A send function for AlertMailer: records each batch, optionally failing first."""

    def __init__(self, failures=()):
        self.batches, self.times, self.failures = [], [], list(failures)
        self.lock = threading.Lock()

    def __call__(self, ids):
        with self.lock:
            self.times.append(time.time())
            if self.failures:
                raise self.failures.pop(0)
            self.batches.append(list(ids))


def test_alarms_raised_together_share_one_email_and_emails_are_spaced_out():
    send = Recorder()
    mailer = AlertMailer(send, gather=0.1, min_interval=0.6)
    mailer.add(1)
    mailer.add(2)
    assert wait_for(lambda: len(send.batches) == 1)
    assert send.batches == [[1, 2]]

    mailer.add(3)
    mailer.add(4)
    assert wait_for(lambda: len(send.batches) == 2)
    assert send.batches[1] == [3, 4]
    assert send.times[1] - send.times[0] >= 0.6


def test_temporary_failure_is_retried_and_reported_until_it_works():
    send = Recorder(failures=[EmailError("Couldn't reach Resend", retryable=True)])
    mailer = AlertMailer(send, gather=0.05, min_interval=0.2)
    mailer.add(7)
    assert wait_for(lambda: mailer.last_error is not None)
    assert "Couldn't reach Resend" in mailer.last_error
    assert wait_for(lambda: send.batches == [[7]])
    assert mailer.last_error is None


def test_permanent_failure_is_reported_not_retried():
    send = Recorder(failures=[EmailError("Resend refused the email (403): domain not verified")])
    mailer = AlertMailer(send, gather=0.05, min_interval=0.1)
    mailer.add(7)
    assert wait_for(lambda: mailer.last_error is not None)
    time.sleep(0.4)
    assert send.batches == [] and len(send.times) == 1
    assert "domain not verified" in mailer.last_error


def test_alert_email_lists_the_new_alarms_and_what_is_still_open(monkeypatch):
    monkeypatch.setattr(config, "RESEND_API_KEY", "re_test")
    monkeypatch.setattr(config, "ALARM_EMAIL_TO", ["guard@example.com"])
    monkeypatch.setattr(email_report, "_camera_status", lambda: {"Entrance": True, "Vault": False})
    calls = []

    def post(url, headers=None, json=None, timeout=None):
        calls.append({"headers": headers, "json": json})
        response = requests.Response()
        response.status_code = 200
        response._content = b'{"id": "email-1"}'
        return response

    monkeypatch.setattr(email_report.requests, "post", post)
    older = database.add_alarm("unknown_person", "high", "Unknown person at Entrance", zone="Entrance")
    first = database.add_alarm("camera_offline", "high", "Camera at Vault is offline", zone="Vault")
    second = database.add_alarm("spoof_suspected", "medium", "Possible photo held up at Vault", zone="Vault")

    email_report.send_alert([first, second])
    email_report.send_alert([first, second])   # a retry of the same alarms
    [one, again] = calls
    sent = one["json"]
    assert sent["to"] == ["guard@example.com"] and sent["subject"] == "Gatekeeper: 2 new alarms"
    new_part, _, older_part = sent["text"].partition("STILL OPEN FROM BEFORE (1)")
    assert "Camera at Vault is offline" in new_part and "Possible photo held up" in new_part
    assert "Unknown person at Entrance" in older_part and "Vault       OFFLINE" in sent["text"]
    # Same alarms, same key: Resend won't deliver a retried email twice.
    assert one["headers"]["Idempotency-Key"] == again["headers"]["Idempotency-Key"]

    calls.clear()
    database.delete_alarms(older)
    email_report.send_alert([older])           # cleared before the email went out
    assert calls == []
    email_report.send_alert([first])
    assert calls[0]["json"]["subject"] == "Gatekeeper alarm: Camera at Vault is offline"


def test_every_alarm_except_info_is_queued_for_email(monkeypatch):
    queued = []
    monkeypatch.setattr(email_report, "alarm_raised", queued.append)
    high = alarms.record_alarm("unknown_person", "high", "Unknown person at Entrance", zone="Entrance")
    alarms.record_alarm("camera_restored", "info", "Cleared: camera at vault is offline", zone="Vault")
    assert queued == [high]


@pytest.mark.parametrize("key,auto,to,expected", [
    ("re_test", True, ["guard@example.com"], True),
    (None, True, ["guard@example.com"], False),
    ("re_test", False, ["guard@example.com"], False),
    ("re_test", True, [], False),
])
def test_automatic_email_needs_a_key_and_recipients(monkeypatch, key, auto, to, expected):
    monkeypatch.setattr(config, "RESEND_API_KEY", key)
    monkeypatch.setattr(config, "ALARM_EMAIL_AUTOMATIC", auto)
    monkeypatch.setattr(config, "ALARM_EMAIL_TO", to)
    assert email_report.automatic() is expected

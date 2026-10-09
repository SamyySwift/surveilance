"""
API tests: authentication, enrollment checks, photo/staff deletion, badge,
alarms, audit trail, snapshots and the live stream. Cameras are stubbed; the
pipeline itself is covered by test_pipeline.py and test_worker.py.
"""

import json as jsonlib
import os
import time

import cv2
import numpy as np
import pytest
import requests
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

import alarms
import config
import database
import email_report
import server
from conftest import BLUE, GREEN, RED, ColorEngine, solid

TOKEN = "test-token"
AUTH = {"Authorization": f"Bearer {TOKEN}"}


class StubWorker:
    camera_connected = True

    def __init__(self):
        self.badges, self.crops, self.reasons = [], [], {}
        self.video_viewers = 0

    def snapshot(self):
        jpeg = b"\xff\xd8fake" if self.video_viewers else None
        return jpeg, [{"bbox": [1, 2, 3, 4], "status": "verified", "label": "Alice 0.91", "matched": True}]

    def add_video_viewer(self):
        self.video_viewers += 1

    def remove_video_viewer(self):
        self.video_viewers -= 1

    def status(self):
        return {"camera_connected": True}

    def badge_check(self, badge_id):
        self.badges.append(badge_id)
        return {"decision": "grant", "reason": "face matches badge holder", "zone": "entrance"}

    def capture_enrollment_crops(self, seconds, max_photos):
        return self.crops, self.reasons

    def stop(self):
        pass


@pytest.fixture
def client(fake_engine, monkeypatch):
    workers = {"entrance": StubWorker(), "vault": StubWorker()}
    monkeypatch.setattr(server, "API_TOKEN", TOKEN)
    monkeypatch.setattr(server, "build_engine", ColorEngine)
    monkeypatch.setattr(server, "build_liveness", lambda: None)
    monkeypatch.setattr(server, "build_workers", lambda engine, live: workers)
    monkeypatch.setattr(config, "CAMERAS", {"entrance": 0, "vault": 1})
    monkeypatch.setattr(server, "WS_AUTH_TIMEOUT_SECONDS", 0.5)
    with TestClient(server.app) as c:
        c.workers = workers
        yield c


def jpeg(bgr, **kw):
    return cv2.imencode(".jpg", solid(bgr, **kw))[1].tobytes()


def enroll(client, name, image, **form):
    return client.post("/api/staff", headers=AUTH, data={"name": name, **form},
                       files=[("photos", ("photo.jpg", image, "image/jpeg"))])


# ---------- authentication ----------

def test_every_api_route_requires_the_token(client):
    for path in ["/api/staff", "/api/zones", "/api/status", "/api/alarms", "/api/access-events",
                 "/api/snapshots/x.jpg", "/api/staff/1/photos"]:
        assert client.get(path).status_code == 401, path
        assert client.get(path, headers={"Authorization": "Bearer nope"}).status_code == 401, path
    assert client.post("/api/access/badge", json={"zone": "entrance", "badgeId": "1"}).status_code == 401
    assert client.get("/api/staff", headers={"Authorization": "Bearer tést".encode()}).status_code == 401
    assert client.get("/api/staff", headers=AUTH).status_code == 200


def test_cors_preflight_allows_the_auth_header(client):
    r = client.options("/api/staff", headers={"Origin": "http://localhost:5173",
                                              "Access-Control-Request-Method": "GET",
                                              "Access-Control-Request-Headers": "authorization"})
    assert r.status_code == 200 and "authorization" in r.headers["access-control-allow-headers"].lower()


def test_enrollment_photos_are_not_public(client):
    assert client.get("/enrollment_photos/anything.jpg").status_code == 404


# ---------- enrollment ----------

def test_enroll_list_and_view_photos(client):
    r = enroll(client, "Alice", jpeg(BLUE), zones="entrance", badge_id="1001")
    assert r.status_code == 200 and r.json()["enrolled"] == 1
    [alice] = client.get("/api/staff", headers=AUTH).json()
    assert alice["badgeId"] == "1001" and alice["zones"] == ["entrance"] and alice["numEmbeddings"] == 1
    [photo] = client.get(f"/api/staff/{alice['id']}/photos", headers=AUTH).json()
    url = f"/api/staff/{alice['id']}/photos/{photo['id']}/image"
    assert client.get(url).status_code == 401
    r = client.get(url, headers=AUTH)
    assert r.status_code == 200 and r.content[:2] == b"\xff\xd8"


def test_poor_photo_is_refused_with_a_reason_and_no_record_is_left(client):
    r = enroll(client, "Dim", jpeg((20, 10, 10)))
    assert r.status_code == 400
    assert any("too dark" in e for e in r.json()["detail"]["errors"])
    assert client.get("/api/staff", headers=AUTH).json() == []


def test_photo_with_two_people_is_refused(client):
    image = solid(BLUE)
    image[0, 0] = (255, 255, 255)
    r = enroll(client, "Pair", cv2.imencode(".png", image)[1].tobytes())
    assert r.status_code == 400 and "people in the photo" in r.json()["detail"]["errors"][0]


def test_someone_already_enrolled_cannot_be_enrolled_again_under_another_name(client):
    assert enroll(client, "Alice", jpeg(BLUE)).status_code == 200
    r = enroll(client, "Mallory", jpeg(BLUE, seed=5))
    assert r.status_code == 400 and "looks like Alice" in r.json()["detail"]["errors"][0]


def test_adding_someone_elses_photo_to_a_person_is_refused(client):
    enroll(client, "Alice", jpeg(BLUE))
    alice_id = client.get("/api/staff", headers=AUTH).json()[0]["id"]
    files = [("photos", ("bob.jpg", jpeg(RED), "image/jpeg"))]
    r = client.post(f"/api/staff/{alice_id}/photos", headers=AUTH, files=files)
    assert r.json()["enrolled"] == 0 and "doesn't look like this person" in r.json()["errors"][0]


def test_badge_conflicts_are_rejected(client):
    enroll(client, "Alice", jpeg(BLUE), badge_id="1001")
    assert enroll(client, "Bob", jpeg(RED), badge_id="1001").status_code == 409
    enroll(client, "Bob", jpeg(RED))
    bob = next(s for s in client.get("/api/staff", headers=AUTH).json() if s["name"] == "Bob")
    assert client.patch(f"/api/staff/{bob['id']}", headers=AUTH, json={"badgeId": "1001"}).status_code == 409
    assert client.patch(f"/api/staff/{bob['id']}", headers=AUTH, json={"badgeId": "2002"}).status_code == 200
    assert client.patch(f"/api/staff/{bob['id']}", headers=AUTH, json={"active": False}).status_code == 200
    bob = next(s for s in client.get("/api/staff", headers=AUTH).json() if s["name"] == "Bob")
    assert bob["badgeId"] == "2002" and bob["active"] is False   # untouched fields keep their value
    client.patch(f"/api/staff/{bob['id']}", headers=AUTH, json={"badgeId": None})
    assert database.get_staff_by_id(bob["id"])["badge_id"] is None


def test_enroll_from_camera(client):
    enroll(client, "Alice", jpeg(BLUE))
    alice_id = client.get("/api/staff", headers=AUTH).json()[0]["id"]
    door = client.workers["entrance"]
    door.reasons = {"too far away / too small": 12, "blurry": 3}
    r = client.post(f"/api/staff/{alice_id}/enroll-from-camera", headers=AUTH, json={"zone": "entrance"})
    assert r.status_code == 422 and "too far away / too small (12)" in r.json()["detail"]["errors"][0]
    door.crops = [solid(BLUE, seed=1), solid(BLUE, seed=2)]
    r = client.post(f"/api/staff/{alice_id}/enroll-from-camera", headers=AUTH, json={"zone": "entrance"})
    assert r.status_code == 200 and r.json()["enrolled"] == 2
    assert client.post(f"/api/staff/{alice_id}/enroll-from-camera", headers=AUTH,
                       json={"zone": "nowhere"}).status_code == 404


# ---------- deleting photos and people ----------

def test_photo_delete_checks_ownership_and_removes_file_and_every_model(client):
    enroll(client, "Alice", jpeg(BLUE))
    enroll(client, "Bob", jpeg(RED))
    staff = {s["name"]: s["id"] for s in client.get("/api/staff", headers=AUTH).json()}
    [photo] = client.get(f"/api/staff/{staff['Alice']}/photos", headers=AUTH).json()
    # The same photo embedded by a second recognizer (as reembed.py would).
    database.add_embedding(staff["Alice"], np.ones(4), source_photo=photo["sourcePhoto"], model_id="other:m")
    path = os.path.join(config.ENROLLMENT_DIR, photo["sourcePhoto"])

    r = client.delete(f"/api/staff/{staff['Bob']}/photos/{photo['id']}", headers=AUTH)
    assert r.status_code == 404 and os.path.isfile(path)
    r = client.delete(f"/api/staff/{staff['Alice']}/photos/{photo['id']}", headers=AUTH)
    assert r.status_code == 200 and not os.path.exists(path)
    assert database.list_source_photos(staff["Alice"]) == []


def test_deleting_a_person_deletes_their_photo_files(client):
    enroll(client, "Alice", jpeg(BLUE))
    alice_id = client.get("/api/staff", headers=AUTH).json()[0]["id"]
    files = [os.path.join(config.ENROLLMENT_DIR, p) for p in database.list_source_photos(alice_id)]
    assert files and all(os.path.isfile(f) for f in files)
    assert client.delete(f"/api/staff/{alice_id}", headers=AUTH).status_code == 200
    assert not any(os.path.exists(f) for f in files)


# ---------- badge, alarms, audit ----------

def test_badge_checks_can_be_switched_off(client, monkeypatch):
    monkeypatch.setattr(config, "BADGE_CHECKS_ENABLED", False)
    r = client.post("/api/access/badge", headers=AUTH, json={"zone": "entrance", "badgeId": "1001"})
    assert r.status_code == 503 and client.workers["entrance"].badges == []
    assert client.get("/api/status", headers=AUTH).json()["badgeChecks"] is False


def test_badge_endpoint_routes_to_the_zone(client, monkeypatch):
    monkeypatch.setattr(config, "BADGE_CHECKS_ENABLED", True)
    r = client.post("/api/access/badge", headers=AUTH, json={"zone": "entrance", "badgeId": " 1001 "})
    assert r.status_code == 200 and r.json()["decision"] == "grant"
    assert client.workers["entrance"].badges == ["1001"]
    assert client.post("/api/access/badge", headers=AUTH, json={"zone": "roof", "badgeId": "1"}).status_code == 404
    assert client.post("/api/access/badge", headers=AUTH, json={"zone": "entrance", "badgeId": ""}).status_code == 422


def test_zones_are_saved_as_the_cameras_spell_them(client):
    r = enroll(client, "Alice", jpeg(BLUE), zones="ENTRANCE")
    assert r.status_code == 200
    [alice] = client.get("/api/staff", headers=AUTH).json()
    assert alice["zones"] == ["entrance"]
    r = client.patch(f"/api/staff/{alice['id']}", headers=AUTH, json={"zones": ["Vault", "vault ", "entrance"]})
    assert r.status_code == 200
    assert sorted(database.get_staff_zones(alice["id"])) == ["entrance", "vault"]
    r = client.patch(f"/api/staff/{alice['id']}", headers=AUTH, json={"zones": ["vault", "garage"]})
    assert r.status_code == 422 and "garage" in r.json()["detail"]
    assert enroll(client, "Bob", jpeg(RED), zones="nowhere").status_code == 422
    assert [s["name"] for s in client.get("/api/staff", headers=AUTH).json()] == ["Alice"]


def test_alarm_with_snapshot_and_acknowledgement(client):
    alarm_id = alarms.record_alarm("spoof_suspected", "high", "Photo held up at entrance", zone="entrance",
                                   frame=solid(GREEN), bbox=np.array([10, 10, 100, 100]))
    [a] = client.get("/api/alarms", headers=AUTH).json()
    assert a["id"] == alarm_id and a["kind"] == "spoof_suspected" and a["snapshot"]
    r = client.get(f"/api/snapshots/{a['snapshot']}", headers=AUTH)
    assert r.status_code == 200 and r.content[:2] == b"\xff\xd8"
    assert oct(os.stat(os.path.join(config.SNAPSHOT_DIR, a["snapshot"])).st_mode & 0o777) == "0o600"

    assert client.post(f"/api/alarms/{alarm_id}/ack", headers=AUTH, json={"note": "drill"}).status_code == 200
    assert client.post(f"/api/alarms/{alarm_id}/ack", headers=AUTH, json={}).status_code == 404
    assert client.get("/api/alarms?unacknowledged=true", headers=AUTH).json() == []


def test_clearing_alarms_deletes_them_and_their_snapshots(client):
    alarms.record_alarm("unknown_person", "high", "Unknown person at entrance", zone="entrance",
                        frame=solid(GREEN))
    shared = alarms.record_alarm("spoof_suspected", "high", "Photo held up at vault", zone="vault",
                                 frame=solid(RED))
    first_snap, shared_snap = (r["snapshot"] for r in reversed(database.list_alarms()))
    database.add_access_event("vault", "deny", "liveness failed", snapshot=shared_snap)
    # Raised after the dashboard last refreshed: not visible, so not cleared.
    later = alarms.record_alarm("camera_offline", "high", "Camera at vault is offline", zone="vault")

    r = client.delete(f"/api/alarms?up_to_id={shared}", headers=AUTH)
    assert r.status_code == 200 and r.json() == {"cleared": 2}
    assert [a["id"] for a in client.get("/api/alarms", headers=AUTH).json()] == [later]
    assert not os.path.exists(os.path.join(config.SNAPSHOT_DIR, first_snap))
    # Still part of the badge audit trail, so the file stays.
    assert client.get(f"/api/snapshots/{shared_snap}", headers=AUTH).status_code == 200

    assert client.delete("/api/alarms", headers=AUTH).status_code == 422
    assert client.delete(f"/api/alarms?up_to_id={later}").status_code == 401


class FakeResend:
    """Stands in for requests.post to Resend: records each call, answers with `status`/`reply`."""

    def __init__(self):
        self.calls, self.status, self.reply = [], 200, {"id": "email-1"}

    def __call__(self, url, headers=None, json=None, timeout=None):
        self.calls.append({"url": url, "headers": headers, "json": json})
        response = requests.Response()
        response.status_code = self.status
        response._content = jsonlib.dumps(self.reply).encode()
        return response


@pytest.fixture
def resend(monkeypatch):
    monkeypatch.setattr(config, "RESEND_API_KEY", "re_test")
    monkeypatch.setattr(config, "ALARM_EMAIL_FROM", "Gatekeeper <onboarding@resend.dev>")
    monkeypatch.setattr(config, "ALARM_EMAIL_TO", ["guard@example.com"])
    fake = FakeResend()
    monkeypatch.setattr(email_report.requests, "post", fake)
    return fake


def email(client, *to):
    return client.post("/api/alarms/email", headers=AUTH, json={"to": list(to)})


def test_alarm_status_is_emailed(client, resend):
    alarms.record_alarm("unknown_person", "high", "Unknown person at entrance", zone="entrance",
                        frame=solid(GREEN))
    fixed = alarms.record_alarm("camera_offline", "high", "Camera at vault is offline", zone="vault")
    database.acknowledge_alarm(fixed, "cable re-seated")
    client.workers["vault"].camera_connected = False

    r = email(client, "guard@example.com", " guard@example.com ", "boss@example.com")
    assert r.status_code == 200 and r.json()["to"] == ["guard@example.com", "boss@example.com"]
    [call] = resend.calls
    assert call["url"] == "https://api.resend.com/emails"
    assert call["headers"]["Authorization"] == "Bearer re_test"
    sent = call["json"]
    assert sent["subject"] == "Gatekeeper: 1 open alarm, 1 camera offline"
    assert sent["from"] == "Gatekeeper <onboarding@resend.dev>"
    assert sent["to"] == ["guard@example.com", "boss@example.com"]
    assert "Unknown person at entrance" in sent["text"] and "cable re-seated" in sent["text"]
    assert "OFFLINE" in sent["text"] and "Unknown person at entrance" in sent["html"]
    assert "attachments" not in sent   # snapshots show faces: never emailed


def test_email_rejects_bad_addresses_and_needs_setup(client, resend, monkeypatch):
    assert client.get("/api/status", headers=AUTH).json()["email"] == {
        "configured": True, "automatic": False, "defaultTo": ["guard@example.com"], "problem": None}
    r = email(client, "guard@example.com", "not-an-address")
    assert r.status_code == 422 and "not-an-address" in r.json()["detail"]
    assert email(client).status_code == 422
    assert email(client, "a@example.com\r\nBcc: spy@example.com").status_code == 422
    assert email(client, *[f"g{i}@example.com" for i in range(11)]).status_code == 422

    monkeypatch.setattr(config, "RESEND_API_KEY", None)
    assert email(client, "guard@example.com").status_code == 503
    assert client.get("/api/status", headers=AUTH).json()["email"]["configured"] is False
    assert resend.calls == []
    assert client.post("/api/alarms/email", json={"to": ["guard@example.com"]}).status_code == 401


def test_email_explains_resends_test_sender_limit(client, resend):
    resend.status = 403
    resend.reply = {"statusCode": 403, "name": "validation_error",
                    "message": "You can only send testing emails to your own email address (you@example.com)."}
    r = email(client, "guard@example.com")
    assert r.status_code == 502
    assert "own email address" in r.json()["detail"] and "Verify a domain" in r.json()["detail"]


def test_only_referenced_snapshots_are_served(client):
    os.makedirs(config.SNAPSHOT_DIR, exist_ok=True)
    stray = os.path.join(config.SNAPSHOT_DIR, "stray.jpg")
    open(stray, "wb").write(b"\xff\xd8x")
    assert client.get("/api/snapshots/stray.jpg", headers=AUTH).status_code == 404
    assert client.get("/api/snapshots/..%2F..%2Fetc%2Fpasswd", headers=AUTH).status_code == 404


def test_access_events_are_listed(client):
    database.add_access_event("entrance", "deny", "face does not match the badge holder", badge_id="1001",
                              staff_id=1, staff_name="Alice", match_score=0.12)
    [e] = client.get("/api/access-events", headers=AUTH).json()
    assert e["decision"] == "deny" and e["name"] == "Alice" and e["badgeId"] == "1001"


def test_status_reports_engine_and_liveness(client):
    s = client.get("/api/status", headers=AUTH).json()
    assert s["recognizer"] == ColorEngine.model_id
    assert s["liveness"]["available"] is False and set(s["zones"]) == {"entrance", "vault"}


# ---------- live stream ----------

def ws_close_code(client, send=None):
    try:
        with client.websocket_connect("/ws/stream/entrance") as ws:
            if send is not None:
                ws.send_text(send)
            ws.receive_text()
    except WebSocketDisconnect as e:
        return e.code
    return None


def test_stream_requires_the_token_as_first_message(client):
    assert ws_close_code(client) == 1008
    assert ws_close_code(client, "nope") == 1008
    with client.websocket_connect("/ws/stream/entrance") as ws:
        ws.send_text(TOKEN)
        msg = ws.receive_json()
    assert msg["connected"] and msg["frame"] and msg["detections"][0]["status"] == "verified"


def test_video_can_be_turned_off_while_detections_keep_coming(client):
    door = client.workers["entrance"]
    with client.websocket_connect("/ws/stream/entrance") as ws:
        ws.send_text(TOKEN)
        assert ws.receive_json()["frame"]
        ws.send_text('{"video": false}')
        msg = ws.receive_json()
        while msg["video"]:                      # frames already in flight
            msg = ws.receive_json()
        assert msg["frame"] is None and msg["detections"][0]["label"] == "Alice 0.91"
        assert door.video_viewers == 0           # nobody watching: the worker can skip encoding
        ws.send_text("not json")                 # ignored
        ws.send_text('{"video": true}')
        msg = ws.receive_json()
        while not msg["video"]:
            msg = ws.receive_json()
        assert msg["frame"] and door.video_viewers == 1
    for _ in range(50):                          # disconnecting releases the viewer
        if door.video_viewers == 0:
            break
        time.sleep(0.02)
    assert door.video_viewers == 0

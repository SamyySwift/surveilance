"""
ZoneWorker end to end: a real camera thread reading a video file, the real
pipeline, the database and alarm recording. Only the face models are fake.
"""

import os
import time

import config
import database
from conftest import BLUE, GREEN, ColorEngine, solid, unit, write_video
from pipeline import Settings
from worker import ZoneWorker


def wait_for(predicate, timeout=10.0):
    end = time.time() + timeout
    while time.time() < end:
        if predicate():
            return True
        time.sleep(0.05)
    return False


def start_worker(tmp_path, color, fake_engine, watching=True):
    clip = write_video(tmp_path / "door.mp4", [solid(color, seed=i) for i in range(45)])
    worker = ZoneWorker("entrance", {"source": clip}, ColorEngine(), None,
                        Settings.from_config(liveness_mode="off"))
    if watching:
        worker.add_video_viewer()
        assert wait_for(lambda: worker.snapshot()[0] is not None), "worker never produced a frame"
    return worker


def test_recognition_continues_with_nobody_watching_video(tmp_path, fake_engine):
    worker = start_worker(tmp_path, GREEN, fake_engine, watching=False)   # a stranger at the door
    try:
        assert wait_for(lambda: any(r["kind"] == "unknown_person" for r in database.list_alarms()))
        jpeg, overlay = worker.snapshot()
        assert jpeg is None and overlay and overlay[0]["status"] == "unknown"
    finally:
        worker.stop()


def enroll_alice():
    sid = database.add_staff("Alice", zones=["entrance"], badge_id="1001")
    database.add_embedding(sid, unit(1), model_id=ColorEngine.model_id)
    return sid


def test_badge_holder_at_the_door_is_granted_and_audited(tmp_path, fake_engine):
    sid = enroll_alice()
    worker = start_worker(tmp_path, BLUE, fake_engine)
    try:
        result = worker.badge_check("1001")
        # The live view catches up once monitoring has its own 3 good frames.
        assert wait_for(lambda: [d["status"] for d in worker.snapshot()[1]] == ["verified"])
    finally:
        worker.stop()
    assert result["decision"] == "grant" and result["staffId"] == sid and result["matchScore"] > 0.9
    [event] = database.list_access_events()
    assert event["decision"] == "grant" and event["badge_id"] == "1001" and event["model_id"] == ColorEngine.model_id
    assert event["snapshot"] and os.path.isfile(os.path.join(config.SNAPSHOT_DIR, event["snapshot"]))
    assert worker.snapshot()[0][:2] == b"\xff\xd8"
    assert database.list_alarms() == []   # an authorized person raises nothing


def test_wrong_face_and_unknown_badge_are_denied_with_alarms(tmp_path, fake_engine):
    enroll_alice()
    worker = start_worker(tmp_path, GREEN, fake_engine)   # a stranger is at the door
    try:
        mismatch = worker.badge_check("1001")
        unknown = worker.badge_check("9999")
        assert wait_for(lambda: any(r["kind"] == "unknown_person" for r in database.list_alarms()))
    finally:
        worker.stop()
    assert mismatch["decision"] == "deny" and mismatch["reason"] == "face does not match the badge holder"
    assert unknown["decision"] == "deny" and unknown["reason"] == "unknown badge"
    kinds = {r["kind"] for r in database.list_alarms()}
    assert {"face_mismatch", "unknown_badge", "unknown_person"} <= kinds
    assert [e["decision"] for e in database.list_access_events()] == ["deny", "deny"]


def test_suspended_badge_holder_is_denied(tmp_path, fake_engine):
    sid = enroll_alice()
    database.set_staff_active(sid, False)
    worker = start_worker(tmp_path, BLUE, fake_engine)
    try:
        result = worker.badge_check("1001")
    finally:
        worker.stop()
    assert result["decision"] == "deny" and result["reason"] == "badge holder is suspended"

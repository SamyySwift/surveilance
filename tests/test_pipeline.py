import numpy as np
import pytest

import database
import matcher
from conftest import FakeFace, FakeLiveness, Scene, blend, unit
from pipeline import Settings, ZonePipeline

ALICE, BOB = 1, 2
A, B = unit(1), unit(2)
FPS = 10.0


def make(fake_engine, authorized=(ALICE,), liveness=None, authorizer=None, **overrides):
    gallery = matcher.Gallery.from_arrays(fake_engine.model_id, [ALICE, BOB], ["Alice", "Bob"], np.stack([A, B]))
    overrides.setdefault("liveness_mode", "off")
    settings = Settings.from_config(**overrides)
    return ZonePipeline("entrance", fake_engine, liveness, settings, gallery_fn=lambda: gallery,
                        authorizer=authorizer or (lambda sid: sid in authorized))


def run(pipe, scenes, start=0.0):
    """Feed scenes at FPS; returns (alarms, time after the last frame)."""
    alarms, t = [], start
    for scene in scenes:
        alarms += pipe.process(scene, t)
        t += 1 / FPS
    return alarms, t


def person(embedding, n, **kw):
    return [Scene([FakeFace(embedding, **kw)]) for _ in range(n)]


def kinds(alarms):
    return [a.kind for a in alarms]


def status(pipe, t):
    return [d["status"] for d in pipe.overlay(t)]


# ---------- monitoring ----------

def test_enrolled_authorized_person_is_verified_quietly(fake_engine):
    pipe = make(fake_engine)
    alarms, t = run(pipe, person(blend(A, unit(10), 0.2), 15))
    assert alarms == [] and status(pipe, t - 0.1) == ["verified"]


def test_one_poor_frame_does_not_flag_an_authorized_person(fake_engine):
    # Regression: the old worker combined the smoothed name with the current
    # frame's staff_id, so one non-matching frame looked up zones for "nobody"
    # and built an "Unauthorized access" streak against an authorized person.
    sid = database.add_staff("Alice", zones=["entrance"])
    pipe = make(fake_engine, authorizer=lambda s: "entrance" in database.get_staff_zones(s))
    gallery = matcher.Gallery.from_arrays(fake_engine.model_id, [sid], ["Alice"], np.stack([A]))
    pipe.gallery_fn = lambda: gallery
    scenes = person(A, 6) + person(blend(A, unit(11), 0.6), 1) + person(A, 6)
    alarms, t = run(pipe, scenes * 3)
    assert alarms == [] and status(pipe, t - 0.1) == ["verified"]


def test_stranger_raises_exactly_one_alarm(fake_engine):
    pipe = make(fake_engine)
    alarms, _ = run(pipe, person(unit(50), 40))
    assert kinds(alarms) == ["unknown_person"]


def test_a_glimpse_is_not_enough_to_decide(fake_engine):
    pipe = make(fake_engine)
    glimpse, t = run(pipe, person(unit(50), 2))   # fewer good frames than MIN_SAMPLES_FOR_DECISION
    assert glimpse == [] and status(pipe, t - 0.1) == ["pending"]
    third, _ = run(pipe, person(unit(50), 1), start=t)
    assert kinds(third) == ["unknown_person"]


def test_enrolled_person_without_access_raises_unauthorized(fake_engine):
    pipe = make(fake_engine, authorized=(ALICE,))
    alarms, _ = run(pipe, person(B, 20))
    assert kinds(alarms) == ["unauthorized"] and alarms[0].staff_id == BOB


def test_face_never_clearly_visible_becomes_unverifiable(fake_engine):
    pipe = make(fake_engine)
    early, t = run(pipe, person(A, int(5 * FPS), ipd=40))   # 5 s too small to use
    assert early == []
    late, _ = run(pipe, person(A, int(2 * FPS), ipd=40), start=t)
    assert kinds(late) == ["unverifiable"]


def test_failed_liveness_is_flagged_and_stays_flagged(fake_engine):
    pipe = make(fake_engine, liveness=FakeLiveness(fake_engine), liveness_mode="enforce")
    alarms, t = run(pipe, person(A, 10, live=0.1))
    assert kinds(alarms) == ["spoof_suspected"]
    # Long enough that the failing frames have left the evidence window, so
    # only the stickiness rule keeps the flag.
    more, t = run(pipe, person(A, int(6 * FPS), live=0.99), start=t)
    assert more == [] and status(pipe, t - 0.1) == ["spoof_suspected"]


def test_monitor_mode_records_liveness_without_acting(fake_engine):
    pipe = make(fake_engine, liveness=FakeLiveness(fake_engine), liveness_mode="monitor")
    alarms, t = run(pipe, person(A, 10, live=0.1))
    assert alarms == [] and pipe.overlay(t - 0.1)[0]["liveness"] == pytest.approx(0.1)


def test_someone_stepping_into_a_tracked_box_starts_over(fake_engine):
    pipe = make(fake_engine)
    _, t = run(pipe, person(A, 10))
    assert status(pipe, t - 0.1) == ["verified"]
    alarms, t = run(pipe, person(unit(60), 20), start=t)   # a stranger in the same spot
    assert kinds(alarms) == ["unknown_person"]
    assert status(pipe, t - 0.1) == ["unknown"]


# ---------- badge + face ----------

def badge(pipe, staff_id, scenes, start=0.0):
    """Show scenes, tap the badge after them, then poll like the worker does."""
    _, t = run(pipe, scenes, start)
    started = t
    while True:
        d = pipe.badge_decision(staff_id, t, started)
        if d.final:
            return d
        pipe.process(scenes[-1] if scenes else Scene([]), t)
        t += 1 / FPS


def test_badge_holder_is_granted(fake_engine):
    d = badge(make(fake_engine), ALICE, person(blend(A, unit(10), 0.2), 6))
    assert d.decision == "grant" and d.match_score > 0.9 and d.alarm_kind is None


def test_someone_else_with_the_badge_is_denied(fake_engine):
    d = badge(make(fake_engine), ALICE, person(B, 6))
    assert d.decision == "deny" and d.alarm_kind == "face_mismatch"


def test_no_face_times_out_to_deny(fake_engine):
    d = badge(make(fake_engine), ALICE, [])
    assert d.decision == "deny" and "no face at the door" in d.reason


def test_face_too_far_away_does_not_count(fake_engine):
    d = badge(make(fake_engine), ALICE, person(A, 10, ipd=45))
    assert d.decision == "deny" and "no face at the door" in d.reason


def test_ambiguous_face_waits_then_denies(fake_engine):
    similar = blend(A, unit(12), 0.72)       # similarity ~0.36: between reject and accept
    assert 0.30 < float(similar @ A) < 0.45
    d = badge(make(fake_engine), ALICE, person(similar, 6))
    assert d.decision == "deny" and "face not confirmed" in d.reason


def test_badge_holder_not_allowed_here_is_denied_immediately(fake_engine):
    d = badge(make(fake_engine, authorized=()), ALICE, person(A, 6))
    assert d.decision == "deny" and d.alarm_kind == "unauthorized"


def test_photo_of_badge_holder_fails_liveness(fake_engine):
    pipe = make(fake_engine, liveness=FakeLiveness(fake_engine), liveness_mode="enforce")
    d = badge(pipe, ALICE, person(A, 6, live=0.15))
    assert d.decision == "deny" and d.alarm_kind == "spoof_suspected"


def test_liveness_unavailable_fails_closed(fake_engine):
    class Broken:
        model_id = "unavailable"

        def score(self, frame, bbox):
            return None

    pipe = make(fake_engine, liveness=Broken(), liveness_mode="enforce")
    d = badge(pipe, ALICE, person(A, 6))
    assert d.decision == "deny" and d.reason == "liveness check unavailable"


def test_second_person_at_door_is_reported_or_refused(fake_engine):
    two = [Scene([FakeFace(A, x=400), FakeFace(unit(70), x=1100, ipd=90)]) for _ in range(6)]
    d = badge(make(fake_engine), ALICE, two)
    assert d.decision == "grant" and d.alarm_kind == "multiple_faces"
    d = badge(make(fake_engine, deny_multiple_faces=True), ALICE, two)
    assert d.decision == "deny" and d.alarm_kind == "multiple_faces"


def test_tailgating_after_a_grant(fake_engine):
    pipe = make(fake_engine)
    d = badge(pipe, ALICE, person(A, 6))
    assert d.decision == "grant"
    t = 1.0
    follower = [Scene([FakeFace(A, x=400), FakeFace(unit(80), x=1100)]) for _ in range(int(4 * FPS))]
    alarms, _ = run(pipe, follower, start=t)
    assert kinds(alarms).count("tailgating") == 1


def test_second_person_who_badges_in_is_not_a_tailgater(fake_engine):
    pipe = make(fake_engine, authorized=(ALICE, BOB))
    both = [Scene([FakeFace(A, x=400, ipd=110), FakeFace(B, x=1100)]) for _ in range(6)]
    _, t = run(pipe, both)
    assert pipe.badge_decision(ALICE, t, t).decision == "grant"
    # Bob is further from the camera than Alice here, so badge him once he's closest.
    bob_closest = [Scene([FakeFace(A, x=400, ipd=70), FakeFace(B, x=1100, ipd=110)]) for _ in range(5)]
    _, t = run(pipe, bob_closest, start=t)
    assert pipe.badge_decision(BOB, t, t).decision == "grant"
    alarms, _ = run(pipe, bob_closest * 8, start=t)
    assert "tailgating" not in kinds(alarms)

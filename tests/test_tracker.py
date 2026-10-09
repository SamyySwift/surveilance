import numpy as np

from tracker import IoUTracker


def box(cx, cy, size=100):
    return np.array([cx - size / 2, cy - size / 2, cx + size / 2, cy + size / 2], dtype=np.float32)


def test_walking_face_keeps_its_id():
    # The old 50 px position buckets changed key as soon as someone moved.
    tracker = IoUTracker()
    ids = [tracker.update([box(200 + 15 * i, 300, 100 + 3 * i)], now=i * 0.1)[0][0] for i in range(30)]
    assert len(set(ids)) == 1


def test_two_people_keep_separate_ids_while_moving():
    tracker = IoUTracker()
    seen = set()
    for i in range(20):
        a, b = box(200 + 10 * i, 300), box(900 - 10 * i, 300)
        ids, _ = tracker.update([a, b], now=i * 0.1)
        seen.add(tuple(ids))
    assert seen == {(1, 2)}


def test_fast_move_matches_on_centre_distance():
    tracker = IoUTracker()
    first = tracker.update([box(200, 300)], now=0.0)[0][0]
    # Jumped 55 px (no overlap at IoU 0.25 for a 100 px box), same size.
    second = tracker.update([box(255, 300)], now=0.1)[0][0]
    assert first == second


def test_far_jump_or_size_change_is_a_new_person():
    tracker = IoUTracker()
    first = tracker.update([box(200, 300)], now=0.0)[0][0]
    assert tracker.update([box(600, 300)], now=0.1)[0][0] != first
    tracker = IoUTracker()
    first = tracker.update([box(200, 300, 100)], now=0.0)[0][0]
    assert tracker.update([box(205, 300, 260)], now=0.1)[0][0] != first


def test_tracks_expire_and_are_reported():
    tracker = IoUTracker(max_age=1.0)
    first = tracker.update([box(200, 300)], now=0.0)[0][0]
    ids, expired = tracker.update([], now=1.5)
    assert ids == [] and expired == [first]
    assert tracker.update([box(200, 300)], now=1.6)[0][0] != first

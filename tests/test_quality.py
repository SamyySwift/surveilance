import math

import numpy as np
import pytest

import config
import quality
from conftest import FakeFace, face_texture

# A generic 3D face (x right, y up, z towards the camera), eyes 1 unit apart,
# matching the proportions quality.py assumes for yaw.
FACE_3D = np.array([
    [-0.5, 0.571, -0.43],   # left eye
    [0.5, 0.571, -0.43],    # right eye
    [0.0, 0.0, 0.0],        # nose tip
    [-0.4, -0.585, -0.30],  # left mouth corner
    [0.4, -0.585, -0.30],   # right mouth corner
])


def project(yaw_deg=0.0, pitch_deg=0.0, roll_deg=0.0, ipd_px=100.0):
    y, p, r = (math.radians(v) for v in (yaw_deg, pitch_deg, roll_deg))
    ry = np.array([[math.cos(y), 0, math.sin(y)], [0, 1, 0], [-math.sin(y), 0, math.cos(y)]])
    rx = np.array([[1, 0, 0], [0, math.cos(p), -math.sin(p)], [0, math.sin(p), math.cos(p)]])
    rz = np.array([[math.cos(r), -math.sin(r), 0], [math.sin(r), math.cos(r), 0], [0, 0, 1]])
    pts = FACE_3D @ (rz @ rx @ ry).T
    # Image coordinates: x right, y down.
    return np.stack([500 + pts[:, 0] * ipd_px, 400 - pts[:, 1] * ipd_px], axis=1)


@pytest.mark.parametrize("yaw", [-40, -20, 0, 15, 35])
def test_yaw_estimate_tracks_true_yaw(yaw):
    est_yaw, _, _ = quality.estimate_pose(project(yaw_deg=yaw))
    assert abs(abs(est_yaw) - abs(yaw)) < 3


def test_roll_is_undone_before_yaw():
    est_yaw, _, roll = quality.estimate_pose(project(yaw_deg=20, roll_deg=25))
    assert abs(abs(roll) - 25) < 3 and abs(abs(est_yaw) - 20) < 5


def test_frontal_face_passes_live_quality():
    face = FakeFace(embedding=np.ones(4))
    report = quality.assess(face.detection(), face.crop, config.LIVE_QUALITY)
    assert report.passed, report.failures


@pytest.mark.parametrize("change, reason", [
    (dict(ipd=40), "too far away / too small"),
    (dict(det_score=0.3), "face not clearly detected"),
    (dict(crop=face_texture(blur=6)), "blurry"),
    (dict(crop=face_texture(brightness=0.2)), "too dark"),
    (dict(kps=project(yaw_deg=55)), "head turned"),
])
def test_bad_faces_fail_with_the_right_reason(change, reason):
    face = FakeFace(embedding=np.ones(4), **change)
    report = quality.assess(face.detection(), face.crop, config.LIVE_QUALITY)
    assert reason in report.failures


def test_dark_face_is_not_called_blurry():
    face = FakeFace(embedding=np.ones(4), crop=face_texture(brightness=0.25))
    failures = quality.assess(face.detection(), face.crop, config.LIVE_QUALITY).failures
    assert "too dark" in failures and "blurry" not in failures


def test_better_frames_weigh_more():
    near = quality.assess(FakeFace(np.ones(4), ipd=100).detection(), face_texture(), config.LIVE_QUALITY)
    far = quality.assess(FakeFace(np.ones(4), ipd=62).detection(), face_texture(), config.LIVE_QUALITY)
    assert near.weight > far.weight

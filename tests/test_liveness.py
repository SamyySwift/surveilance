import json
import math
import os
import shutil

import numpy as np
import pytest

import config
from liveness import FLAT, INCONCLUSIVE, THREE_D, MiniFASNetLiveness, planarity_verdict
from test_quality import FACE_3D

FLAT_FACE = FACE_3D * np.array([1.0, 1.0, 0.0])  # a printed photo: same layout, no depth


def view(points, yaw_deg=0.0, pitch_deg=0.0, noise=0.0, seed=0, distance=8.0, focal=900.0):
    """Perspective projection of a face model (eyes 1 unit apart) at a given pose."""
    y, p = math.radians(yaw_deg), math.radians(pitch_deg)
    ry = np.array([[math.cos(y), 0, math.sin(y)], [0, 1, 0], [-math.sin(y), 0, math.cos(y)]])
    rx = np.array([[1, 0, 0], [0, math.cos(p), -math.sin(p)], [0, math.sin(p), math.cos(p)]])
    pts = points @ (rx @ ry).T
    z = distance - pts[:, 2]
    img = np.stack([960 + focal * pts[:, 0] / z, 540 - focal * pts[:, 1] / z], axis=1)
    return img + np.random.default_rng(seed).normal(scale=noise, size=img.shape)


def sweep(points, yaws, noise=0.7):
    return [view(points, yaw_deg=a, noise=noise, seed=i) for i, a in enumerate(yaws)]


def test_real_head_turning_shows_depth():
    assert planarity_verdict(sweep(FACE_3D, [0, 8, 16, 24, 30])) == THREE_D


def test_tilted_photo_is_flat():
    assert planarity_verdict(sweep(FLAT_FACE, [0, 8, 16, 24, 30])) == FLAT


def test_little_movement_is_inconclusive_not_flat():
    # A real person walking straight at the camera barely turns: no verdict.
    assert planarity_verdict(sweep(FACE_3D, [0, 1, 2, 1, 0])) == INCONCLUSIVE
    assert planarity_verdict(sweep(FLAT_FACE, [0, 1, 2, 1, 0])) == INCONCLUSIVE


MODEL_DIR = config.LIVENESS_MODEL_DIR


@pytest.mark.skipif(not os.path.exists(os.path.join(MODEL_DIR, "MANIFEST.json")),
                    reason="liveness models not installed")
def test_minifasnet_scores_are_probabilities():
    model = MiniFASNetLiveness(MODEL_DIR)
    frame = np.random.default_rng(0).integers(0, 255, size=(720, 1280, 3)).astype(np.uint8)
    score = model.score(frame, np.array([600, 300, 700, 420], dtype=np.float32))
    assert 0.0 <= score <= 1.0
    # A box at the frame edge still crops inside the frame.
    assert 0.0 <= model.score(frame, np.array([1200, 650, 1279, 719], dtype=np.float32)) <= 1.0


@pytest.mark.skipif(not os.path.exists(os.path.join(MODEL_DIR, "MANIFEST.json")),
                    reason="liveness models not installed")
def test_altered_model_file_is_refused(tmp_path):
    copy = tmp_path / "liveness"
    shutil.copytree(MODEL_DIR, copy)
    manifest = json.loads((copy / "MANIFEST.json").read_text())
    target = copy / manifest["models"][0]["file"]
    data = bytearray(target.read_bytes())
    data[-10] ^= 0xFF
    target.write_bytes(bytes(data))
    with pytest.raises(ValueError, match="doesn't match MANIFEST"):
        MiniFASNetLiveness(str(copy))

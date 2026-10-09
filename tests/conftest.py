"""
Shared test fixtures.

Every test gets its own empty database, enrollment folder and snapshot folder
(never the real staff.db), alarm sounds off, and no webhook or email.
FakeEngine stands in for the face models so pipeline logic can be tested
deterministically: a "frame" is a Scene listing which fake faces are visible.
"""

import os
import sys
from dataclasses import dataclass, field

import cv2
import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import config  # noqa: E402
import database  # noqa: E402
from face_engine import Detection  # noqa: E402

# ArcFace's 5-point alignment template (eyes 35.2 px apart in a 112x112 crop).
TEMPLATE = np.array([[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
                     [41.5493, 92.3655], [70.7299, 92.2041]], dtype=np.float32)


@pytest.fixture(autouse=True)
def isolated_storage(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DB_PATH", str(tmp_path / "test.db"))
    monkeypatch.setattr(config, "ENROLLMENT_DIR", str(tmp_path / "enrollment_photos"))
    monkeypatch.setattr(config, "SNAPSHOT_DIR", str(tmp_path / "snapshots"))
    monkeypatch.setattr(config, "ALARM_SOUND_ENABLED", False)
    monkeypatch.setattr(config, "ALARM_WEBHOOK_URL", None)
    monkeypatch.setattr(config, "RESEND_API_KEY", None)
    monkeypatch.setattr(config, "ALARM_EMAIL_AUTOMATIC", False)
    monkeypatch.setattr(config, "CAMERAS", {"entrance": 0, "vault": 1})  # never the real cameras
    os.makedirs(config.ENROLLMENT_DIR)
    database.init_db()
    yield tmp_path


def unit(seed: int, dim: int = 256) -> np.ndarray:
    v = np.random.default_rng(seed).normal(size=dim).astype(np.float32)
    return v / np.linalg.norm(v)


def blend(a: np.ndarray, b: np.ndarray, weight_b: float) -> np.ndarray:
    v = (1 - weight_b) * a + weight_b * b
    return (v / np.linalg.norm(v)).astype(np.float32)


def face_texture(seed: int = 0, brightness: float = 1.0, blur: float = 0.0) -> np.ndarray:
    """A 112x112 stand-in for an aligned face crop: textured, mid-grey."""
    rng = np.random.default_rng(seed)
    img = rng.integers(60, 200, size=(112, 112, 3)).astype(np.uint8)
    img = cv2.GaussianBlur(img, (0, 0), 1.0)
    if blur:
        img = cv2.GaussianBlur(img, (0, 0), blur)
    return np.clip(img.astype(np.float32) * brightness, 0, 255).astype(np.uint8)


@dataclass
class FakeFace:
    """One person in a Scene."""
    embedding: np.ndarray
    x: float = 400.0               # face centre in the frame
    y: float = 300.0
    ipd: float = 100.0             # pixels between the eyes
    live: float = 0.95             # what the fake liveness model reports
    det_score: float = 0.9
    crop: np.ndarray = field(default_factory=face_texture)
    kps: np.ndarray = None         # override landmarks (e.g. for parallax tests)

    def landmarks(self) -> np.ndarray:
        if self.kps is not None:
            return self.kps.astype(np.float32)
        scale = self.ipd / 35.237
        centre = TEMPLATE.mean(axis=0)
        return ((TEMPLATE - centre) * scale + np.array([self.x, self.y])).astype(np.float32)

    def detection(self) -> Detection:
        kps = self.landmarks()
        w = 2.4 * self.ipd
        cx, cy = kps[:, 0].mean(), kps[:, 1].mean()
        bbox = np.array([cx - w / 2, cy - 0.6 * w, cx + w / 2, cy + 0.6 * w], dtype=np.float32)
        return Detection(bbox=bbox, kps=kps, det_score=self.det_score)


@dataclass
class Scene:
    """A fake frame: whoever is visible."""
    faces: list
    shape: tuple = (1080, 1920, 3)


class FakeEngine:
    model_id = "fake:test"

    def __init__(self):
        self._by_kps = {}
        self._current = None

    def detect(self, scene):
        dets = []
        for face in scene.faces:
            det = face.detection()
            self._by_kps[det.kps.tobytes()] = face
            dets.append(det)
        return dets

    def align(self, scene, kps):
        self._current = self._by_kps[np.asarray(kps, dtype=np.float32).tobytes()]
        return self._current.crop

    def embed_aligned(self, crop):
        return self._current.embedding

    def embed(self, scene, kps):
        self.align(scene, kps)
        return self._current.embedding


class ColorEngine:
    """
    A fake engine for real image arrays (video files, uploads). Identity comes
    from the dominant colour: blue = person A (unit(1)), red = person B
    (unit(2)), green = a stranger (unit(3)). A frame whose top-left pixel is
    pure white contains two people. Brightness carries through to the face crop.
    """

    model_id = FakeEngine.model_id
    IDENTITIES = {0: 1, 2: 2, 1: 3}  # BGR channel -> unit() seed

    def _face(self, frame):
        h, w = frame.shape[:2]
        face = FakeFace(unit(self.IDENTITIES[int(np.argmax(frame.reshape(-1, 3).mean(axis=0)))]),
                        x=w / 2, y=h / 2, ipd=min(100.0, w / 4))
        brightness = float(frame.mean()) / 128.0
        face.crop = face_texture(brightness=min(brightness, 1.6))
        return face

    def detect(self, frame):
        if float(frame.mean()) < 8:
            return []
        face = self._face(frame)
        dets = [face.detection()]
        if tuple(int(v) for v in frame[0, 0]) == (255, 255, 255):
            other = FakeFace(unit(4), x=face.x / 2, y=face.y, ipd=face.ipd * 0.9)
            dets.append(other.detection())
        return dets

    def align(self, frame, kps):
        self._current = self._face(frame)
        return self._current.crop

    def embed_aligned(self, crop):
        return self._current.embedding

    def embed(self, frame, kps):
        self.align(frame, kps)
        return self._current.embedding


def solid(bgr, size=(480, 640), seed=0) -> np.ndarray:
    """A noisy solid-colour frame (the noise keeps camera-health from calling it frozen)."""
    noise = np.random.default_rng(seed).integers(-12, 12, size=(*size, 3))
    return np.clip(np.array(bgr)[None, None, :] + noise, 0, 255).astype(np.uint8)


def write_video(path, frames, fps=15.0):
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for f in frames:
        writer.write(f)
    writer.release()
    return str(path)


BLUE, RED, GREEN = (200, 60, 60), (60, 60, 200), (60, 200, 60)


class FakeLiveness:
    model_id = "fake-liveness"

    def __init__(self, engine):
        self.engine = engine

    def score(self, scene, bbox):
        return self.engine._current.live


@pytest.fixture
def fake_engine(monkeypatch):
    monkeypatch.setitem(config.MATCH_THRESHOLDS, FakeEngine.model_id, {
        "verify_accept": 0.45, "identify_accept": 0.50, "identify_margin": 0.08,
        "reject": 0.30, "duplicate": 0.45,
    })
    return FakeEngine()

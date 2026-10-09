import platform

import numpy as np
import pytest

import config
from camera import CameraHealth, camera_index, is_camera_name, moved_usb_camera, normalize_camera


@pytest.fixture(autouse=True)
def short_delays(monkeypatch):
    monkeypatch.setattr(config, "CAMERA_OFFLINE_ALARM_SECONDS", 10)
    monkeypatch.setattr(config, "CAMERA_FROZEN_SECONDS", 10)
    monkeypatch.setattr(config, "CAMERA_OBSTRUCTED_SECONDS", 5)
    monkeypatch.setattr(config, "CAMERA_SCENE_CHANGE_SECONDS", 20)


def feed(health, start, seconds, frame_fn, connected=True, faces=False, step=0.5):
    events, t = [], start
    while t < start + seconds:
        events += health.update(t, connected, frame_fn(t) if connected else None, faces_present=faces)
        t += step
    return events, t


def kinds(events):
    return [k for k, _, _ in events]


def scene(seed, tint=(0, 0, 0), layout=0):
    """A door-like view: a floor-to-ceiling gradient, a door frame and a wall panel, plus sensor noise."""
    img = np.tile(np.linspace(60, 190, 480, dtype=np.float32)[:, None, None], (1, 640, 3))
    x = 150 + 180 * layout
    img[80:440, x:x + 160] = (70, 55, 45)              # door
    img[60:200, 420 - 120 * layout:600 - 120 * layout] = 215  # window / panel
    img += np.array(tint, dtype=np.float32)
    img += np.random.default_rng(seed).normal(scale=3, size=img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


def live_scene(t):
    return scene(int(t * 10))


def test_healthy_camera_is_quiet():
    events, _ = feed(CameraHealth("door", now=0), 0, 60, live_scene)
    assert events == []


def test_offline_camera_alarms_once_then_reports_recovery():
    h = CameraHealth("door", now=0)
    events, t = feed(h, 0, 30, live_scene, connected=False)
    assert kinds(events) == ["camera_offline"]
    events, _ = feed(h, t, 5, live_scene)
    assert kinds(events) == ["camera_restored"]


def test_frozen_picture_alarms():
    still = scene(1)
    events, _ = feed(CameraHealth("door", now=0), 0, 15, lambda t: still)
    assert kinds(events) == ["camera_frozen"]


def test_covered_lens_alarms():
    covered = np.full((480, 640, 3), 18, dtype=np.uint8)
    h = CameraHealth("door", now=0)
    feed(h, 0, 5, live_scene)
    # Slight sensor noise so this is "covered", not "frozen".
    events, _ = feed(h, 5, 10, lambda t: np.clip(covered + (int(t * 2) % 2), 0, 255).astype(np.uint8))
    assert "camera_obstructed" in kinds(events)


def person_close_up(t):
    img = scene(int(t * 10))
    img[40:480, 120:520] = (90, 120, 170)   # someone filling most of the view
    return img


def test_someone_lingering_at_the_door_is_not_a_moved_camera():
    h = CameraHealth("door", now=0)
    feed(h, 0, 10, live_scene)
    events, t = feed(h, 10, 90, person_close_up, faces=True)   # 90 s at the reader
    assert events == []
    events, _ = feed(h, t, 60, live_scene)                      # they leave
    assert events == []


def test_camera_turned_away_alarms_once():
    h = CameraHealth("door", now=0)
    feed(h, 0, 10, live_scene)
    # The camera now sees a completely different scene with nobody in it.
    events, _ = feed(h, 10, 90, lambda t: scene(int(t * 10), tint=(-40, 30, 50), layout=1))
    assert kinds(events) == ["camera_moved"]


def test_camera_config_accepts_bare_sources():
    assert normalize_camera(0) == {"source": 0, "width": None, "height": None}
    assert normalize_camera({"source": "rtsp://x", "width": 1920})["height"] is None


# In OpenCV's macOS order (sorted by unique ID): plugging in the USB camera
# moved the FaceTime camera from index 1 to 2.
MAC_CAMERAS = [
    ("iPhone Camera", "0DE66902-3E03-428B-92FE-AB2400000001"),
    ("USB2.0 PC CAMERA", "0x14000019082311"),
    ("FaceTime HD Camera", "EBB0ACC7-C7DB-49F1-B1FD-C77F4E234E8C"),
]


def test_named_camera_resolves_to_its_current_index():
    assert camera_index("FaceTime HD Camera", MAC_CAMERAS) == 2
    assert camera_index("usb2.0 pc camera", MAC_CAMERAS) == 1
    assert camera_index("0x14000019082311", MAC_CAMERAS) == 1
    assert camera_index("FaceTime HD Camera", [c for c in MAC_CAMERAS if "USB" not in c[0]]) == 1


def test_missing_or_ambiguous_camera_name_resolves_to_nothing():
    assert camera_index("FaceTime HD Camera", MAC_CAMERAS[:2]) is None
    twins = [("USB Camera", "0x1100"), ("USB Camera", "0x1200")]
    assert camera_index("USB Camera", twins) is None
    assert camera_index("0x1200", twins) == 1


def test_only_plain_strings_on_a_mac_are_camera_names(monkeypatch, tmp_path):
    monkeypatch.setattr(platform, "system", lambda: "Darwin")
    clip = tmp_path / "door.mp4"
    clip.write_bytes(b"")
    assert is_camera_name("FaceTime HD Camera")
    assert not is_camera_name(0)
    assert not is_camera_name("rtsp://user:pass@10.0.0.20/stream1")
    assert not is_camera_name(str(clip))
    monkeypatch.setattr(platform, "system", lambda: "Linux")
    assert not is_camera_name("FaceTime HD Camera")


def test_usb_camera_moved_to_another_port_is_recognized():
    # The ID starts with the USB port, so swapping ports changes it.
    cameras = [("USB Camera", "0x1300000bda3035"), ("2M WebCam", "0x1400000c456366"),
               ("FaceTime HD Camera", "EBB0ACC7-C7DB-49F1-B1FD-C77F4E234E8C")]
    assert camera_index("0x1400000bda3035", cameras) is None
    assert moved_usb_camera("0x1400000bda3035", cameras) == "USB Camera"
    assert moved_usb_camera("0x1300000c456366", cameras) == "2M WebCam"
    assert moved_usb_camera("0x1300000bda3035", cameras) is None   # connected as configured
    assert moved_usb_camera("0x1500000aaaa1111", cameras) is None  # a model that isn't plugged in
    assert moved_usb_camera("FaceTime HD Camera", cameras) is None

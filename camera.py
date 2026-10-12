"""
Camera capture with automatic reconnect, plus camera-health checks.

CameraReader reads frames in a background thread and keeps only the newest
one, so processing never falls behind real time: frames that arrive while
the previous one is still being processed are dropped, not queued.

CameraHealth watches for a camera that's offline, frozen, covered or
blinded, or pointed somewhere else. A broken or tampered camera then raises
an alarm instead of silently producing no alarms at all.
"""

import os
import platform
import re
import threading
import time
from typing import Optional

import cv2
import numpy as np

import config


def normalize_camera(value) -> dict:
    """Accept either {"source": ..., "width": ..., "height": ...} or a bare source."""
    cfg = dict(value) if isinstance(value, dict) else {"source": value}
    cfg.setdefault("width", None)
    cfg.setdefault("height", None)
    return cfg


def is_camera_name(source) -> bool:
    """True for a Mac webcam named in config (e.g. "FaceTime HD Camera") rather than an index, URL or file."""
    return (isinstance(source, str) and platform.system() == "Darwin"
            and "://" not in source and not os.path.exists(source))


def mac_cameras() -> list[tuple[str, str]]:
    """
    (name, unique ID) of each connected camera, in the order OpenCV numbers
    them on macOS: sorted by unique ID. That's why plugging a camera in or
    out (or an iPhone coming into Continuity Camera range) can renumber the
    others.
    """
    return sorted(((str(d.localizedName()), str(d.uniqueID())) for d in _mac_devices()), key=lambda c: c[1])


def _mac_devices() -> list:
    import AVFoundation  # pyobjc; macOS only

    return (list(AVFoundation.AVCaptureDevice.devicesWithMediaType_(AVFoundation.AVMediaTypeVideo))
            + list(AVFoundation.AVCaptureDevice.devicesWithMediaType_(AVFoundation.AVMediaTypeMuxed)))


def mac_camera_sizes() -> dict[str, list[tuple[int, int, float]]]:
    """
    Unique ID -> every (width, height, max frames per second) the camera
    supports, widest first: pixels across the image are what put pixels on a
    face (see config.py "Cameras").
    """
    import CoreMedia  # installed with pyobjc's AVFoundation

    out = {}
    for d in _mac_devices():
        sizes: dict[tuple[int, int], float] = {}
        for f in d.formats():
            dims = CoreMedia.CMVideoFormatDescriptionGetDimensions(f.formatDescription())
            fps = max((r.maxFrameRate() for r in f.videoSupportedFrameRateRanges()), default=0.0)
            sizes[(dims.width, dims.height)] = max(sizes.get((dims.width, dims.height), 0.0), fps)
        out[str(d.uniqueID())] = sorted(((w, h, fps) for (w, h), fps in sizes.items()), key=lambda s: (-s[0], -s[1]))
    return out


def camera_index(name: str, cameras: list[tuple[str, str]]) -> Optional[int]:
    """
    OpenCV index of the camera with this name or unique ID (any case), or
    None if none is connected. Also None when several cameras share the name
    (two of the same model): name one by its unique ID instead.
    """
    wanted = name.casefold()
    matches = [i for i, (cam_name, uid) in enumerate(cameras) if wanted in (cam_name.casefold(), uid.casefold())]
    return matches[0] if len(matches) == 1 else None


def moved_usb_camera(unique_id: str, cameras: list[tuple[str, str]]) -> Optional[str]:
    """
    Name of a connected camera that looks like `unique_id` plugged into another
    port, or None. A Mac USB camera's ID is its port (0x14000000) followed by
    its vendor and product IDs (0bda3035), so only the start changes.
    """
    if not re.fullmatch(r"0x[0-9a-f]{14}", unique_id, re.IGNORECASE):
        return None
    model = unique_id[-8:].lower()
    return next((name for name, uid in cameras
                 if uid.lower() != unique_id.lower() and re.fullmatch(r"0x[0-9a-f]{14}", uid, re.IGNORECASE)
                 and uid[-8:].lower() == model), None)


class CameraReader:
    def __init__(self, source, width: Optional[int] = None, height: Optional[int] = None, name: str = "camera"):
        self.source = source
        self.width = width
        self.height = height
        self.name = name
        self.connected = False
        self.frame_size: Optional[tuple[int, int]] = None  # what the camera actually delivers
        self._is_file = isinstance(source, str) and os.path.isfile(source)
        self._by_name = is_camera_name(source)
        if self._by_name:
            mac_cameras()  # a missing pyobjc fails here, not silently in the capture thread
        self._lookup_failed = False
        self._lock = threading.Lock()
        self._frame: Optional[np.ndarray] = None
        self._frame_id = 0
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True, name=f"camera-{name}")
        self._thread.start()

    def _find_camera(self) -> Optional[int]:
        # Looked up on every (re)connect: the index changes whenever another
        # camera is plugged in or removed.
        cameras = mac_cameras()
        index = camera_index(self.source, cameras)
        if index is None and not self._lookup_failed:
            connected = ", ".join(f"{n!r} ({uid})" for n, uid in cameras) or "none"
            print(f"[{self.name}] No single camera named {self.source!r} is connected. Connected: {connected}")
            moved = moved_usb_camera(self.source, cameras)
            if moved:
                print(f"[{self.name}] {moved!r} looks like the same camera on a different USB port. "
                      f"Name it in config.CAMERAS instead of using its ID, unless another camera shares that name.")
        self._lookup_failed = index is None
        return index

    def _open(self):
        if self._by_name:
            index = self._find_camera()
            if index is None:
                return None
            cap = cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION)
        else:
            cap = cv2.VideoCapture(self.source)
        if self.width:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        if self.height:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        if not cap.isOpened():
            cap.release()
            return None
        return cap

    def _run(self):
        cap, delay = None, 0.0
        while self._running:
            if cap is None:
                cap = self._open()
                if cap is None:
                    self.connected = False
                    time.sleep(1.0)
                    continue
                # Video files play back at their own frame rate (and loop), so
                # a recorded clip behaves like a live camera.
                delay = 1.0 / (cap.get(cv2.CAP_PROP_FPS) or 25.0) if self._is_file else 0.0

            ok, frame = cap.read()
            if not ok:
                self.connected = False
                cap.release()
                cap = None
                time.sleep(0.2 if self._is_file else 0.5)
                continue

            with self._lock:
                self._frame = frame
                self._frame_id += 1
            self.frame_size = (frame.shape[1], frame.shape[0])
            self.connected = True
            if delay:
                time.sleep(delay)
        if cap is not None:
            cap.release()

    def latest(self) -> tuple[Optional[np.ndarray], int]:
        """(newest frame, its sequence number). The frame must not be modified."""
        with self._lock:
            return self._frame, self._frame_id

    def stop(self):
        self._running = False
        self._thread.join(timeout=2)


def _thumbnail(frame: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    return cv2.resize(gray, (64, 36), interpolation=cv2.INTER_AREA).astype(np.float32)


class CameraHealth:
    """
    Feed it every loop iteration; it returns (kind, severity, message) events
    when a problem has lasted long enough, and an "info" event when it clears.
    """

    _MESSAGES = {
        "camera_offline": "Camera at {zone} is offline",
        "camera_frozen": "Camera at {zone} is sending a frozen picture",
        "camera_obstructed": "Camera at {zone} looks covered or blinded",
        "camera_moved": "Camera view at {zone} changed a lot (moved, blocked, or lighting changed)",
    }

    def __init__(self, zone: str, now: Optional[float] = None):
        now = time.time() if now is None else now
        self.zone = zone
        self._delays = {
            "camera_offline": config.CAMERA_OFFLINE_ALARM_SECONDS,
            "camera_frozen": config.CAMERA_FROZEN_SECONDS,
            "camera_obstructed": config.CAMERA_OBSTRUCTED_SECONDS,
            "camera_moved": config.CAMERA_SCENE_CHANGE_SECONDS,
        }
        self._since: dict[str, float] = {}
        self._alarmed: set[str] = set()
        self._last_thumb: Optional[np.ndarray] = None
        self._last_change = now
        self._background: Optional[np.ndarray] = None
        self._obstructed = False
        self._moved = False
        self._was_connected = False

    def update(self, now: float, connected: bool, frame: Optional[np.ndarray] = None,
               faces_present: bool = False) -> list[tuple[str, str, str]]:
        if connected and not self._was_connected:
            self._last_change = now
        self._was_connected = connected

        if frame is not None:
            self._observe(now, _thumbnail(frame), faces_present)

        conditions = {
            "camera_offline": not connected,
            # Measured from the last time the picture changed at all.
            "camera_frozen": connected and now - self._last_change >= self._delays["camera_frozen"],
            "camera_obstructed": connected and self._obstructed,
            "camera_moved": connected and self._moved,
        }
        events = []
        for kind, active in conditions.items():
            if active:
                onset = self._since.setdefault(kind, now)
                delay = 0 if kind == "camera_frozen" else self._delays[kind]
                if kind not in self._alarmed and now - onset >= delay:
                    self._alarmed.add(kind)
                    events.append((kind, "high", self._MESSAGES[kind].format(zone=self.zone)))
                    if kind == "camera_moved":
                        # Learn the new view so it alarms once, not forever.
                        self._background = self._last_thumb.copy() if self._last_thumb is not None else None
                        self._moved = False
                        self._since.pop(kind, None)
                        self._alarmed.discard(kind)
            else:
                self._since.pop(kind, None)
                if kind in self._alarmed:
                    self._alarmed.discard(kind)
                    events.append(("camera_restored", "info",
                                   f"Cleared: {self._MESSAGES[kind].format(zone=self.zone).lower()}"))
        return events

    def _observe(self, now: float, thumb: np.ndarray, faces_present: bool):
        if self._last_thumb is None or float(np.abs(thumb - self._last_thumb).mean()) >= 0.05:
            self._last_change = now
        self._last_thumb = thumb

        mean, std = float(thumb.mean()), float(thumb.std())
        self._obstructed = std < config.CAMERA_OBSTRUCTED_MAX_STDDEV or mean < 12 or mean > 243

        if self._background is None:
            self._background = thumb.copy()
            return
        diff = float(np.abs(thumb - self._background).mean())
        # People walking past change part of the view; only a sustained,
        # whole-scene change with nobody in front of the camera counts.
        self._moved = diff > config.CAMERA_SCENE_CHANGE_THRESHOLD and not faces_present
        # Learn slow changes (daylight) only from empty scenes, so someone
        # lingering at the door isn't absorbed into the background and then
        # mistaken for a moved camera after they leave.
        if not self._moved and not faces_present:
            self._background += 0.02 * (thumb - self._background)


def _print_mac_cameras():
    """`uv run camera.py`: each connected camera, the zone using it, and the sizes it supports."""
    cameras, sizes = mac_cameras(), mac_camera_sizes()
    zones: dict[int, list[str]] = {}
    for zone, value in config.CAMERAS.items():
        cfg = normalize_camera(value)
        source = cfg["source"]
        index = camera_index(source, cameras) if is_camera_name(source) else source if isinstance(source, int) else None
        if index is not None:
            zones.setdefault(index, []).append(f"{zone} (asks for {cfg['width']}x{cfg['height']})"
                                               if cfg["width"] else zone)
    for i, (name, uid) in enumerate(cameras):
        print(f"{i}  {name}   [{uid}]" + (f"   used by {', '.join(zones[i])}" if i in zones else ""))
        modes = [f"{w}x{h} @ {fps:.0f} fps" for w, h, fps in sizes.get(uid, [])]
        print(f"   {', '.join(modes) or 'no sizes reported'}")
    print("\nIn config.CAMERAS, set a zone's width/height to the first (widest) size listed for its camera.\n"
          "A camera asked for a size it doesn't have sends the nearest one it can; the dashboard shows what\n"
          "each camera actually sends under its feed. Cameras that aren't plugged in aren't listed.")


if __name__ == "__main__":
    _print_mac_cameras()

"""
One ZoneWorker per camera zone. It owns the camera, runs the decision
pipeline on the newest frame in a background thread, watches camera
health, records alarms, and answers badge checks from the API.

Processing runs in a plain thread (not asyncio) because face detection is
CPU/GPU-bound; inside an async endpoint it would block every other request.
The camera reconnects in the background, so the API stays usable even while
a camera is unplugged.
"""

import threading
import time
import traceback
from typing import Optional

import cv2

import alarms
import config
import database
import quality
from camera import CameraHealth, CameraReader, normalize_camera
from enrollment import face_crop
from pipeline import BadgeDecision, ZonePipeline


class ZoneWorker:
    def __init__(self, zone: str, camera_cfg, engine, liveness=None, settings=None):
        cfg = normalize_camera(camera_cfg)
        self.zone = zone
        self.engine = engine
        self.liveness = liveness
        self.reader = CameraReader(cfg["source"], cfg["width"], cfg["height"], name=zone)
        self.pipeline = ZonePipeline(zone, engine, liveness, settings)
        self.health = CameraHealth(zone)
        self.processing_fps = 0.0

        self._pipeline_lock = threading.Lock()
        self._preview_lock = threading.Lock()
        self._preview_jpeg: Optional[bytes] = None
        self._preview_overlay: list = []
        # Dashboards currently showing this camera's video. With none, frames
        # are still processed but no preview JPEG is encoded.
        self._video_viewers = 0
        self._error_reported = False

        self.running = True
        self._thread = threading.Thread(target=self._run, daemon=True, name=f"worker-{zone}")
        self._thread.start()

    @property
    def camera_connected(self) -> bool:
        return self.reader.connected

    # ---------- processing loop ----------

    def _run(self):
        last_id, last_t = 0, None
        while self.running:
            frame, frame_id = self.reader.latest()
            now = time.time()
            if frame is None or frame_id == last_id:
                self._record_health(self.health.update(now, self.reader.connected))
                time.sleep(0.005)
                continue
            last_id = frame_id

            try:
                with self._pipeline_lock:
                    raised = self.pipeline.process(frame, now)
                    overlay = self.pipeline.overlay(now)
                self._error_reported = False
            except Exception:
                # Keep the camera loop alive, but make the failure visible once.
                traceback.print_exc()
                if not self._error_reported:
                    self._error_reported = True
                    alarms.record_alarm("processing_error", "high",
                                        f"Face processing failed at {self.zone}; see server log", zone=self.zone)
                time.sleep(0.5)
                continue

            self._record_health(self.health.update(now, True, frame, faces_present=bool(overlay)))
            for a in raised:
                alarms.record_alarm(a.kind, a.severity, a.message, zone=self.zone, staff_id=a.staff_id,
                                    track_id=a.track_id, frame=a.frame, bbox=a.bbox, details=a.details)
            self._publish_preview(frame, overlay)

            if last_t is not None and now > last_t:
                self.processing_fps = 0.9 * self.processing_fps + 0.1 * (1.0 / (now - last_t))
            last_t = now

    def _record_health(self, events):
        for kind, severity, message in events:
            frame, _ = self.reader.latest()
            alarms.record_alarm(kind, severity, message, zone=self.zone,
                                frame=frame if kind in ("camera_obstructed", "camera_moved") else None)

    def _publish_preview(self, frame, overlay):
        scale = min(1.0, config.STREAM_MAX_WIDTH / frame.shape[1])
        scaled = [{**d, "bbox": [v * scale for v in d["bbox"]]} for d in overlay]
        jpeg = None
        if self._video_viewers > 0:
            small = frame if scale == 1.0 else cv2.resize(
                frame, (int(frame.shape[1] * scale), int(frame.shape[0] * scale)), interpolation=cv2.INTER_AREA)
            ok, encoded = cv2.imencode(".jpg", small, [int(cv2.IMWRITE_JPEG_QUALITY), config.STREAM_JPEG_QUALITY])
            jpeg = encoded.tobytes() if ok else None
        with self._preview_lock:
            self._preview_jpeg = jpeg
            self._preview_overlay = scaled

    def snapshot(self):
        """(latest preview JPEG, or None while nobody is watching video; overlay for it)."""
        with self._preview_lock:
            return self._preview_jpeg, list(self._preview_overlay)

    def add_video_viewer(self):
        with self._preview_lock:
            self._video_viewers += 1

    def remove_video_viewer(self):
        with self._preview_lock:
            self._video_viewers = max(0, self._video_viewers - 1)

    def status(self) -> dict:
        with self._pipeline_lock:
            tracks = len(self.pipeline.tracks)
        return {
            "camera_connected": self.reader.connected,
            "frame_size": self.reader.frame_size,
            "processing_fps": round(self.processing_fps, 1),
            "tracks": tracks,
            "video_viewers": self._video_viewers,
        }

    # ---------- badge + face ----------

    def badge_check(self, badge_id: str) -> dict:
        """
        Called by the API when a badge is read at this zone's door. Waits (up
        to config.BADGE_DECISION_TIMEOUT_SECONDS) for enough frames of the
        person at the door, then grants or denies. Every outcome is written
        to the audit trail, and every deny raises an alarm.
        """
        staff = database.get_staff_by_badge(badge_id)
        if staff is None:
            decision = BadgeDecision("deny", "unknown badge", None, alarm_kind="unknown_badge", severity="medium")
        elif not staff["active"]:
            decision = BadgeDecision("deny", "badge holder is suspended", staff["id"],
                                     alarm_kind="suspended_badge", severity="high")
        else:
            started = time.time()
            while True:
                with self._pipeline_lock:
                    decision = self.pipeline.badge_decision(staff["id"], time.time(), started)
                if decision.final:
                    break
                time.sleep(0.1)

        name = staff["name"] if staff is not None else None
        current, _ = self.reader.latest()
        live_id = getattr(self.liveness, "model_id", None)
        event_id, snapshot = alarms.record_access_event(self.zone, decision, badge_id, name,
                                                         self.engine.model_id, live_id, fallback_frame=current)
        if decision.alarm_kind:
            who = f"{name} (badge {badge_id})" if name else f"badge {badge_id}"
            verb = "Access granted" if decision.decision == "grant" else "Access denied"
            alarms.record_alarm(decision.alarm_kind, decision.severity,
                                f"{verb} at {self.zone} for {who}: {decision.reason}"
                                + (" - more than one person at the door" if decision.decision == "grant" else ""),
                                zone=self.zone, staff_id=decision.staff_id, track_id=decision.track_id,
                                frame=decision.frame if decision.frame is not None else current,
                                bbox=decision.bbox, details={"access_event_id": event_id, **decision.details})
        return {
            "decision": decision.decision,
            "reason": decision.reason,
            "zone": self.zone,
            "staffId": decision.staff_id,
            "name": name,
            "matchScore": None if decision.match_score is None else round(decision.match_score, 3),
            "livenessScore": None if decision.liveness_score is None else round(decision.liveness_score, 3),
            "planarity": decision.planarity,
            "samples": decision.samples,
            "eventId": event_id,
        }

    # ---------- enrollment from the door camera ----------

    def capture_enrollment_crops(self, seconds: float = 6.0, max_photos: int = 3,
                                 min_gap: float = 0.6) -> tuple[list, dict]:
        """
        Watch the camera for a few seconds and keep up to max_photos frames in
        which exactly one person is at the door and their face passes the
        enrollment quality bar. Returns (face crops, counts of rejection reasons).
        """
        candidates, reasons = [], {}
        last_id, end = 0, time.time() + seconds

        def reject(reason):
            reasons[reason] = reasons.get(reason, 0) + 1

        while time.time() < end:
            frame, frame_id = self.reader.latest()
            if frame is None or frame_id == last_id:
                time.sleep(0.05)
                continue
            last_id = frame_id
            detections = self.engine.detect(frame)
            if not detections:
                reject("no face")
                continue
            main = max(detections, key=lambda d: d.ipd)
            if any(d is not main and d.ipd >= 0.4 * main.ipd for d in detections):
                reject("more than one person in view")
                continue
            report = quality.assess(main, self.engine.align(frame, main.kps), config.ENROLL_QUALITY)
            if not report.passed:
                for f in report.failures:
                    reject(f)
                continue
            candidates.append((time.time(), report.weight, face_crop(frame, main.bbox)))
            time.sleep(0.15)

        chosen = []
        for t, _, crop in sorted(candidates, key=lambda c: c[1], reverse=True):
            if all(abs(t - t2) >= min_gap for t2, _ in chosen):
                chosen.append((t, crop))
            if len(chosen) >= max_photos:
                break
        return [crop for _, crop in chosen], reasons

    def stop(self):
        self.running = False
        self._thread.join(timeout=2)
        self.reader.stop()

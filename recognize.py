"""
Live access-control view in a local window: reads one camera, runs the same
decision pipeline as the server (pipeline.py), draws each person's status,
and raises alarms through alarms.py (database + sound + webhook).

    uv run recognize.py              # first zone in config.CAMERAS
    uv run recognize.py --zone vault

This is a reference implementation, not a certified security product. See
README "Before you rely on this operationally".
"""

import argparse
import time

import cv2

import alarms
import config
import database
from camera import CameraReader, normalize_camera
from face_engine import load_engine
from liveness import load_liveness
from pipeline import ZonePipeline

# BGR colours per status.
_COLORS = {
    "verified": (0, 200, 0),
    "pending": (0, 200, 255),
}
_ALERT = (0, 0, 220)


def main():
    parser = argparse.ArgumentParser(description="Run the access-control pipeline with a local video window.")
    parser.add_argument("--zone", default=next(iter(config.CAMERAS)), help="Zone from config.CAMERAS")
    args = parser.parse_args()
    if args.zone not in config.CAMERAS:
        parser.error(f"unknown zone {args.zone!r} (configured: {', '.join(config.CAMERAS)})")

    database.init_db()
    engine = load_engine()
    pipeline = ZonePipeline(args.zone, engine, load_liveness())
    cam = normalize_camera(config.CAMERAS[args.zone])
    reader = CameraReader(cam["source"], cam["width"], cam["height"], name=args.zone)

    print(f"Access control running for {args.zone}. Press 'q' in the video window to quit.")
    last_id = 0
    try:
        while True:
            frame, frame_id = reader.latest()
            if frame is None or frame_id == last_id:
                time.sleep(0.005)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                continue
            last_id = frame_id
            now = time.time()

            for a in pipeline.process(frame, now):
                alarms.record_alarm(a.kind, a.severity, a.message, zone=args.zone, staff_id=a.staff_id,
                                    track_id=a.track_id, frame=a.frame, bbox=a.bbox, details=a.details)

            view = frame.copy()
            thickness = max(2, view.shape[1] // 640)
            for det in pipeline.overlay(now):
                x1, y1, x2, y2 = (int(v) for v in det["bbox"])
                color = _COLORS.get(det["status"], _ALERT)
                cv2.rectangle(view, (x1, y1), (x2, y2), color, thickness)
                cv2.putText(view, det["label"], (x1, max(20, y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6 * thickness / 2, color, thickness)

            cv2.imshow("Access control", view)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        reader.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

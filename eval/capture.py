"""
Record labelled test clips at a door camera, for eval/run_eval.py.

Each run records one clip and appends one line to <out>/labels.jsonl:

  # Reference photos/clips used to build each person's template
  uv run eval/capture.py --zone entrance --kind enroll   --person P01 --seconds 5
  # The person walking up to the door as they normally would
  uv run eval/capture.py --zone entrance --kind genuine  --person P01 --lighting day
  # Someone who isn't enrolled walking up (or presenting P01's badge)
  uv run eval/capture.py --zone entrance --kind impostor --person V07 --claimed P01
  # An attack: a photo or screen of P01 held up to the camera
  uv run eval/capture.py --zone entrance --kind attack   --person P01 --attack print_matte

Use stable pseudonymous IDs (P01, V07...) rather than names, and record a
demographic group with --group only where people have consented to it.
Attack types are free text, but keep them consistent, e.g. print_matte,
print_glossy, print_cutout_eyes, phone_photo, tablet_photo, phone_video,
tablet_video. The clips contain faces: keep <out> access-restricted and
delete it when the evaluation is done.
"""

import argparse
import datetime
import json
import os
import sys
import time

import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config  # noqa: E402
from camera import CameraReader, normalize_camera  # noqa: E402

KINDS = ("enroll", "genuine", "impostor", "attack")


def main():
    parser = argparse.ArgumentParser(description="Record a labelled test clip.")
    parser.add_argument("--zone", default=next(iter(config.CAMERAS)))
    parser.add_argument("--kind", required=True, choices=KINDS)
    parser.add_argument("--person", required=True, help="Pseudonymous ID of the person in the clip")
    parser.add_argument("--claimed", help="Identity being claimed (impostor/attack); defaults to --person")
    parser.add_argument("--attack", help="Attack type (required for --kind attack)")
    parser.add_argument("--lighting", default="", help="Free text, e.g. day, night, backlit")
    parser.add_argument("--group", default="", help="Demographic group, only with consent")
    parser.add_argument("--note", default="")
    parser.add_argument("--seconds", type=float, default=8.0)
    parser.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"))
    args = parser.parse_args()
    if args.kind == "attack" and not args.attack:
        parser.error("--attack is required for --kind attack")

    os.makedirs(args.out, mode=0o700, exist_ok=True)
    cam = normalize_camera(config.CAMERAS[args.zone])
    reader = CameraReader(cam["source"], cam["width"], cam["height"], name=args.zone)

    print("Waiting for the camera...")
    deadline = time.time() + 15
    while reader.latest()[0] is None and time.time() < deadline:
        time.sleep(0.1)
    frame, _ = reader.latest()
    if frame is None:
        reader.stop()
        raise SystemExit("No frames from the camera.")

    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    clip_name = f"{stamp}_{args.zone}_{args.kind}_{args.person}.mp4"
    path = os.path.join(args.out, clip_name)
    h, w = frame.shape[:2]
    fps = 15.0
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    print(f"Recording {args.seconds:.0f}s at {w}x{h}...")
    last_id, frames, start = 0, 0, time.time()
    while time.time() - start < args.seconds:
        frame, frame_id = reader.latest()
        if frame_id != last_id:
            last_id = frame_id
            writer.write(frame)
            frames += 1
        time.sleep(max(0.0, 1.0 / fps - 0.002))
    elapsed = time.time() - start
    writer.release()
    reader.stop()

    label = {
        "clip": clip_name,
        "kind": args.kind,
        "person": args.person,
        "claimed": args.claimed or args.person,
        "attack": args.attack,
        "lighting": args.lighting,
        "group": args.group,
        "zone": args.zone,
        "note": args.note,
        # The rate frames were actually captured at; run_eval.py uses it to
        # replay the clip with real timings.
        "fps": round(frames / elapsed, 2) if elapsed > 0 else fps,
        "frames": frames,
        "recorded_at": stamp,
    }
    with open(os.path.join(args.out, "labels.jsonl"), "a") as f:
        f.write(json.dumps(label) + "\n")
    print(f"Saved {frames} frames to {path}")


if __name__ == "__main__":
    main()

"""
Build embeddings for another recognizer from the enrollment photos already
on disk, so you can switch models (or compare them) without re-enrolling
everyone.

    python reembed.py --recognizer onnx:/models/adaface_ir101.onnx
    python reembed.py --recognizer insightface:antelopev2

New rows are tagged with the recognizer's model_id; existing embeddings are
kept, so switching config.RECOGNIZER back and forth is safe. Photos that
already have an embedding for this recognizer are skipped.
"""

import argparse
from pathlib import Path

import cv2

import config
import database
from face_engine import load_engine


def main():
    parser = argparse.ArgumentParser(description="Re-embed stored enrollment photos with another recognizer.")
    parser.add_argument("--recognizer", required=True, help='e.g. "onnx:/path/model.onnx"')
    parser.add_argument("--detector", default=config.DETECTOR)
    args = parser.parse_args()

    database.init_db()
    engine = load_engine(args.detector, args.recognizer)
    print(f"Re-embedding with {engine.model_id}")

    added = skipped = failed = 0
    for staff in database.list_staff():
        for photo in database.list_source_photos(staff["id"]):
            if database.has_embedding(staff["id"], photo, engine.model_id):
                skipped += 1
                continue
            image = cv2.imread(str(Path(config.ENROLLMENT_DIR) / Path(photo).name))
            detections = engine.detect(image) if image is not None else []
            if not detections:
                print(f"  [fail] {staff['name']}: {photo}: no face found")
                failed += 1
                continue
            # The photo already passed enrollment checks; take its main face.
            main_face = max(detections, key=lambda d: d.ipd)
            database.add_embedding(staff["id"], engine.embed(image, main_face.kps),
                                   source_photo=photo, model_id=engine.model_id)
            added += 1

    print(f"Done: {added} added, {skipped} already present, {failed} failed.")
    if engine.model_id not in config.MATCH_THRESHOLDS:
        print(f"Next: calibrate thresholds for {engine.model_id} with eval/run_eval.py and add them "
              "to config.MATCH_THRESHOLDS before switching config.RECOGNIZER to it.")


if __name__ == "__main__":
    main()

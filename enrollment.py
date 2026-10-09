"""
Enrollment: turning a photo (uploaded in the dashboard, given to enroll.py,
or captured at a door camera) into a stored face embedding.

Enrollment photos become the reference for every later decision, so each
one is checked before it's accepted:
  - exactly one person in the photo (no silently picking the biggest face);
  - quality at config.ENROLL_QUALITY: close enough, sharp, well lit, frontal;
  - it looks like this person's existing photos, if they have any;
  - it doesn't look like someone else who's already enrolled.
"""

import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

import config
import database
import matcher
import quality


@dataclass
class EnrollResult:
    ok: bool
    message: str                      # what went wrong, or the stored file name
    source_photo: Optional[str] = None
    quality: Optional[dict] = None


def decode_image(data: bytes) -> Optional[np.ndarray]:
    return cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)


def enroll_image(engine, staff_id: int, image: np.ndarray, filename: str,
                 original_bytes: Optional[bytes] = None) -> EnrollResult:
    """
    Check one photo and, if it passes, store it and its embedding. The
    original file bytes are kept when given (uploads); otherwise the image
    is saved as JPEG (camera captures).
    """
    label = Path(filename or "photo.jpg").name
    if image is None:
        return EnrollResult(False, f"{label}: could not read image")

    detections = engine.detect(image)
    if not detections:
        return EnrollResult(False, f"{label}: no face detected")
    main = max(detections, key=lambda d: d.ipd)
    others = [d for d in detections if d is not main and d.ipd >= 0.4 * main.ipd and d.det_score >= 0.5]
    if others:
        return EnrollResult(False, f"{label}: {len(others) + 1} people in the photo; use one with only this person")

    crop = engine.align(image, main.kps)
    report = quality.assess(main, crop, config.ENROLL_QUALITY)
    if not report.passed:
        return EnrollResult(False, f"{label}: photo not good enough ({', '.join(report.failures)})",
                            quality=report.to_dict())

    embedding = engine.embed_aligned(crop)
    gallery = matcher.get_gallery(engine.model_id, include_inactive=True)
    limits = matcher.thresholds_for(engine.model_id)

    own_score, _ = matcher.verify(embedding, staff_id, gallery)
    if 0 <= own_score < limits["reject"]:
        return EnrollResult(False, f"{label}: doesn't look like this person's other photos "
                                   f"(similarity {own_score:.2f})", quality=report.to_dict())
    other = matcher.best_other(embedding, gallery, exclude_staff_id=staff_id)
    if other and other[2] >= limits["duplicate"]:
        return EnrollResult(False, f"{label}: looks like {other[1]}, who is already enrolled "
                                   f"(similarity {other[2]:.2f})", quality=report.to_dict())

    dest_name = f"{staff_id}_{uuid.uuid4().hex[:8]}_{label}"
    dest = Path(config.ENROLLMENT_DIR) / dest_name
    os.makedirs(config.ENROLLMENT_DIR, exist_ok=True)
    if original_bytes is not None:
        dest.write_bytes(original_bytes)
    else:
        if dest.suffix.lower() not in (".jpg", ".jpeg"):
            dest = dest.with_suffix(".jpg")
            dest_name = dest.name
        cv2.imwrite(str(dest), image, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

    database.add_embedding(staff_id, embedding, source_photo=dest_name,
                           model_id=engine.model_id, quality=report.to_dict())
    return EnrollResult(True, dest_name, source_photo=dest_name, quality=report.to_dict())


def face_crop(frame: np.ndarray, bbox, margin: float = 1.0) -> np.ndarray:
    """The face plus `margin` box-widths of context on each side, clipped to the frame."""
    x1, y1, x2, y2 = (float(v) for v in bbox[:4])
    w, h = x2 - x1, y2 - y1
    fh, fw = frame.shape[:2]
    left, top = max(0, int(x1 - margin * w)), max(0, int(y1 - margin * h))
    right, bottom = min(fw, int(x2 + margin * w)), min(fh, int(y2 + margin * h))
    return frame[top:bottom, left:right].copy()

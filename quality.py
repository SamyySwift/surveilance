"""
Per-face quality checks.

Every detected face is checked before it's used, so tiny, blurry, dark or
turned-away faces never drive a decision. The checks are cheap: they use the
detector's 5 landmarks and the aligned 112x112 face crop.

Limits come from config.LIVE_QUALITY (live video) and config.ENROLL_QUALITY
(enrollment, stricter). Calibrate them with eval/run_eval.py.
"""

import math
from dataclasses import asdict, dataclass, field

import cv2
import numpy as np

# Generic face proportions for the pose estimate, in units of the distance
# between the eyes. Vertical spacing comes from the ArcFace alignment
# template. The yaw depth was fitted against InsightFace's 3D landmark model
# (agreement within ~6 degrees on test faces). Pitch from 5 points is rough
# (~15 degrees, since nose position varies between people), so its limits
# are looser. Good enough to spot an obviously turned face; not a measurement.
_EYES_TO_NOSE = 0.571      # vertical, eyes down to nose tip
_NOSE_TO_MOUTH = 0.585     # vertical, nose tip down to mouth corners
_YAW_DEPTH = 0.43          # nose tip in front of the eyes, for yaw
_EYE_DEPTH = 0.30          # for pitch: eyes behind the nose tip...
_MOUTH_DEPTH = 0.25        # ...and mouth corners behind the nose tip

# Pixels between the eyes at which a face counts as full resolution when
# weighting frames against each other.
_FULL_WEIGHT_IPD = 90.0


@dataclass
class QualityReport:
    ipd_px: float
    det_score: float
    yaw_deg: float
    pitch_deg: float
    roll_deg: float
    sharpness: float
    brightness: float
    clipped_fraction: float
    failures: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.failures

    @property
    def weight(self) -> float:
        """How much this frame counts when a person's frames are averaged."""
        w = (
            self.det_score
            * min(1.0, self.ipd_px / _FULL_WEIGHT_IPD)
            * math.cos(math.radians(min(abs(self.yaw_deg), 80)))
            * math.cos(math.radians(min(abs(self.pitch_deg), 80)))
        )
        return max(0.05, float(w))

    def to_dict(self) -> dict:
        d = asdict(self)
        for k, v in d.items():
            if isinstance(v, float):
                d[k] = round(v, 3)
        return d


def estimate_pose(kps: np.ndarray) -> tuple[float, float, float]:
    """
    Approximate (yaw, pitch, roll) in degrees from 5 landmarks. Yaw comes
    from how far the nose sits from the midpoint between the eyes, pitch from
    where it sits between the eye line and the mouth line, after undoing roll.
    """
    kps = np.asarray(kps, dtype=np.float64)
    left_eye, right_eye, nose, mouth_l, mouth_r = kps
    eye_vec = right_eye - left_eye
    roll = math.atan2(eye_vec[1], eye_vec[0])

    # Rotate everything so the eyes are level.
    c, s = math.cos(-roll), math.sin(-roll)
    rot = np.array([[c, -s], [s, c]])
    centre = (left_eye + right_eye) / 2
    le, re, no, ml, mr = ((p - centre) @ rot.T for p in (left_eye, right_eye, nose, mouth_l, mouth_r))

    d_left = no[0] - le[0]
    d_right = re[0] - no[0]
    if d_left + d_right <= 1e-6:
        return 90.0, 0.0, math.degrees(roll)
    r = (d_left - d_right) / (d_left + d_right)
    yaw = math.degrees(math.atan(r / (2 * _YAW_DEPTH)))

    eye_y = (le[1] + re[1]) / 2
    mouth_y = (ml[1] + mr[1]) / 2
    span = mouth_y - eye_y
    if span <= 1e-6:
        return yaw, 90.0, math.degrees(roll)
    t = (no[1] - eye_y) / span
    a, b = _EYES_TO_NOSE, _NOSE_TO_MOUTH
    tan_pitch = (a - t * (a + b)) / (t * (_EYE_DEPTH - _MOUTH_DEPTH) - _EYE_DEPTH)
    pitch = math.degrees(math.atan(tan_pitch))
    return yaw, pitch, math.degrees(roll)


def _image_stats(aligned_crop: np.ndarray) -> tuple[float, float, float]:
    """(sharpness, brightness, clipped fraction) of the aligned face crop."""
    gray = aligned_crop if aligned_crop.ndim == 2 else cv2.cvtColor(aligned_crop, cv2.COLOR_BGR2GRAY)
    gray = gray.astype(np.float64)
    # The centre of the aligned crop is face; the corners can be background.
    h, w = gray.shape
    inner = (slice(h // 7, h - h // 7), slice(w // 7, w - w // 7))
    centre = gray[inner]
    # Fine detail relative to the face's overall contrast, so a dim but
    # sharp face doesn't read as blurry. The +100 (a contrast floor of ~10
    # grey levels) keeps a featureless, washed-out crop from scoring as sharp.
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F)[inner].var() / (centre.var() + 100.0))
    brightness = float(centre.mean())
    clipped = float(np.mean((centre <= 5) | (centre >= 250)))
    return sharpness, brightness, clipped


def assess(detection, aligned_crop: np.ndarray, limits: dict) -> QualityReport:
    """Measure one face and list every limit it fails (an empty list means usable)."""
    yaw, pitch, roll = estimate_pose(detection.kps)
    sharpness, brightness, clipped = _image_stats(aligned_crop)
    report = QualityReport(
        ipd_px=detection.ipd,
        det_score=float(detection.det_score),
        yaw_deg=yaw,
        pitch_deg=pitch,
        roll_deg=roll,
        sharpness=sharpness,
        brightness=brightness,
        clipped_fraction=clipped,
    )

    f = report.failures
    if report.ipd_px < limits["min_ipd_px"]:
        f.append("too far away / too small")
    if report.det_score < limits["min_det_score"]:
        f.append("face not clearly detected")
    if abs(yaw) > limits["max_yaw_deg"]:
        f.append("head turned")
    if abs(pitch) > limits["max_pitch_deg"]:
        f.append("head tilted up or down")
    if sharpness < limits["min_sharpness"]:
        f.append("blurry")
    if brightness < limits["min_brightness"]:
        f.append("too dark")
    if brightness > limits["max_brightness"]:
        f.append("too bright")
    if clipped > limits["max_clipped_fraction"]:
        f.append("over- or under-exposed")
    return report

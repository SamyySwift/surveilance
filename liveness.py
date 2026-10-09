"""
Liveness: is this a live face, or a photo or screen held up to the camera?

Layer 1, passive model. Scores each good-quality face crop. The open
baseline is MiniFASNet from Silent-Face-Anti-Spoofing (Apache-2.0), converted
to ONNX by tools/convert_minifasnet.py; models/liveness/MANIFEST.json records
where the weights came from and their SHA-256 hashes, which are checked on
every load. A commercial liveness SDK can replace it by providing the same
model_id attribute and score() method.

Layer 2, experimental parallax check across frames, from the 5 landmarks the
detector already gives us. A photo or screen is flat: when it's tilted, the
nose lands exactly where a flat surface would put it. A real head turning
shows the nose shifting relative to the eyes and mouth. Only a clear "flat"
result counts against someone; everything else is "inconclusive". It can't
catch a video replay of a real head (layer 1's job).

Neither layer is certified, and with ordinary RGB cameras open-source
liveness misses many attacks. Treat it as one layer behind the badge, and
measure it on your own attack set with eval/run_eval.py.
"""

import hashlib
import itertools
import json
import os
from typing import Optional

import cv2
import numpy as np

import config

# ---------- Layer 1: passive model ----------


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _context_crop(frame: np.ndarray, bbox, scale: float, out_w: int, out_h: int) -> np.ndarray:
    """
    The face box enlarged by `scale` (so the model can see phone bezels,
    paper edges and fingers), shifted to stay inside the frame, resized.
    Same arithmetic as Silent-Face's CropImage, which the models expect.
    """
    src_h, src_w = frame.shape[:2]
    x, y = float(bbox[0]), float(bbox[1])
    box_w = float(bbox[2] - bbox[0] + 1)
    box_h = float(bbox[3] - bbox[1] + 1)
    scale = min((src_h - 1) / box_h, (src_w - 1) / box_w, scale)
    new_w, new_h = box_w * scale, box_h * scale
    cx, cy = x + box_w / 2, y + box_h / 2
    left, top = cx - new_w / 2, cy - new_h / 2
    right, bottom = cx + new_w / 2, cy + new_h / 2
    if left < 0:
        right -= left
        left = 0
    if top < 0:
        bottom -= top
        top = 0
    if right > src_w - 1:
        left -= right - src_w + 1
        right = src_w - 1
    if bottom > src_h - 1:
        top -= bottom - src_h + 1
        bottom = src_h - 1
    crop = frame[int(top): int(bottom) + 1, int(left): int(right) + 1]
    return cv2.resize(crop, (out_w, out_h))


class MiniFASNetLiveness:
    def __init__(self, model_dir: str):
        import onnxruntime as ort

        with open(os.path.join(model_dir, "MANIFEST.json")) as f:
            manifest = json.load(f)
        self._models = []
        for m in manifest["models"]:
            path = os.path.join(model_dir, m["file"])
            actual = _sha256(path)
            if actual != m["onnx_sha256"]:
                raise ValueError(f"{m['file']} doesn't match MANIFEST.json (sha256 {actual[:12]}...); "
                                 "refusing to load a model that may have been altered")
            # These models are tiny; CPU is faster than handing them to CoreML.
            session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
            w, h = m["input_size"]
            self._models.append((session, session.get_inputs()[0].name, float(m["scale"]),
                                 int(w), int(h), int(m["live_class_index"])))
        self.model_id = f"minifasnet@{manifest['source_commit'][:8]}"

    def score(self, frame: np.ndarray, bbox) -> float:
        """Probability (0-1) that this is a live face, averaged over the models."""
        probs = []
        for session, input_name, scale, w, h, live_idx in self._models:
            crop = _context_crop(frame, bbox, scale, w, h)
            # BGR, 0-255, not normalised: what the original models were trained on.
            x = crop.astype(np.float32).transpose(2, 0, 1)[None]
            logits = session.run(None, {input_name: x})[0][0]
            e = np.exp(logits - logits.max())
            probs.append(float(e[live_idx] / e.sum()))
        return float(np.mean(probs))


class LivenessUnavailable:
    """
    Stands in when liveness is switched on but its model can't be loaded.
    score() returns None, so "enforce" mode denies every badge entry instead
    of quietly letting people through unchecked.
    """

    model_id = "unavailable"

    def __init__(self, reason: str):
        self.reason = reason

    def score(self, frame, bbox) -> Optional[float]:
        return None


def load_liveness():
    """The configured liveness model, LivenessUnavailable if it can't load, or None if off."""
    if config.LIVENESS_MODE == "off":
        return None
    if config.LIVENESS_BACKEND != "minifasnet":
        raise ValueError(f"Unknown LIVENESS_BACKEND {config.LIVENESS_BACKEND!r}")
    try:
        model = MiniFASNetLiveness(config.LIVENESS_MODEL_DIR)
    except Exception as e:
        print(f"[liveness] ERROR: could not load the liveness model from {config.LIVENESS_MODEL_DIR}: {e}")
        if config.LIVENESS_MODE == "enforce":
            print("[liveness] LIVENESS_MODE is 'enforce', so badge entries will be denied until this is fixed.")
        return LivenessUnavailable(str(e))
    print(f"[liveness] {model.model_id} loaded, mode={config.LIVENESS_MODE}")
    return model


# ---------- Layer 2: parallax (flat-face) check ----------

FLAT = "flat"
THREE_D = "3d"
INCONCLUSIVE = "inconclusive"

# In units of the distance between the eyes.
_MIN_SHAPE_CHANGE = 0.08  # the eyes/mouth outline must have changed shape this much (something turned)
_FLAT_RESIDUAL = 0.03     # nose within this of where a flat surface puts it -> consistent with flat
_DEPTH_RESIDUAL = 0.06    # nose at least this far off -> real depth


def _outline_aspect(kps: np.ndarray) -> float:
    eye_mid = (kps[0] + kps[1]) / 2
    mouth_mid = (kps[3] + kps[4]) / 2
    return float(np.linalg.norm(kps[1] - kps[0]) / max(np.linalg.norm(mouth_mid - eye_mid), 1e-6))


def parallax(kps_a: np.ndarray, kps_b: np.ndarray) -> tuple[float, float]:
    """
    (shape_change, nose_residual) between two frames of the same face.
    Maps the eyes and mouth corners from frame a to frame b with a
    homography - exact for anything flat - and measures how far the real
    nose is from where that mapping puts it.
    """
    src = np.asarray(kps_a, dtype=np.float32)[[0, 1, 3, 4]]
    dst = np.asarray(kps_b, dtype=np.float32)[[0, 1, 3, 4]]
    h = cv2.getPerspectiveTransform(src, dst)
    nose = np.asarray(kps_a[2], dtype=np.float32).reshape(1, 1, 2)
    predicted = cv2.perspectiveTransform(nose, h)[0, 0]
    ipd_b = max(float(np.linalg.norm(kps_b[1] - kps_b[0])), 1e-6)
    residual = float(np.linalg.norm(np.asarray(kps_b[2]) - predicted)) / ipd_b
    change = abs(_outline_aspect(kps_b) / _outline_aspect(kps_a) - 1.0)
    return change, residual


def planarity_verdict(kps_list: list[np.ndarray], max_frames: int = 12) -> str:
    """
    FLAT only when the face visibly turned in at least two frame pairs and
    every such pair stayed consistent with a flat surface; THREE_D as soon
    as one pair shows real depth; otherwise INCONCLUSIVE.
    """
    frames = kps_list[-max_frames:]
    turned = flat = 0
    for a, b in itertools.combinations(frames, 2):
        try:
            change, residual = parallax(a, b)
        except cv2.error:
            continue
        if change < _MIN_SHAPE_CHANGE:
            continue
        turned += 1
        if residual >= _DEPTH_RESIDUAL:
            return THREE_D
        if residual <= _FLAT_RESIDUAL:
            flat += 1
    if turned >= 2 and flat == turned:
        return FLAT
    return INCONCLUSIVE

"""
Face detection and recognition engines.

The rest of the codebase talks to a FaceEngine, which pairs a detector with
a recognizer. Both are picked by name in config.py (DETECTOR, RECOGNIZER) so
models can be swapped and compared with eval/run_eval.py. A recognizer's
model_id is stored with every embedding it makes, and embeddings from
different recognizers are never compared.

Detection and recognition are separate steps on purpose: the pipeline only
embeds faces that pass the quality gate, instead of embedding every face
the detector finds.

To plug in another engine (a different open model, or a commercial SDK),
write a class with the same methods as InsightFaceDetector or
OnnxRecognizer and register it in _DETECTORS / _RECOGNIZERS.
"""

import hashlib
import os
from dataclasses import dataclass

import numpy as np

import config


@dataclass
class Detection:
    bbox: np.ndarray       # x1, y1, x2, y2 in full-frame pixels
    kps: np.ndarray        # 5x2: left eye, right eye, nose, left mouth, right mouth (image left/right)
    det_score: float

    @property
    def ipd(self) -> float:
        """Pixels between the eyes - the standard measure of face resolution."""
        return float(np.linalg.norm(self.kps[1] - self.kps[0]))


def _insightface_pack(pack: str):
    """
    Load an InsightFace pack's detection and recognition models. buffalo_l
    also ships landmark/age/gender models we never use; skipping them saves
    three extra model runs per face.
    """
    from insightface.app import FaceAnalysis

    app = FaceAnalysis(
        name=pack,
        allowed_modules=["detection", "recognition"],
        providers=config.EXECUTION_PROVIDERS,
    )
    # ctx_id selects a "device index" - meaningful for CUDA, harmless for
    # CoreML/CPU. A NEGATIVE ctx_id forces CPU-only in InsightFace and would
    # silently disable CoreML/CUDA acceleration, so this must stay >= 0.
    app.prepare(ctx_id=0, det_size=config.DETECTION_SIZE, det_thresh=config.DETECTION_THRESHOLD)
    return app


_PACK_CACHE = {}


def _cached_pack(pack: str):
    if pack not in _PACK_CACHE:
        _PACK_CACHE[pack] = _insightface_pack(pack)
    return _PACK_CACHE[pack]


def _log_providers(label: str, model):
    """
    ONNX Runtime falls back to CPU without raising an error if a requested
    provider (e.g. CoreML) isn't available or fails to load - so print what's
    actually running rather than assuming the GPU is in use.
    """
    session = getattr(model, "session", None)
    if session is not None:
        print(f"[face_engine] {label} running on: {session.get_providers()}")


class InsightFaceDetector:
    def __init__(self, pack: str):
        self.model_id = f"insightface:{pack}"
        self._model = _cached_pack(pack).det_model
        _log_providers(f"detector {self.model_id}", self._model)

    def detect(self, frame: np.ndarray) -> list[Detection]:
        bboxes, kpss = self._model.detect(frame, max_num=0, metric="default")
        if kpss is None:
            return []
        return [
            Detection(bbox=b[:4].astype(np.float32), kps=k.astype(np.float32), det_score=float(b[4]))
            for b, k in zip(bboxes, kpss)
        ]


class OnnxRecognizer:
    """
    ArcFace-style recognizer: aligns the face to the standard 5-point template
    and returns a unit-length embedding. Wraps InsightFace's ArcFaceONNX, which
    reads the model's input size and normalisation from the ONNX file.
    """

    def __init__(self, model, model_id: str):
        from insightface.utils import face_align

        self._model = model
        self._norm_crop = face_align.norm_crop
        self.model_id = model_id
        self.input_size = int(model.input_size[0])
        _log_providers(f"recognizer {model_id}", model)

    def align(self, frame: np.ndarray, kps: np.ndarray) -> np.ndarray:
        # Landmarks are in full-frame pixels, so the crop uses the camera's
        # full resolution rather than the detector's downscaled copy.
        return self._norm_crop(frame, landmark=kps, image_size=self.input_size)

    def embed_aligned(self, crop: np.ndarray) -> np.ndarray:
        feat = self._model.get_feat(crop).flatten().astype(np.float32)
        return feat / (np.linalg.norm(feat) + 1e-10)


def _insightface_recognizer(pack: str) -> OnnxRecognizer:
    app = _cached_pack(pack)
    if "recognition" not in app.models:
        raise ValueError(f"InsightFace pack {pack!r} has no recognition model")
    return OnnxRecognizer(app.models["recognition"], model_id=f"insightface:{pack}")


def _onnx_recognizer(path: str) -> OnnxRecognizer:
    from insightface.model_zoo import model_zoo

    path = os.path.expanduser(path)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Recognizer model not found: {path}")
    model = model_zoo.get_model(path, providers=config.EXECUTION_PROVIDERS)
    if model is None or getattr(model, "taskname", None) != "recognition":
        raise ValueError(f"{path} doesn't look like an ArcFace-style recognition model")
    with open(path, "rb") as f:
        digest = hashlib.sha256(f.read()).hexdigest()[:12]
    # The hash means a different file under the same name gets a new id, so
    # its embeddings are never mixed with the old file's.
    return OnnxRecognizer(model, model_id=f"onnx:{os.path.basename(path)}@{digest}")


_DETECTORS = {
    "insightface": InsightFaceDetector,
}
_RECOGNIZERS = {
    "insightface": _insightface_recognizer,
    "onnx": _onnx_recognizer,
}


def _split_spec(spec: str):
    kind, sep, arg = spec.partition(":")
    if not sep or not arg:
        raise ValueError(f"Engine spec should look like 'kind:argument', got {spec!r}")
    return kind, arg


class FaceEngine:
    def __init__(self, detector, recognizer):
        self.detector = detector
        self.recognizer = recognizer

    @property
    def model_id(self) -> str:
        """Identifies the recognizer; stored with every embedding."""
        return self.recognizer.model_id

    def detect(self, frame: np.ndarray) -> list[Detection]:
        return self.detector.detect(frame)

    def align(self, frame: np.ndarray, kps: np.ndarray) -> np.ndarray:
        return self.recognizer.align(frame, kps)

    def embed_aligned(self, crop: np.ndarray) -> np.ndarray:
        return self.recognizer.embed_aligned(crop)

    def embed(self, frame: np.ndarray, kps: np.ndarray) -> np.ndarray:
        return self.embed_aligned(self.align(frame, kps))


def load_engine(detector_spec: str = None, recognizer_spec: str = None) -> FaceEngine:
    detector_spec = detector_spec or config.DETECTOR
    recognizer_spec = recognizer_spec or config.RECOGNIZER

    det_kind, det_arg = _split_spec(detector_spec)
    rec_kind, rec_arg = _split_spec(recognizer_spec)
    if det_kind not in _DETECTORS:
        raise ValueError(f"Unknown detector {det_kind!r}; known: {', '.join(_DETECTORS)}")
    if rec_kind not in _RECOGNIZERS:
        raise ValueError(f"Unknown recognizer {rec_kind!r}; known: {', '.join(_RECOGNIZERS)}")
    return FaceEngine(_DETECTORS[det_kind](det_arg), _RECOGNIZERS[rec_kind](rec_arg))

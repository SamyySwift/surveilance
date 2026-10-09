"""
Matching face embeddings against enrolled staff.

  verify()    1:1 - is this the person the badge belongs to?
  identify()  1:N - who among everyone enrolled is this, if anyone?

The gallery is cached in memory per recognizer model and reloaded only when
enrolled faces change (database.get_gallery_version, which every write
bumps - including writes from the CLI tools in another process).
"""

import threading
from dataclasses import dataclass
from typing import Optional

import numpy as np

import config
import database


def cosine_similarity_batch(probe: np.ndarray, gallery: np.ndarray) -> np.ndarray:
    """probe: (D,), gallery: (N, D). Returns (N,) similarity scores in [-1, 1]."""
    if gallery.shape[0] == 0:
        return np.zeros((0,), dtype=np.float32)
    probe_n = probe / (np.linalg.norm(probe) + 1e-10)
    gallery_n = gallery / (np.linalg.norm(gallery, axis=1, keepdims=True) + 1e-10)
    return gallery_n @ probe_n


def aggregate_embeddings(embeddings: list[np.ndarray], weights: Optional[list[float]] = None) -> np.ndarray:
    """Weighted mean of unit embeddings, renormalised: one template for a person's frames."""
    stack = np.stack(embeddings).astype(np.float32)
    w = np.ones(len(embeddings), dtype=np.float32) if weights is None else np.asarray(weights, dtype=np.float32)
    mean = (stack * w[:, None]).sum(axis=0) / max(float(w.sum()), 1e-10)
    return mean / (np.linalg.norm(mean) + 1e-10)


@dataclass
class Gallery:
    model_id: str
    staff_ids: np.ndarray   # (N,) one entry per stored embedding
    names: list[str]
    vectors: np.ndarray     # (N, D) unit vectors

    @classmethod
    def from_arrays(cls, model_id, staff_ids, names, vectors):
        vectors = np.asarray(vectors, dtype=np.float32)
        if len(vectors):
            vectors = vectors / (np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-10)
        return cls(model_id, np.asarray(staff_ids, dtype=np.int64), list(names), vectors)

    def __len__(self):
        return len(self.names)

    def identity_scores(self, embedding: np.ndarray) -> dict[int, tuple[str, float]]:
        """Best score per enrolled person: {staff_id: (name, score)}."""
        if len(self) == 0:
            return {}
        scores = cosine_similarity_batch(embedding, self.vectors)
        best: dict[int, tuple[str, float]] = {}
        for sid, name, s in zip(self.staff_ids.tolist(), self.names, scores.tolist()):
            if sid not in best or s > best[sid][1]:
                best[sid] = (name, s)
        return best


_cache: dict[tuple, tuple[int, Gallery]] = {}
_cache_lock = threading.Lock()


def get_gallery(model_id: str, include_inactive: bool = False) -> Gallery:
    """Enrolled embeddings for one recognizer, reloaded only when they change."""
    version = database.get_gallery_version()
    key = (model_id, include_inactive)
    with _cache_lock:
        cached = _cache.get(key)
        if cached and cached[0] == version:
            return cached[1]
    staff_ids, names, vectors = database.load_all_embeddings(model_id, include_inactive=include_inactive)
    gallery = Gallery.from_arrays(model_id, staff_ids, names, vectors)
    with _cache_lock:
        _cache[key] = (version, gallery)
    return gallery


_warned_models = set()


def thresholds_for(model_id: str) -> dict:
    if model_id in config.MATCH_THRESHOLDS:
        return config.MATCH_THRESHOLDS[model_id]
    if model_id not in _warned_models:
        _warned_models.add(model_id)
        print(f"[matcher] WARNING: no thresholds configured for {model_id}; using defaults. "
              "Calibrate with eval/run_eval.py and add them to config.MATCH_THRESHOLDS.")
    return config.DEFAULT_MATCH_THRESHOLDS


@dataclass
class IdentifyResult:
    staff_id: Optional[int]
    name: Optional[str]
    score: float          # best person's score (-1 if nobody is enrolled)
    runner_up: float      # second-best person's score (-1 if none)
    accepted: bool


def identify(embedding: np.ndarray, gallery: Gallery, accept: Optional[float] = None,
             margin: Optional[float] = None) -> IdentifyResult:
    """
    1:N search. Accepted only if the best person clears `accept` AND beats the
    runner-up by `margin` - two similar-looking enrolled people shouldn't be
    a coin toss. staff_id/name are always the best candidate, accepted or not.
    """
    t = thresholds_for(gallery.model_id)
    accept = t["identify_accept"] if accept is None else accept
    margin = t["identify_margin"] if margin is None else margin

    ranked = sorted(gallery.identity_scores(embedding).items(), key=lambda kv: kv[1][1], reverse=True)
    if not ranked:
        return IdentifyResult(None, None, -1.0, -1.0, False)
    best_id, (best_name, best) = ranked[0]
    runner_up = ranked[1][1][1] if len(ranked) > 1 else -1.0
    accepted = best >= accept and (best - runner_up) >= margin
    return IdentifyResult(best_id, best_name, float(best), float(runner_up), accepted)


def verify(embedding: np.ndarray, staff_id: int, gallery: Gallery,
           accept: Optional[float] = None) -> tuple[float, bool]:
    """
    1:1 check against one person's templates. Returns (score, accepted);
    score is -1 if that person has no embeddings for this model.
    """
    accept = thresholds_for(gallery.model_id)["verify_accept"] if accept is None else accept
    mask = gallery.staff_ids == staff_id
    if not mask.any():
        return -1.0, False
    score = float(cosine_similarity_batch(embedding, gallery.vectors[mask]).max())
    return score, score >= accept


def best_other(embedding: np.ndarray, gallery: Gallery, exclude_staff_id: Optional[int]):
    """Closest enrolled person other than exclude_staff_id: (staff_id, name, score) or None."""
    others = [(sid, ns) for sid, ns in gallery.identity_scores(embedding).items() if sid != exclude_staff_id]
    if not others:
        return None
    sid, (name, score) = max(others, key=lambda kv: kv[1][1])
    return sid, name, float(score)

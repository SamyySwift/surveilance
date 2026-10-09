"""
Face tracker: gives each face a stable ID across frames, so evidence
(embeddings, quality, liveness) builds up per person rather than per screen
position.

Faces are matched to existing tracks by box overlap (IoU) using Hungarian
assignment. A face that moved further than overlap allows between processed
frames can still match on centre distance if its size is similar. Tracks not
seen for max_age seconds are dropped.
"""

import itertools
from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

_NO_MATCH = 1e6


@dataclass
class Track:
    track_id: int
    bbox: np.ndarray
    first_seen: float
    last_seen: float
    hits: int = 1


def iou(a: np.ndarray, b: np.ndarray) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter <= 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return float(inter / (area_a + area_b - inter))


class IoUTracker:
    def __init__(self, iou_threshold: float = 0.25, max_center_shift: float = 0.6,
                 max_size_change: float = 1.6, max_age: float = 1.0):
        """
        iou_threshold     minimum overlap for a normal match
        max_center_shift  fallback: centre may move this many box-widths...
        max_size_change   ...if the box didn't grow or shrink more than this factor
        max_age           seconds a track survives without being seen
        """
        self.iou_threshold = iou_threshold
        self.max_center_shift = max_center_shift
        self.max_size_change = max_size_change
        self.max_age = max_age
        self.tracks: dict[int, Track] = {}
        self._ids = itertools.count(1)

    def _cost(self, track_box: np.ndarray, det_box: np.ndarray) -> float:
        overlap = iou(track_box, det_box)
        if overlap >= self.iou_threshold:
            return 1.0 - overlap
        tw, th = track_box[2] - track_box[0], track_box[3] - track_box[1]
        dw, dh = det_box[2] - det_box[0], det_box[3] - det_box[1]
        size = max(tw, th, 1e-6)
        ratio = max(dw, dh) / size
        if not (1 / self.max_size_change <= ratio <= self.max_size_change):
            return _NO_MATCH
        tc = np.array([(track_box[0] + track_box[2]) / 2, (track_box[1] + track_box[3]) / 2])
        dc = np.array([(det_box[0] + det_box[2]) / 2, (det_box[1] + det_box[3]) / 2])
        shift = float(np.linalg.norm(tc - dc)) / size
        if shift > self.max_center_shift:
            return _NO_MATCH
        return 1.0 + shift  # always worse than any overlap match

    def update(self, boxes: list[np.ndarray], now: float) -> tuple[list[int], list[int]]:
        """
        Assign a track ID to each box (returned in the same order) and drop
        stale tracks. Returns (ids, expired_ids).
        """
        expired = [tid for tid, t in self.tracks.items() if now - t.last_seen > self.max_age]
        for tid in expired:
            del self.tracks[tid]

        ids: list[int] = [-1] * len(boxes)
        track_ids = list(self.tracks)
        if boxes and track_ids:
            cost = np.array([[self._cost(self.tracks[tid].bbox, box) for box in boxes] for tid in track_ids])
            rows, cols = linear_sum_assignment(cost)
            for r, c in zip(rows, cols):
                if cost[r, c] >= _NO_MATCH:
                    continue
                track = self.tracks[track_ids[r]]
                track.bbox = np.asarray(boxes[c], dtype=np.float32)
                track.last_seen = now
                track.hits += 1
                ids[c] = track.track_id

        for i, box in enumerate(boxes):
            if ids[i] == -1:
                tid = next(self._ids)
                self.tracks[tid] = Track(tid, np.asarray(box, dtype=np.float32), now, now)
                ids[i] = tid
        return ids, expired

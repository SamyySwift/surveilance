"""
Per-zone decision pipeline, shared by the server (worker.py), the CLI viewer
(recognize.py) and the evaluation harness (eval/run_eval.py), so what gets
measured is exactly what runs at the door.

For each frame:
    detect faces -> track them -> quality-check each face ->
    embed and liveness-score the faces that pass -> update each person's
    evidence -> decide.

Decisions are made per tracked person from several good frames, never from
one frame. A person who never shows a usable face can't slip through
silently: after a while they become UNVERIFIABLE, which raises an alarm.

Two kinds of decision:
  - Monitoring (1:N): every tracked person is identified against everyone
    enrolled, and unknown, unauthorized or spoof-suspected people raise alarms.
  - Badge (1:1): badge_decision() checks that the face at the door is the
    badge holder's, is live, and is allowed in this zone.
"""

from collections import Counter, deque
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Callable, Optional

import numpy as np

import config
import database
import matcher
import quality
from liveness import FLAT, INCONCLUSIVE, planarity_verdict
from tracker import IoUTracker


class Status(str, Enum):
    PENDING = "pending"                  # collecting good frames
    VERIFIED = "verified"                # recognised and allowed in this zone
    UNAUTHORIZED = "unauthorized"        # recognised, but not allowed in this zone
    UNKNOWN = "unknown"                  # doesn't match anyone enrolled
    SPOOF_SUSPECTED = "spoof_suspected"  # failed the liveness check
    UNVERIFIABLE = "unverifiable"        # present, but never showed a usable face


ALARM_STATUSES = {Status.UNAUTHORIZED, Status.UNKNOWN, Status.SPOOF_SUSPECTED, Status.UNVERIFIABLE}

# Short prompts shown on the live view while a face isn't usable yet.
_HINTS = {
    "too far away / too small": "Move closer",
    "face not clearly detected": "Face the camera",
    "head turned": "Face the camera",
    "head tilted up or down": "Face the camera",
    "blurry": "Hold still",
    "too dark": "Too dark",
    "too bright": "Too bright",
    "over- or under-exposed": "Lighting problem",
}


@dataclass
class Settings:
    live_quality: dict
    evidence_window: float
    min_samples: int
    unverifiable_after: float
    decision_timeout: float
    track_max_age: float
    badge_min_samples: int
    badge_timeout: float
    door_min_ipd: float
    tailgate_window: float
    tailgate_grace: float
    deny_multiple_faces: bool
    liveness_mode: str
    liveness_threshold: float
    liveness_min_samples: int
    planarity_mode: str
    max_samples_per_track: int = 40

    @classmethod
    def from_config(cls, **overrides) -> "Settings":
        s = cls(
            live_quality=dict(config.LIVE_QUALITY),
            evidence_window=config.EVIDENCE_WINDOW_SECONDS,
            min_samples=config.MIN_SAMPLES_FOR_DECISION,
            unverifiable_after=config.UNVERIFIABLE_AFTER_SECONDS,
            decision_timeout=config.DECISION_TIMEOUT_SECONDS,
            track_max_age=config.TRACK_MAX_AGE_SECONDS,
            badge_min_samples=config.BADGE_MIN_SAMPLES,
            badge_timeout=config.BADGE_DECISION_TIMEOUT_SECONDS,
            door_min_ipd=config.DOOR_MIN_IPD_PX,
            tailgate_window=config.TAILGATE_WINDOW_SECONDS,
            tailgate_grace=config.TAILGATE_GRACE_SECONDS,
            deny_multiple_faces=config.DENY_WHEN_MULTIPLE_FACES,
            liveness_mode=config.LIVENESS_MODE,
            liveness_threshold=config.LIVENESS_THRESHOLD,
            liveness_min_samples=config.LIVENESS_MIN_SAMPLES,
            planarity_mode=config.PLANARITY_MODE,
        )
        return replace(s, **overrides)


@dataclass
class Sample:
    t: float
    kps: np.ndarray
    quality: quality.QualityReport
    embedding: np.ndarray
    liveness: Optional[float]


@dataclass
class Evidence:
    embedding: np.ndarray
    liveness: Optional[float]   # mean live-probability, None if not scored
    liveness_count: int
    planarity: str


@dataclass
class TrackState:
    track_id: int
    first_seen: float
    last_seen: float = 0.0
    hits: int = 0
    bbox: Optional[np.ndarray] = None
    ipd: float = 0.0
    samples: deque = field(default_factory=deque)
    last_quality: Optional[quality.QualityReport] = None
    recent_failures: Counter = field(default_factory=Counter)
    status: Status = Status.PENDING
    reason: str = ""
    staff_id: Optional[int] = None
    name: Optional[str] = None
    match_score: Optional[float] = None
    runner_up: Optional[float] = None
    liveness_score: Optional[float] = None
    planarity: str = INCONCLUSIVE
    alarmed: set = field(default_factory=set)
    granted_at: Optional[float] = None
    best_frame: Optional[np.ndarray] = None
    best_bbox: Optional[np.ndarray] = None
    best_weight: float = 0.0


@dataclass
class Alarm:
    kind: str
    severity: str          # "high", "medium" or "info"
    message: str
    staff_id: Optional[int] = None
    track_id: Optional[int] = None
    frame: Optional[np.ndarray] = None
    bbox: Optional[np.ndarray] = None
    details: dict = field(default_factory=dict)


@dataclass
class BadgeDecision:
    decision: str           # "grant", "deny" or "pending"
    reason: str
    staff_id: int
    track_id: Optional[int] = None
    match_score: Optional[float] = None
    liveness_score: Optional[float] = None
    planarity: Optional[str] = None
    samples: int = 0
    alarm_kind: Optional[str] = None   # alarm to raise with this decision, if any
    severity: str = "medium"
    frame: Optional[np.ndarray] = None
    bbox: Optional[np.ndarray] = None
    details: dict = field(default_factory=dict)

    @property
    def final(self) -> bool:
        return self.decision != "pending"


class ZonePipeline:
    def __init__(self, zone: str, engine, liveness=None, settings: Optional[Settings] = None,
                 gallery_fn: Optional[Callable] = None, authorizer: Optional[Callable] = None):
        """
        engine      face_engine.FaceEngine (or anything with detect/align/embed_aligned/model_id)
        liveness    liveness model (score(frame, bbox) -> float or None), or None
        gallery_fn  returns the matcher.Gallery to match against (default: enrolled staff)
        authorizer  staff_id -> may this person be in this zone? (default: their zones in the database)
        """
        self.zone = zone
        self.engine = engine
        self.settings = settings or Settings.from_config()
        self.liveness = liveness if self.settings.liveness_mode != "off" else None
        self.gallery_fn = gallery_fn or (lambda: matcher.get_gallery(engine.model_id))
        self.authorizer = authorizer or (lambda staff_id: zone in database.get_staff_zones(staff_id))
        self.tracker = IoUTracker(max_age=self.settings.track_max_age)
        self.tracks: dict[int, TrackState] = {}
        self._grant: Optional[tuple[int, float]] = None  # (track_id, time) of the last badge grant

    # ---------- frame processing ----------

    def process(self, frame: np.ndarray, now: float) -> list[Alarm]:
        detections = self.engine.detect(frame)
        ids, expired = self.tracker.update([d.bbox for d in detections], now)
        for tid in expired:
            self.tracks.pop(tid, None)

        seen = []
        for det, tid in zip(detections, ids):
            tr = self.tracks.get(tid)
            if tr is None:
                tr = self.tracks[tid] = TrackState(track_id=tid, first_seen=now)
            seen.append(self._observe(tr, frame, det, now))

        alarms = []
        for tr in seen:
            alarms.extend(self._evaluate(tr, now, frame))
        alarms.extend(self._check_tailgating(now, frame))
        return alarms

    def _observe(self, tr: TrackState, frame, det, now: float) -> TrackState:
        """Add one detection to a track. Returns the track (a fresh one if the person changed)."""
        tr.last_seen = now
        tr.hits += 1
        tr.bbox = det.bbox
        tr.ipd = det.ipd
        crop = self.engine.align(frame, det.kps)
        q = quality.assess(det, crop, self.settings.live_quality)
        tr.last_quality = q
        if not q.passed:
            tr.recent_failures.update(q.failures)
            return tr

        embedding = self.engine.embed_aligned(crop)
        if self._is_someone_else(tr, embedding, now):
            # The tracker kept the box but the face changed (one person
            # stepped into another's place). Start over rather than let the
            # newcomer inherit the previous person's decision.
            self.tracks[tr.track_id] = tr = TrackState(track_id=tr.track_id, first_seen=now,
                                                       last_seen=now, hits=1, bbox=det.bbox, ipd=det.ipd,
                                                       last_quality=q)
        live = self.liveness.score(frame, det.bbox) if self.liveness is not None else None
        tr.samples.append(Sample(now, det.kps.copy(), q, embedding, live))
        while len(tr.samples) > self.settings.max_samples_per_track:
            tr.samples.popleft()
        if q.weight >= tr.best_weight:
            tr.best_weight, tr.best_frame, tr.best_bbox = q.weight, frame, det.bbox.copy()
        return tr

    def _is_someone_else(self, tr: TrackState, embedding: np.ndarray, now: float) -> bool:
        samples = self._recent(tr, now)
        if len(samples) < 2:
            return False
        reference = matcher.aggregate_embeddings([s.embedding for s in samples])
        reject = matcher.thresholds_for(self.engine.model_id)["reject"]
        return float(reference @ embedding) < reject

    def _recent(self, tr: TrackState, now: float) -> list[Sample]:
        while tr.samples and now - tr.samples[0].t > self.settings.evidence_window:
            tr.samples.popleft()
        return list(tr.samples)

    def _evidence(self, samples: list[Sample]) -> Evidence:
        embedding = matcher.aggregate_embeddings([s.embedding for s in samples],
                                                 [s.quality.weight for s in samples])
        scores = [s.liveness for s in samples if s.liveness is not None]
        planarity = (planarity_verdict([s.kps for s in samples])
                     if self.settings.planarity_mode != "off" else INCONCLUSIVE)
        return Evidence(embedding, float(np.mean(scores)) if scores else None, len(scores), planarity)

    def _spoof_reason(self, ev: Evidence) -> Optional[str]:
        """Why this evidence fails liveness, if it does (only in enforce mode)."""
        s = self.settings
        if (s.liveness_mode == "enforce" and ev.liveness is not None
                and ev.liveness_count >= s.liveness_min_samples and ev.liveness < s.liveness_threshold):
            return f"liveness {ev.liveness:.2f} below {s.liveness_threshold:.2f}"
        if s.planarity_mode == "enforce" and ev.planarity == FLAT:
            return "face moved like a flat surface"
        return None

    # ---------- monitoring (1:N) ----------

    def _evaluate(self, tr: TrackState, now: float, frame) -> list[Alarm]:
        s = self.settings
        samples = self._recent(tr, now)
        age = now - tr.first_seen
        if len(samples) < s.min_samples:
            if tr.status == Status.PENDING and age >= s.unverifiable_after and tr.hits >= 3:
                problem = tr.recent_failures.most_common(1)[0][0] if tr.recent_failures else "no clear view"
                return self._set_status(tr, Status.UNVERIFIABLE, f"face not clearly visible ({problem})", frame)
            return []

        ev = self._evidence(samples)
        tr.liveness_score, tr.planarity = ev.liveness, ev.planarity
        gallery = self.gallery_fn()
        result = matcher.identify(ev.embedding, gallery)
        tr.staff_id, tr.name = result.staff_id, result.name
        tr.match_score, tr.runner_up = result.score, result.runner_up
        limits = matcher.thresholds_for(gallery.model_id)

        spoof = self._spoof_reason(ev)
        if spoof:
            return self._set_status(tr, Status.SPOOF_SUSPECTED, spoof, frame)
        if result.accepted:
            if self.authorizer(result.staff_id):
                return self._set_status(tr, Status.VERIFIED, "", frame)
            return self._set_status(tr, Status.UNAUTHORIZED, f"{result.name} has no access to {self.zone}", frame)
        if result.score < limits["reject"]:
            return self._set_status(tr, Status.UNKNOWN, "matches nobody enrolled", frame)
        if age >= s.decision_timeout and tr.status == Status.PENDING:
            why = ("too close to call between two enrolled people"
                   if result.score >= limits["identify_accept"] else "match too weak to confirm")
            return self._set_status(tr, Status.UNKNOWN, why, frame)
        # Between reject and accept: keep collecting. A person already
        # decided keeps their status rather than flipping back and forth.
        return []

    def _set_status(self, tr: TrackState, status: Status, reason: str, frame) -> list[Alarm]:
        if status == tr.status:
            return []
        if tr.status == Status.SPOOF_SUSPECTED:
            # Once a face has failed liveness it stays suspect for as long as
            # it's tracked; a genuine person can step away and come back.
            return []
        tr.status, tr.reason = status, reason
        if status not in ALARM_STATUSES or status in tr.alarmed:
            return []
        tr.alarmed.add(status)
        return [self._alarm_for(tr, frame)]

    def _alarm_for(self, tr: TrackState, frame) -> Alarm:
        z = self.zone
        details = {
            "reason": tr.reason,
            "closest_match": tr.name,
            "match_score": None if tr.match_score is None else round(tr.match_score, 3),
            "liveness": None if tr.liveness_score is None else round(tr.liveness_score, 3),
            "planarity": tr.planarity,
        }
        snap = tr.best_frame if tr.best_frame is not None else frame
        bbox = tr.best_bbox if tr.best_frame is not None else tr.bbox
        if tr.status == Status.UNKNOWN:
            return Alarm("unknown_person", "medium", f"Unrecognized person at {z}: {tr.reason}",
                         track_id=tr.track_id, frame=snap, bbox=bbox, details=details)
        if tr.status == Status.UNAUTHORIZED:
            return Alarm("unauthorized", "high", f"{tr.name} is not authorized for {z}",
                         staff_id=tr.staff_id, track_id=tr.track_id, frame=snap, bbox=bbox, details=details)
        if tr.status == Status.SPOOF_SUSPECTED:
            return Alarm("spoof_suspected", "high",
                         f"Possible photo or screen held up at {z} ({tr.reason})",
                         track_id=tr.track_id, frame=snap, bbox=bbox, details=details)
        return Alarm("unverifiable", "medium", f"Person at {z} could not be verified: {tr.reason}",
                     track_id=tr.track_id, frame=frame, bbox=tr.bbox, details=details)

    # ---------- badge + face (1:1) ----------

    def faces_at_door(self, now: float) -> list[TrackState]:
        """People visible right now and close enough to be at the door, closest first."""
        at_door = [tr for tr in self.tracks.values()
                   if now - tr.last_seen <= 0.7 and tr.ipd >= self.settings.door_min_ipd]
        return sorted(at_door, key=lambda t: t.ipd, reverse=True)

    def track_evidence(self, tr: TrackState, now: float) -> tuple[Optional[Evidence], int]:
        """A person's combined evidence from their recent good frames, and how many frames that is."""
        samples = self._recent(tr, now)
        return (self._evidence(samples) if samples else None), len(samples)

    def badge_decision(self, staff_id: int, now: float, started_at: float) -> BadgeDecision:
        """
        Decide whether the person at the door is the badge holder. Returns a
        "pending" decision while more frames are needed; the caller polls
        until the decision is final. After the timeout, pending becomes deny.
        """
        s = self.settings
        timed_out = now - started_at >= s.badge_timeout

        def deny(reason, kind, severity="high", **kw):
            return BadgeDecision("deny", reason, staff_id, alarm_kind=kind, severity=severity, **kw)

        def wait(reason, **kw):
            if timed_out:
                return deny(f"timed out: {reason}", "access_denied", "medium", **kw)
            return BadgeDecision("pending", reason, staff_id, **kw)

        if not self.authorizer(staff_id):
            return deny(f"not authorized for {self.zone}", "unauthorized")

        at_door = self.faces_at_door(now)
        if not at_door:
            return wait("no face at the door")
        tr = at_door[0]
        others = len(at_door) - 1
        ev, n_samples = self.track_evidence(tr, now)
        snapshot = dict(frame=tr.best_frame, bbox=tr.best_bbox)
        if n_samples < s.badge_min_samples:
            hint = tr.recent_failures.most_common(1)[0][0] if tr.recent_failures else "collecting frames"
            return wait(f"waiting for a clear view of the face ({hint})", track_id=tr.track_id, **snapshot)

        gallery = self.gallery_fn()
        score, accepted = matcher.verify(ev.embedding, staff_id, gallery)
        info = dict(track_id=tr.track_id, match_score=score, liveness_score=ev.liveness,
                    planarity=ev.planarity, samples=n_samples, details={"other_faces_at_door": others},
                    **snapshot)

        if score < 0:
            return deny("no enrolled face for this badge holder with the current recognizer",
                        "access_denied", "medium", **info)
        if s.liveness_mode == "enforce":
            if ev.liveness is None:
                return deny("liveness check unavailable", "liveness_unavailable", **info)
            if ev.liveness_count < s.liveness_min_samples:
                return wait("checking liveness", **info)
        spoof = self._spoof_reason(ev)
        if spoof:
            return deny(f"liveness check failed: {spoof}", "spoof_suspected", **info)
        if not accepted:
            if score < matcher.thresholds_for(gallery.model_id)["reject"]:
                return deny("face does not match the badge holder", "face_mismatch", **info)
            return wait(f"face not confirmed (similarity {score:.2f})", **info)
        if others and s.deny_multiple_faces:
            return deny("more than one person at the door", "multiple_faces", **info)

        tr.granted_at = now
        self._grant = (tr.track_id, now)
        decision = BadgeDecision("grant", "face matches badge holder", staff_id, **info)
        if others:
            decision.alarm_kind, decision.severity = "multiple_faces", "medium"
        return decision

    def _check_tailgating(self, now: float, frame) -> list[Alarm]:
        if self._grant is None:
            return []
        s = self.settings
        granted_track, granted_at = self._grant
        if now - granted_at > s.tailgate_window:
            self._grant = None
            return []
        alarms = []
        for tr in self.tracks.values():
            if (tr.track_id == granted_track or tr.granted_at is not None or "tailgating" in tr.alarmed
                    or now - tr.last_seen > 0.5 or tr.ipd < s.door_min_ipd):
                continue
            if now - max(tr.first_seen, granted_at) >= s.tailgate_grace:
                tr.alarmed.add("tailgating")
                alarms.append(Alarm(
                    "tailgating", "high",
                    f"Possible tailgating at {self.zone}: a second person is at the door after a badge grant",
                    track_id=tr.track_id, frame=frame, bbox=tr.bbox,
                    details={"granted_track": granted_track},
                ))
        return alarms

    # ---------- display ----------

    def overlay(self, now: float) -> list[dict]:
        """What to draw on the live view for each face currently visible."""
        out = []
        for tr in self.tracks.values():
            if now - tr.last_seen > 0.5 or tr.bbox is None:
                continue
            out.append({
                "trackId": tr.track_id,
                "bbox": [float(v) for v in tr.bbox],
                "status": tr.status.value,
                "label": self._label(tr),
                "name": tr.name if tr.status in (Status.VERIFIED, Status.UNAUTHORIZED) else None,
                "score": None if tr.match_score is None else round(tr.match_score, 3),
                "liveness": None if tr.liveness_score is None else round(tr.liveness_score, 3),
                "matched": tr.status == Status.VERIFIED,
            })
        return out

    def _label(self, tr: TrackState) -> str:
        if tr.status == Status.VERIFIED:
            return f"{tr.name} {tr.match_score:.2f}"
        if tr.status == Status.UNAUTHORIZED:
            return f"{tr.name}: not authorized"
        if tr.status == Status.UNKNOWN:
            return "Unrecognized"
        if tr.status == Status.SPOOF_SUSPECTED:
            return "Liveness failed"
        if tr.status == Status.UNVERIFIABLE:
            return "Face not visible"
        if tr.last_quality is not None and tr.last_quality.failures:
            return _HINTS.get(tr.last_quality.failures[0], "Checking")
        return "Checking"

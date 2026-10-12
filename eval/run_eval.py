"""
Measure accuracy on clips recorded at your own doors, and suggest thresholds.

    uv run eval/run_eval.py --data eval/data --out eval/report
    uv run eval/run_eval.py --data eval/data --recognizer onnx:/models/adaface.onnx --out eval/report-adaface

Clips come from eval/capture.py (labels.jsonl plus videos; "enroll" items
may also be still photos). Every probe clip is replayed through the same
pipeline the doors run (pipeline.py) with its real timings, so the numbers
describe the deployed system:

  Badge + face (1:1)      FNMR  staff wrongly rejected     FMR   impostors wrongly accepted
  Unknown people (1:N)    FNIR  staff missed/misidentified FPIR  strangers matched to someone
  Liveness (ISO/IEC 30107-3)  APCER attacks accepted, per attack type
                              BPCER real people rejected
  Failure to acquire (FTA): clips where no usable face was ever captured

Scores follow the door's behaviour: once a person first has enough good
frames, the door keeps re-checking until BADGE_DECISION_TIMEOUT_SECONDS, so a
clip's score is the best one reached in that window. That gives impostors
and attacks every chance the real door would - the conservative choice.

Writes report.html (open it in a browser), summary.json and scores.csv to
--out. They describe people's faces: keep them access-restricted.
"""

import argparse
import csv
import datetime
import html
import json
import math
import os
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import config  # noqa: E402
import matcher  # noqa: E402
import quality  # noqa: E402
from face_engine import load_engine  # noqa: E402
from liveness import FLAT, THREE_D, MiniFASNetLiveness  # noqa: E402
from pipeline import Settings, ZonePipeline  # noqa: E402

KINDS = ("enroll", "genuine", "impostor", "attack")
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
FMR_TARGETS = (1e-2, 1e-3, 1e-4)
GRID = np.round(np.arange(0.0, 1.0001, 0.01), 2)


# ---------- loading ----------

def load_labels(data_dir: str) -> list[dict]:
    items = []
    with open(os.path.join(data_dir, "labels.jsonl")) as f:
        for n, line in enumerate(f, 1):
            if not line.strip():
                continue
            item = json.loads(line)
            for key in ("clip", "kind", "person"):
                if key not in item:
                    raise SystemExit(f"labels.jsonl line {n}: missing {key!r}")
            if item["kind"] not in KINDS:
                raise SystemExit(f"labels.jsonl line {n}: kind must be one of {', '.join(KINDS)}")
            item["claimed"] = item.get("claimed") or item["person"]
            item["path"] = os.path.join(data_dir, item["clip"])
            if not os.path.isfile(item["path"]):
                raise SystemExit(f"labels.jsonl line {n}: {item['clip']} not found")
            items.append(item)
    return items


def read_frames(path: str):
    if os.path.splitext(path)[1].lower() in IMAGE_EXT:
        image = cv2.imread(path)
        if image is not None:
            yield image
        return
    cap = cv2.VideoCapture(path)
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            yield frame
    finally:
        cap.release()


# ---------- gallery from the "enroll" items ----------

def build_gallery(engine, items, per_clip: int = 3):
    """Templates per person, chosen the way camera enrollment does (enrollment quality, one person)."""
    people = sorted({it["person"] for it in items if it["kind"] == "enroll"})
    index = {p: i + 1 for i, p in enumerate(people)}
    staff_ids, names, vectors = [], [], []
    templates = Counter()
    for it in (i for i in items if i["kind"] == "enroll"):
        candidates = []
        for n, frame in enumerate(read_frames(it["path"])):
            detections = engine.detect(frame)
            if not detections:
                continue
            main = max(detections, key=lambda d: d.ipd)
            if any(d is not main and d.ipd >= 0.4 * main.ipd for d in detections):
                continue
            crop = engine.align(frame, main.kps)
            report = quality.assess(main, crop, config.ENROLL_QUALITY)
            if report.passed:
                candidates.append((report.weight, n, crop))
        chosen = []
        for _, n, crop in sorted(candidates, key=lambda c: -c[0]):
            if all(abs(n - m) >= 8 for m, _ in chosen):
                chosen.append((n, crop))
            if len(chosen) >= per_clip:
                break
        for _, crop in chosen:
            staff_ids.append(index[it["person"]])
            names.append(it["person"])
            vectors.append(engine.embed_aligned(crop))
            templates[it["person"]] += 1
    gallery = matcher.Gallery.from_arrays(engine.model_id, staff_ids, names,
                                          np.stack(vectors) if vectors else np.zeros((0, 1)))
    return gallery, index, templates, [p for p in people if templates[p] == 0]


# ---------- replaying probe clips ----------

@dataclass
class Window:
    t: float
    embedding: np.ndarray
    liveness: Optional[float]
    liveness_count: int
    planarity: str


def replay(engine, live_model, gallery, settings, item) -> tuple[list[Window], Optional[float]]:
    """Feed a clip through a fresh pipeline; collect the evidence the door would decide on."""
    pipe = ZonePipeline("eval", engine, live_model, settings,
                        gallery_fn=lambda: gallery, authorizer=lambda staff_id: True)
    fps = float(item.get("fps") or 15.0)
    windows, t0, appeared = [], None, None
    for n, frame in enumerate(read_frames(item["path"])):
        t = n / fps
        pipe.process(frame, t)
        at_door = pipe.faces_at_door(t)
        if not at_door:
            continue
        tr = at_door[0]
        appeared = tr.first_seen if appeared is None else appeared
        ev, n_samples = pipe.track_evidence(tr, t)
        if ev is None or n_samples < settings.badge_min_samples:
            continue
        t0 = t if t0 is None else t0
        if t - t0 > settings.badge_timeout:
            break
        windows.append(Window(t, ev.embedding, ev.liveness, ev.liveness_count, ev.planarity))
    return windows, (None if t0 is None else t0 - appeared)


def summarize_clip(windows, gallery, index, settings) -> dict:
    per_person = {}
    for person, sid in index.items():
        scores = [matcher.verify(w.embedding, sid, gallery, accept=2.0)[0] for w in windows]
        per_person[person] = max(scores) if scores else None
    searches = []
    for w in windows:
        ranked = sorted(gallery.identity_scores(w.embedding).items(), key=lambda kv: -kv[1][1])
        if ranked:
            best_name, best = ranked[0][1]
            runner_up = ranked[1][1][1] if len(ranked) > 1 else -1.0
            searches.append((best_name, best, runner_up))
    lives = [w.liveness for w in windows
             if w.liveness is not None and w.liveness_count >= settings.liveness_min_samples]
    verdicts = {w.planarity for w in windows}
    planarity = FLAT if FLAT in verdicts else THREE_D if THREE_D in verdicts else "inconclusive"
    return {"per_person": per_person, "searches": searches,
            "liveness": max(lives) if lives else None, "planarity": planarity}


# ---------- metrics ----------

def rate(hits: int, total: int) -> Optional[float]:
    return hits / total if total else None


def threshold_for_fmr(impostor: np.ndarray, target: float) -> Optional[float]:
    """Lowest threshold (0.001 steps) at which FMR <= target, if the data can show it."""
    if len(impostor) < 1 / target:
        return None
    for t in np.round(np.arange(-1.0, 1.0005, 0.001), 3):
        if np.mean(impostor >= t) <= target:
            return float(t)
    return None


def evaluate(items, results, index, templates, failed_enroll, model_id, live_model_id, settings) -> dict:
    limits = matcher.thresholds_for(model_id)
    t_verify, t_ident, margin = limits["verify_accept"], limits["identify_accept"], limits["identify_margin"]
    probes = [(it, results[it["clip"]]) for it in items if it["kind"] != "enroll"]

    # Face matching (1:1)
    genuine, impostor, fta = [], [], {"genuine": 0, "impostor": 0, "attack": 0}
    for it, r in probes:
        if not r["acquired"]:
            fta[it["kind"]] += 1
            continue
        if it["kind"] == "attack":
            continue
        for person, score in r["per_person"].items():
            (genuine if person == it["person"] else impostor).append(score)
    genuine, impostor = np.array(genuine), np.array(impostor)
    genuine_clips = sum(1 for it, _ in probes if it["kind"] == "genuine" and it["person"] in index)

    curve = [{"threshold": float(t),
              "fmr": float(np.mean(impostor >= t)) if len(impostor) else None,
              "fnmr": float(np.mean(genuine < t)) if len(genuine) else None} for t in GRID]
    targets = []
    for target in FMR_TARGETS:
        t = threshold_for_fmr(impostor, target)
        targets.append({"target_fmr": target, "threshold": t,
                        "fnmr": None if t is None or not len(genuine) else float(np.mean(genuine < t)),
                        "measurable": t is not None})
    eer = None
    if len(genuine) and len(impostor):
        diffs = [(abs(c["fmr"] - c["fnmr"]), c) for c in curve]
        _, best = min(diffs, key=lambda d: d[0])
        eer = {"threshold": best["threshold"], "rate": (best["fmr"] + best["fnmr"]) / 2}

    # Watching for unknown people (1:N)
    def search_outcome(r, person, threshold, m):
        hit = wrong = False
        for best_name, best, runner_up in r["searches"]:
            if best >= threshold and best - runner_up >= m:
                if best_name == person:
                    hit = True
                else:
                    wrong = True
        return hit, wrong

    def one_to_many(threshold, m):
        mated = [(it, r) for it, r in probes if it["kind"] == "genuine" and it["person"] in index]
        non_mated = [(it, r) for it, r in probes if it["kind"] == "impostor" and it["person"] not in index]
        misses = misid = false_pos = 0
        for it, r in mated:
            hit, wrong = search_outcome(r, it["person"], threshold, m)
            misses += not hit
            misid += wrong
        for it, r in non_mated:
            false_pos += any(best >= threshold and best - ru >= m for _, best, ru in r["searches"])
        return {"threshold": float(threshold), "margin": m,
                "fnir": rate(misses, len(mated)), "misidentified": misid, "mated": len(mated),
                "fpir": rate(false_pos, len(non_mated)), "non_mated": len(non_mated)}

    ident_now = one_to_many(t_ident, margin)
    ident_curve = [one_to_many(t, margin) for t in GRID]

    # Liveness
    bona_fide = [r["liveness"] for it, r in probes
                 if it["kind"] in ("genuine", "impostor") and r["acquired"] and r["liveness"] is not None]
    attacks = defaultdict(list)
    for it, r in probes:
        if it["kind"] == "attack" and r["acquired"] and r["liveness"] is not None:
            attacks[it.get("attack") or "unspecified"].append(r["liveness"])
    have_liveness = bool(bona_fide) or bool(attacks)

    def pad_at(tau):
        per_type = {k: float(np.mean(np.array(v) >= tau)) for k, v in attacks.items()}
        return {"threshold": float(tau),
                "bpcer": float(np.mean(np.array(bona_fide) < tau)) if bona_fide else None,
                "apcer": per_type,
                "apcer_worst": max(per_type.values()) if per_type else None}

    pad_curve = [pad_at(t) for t in GRID] if have_liveness else []
    pad_now = pad_at(settings.liveness_threshold) if have_liveness else None
    suggested_tau = None
    if pad_curve:
        ok = [p for p in pad_curve if p["bpcer"] is not None and p["bpcer"] <= 0.05 and p["apcer_worst"] is not None]
        if ok:
            suggested_tau = min(ok, key=lambda p: (p["apcer_worst"], p["threshold"]))["threshold"]

    # Whole system against attacks: matched the claimed person AND passed liveness.
    attack_clips = [(it, r) for it, r in probes if it["kind"] == "attack"]
    granted = 0
    for it, r in attack_clips:
        if not r["acquired"]:
            continue
        score = r["per_person"].get(it["claimed"])
        live_ok = (settings.liveness_mode != "enforce" or
                   (r["liveness"] is not None and r["liveness"] >= settings.liveness_threshold))
        flat = settings.planarity_mode == "enforce" and r["planarity"] == FLAT
        granted += bool(score is not None and score >= t_verify and live_ok and not flat)

    planarity_table = defaultdict(Counter)
    for it, r in probes:
        if r["acquired"]:
            key = it["kind"] if it["kind"] != "attack" else f"attack: {it.get('attack') or 'unspecified'}"
            planarity_table[key][r["planarity"]] += 1

    def breakdown(field):
        rows = defaultdict(lambda: {"clips": 0, "fta": 0, "rejected": 0})
        for it, r in probes:
            if it["kind"] != "genuine" or it["person"] not in index:
                continue
            row = rows[it.get(field) or "(not recorded)"]
            row["clips"] += 1
            if not r["acquired"]:
                row["fta"] += 1
            elif r["per_person"][it["person"]] < t_verify:
                row["rejected"] += 1
        return dict(rows)

    times = [r["time_to_decision"] for it, r in probes
             if it["kind"] == "genuine" and r["time_to_decision"] is not None]

    return {
        "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "recognizer": model_id,
        "liveness_model": live_model_id,
        "people_enrolled": len(index) - len(failed_enroll),
        "failed_to_enroll": failed_enroll,
        "templates": dict(templates),
        "clips": dict(Counter(it["kind"] for it in items)),
        "comparisons": {"genuine": int(len(genuine)), "impostor": int(len(impostor))},
        "failure_to_acquire": fta,
        "verify": {
            "configured_threshold": t_verify,
            "fmr": float(np.mean(impostor >= t_verify)) if len(impostor) else None,
            "fnmr": float(np.mean(genuine < t_verify)) if len(genuine) else None,
            "fnmr_incl_fta": rate(int(np.sum(genuine < t_verify)) + fta["genuine"], genuine_clips),
            "targets": targets,
            "eer": eer,
            "curve": curve,
        },
        "identify": {"configured": ident_now, "curve": ident_curve},
        "liveness": {
            "configured_threshold": settings.liveness_threshold,
            "at_configured": pad_now,
            "suggested_threshold": suggested_tau,
            "bona_fide_clips": len(bona_fide),
            "attack_clips": {k: len(v) for k, v in attacks.items()},
            "curve": pad_curve,
        },
        "attacks_granted": {"granted": granted, "total": len(attack_clips),
                            "rate": rate(granted, len(attack_clips))},
        "planarity": {k: dict(v) for k, v in planarity_table.items()},
        "by_lighting": breakdown("lighting"),
        "by_group": breakdown("group"),
        "time_to_decision": {
            "median": float(np.median(times)) if times else None,
            "p90": float(np.percentile(times, 90)) if times else None,
        },
    }


# ---------- report ----------

def pct(value: Optional[float], digits: int = 2) -> str:
    if value is None:
        return "-"
    if value == 0:
        return "0%"
    p = value * 100
    return f"{p:.{digits}f}%" if p < 10 else f"{p:.1f}%"


def _ticks_log(floor: float):
    ticks, v = [], 1.0
    while v >= floor * 0.999:
        ticks.append(v)
        v /= 10
    return ticks


def line_chart(chart_id: str, title: str, subtitle: str, xs, series, log_y: bool, marker_x: Optional[float],
               marker_label: str, x_label: str) -> str:
    """
    A two-series line chart as inline SVG plus the data its hover layer reads.
    series: [(name, slot, values)], values are rates 0-1 (None = no data).
    """
    W, H, L, R, T, B = 720, 300, 64, 24, 20, 44
    pw, ph = W - L - R, H - T - B
    positives = [v for _, _, vals in series for v in vals if v]
    floor = 10 ** math.floor(math.log10(min(positives))) if (log_y and positives) else 1e-4

    def x_px(x):
        return L + (x - xs[0]) / (xs[-1] - xs[0]) * pw

    def y_px(v):
        if log_y:
            v = max(v, floor)
            return T + (0 - math.log10(v)) / (0 - math.log10(floor)) * ph
        return T + (1 - v) * ph

    parts = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-labelledby="{chart_id}-title" class="chart-svg">']
    y_ticks = _ticks_log(floor) if log_y else [0, 0.25, 0.5, 0.75, 1.0]
    for v in y_ticks:
        y = y_px(v)
        parts.append(f'<line class="grid" x1="{L}" x2="{L + pw}" y1="{y:.1f}" y2="{y:.1f}"/>')
        parts.append(f'<text class="tick" x="{L - 8}" y="{y + 4:.1f}" text-anchor="end">{html.escape(pct(v, 3))}</text>')
    for x in (0, 0.2, 0.4, 0.6, 0.8, 1.0):
        parts.append(f'<text class="tick" x="{x_px(x):.1f}" y="{T + ph + 18}" text-anchor="middle">{x:.1f}</text>')
    parts.append(f'<line class="axis" x1="{L}" x2="{L + pw}" y1="{T + ph}" y2="{T + ph}"/>')
    parts.append(f'<text class="axis-label" x="{L + pw / 2:.1f}" y="{H - 6}" text-anchor="middle">'
                 f'{html.escape(x_label)}</text>')
    if marker_x is not None:
        mx = x_px(marker_x)
        parts.append(f'<line class="marker-rule" x1="{mx:.1f}" x2="{mx:.1f}" y1="{T}" y2="{T + ph}"/>')
        anchor = "end" if marker_x > 0.75 else "start"
        dx = -6 if anchor == "end" else 6
        parts.append(f'<text class="marker-label" x="{mx + dx:.1f}" y="{T + 12}" text-anchor="{anchor}">'
                     f'{html.escape(marker_label)}</text>')
    for name, slot, vals in series:
        segments, current = [], []
        for x, v in zip(xs, vals):
            if v is None or (log_y and v <= 0):
                if len(current) > 1:
                    segments.append(current)
                current = []
            else:
                current.append(f"{x_px(x):.1f},{y_px(v):.1f}")
        if len(current) > 1:
            segments.append(current)
        for seg in segments:
            parts.append(f'<polyline class="line s{slot}" points="{" ".join(seg)}"/>')
        if marker_x is not None:
            i = int(np.argmin(np.abs(np.array(xs) - marker_x)))
            v = vals[i]
            if v is not None and (v > 0 or not log_y):
                parts.append(f'<circle class="dot s{slot}" cx="{x_px(xs[i]):.1f}" cy="{y_px(v):.1f}" r="4"/>')
    parts.append(f'<line class="crosshair" x1="0" x2="0" y1="{T}" y2="{T + ph}" visibility="hidden"/>')
    parts.append(f'<rect class="hit" x="{L}" y="{T}" width="{pw}" height="{ph}" tabindex="0" '
                 f'aria-label="{html.escape(title)}: use arrow keys to read values"/>')
    parts.append("</svg>")

    data = {"xs": [float(x) for x in xs], "L": L, "pw": pw, "W": W,
            "series": [{"name": n, "slot": s, "values": v} for n, s, v in series]}
    legend = "".join(f'<span class="key"><span class="line-key s{slot}"></span>{html.escape(name)}</span>'
                     for name, slot, _ in series)
    return (f'<figure class="chart" id="{chart_id}"><figcaption><h3 id="{chart_id}-title">{html.escape(title)}</h3>'
            f'<p class="sub">{html.escape(subtitle)}</p><div class="legend">{legend}</div></figcaption>'
            f'<div class="plot">{"".join(parts)}<div class="tooltip" hidden></div></div>'
            f'<script type="application/json" class="chart-data">{json.dumps(data)}</script></figure>')


_CSS = """
.viz-root{color-scheme:light;--page:#f9f9f7;--surface-1:#fcfcfb;--text-primary:#0b0b0b;--text-secondary:#52514e;
--muted:#898781;--grid:#e1e0d9;--axis:#c3c2b7;--border:rgba(11,11,11,0.10);--series-1:#2a78d6;--series-2:#eb6834;
--critical:#d03b3b}
@media (prefers-color-scheme:dark){:root:where(:not([data-theme="light"])) .viz-root{color-scheme:dark;--page:#0d0d0d;
--surface-1:#1a1a19;--text-primary:#ffffff;--text-secondary:#c3c2b7;--grid:#2c2c2a;--axis:#383835;
--border:rgba(255,255,255,0.10);--series-1:#3987e5;--series-2:#d95926}}
:root[data-theme="dark"] .viz-root{color-scheme:dark;--page:#0d0d0d;--surface-1:#1a1a19;--text-primary:#ffffff;
--text-secondary:#c3c2b7;--grid:#2c2c2a;--axis:#383835;--border:rgba(255,255,255,0.10);--series-1:#3987e5;--series-2:#d95926}
*{box-sizing:border-box}
body{margin:0;background:var(--page)}
.viz-root{font-family:system-ui,-apple-system,"Segoe UI",sans-serif;color:var(--text-primary);background:var(--page);
padding:32px 16px;line-height:1.5}
main{max-width:960px;margin:0 auto}
h1{font-size:24px;margin:0 0 4px}h2{font-size:18px;margin:40px 0 8px}h3{font-size:15px;margin:0}
p{margin:4px 0 12px;color:var(--text-secondary)}.sub{font-size:13px;margin:2px 0 8px}
.meta{font-size:13px;color:var(--text-secondary)}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:12px;margin:20px 0}
.tile{background:var(--surface-1);border:1px solid var(--border);border-radius:8px;padding:12px 14px}
.tile .label{font-size:13px;color:var(--text-secondary)}.tile .value{font-size:24px;font-weight:600}
.tile .note{font-size:12px;color:var(--muted)}
.chart{background:var(--surface-1);border:1px solid var(--border);border-radius:8px;padding:16px;margin:16px 0}
.legend{display:flex;gap:16px;flex-wrap:wrap;font-size:13px;color:var(--text-secondary)}
.key{display:inline-flex;align-items:center;gap:6px}
.line-key{display:inline-block;width:16px;height:2px;border-radius:1px}
.line-key.s1{background:var(--series-1)}.line-key.s2{background:var(--series-2)}
.plot{position:relative}.chart-svg{width:100%;height:auto;display:block;overflow:visible}
.grid{stroke:var(--grid);stroke-width:1}.axis{stroke:var(--axis);stroke-width:1}
.tick{fill:var(--muted);font-size:11px;font-variant-numeric:tabular-nums}
.axis-label{fill:var(--text-secondary);font-size:12px}
.line{fill:none;stroke-width:2;stroke-linejoin:round;stroke-linecap:round}
.line.s1{stroke:var(--series-1)}.line.s2{stroke:var(--series-2)}
.dot{stroke:var(--surface-1);stroke-width:2}.dot.s1{fill:var(--series-1)}.dot.s2{fill:var(--series-2)}
.marker-rule{stroke:var(--axis);stroke-width:1}.marker-label{fill:var(--text-secondary);font-size:12px}
.crosshair{stroke:var(--muted);stroke-width:1}.hit{fill:transparent;cursor:crosshair}
.hit:focus{outline:none}.hit:focus-visible{outline:2px solid var(--series-1);outline-offset:2px}
.tooltip{position:absolute;top:0;pointer-events:none;background:var(--surface-1);border:1px solid var(--border);
border-radius:6px;padding:8px 10px;font-size:12px;box-shadow:0 2px 8px rgba(0,0,0,0.12);min-width:150px}
.tooltip .row{display:flex;align-items:center;gap:8px}.tooltip strong{font-variant-numeric:tabular-nums}
.tooltip .name{color:var(--text-secondary)}.tooltip .head{color:var(--muted);margin-bottom:4px}
table{width:100%;border-collapse:collapse;font-size:13px;background:var(--surface-1);border:1px solid var(--border);
border-radius:8px;overflow:hidden;margin:8px 0}
th,td{text-align:left;padding:7px 10px;border-bottom:1px solid var(--grid)}
th{color:var(--text-secondary);font-weight:600}td.num,th.num{text-align:right;font-variant-numeric:tabular-nums}
tr:last-child td{border-bottom:none}
.warn{border-left:3px solid var(--critical);padding:8px 12px;background:var(--surface-1);border-radius:4px;
color:var(--text-primary);font-size:14px}
.table-wrap{overflow-x:auto}
"""

_JS = """
document.querySelectorAll('figure.chart').forEach(function (fig) {
  var data = JSON.parse(fig.querySelector('.chart-data').textContent);
  var svg = fig.querySelector('svg'), hit = fig.querySelector('.hit'), cross = fig.querySelector('.crosshair');
  var tip = fig.querySelector('.tooltip'), idx = -1;
  function fmt(v) {
    if (v === null || v === undefined) return 'no data';
    if (v === 0) return '0%';
    var p = v * 100; return (p < 10 ? p.toPrecision(2) : p.toFixed(1)) + '%';
  }
  function show(i) {
    idx = Math.max(0, Math.min(data.xs.length - 1, i));
    var x = data.L + (data.xs[idx] - data.xs[0]) / (data.xs[data.xs.length - 1] - data.xs[0]) * data.pw;
    cross.setAttribute('x1', x); cross.setAttribute('x2', x); cross.setAttribute('visibility', 'visible');
    tip.replaceChildren();
    var head = document.createElement('div'); head.className = 'head';
    head.textContent = 'Threshold ' + data.xs[idx].toFixed(2); tip.appendChild(head);
    data.series.forEach(function (s) {
      var row = document.createElement('div'); row.className = 'row';
      var key = document.createElement('span'); key.className = 'line-key s' + s.slot;
      var val = document.createElement('strong'); val.textContent = fmt(s.values[idx]);
      var name = document.createElement('span'); name.className = 'name'; name.textContent = s.name;
      row.append(key, val, name); tip.appendChild(row);
    });
    tip.hidden = false;
    var scale = svg.getBoundingClientRect().width / data.W, left = x * scale + 12;
    if (left + tip.offsetWidth > svg.getBoundingClientRect().width) left = x * scale - tip.offsetWidth - 12;
    tip.style.left = left + 'px';
  }
  function hide() { cross.setAttribute('visibility', 'hidden'); tip.hidden = true; }
  hit.addEventListener('pointermove', function (e) {
    var r = svg.getBoundingClientRect(), x = (e.clientX - r.left) / (r.width / data.W);
    var frac = (x - data.L) / data.pw;
    show(Math.round(frac * (data.xs.length - 1)));
  });
  hit.addEventListener('pointerleave', hide);
  hit.addEventListener('blur', hide);
  hit.addEventListener('focus', function () { show(idx < 0 ? Math.round((data.xs.length - 1) / 2) : idx); });
  hit.addEventListener('keydown', function (e) {
    if (e.key === 'ArrowRight' || e.key === 'ArrowLeft') {
      e.preventDefault(); show(idx + (e.key === 'ArrowRight' ? 1 : -1));
    }
  });
});
"""


def _table(headers, rows, numeric=()) -> str:
    head = "".join(f'<th class="{"num" if i in numeric else ""}">{html.escape(h)}</th>' for i, h in enumerate(headers))
    body = "".join("<tr>" + "".join(f'<td class="{"num" if i in numeric else ""}">{html.escape(str(c))}</td>'
                                    for i, c in enumerate(r)) + "</tr>" for r in rows)
    return f'<div class="table-wrap"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def _tile(label, value, note="") -> str:
    return (f'<div class="tile"><div class="label">{html.escape(label)}</div>'
            f'<div class="value">{html.escape(value)}</div><div class="note">{html.escape(note)}</div></div>')


def render_html(s: dict) -> str:
    v, ident, live = s["verify"], s["identify"], s["liveness"]
    out = ['<!doctype html><html lang="en"><head><meta charset="utf-8">'
           '<meta name="viewport" content="width=device-width,initial-scale=1">'
           f'<title>Face access evaluation</title><style>{_CSS}</style></head>'
           '<body class="viz-root"><main>']
    out.append("<h1>Face access evaluation</h1>")
    out.append(f'<p class="meta">Recognizer {html.escape(s["recognizer"])} · liveness '
               f'{html.escape(str(s["liveness_model"]))} · generated {html.escape(s["generated_at"])}</p>')
    clips = s["clips"]
    out.append(f'<p>{s["people_enrolled"]} people enrolled from {clips.get("enroll", 0)} enrollment items; '
               f'{clips.get("genuine", 0)} genuine, {clips.get("impostor", 0)} impostor and '
               f'{clips.get("attack", 0)} attack clips. {s["comparisons"]["impostor"]:,} impostor and '
               f'{s["comparisons"]["genuine"]:,} genuine comparisons.</p>')
    if s["failed_to_enroll"]:
        out.append(f'<p class="warn">No usable enrollment frames for: {html.escape(", ".join(s["failed_to_enroll"]))}. '
                   "Their clips count as impostors against everyone else only.</p>")

    fta = s["failure_to_acquire"]
    ag = s["attacks_granted"]
    out.append('<div class="tiles">')
    out.append(_tile("Staff rejected (FNMR)", pct(v["fnmr"]), f"at threshold {v['configured_threshold']:.2f}"))
    out.append(_tile("Impostors accepted (FMR)", pct(v["fmr"]), f"of {s['comparisons']['impostor']:,} comparisons"))
    if live["at_configured"]:
        out.append(_tile("Attacks passing liveness", pct(live["at_configured"]["apcer_worst"]),
                         "worst attack type (APCER)"))
        out.append(_tile("Real people failing liveness", pct(live["at_configured"]["bpcer"]), "BPCER"))
    out.append(_tile("Attacks let in", pct(ag["rate"]), f"{ag['granted']} of {ag['total']} attack clips"))
    out.append(_tile("No usable face", f"{fta['genuine']} clips", "genuine clips never acquired"))
    out.append("</div>")

    # Face matching
    out.append("<h2>Badge + face matching (1:1)</h2>")
    out.append("<p>How the two errors trade off as the match threshold moves. Higher thresholds let fewer "
               "impostors in and turn away more of your own staff. The vertical rule is the current setting.</p>")
    xs = [c["threshold"] for c in v["curve"]]
    out.append(line_chart(
        "verify", "Error rates by match threshold",
        "Log scale. A line stops where its rate reaches zero in this data.", xs,
        [("Staff wrongly rejected (FNMR)", 1, [c["fnmr"] for c in v["curve"]]),
         ("Impostors wrongly accepted (FMR)", 2, [c["fmr"] for c in v["curve"]])],
        log_y=True, marker_x=v["configured_threshold"], marker_label=f"current {v['configured_threshold']:.2f}",
        x_label="Match threshold (cosine similarity)"))
    rows = []
    for t in v["targets"]:
        if t["measurable"]:
            rows.append([f"FMR ≤ {pct(t['target_fmr'])}", f"{t['threshold']:.3f}", pct(t["fnmr"])])
        else:
            need = int(math.ceil(1 / t["target_fmr"]))
            rows.append([f"FMR ≤ {pct(t['target_fmr'])}", "not measurable",
                         f"needs ≥ {need:,} impostor comparisons"])
    if v["eer"]:
        rows.append(["Equal error rate", f"{v['eer']['threshold']:.2f}", pct(v["eer"]["rate"])])
    rows.append(["Current setting", f"{v['configured_threshold']:.2f}",
                 f"{pct(v['fnmr'])} (with failures to acquire: {pct(v['fnmr_incl_fta'])})"])
    out.append(_table(["Operating point", "Threshold", "Staff rejected (FNMR)"], rows, numeric=(1,)))

    # 1:N
    c = ident["configured"]
    out.append("<h2>Watching for unknown people (1:N)</h2>")
    out.append(f"<p>At the current identify threshold {c['threshold']:.2f} and margin {c['margin']:.2f}: "
               f"{pct(c['fnir'])} of enrolled staff missed or misidentified ({c['misidentified']} misidentified "
               f"of {c['mated']}), {pct(c['fpir'])} of strangers matched to someone ({c['non_mated']} clips).</p>")
    sweep = [r for r in ident["curve"] if round(r["threshold"] * 100) % 5 == 0 and 0.2 <= r["threshold"] <= 0.8]
    out.append(_table(["Threshold", "Staff missed (FNIR)", "Strangers matched (FPIR)"],
                      [[f"{r['threshold']:.2f}", pct(r["fnir"]), pct(r["fpir"])] for r in sweep], numeric=(0, 1, 2)))

    # Liveness
    out.append("<h2>Liveness</h2>")
    if not live["curve"]:
        out.append("<p>No liveness scores: run with liveness on and record attack clips to measure it.</p>")
    else:
        out.append("<p>Attacks are reported per attack type (ISO/IEC 30107-3); the chart shows the worst type. "
                   "Open-source passive liveness is not certified: decide using these numbers, not assumptions.</p>")
        lx = [p["threshold"] for p in live["curve"]]
        out.append(line_chart(
            "liveness", "Liveness errors by threshold",
            "Share of clips. The vertical rule is the current setting.", lx,
            [("Real people rejected (BPCER)", 1, [p["bpcer"] for p in live["curve"]]),
             ("Attacks accepted, worst type (APCER)", 2, [p["apcer_worst"] for p in live["curve"]])],
            log_y=False, marker_x=live["configured_threshold"],
            marker_label=f"current {live['configured_threshold']:.2f}", x_label="Liveness threshold"))
        now = live["at_configured"]
        rows = [[k, live["attack_clips"][k], pct(rate_)] for k, rate_ in sorted(now["apcer"].items())]
        rows.append(["Real people rejected (BPCER)", live["bona_fide_clips"], pct(now["bpcer"])])
        out.append(_table(["Attack type", "Clips", f"Accepted at {live['configured_threshold']:.2f}"], rows,
                          numeric=(1, 2)))
        if live["suggested_threshold"] is not None:
            out.append(f"<p>Suggested liveness threshold: {live['suggested_threshold']:.2f} (fewest attacks "
                       "accepted while rejecting at most 5% of real people).</p>")

    if s["planarity"]:
        out.append("<h2>Flat-face check (experimental)</h2>")
        out.append("<p>Only enforce it if genuine and impostor clips are never judged flat.</p>")
        rows = [[k, d.get("flat", 0), d.get("3d", 0), d.get("inconclusive", 0)] for k, d in sorted(s["planarity"].items())]
        out.append(_table(["Clips", "Flat", "3D", "Inconclusive"], rows, numeric=(1, 2, 3)))

    for title, key in (("By lighting", "by_lighting"), ("By group", "by_group")):
        if s[key]:
            out.append(f"<h2>{title}</h2>")
            rows = [[k, d["clips"], d["fta"], d["rejected"],
                     pct(rate(d["rejected"] + d["fta"], d["clips"]))] for k, d in sorted(s[key].items())]
            out.append(_table(["", "Genuine clips", "No usable face", "Rejected", "Total failed"], rows,
                              numeric=(1, 2, 3, 4)))

    t = s["time_to_decision"]
    if t["median"] is not None:
        out.append(f"<p>Time from a person appearing at the door to enough evidence for a decision: "
                   f"median {t['median']:.1f}s, 90th percentile {t['p90']:.1f}s.</p>")
    out.append("<h2>Limits of these numbers</h2><p>Error rates can only be measured down to about one in the "
               "number of comparisons made. Scores take the best value reached during the decision window, which "
               "favours impostors and attacks, as the door does. These results are for these cameras, this "
               "lighting and these people; they are not a certification.</p>")
    out.append(f"</main><script>{_JS}</script></body></html>")
    return "".join(out)


# ---------- main ----------

def main():
    parser = argparse.ArgumentParser(description="Evaluate face matching and liveness on recorded clips.")
    parser.add_argument("--data", required=True, help="Folder with labels.jsonl and clips")
    parser.add_argument("--out", required=True, help="Folder for report.html, summary.json, scores.csv")
    parser.add_argument("--detector", default=config.DETECTOR)
    parser.add_argument("--recognizer", default=config.RECOGNIZER)
    parser.add_argument("--no-liveness", action="store_true", help="Skip liveness scoring")
    args = parser.parse_args()

    items = load_labels(args.data)
    engine = load_engine(args.detector, args.recognizer)
    live_model = None if args.no_liveness else MiniFASNetLiveness(config.LIVENESS_MODEL_DIR)
    # Monitor mode: record every score without letting liveness change the run.
    settings = Settings.from_config(liveness_mode="monitor", planarity_mode="monitor")
    enforce = Settings.from_config()

    print(f"Building templates from {sum(1 for i in items if i['kind'] == 'enroll')} enrollment items...")
    gallery, index, templates, failed_enroll = build_gallery(engine, items)

    results = {}
    probes = [i for i in items if i["kind"] != "enroll"]
    for n, it in enumerate(probes, 1):
        windows, ttd = replay(engine, live_model, gallery, settings, it)
        r = summarize_clip(windows, gallery, index, settings)
        r.update(acquired=bool(windows), time_to_decision=ttd)
        results[it["clip"]] = r
        print(f"  [{n}/{len(probes)}] {it['kind']:<8} {it['clip']}: "
              f"{'no usable face' if not windows else f'{len(windows)} decision windows'}")

    summary = evaluate(items, results, index, templates, failed_enroll, engine.model_id,
                       getattr(live_model, "model_id", None), enforce)

    os.makedirs(args.out, mode=0o700, exist_ok=True)
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(args.out, "scores.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["clip", "kind", "person", "claimed", "attack", "lighting", "group", "compared_to",
                    "mated", "score", "liveness", "planarity", "acquired"])
        for it in probes:
            r = results[it["clip"]]
            for person, score in (r["per_person"] or {"": None}).items():
                w.writerow([it["clip"], it["kind"], it["person"], it["claimed"], it.get("attack") or "",
                            it.get("lighting") or "", it.get("group") or "", person, person == it["person"],
                            "" if score is None else f"{score:.4f}",
                            "" if r["liveness"] is None else f"{r['liveness']:.4f}", r["planarity"], r["acquired"]])
    with open(os.path.join(args.out, "report.html"), "w") as f:
        f.write(render_html(summary))

    v = summary["verify"]
    print(f"\nAt the current threshold {v['configured_threshold']:.2f}: FNMR {pct(v['fnmr'])}, FMR {pct(v['fmr'])} "
          f"({summary['comparisons']['impostor']:,} impostor comparisons).")
    print(f"Report: {os.path.join(args.out, 'report.html')}")


if __name__ == "__main__":
    main()

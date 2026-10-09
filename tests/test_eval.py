import csv
import json
import os
import sys

import numpy as np
import pytest

from conftest import BLUE, GREEN, RED, ColorEngine, solid, write_video

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "eval"))
import run_eval  # noqa: E402


def clip(folder, name, color, n=24):
    write_video(folder / name, [solid(color, seed=i) for i in range(n)])
    return name


@pytest.fixture
def dataset(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    labels = [
        {"clip": clip(data, "enroll_A.mp4", BLUE), "kind": "enroll", "person": "P01"},
        {"clip": clip(data, "enroll_B.mp4", RED), "kind": "enroll", "person": "P02"},
        {"clip": clip(data, "gen_A1.mp4", BLUE), "kind": "genuine", "person": "P01", "lighting": "day"},
        {"clip": clip(data, "gen_A2.mp4", BLUE), "kind": "genuine", "person": "P01", "lighting": "night"},
        {"clip": clip(data, "gen_B1.mp4", RED), "kind": "genuine", "person": "P02", "lighting": "day"},
        {"clip": clip(data, "imp_V1.mp4", GREEN), "kind": "impostor", "person": "V01", "claimed": "P01"},
        {"clip": clip(data, "imp_B_as_A.mp4", RED), "kind": "impostor", "person": "P02", "claimed": "P01"},
        {"clip": clip(data, "atk_A.mp4", BLUE), "kind": "attack", "person": "P01", "attack": "print_matte"},
        {"clip": clip(data, "empty.mp4", (2, 2, 2)), "kind": "genuine", "person": "P01", "lighting": "dark"},
    ]
    with open(data / "labels.jsonl", "w") as f:
        for item in labels:
            f.write(json.dumps({**item, "fps": 15}) + "\n")
    return data


def test_harness_end_to_end(dataset, tmp_path, fake_engine, monkeypatch):
    out = tmp_path / "report"
    monkeypatch.setattr(run_eval, "load_engine", lambda d, r: ColorEngine())
    monkeypatch.setattr(sys, "argv", ["run_eval.py", "--data", str(dataset), "--out", str(out), "--no-liveness"])
    run_eval.main()

    s = json.loads((out / "summary.json").read_text())
    # Up to 3 well-spaced frames per enrollment clip, like enrolling at the door camera.
    assert s["people_enrolled"] == 2 and s["templates"] == {"P01": 3, "P02": 3}
    # 3 acquired genuine clips (one is too dark: a failure to acquire).
    assert s["failure_to_acquire"]["genuine"] == 1
    # Genuine: 3 genuine clips + the impostor clip of P02 compared with P02's own
    # template. Impostor: each genuine clip vs the other person, the stranger vs
    # both, P02's clip vs P01.
    assert s["comparisons"] == {"genuine": 4, "impostor": 3 * 1 + 2 + 1}
    v = s["verify"]
    assert v["fnmr"] == 0.0 and v["fmr"] == 0.0
    assert v["fnmr_incl_fta"] == pytest.approx(1 / 4)
    assert not v["targets"][0]["measurable"]           # 6 impostor comparisons can't show FMR <= 1%
    ident = s["identify"]["configured"]
    assert ident["fnir"] == pytest.approx(1 / 4) and ident["fpir"] == 0.0
    assert s["by_lighting"]["dark"] == {"clips": 1, "fta": 1, "rejected": 0}
    # Liveness wasn't run, and LIVENESS_MODE is enforce: the attack isn't let in.
    assert s["attacks_granted"] == {"granted": 0, "total": 1, "rate": 0.0}

    page = (out / "report.html").read_text()
    for heading in ("Badge + face matching", "Watching for unknown people", "Liveness", "Limits of these numbers"):
        assert heading in page
    rows = list(csv.DictReader(open(out / "scores.csv")))
    assert {r["clip"] for r in rows} >= {"gen_A1.mp4", "imp_V1.mp4", "atk_A.mp4"}


def test_threshold_for_fmr_needs_enough_comparisons():
    impostor = np.linspace(-0.2, 0.4, 1000)
    t = run_eval.threshold_for_fmr(impostor, 1e-2)
    assert t is not None and np.mean(impostor >= t) <= 1e-2
    assert run_eval.threshold_for_fmr(impostor, 1e-4) is None


def test_chart_handles_gaps_and_zeros():
    html = run_eval.line_chart("c", "T", "S", [0.0, 0.5, 1.0],
                               [("a", 1, [0.5, None, 0.0]), ("b", 2, [0.0, 0.01, 0.2])],
                               log_y=True, marker_x=0.5, marker_label="current 0.50", x_label="x")
    assert "<svg" in html and "current 0.50" in html

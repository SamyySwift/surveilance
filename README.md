# Gatekeeper: staff access control with face recognition

Cameras at each door watch the people approaching. When someone badges in,
Gatekeeper checks that the face at the door belongs to the badge holder, is a
live person rather than a photo or screen, and is allowed through that door.
It answers the door's access control system with grant or deny. In the
background it also watches for unknown people, people in zones they aren't
cleared for, tailgating, and cameras that go offline or are tampered with.
Every decision and alarm is recorded with a snapshot for review.

Detection and recognition use [InsightFace](https://github.com/deepinsight/insightface)
(`buffalo_l` by default, swappable), liveness uses MiniFASNet from
[Silent-Face-Anti-Spoofing](https://github.com/minivision-ai/Silent-Face-Anti-Spoofing),
and storage is SQLite.

## Read this first

- **This is a reference implementation, not a certified product.** It has an
  automated test suite and has been run end to end on sample clips. It has
  **not** been measured on your cameras, lighting or people. Every threshold
  in `config.py` is an uncalibrated starting point until you run the
  evaluation in [Measuring accuracy](#measuring-accuracy).
- **Licensing: the default model can't be deployed as-is.** InsightFace's
  code is MIT-licensed, but its pretrained packs (including `buffalo_l`) are
  licensed for non-commercial research only. Production use needs a
  commercial license from InsightFace or a different model (see
  [Comparing or switching models](#comparing-or-switching-models)). Most open
  models are trained on research-only datasets too, so have every model reviewed.
- **Liveness with ordinary cameras is one layer, not a guarantee.**
  Open-source passive liveness misses many attacks, and even certified
  commercial products vary by attack type. Keep the badge requirement and
  guard review in place, and measure liveness on your own attack set.

## How it works

For every camera frame:

1. **Detect** faces at the camera's full resolution.
2. **Track** each face, so evidence builds up per person instead of per
   frame or screen position.
3. **Quality-check** each face: pixels between the eyes (distance), sharpness,
   brightness and exposure, and head turn and tilt. Faces that fail are never
   used for a decision. This is what stops far-away, blurred, dark or turned
   faces from causing false alarms.
4. **Embed and score liveness** for faces that pass.
5. **Decide per person** from several good frames (by default 3 within 4
   seconds), using a quality-weighted average of their embeddings.

Each tracked person ends up in one of these states:

| Status | Meaning | Alarm |
|---|---|---|
| `pending` | Collecting good frames | - |
| `verified` | Matches an enrolled person who may be in this zone | - |
| `unauthorized` | Matches an enrolled person who may **not** be in this zone | high |
| `unknown` | Matches nobody enrolled, or is too close to call | medium |
| `spoof_suspected` | Failed the liveness check. Stays flagged while tracked | high |
| `unverifiable` | Present for 6+ s without giving enough usable frames | medium |

The last state means someone can't avoid alarms by hiding their face. Alarms
fire once per person, not once per frame.

**Badge + face (1:1).** When your access control system reports a badge
read, the system takes the closest face at the door and checks it against
**only that badge holder's** enrollment. It uses frames from the last few
seconds, so a person who walked up normally gets an answer immediately. The
door opens only on `grant`. Any denial raises an alarm, and so does a
second person lingering at the door after a grant (possible tailgating).
**Badge checks are switched off for now** (`BADGE_CHECKS_ENABLED = False`
in `config.py`) until a reader is connected. While they're off, the badge
endpoint refuses requests and the dashboard hides every badge feature.
Monitoring below runs either way.

**Monitoring (1:N).** Every tracked person is also searched against everyone
enrolled. A match must clear the threshold *and* beat the second-closest
person by a margin, so two similar-looking staff aren't a coin toss.

**Camera health.** An alarm fires if a camera goes offline, sends a frozen
picture, looks covered or blinded, or suddenly shows a different scene with
nobody in front of it. A broken camera is never silent.

## Setup

Dependencies are managed with [uv](https://docs.astral.sh/uv/). It installs
the Python version in `.python-version` and the exact package versions in
`uv.lock` into `.venv/`:

```bash
uv sync
```

Run every command below with `uv run` (as shown), or activate the
environment once with `source .venv/bin/activate` and drop the prefix. Add or
upgrade packages with `uv add <package>` / `uv lock --upgrade-package <package>`,
not pip, so `pyproject.toml` and `uv.lock` stay in step.

The first run downloads the `buffalo_l` model pack (~280 MB). The liveness
models are already in `models/liveness/`. `MANIFEST.json` there records
where they came from and their SHA-256 hashes, and those hashes are checked
on every load.

Set the dashboard/API token in `.env` next to `config.py` (a real environment
variable of the same name takes precedence):

```bash
echo "GATEKEEPER_API_TOKEN=$(uv run python -c 'import secrets; print(secrets.token_urlsafe(32))')" > .env
chmod 600 .env
```

With no token configured, the backend makes up a new one at every start and
prints it, so you'd sign in again after every restart.

The first start after upgrading from an earlier version upgrades `staff.db`
in place. It saves a copy as `staff.db.before-upgrade-<time>` first, and
existing enrollments keep working.

## Camera and lighting

This matters more than the choice of model. Recognition needs pixels on the
face. Measure it as pixels between the eyes at the point where the decision
is made: at least 60 (the quality gate's floor), ideally 90 or more.

```
pixels between eyes ≈ 0.063 m × image width (px) ÷ (2 × distance (m) × tan(horizontal FOV ÷ 2))
```

| Camera | 1 m | 2 m | 3 m |
|---|---|---|---|
| Typical 78° webcam at 960 px wide | 37 | 19 | 12 |
| 1080p, 78° lens | 75 | 37 | 25 |
| **1080p, ~37° lens** | 181 | **90** | 60 |
| **4K, ~68° lens** | 179 | **90** | 60 |

- Aim the camera at the 0.5–2 m zone in front of the badge reader. Mount it
  at face height, tilted no more than 15° vertically.
- Light faces evenly from the front. Avoid backlight from glass doors and
  hard overhead light. Use a camera with good wide dynamic range (WDR) at
  doors that face daylight.
- Low light can't be fixed in software. Dark frames simply fail the quality
  gate, so the system waits instead of guessing.

Set each camera in `config.py`:

```python
CAMERAS = {
    "entrance": {"source": "rtsp://user:pass@10.0.0.20/stream1"},
    "vault": {"source": "FaceTime HD Camera", "width": 1920, "height": 1080},  # request full resolution
}
```

To see each connected camera's name and the sizes it supports, run
`uv run camera.py`, then set `width`/`height` to the first (widest) size
listed. On a Mac, name webcams as that list (or System Information > Camera)
shows them rather than by index. OpenCV numbers Mac cameras by their unique ID, so plugging a
camera in or out (or an iPhone coming into Continuity Camera range)
renumbers the others, and an index would silently switch a zone to a
different camera. A named camera is looked up each time it connects; if it
isn't there, the zone raises `camera_offline` and the log lists the cameras
that are connected. Two cameras of the same model share a name, so name
those by the unique ID shown in that same list.

The dashboard's live view is scaled down to `STREAM_MAX_WIDTH`, but
processing always uses the full frame. A video file also works as a source
(it loops), which is handy for testing.

## Enrolling staff

Enrollment photos become the reference for every later decision, so each
one is checked before it's accepted:

- exactly one person in the photo;
- close enough, sharp, well lit and facing the camera (`config.ENROLL_QUALITY`);
- looks like this person's existing photos, if they have any;
- doesn't look like someone already enrolled. This stops one person being
  enrolled twice, or someone's photo being added to the wrong record.

The best way is **"Capture"** in the dashboard's staff panel. The person
stands alone at a door camera for about 6 seconds and the best frames are
enrolled. That captures them in exactly the conditions the system will see
every day. You can also upload photos in the dashboard, or use the command line:

```bash
uv run enroll.py --name "Jane Doe" --employee-id E1023 --badge-id 40123 --zones entrance,vault jane1.jpg jane2.jpg
```

`--zones` lists the zones from `config.CAMERAS` the person may be in. Access
is deny-by-default: someone with no zones is "unauthorized" everywhere.
`--badge-id` is the number the door reader sends.

```bash
uv run manage.py list                      # everyone, with badge, zones and photo count
uv run manage.py deactivate "Jane Doe"      # revoke access (e.g. on departure)
uv run manage.py badge "Jane Doe" 40123     # link a badge ("" removes it)
```

## Connecting the badge reader

First set `BADGE_CHECKS_ENABLED = True` in `config.py` and restart the
backend. Until then the endpoint answers `503`, and the dashboard shows no
badge features.

When a badge is read, your access control system (or a small bridge
program) calls:

```bash
curl -X POST http://gatekeeper:8000/api/access/badge \
  -H "Authorization: Bearer $GATEKEEPER_API_TOKEN" -H "Content-Type: application/json" \
  -d '{"zone": "entrance", "badgeId": "40123"}'
```

```json
{"decision": "grant", "reason": "face matches badge holder", "zone": "entrance",
 "staffId": 7, "name": "Jane Doe", "matchScore": 0.81, "livenessScore": 0.97,
 "planarity": "inconclusive", "samples": 9, "eventId": 1532}
```

- **Open the door only on `"grant"`**, and treat errors and timeouts as deny.
- The call waits up to `BADGE_DECISION_TIMEOUT_SECONDS` (5 s) for a clear
  view of the person. If they were already in view, the answer is
  immediate.
- How you connect depends on your panel: OSDP, Wiegand through a bridge, or
  the vendor's API. The dashboard's "Test badge" box under each camera
  makes the same call, so you can try the flow before the reader is wired up.

## Liveness

Each good-quality face is scored by two small MiniFASNet models. They look
at a wider crop around the face, which helps them spot phone bezels, paper
edges and fingers holding a photo. A person's score is the average over
their recent frames. The settings are in `config.py`:

- `LIVENESS_MODE = "enforce"`: failing (or being unable to run) the check
  denies entry and raises an alarm. `"monitor"` records scores without
  acting on them; `"off"` skips the check. The dashboard shows a banner
  whenever liveness isn't enforced or isn't working.
- `LIVENESS_THRESHOLD = 0.70`: set this from your own measurements (see below).

How it behaved in testing:
- On the original project's sample images, real faces scored 0.99–1.0 and
  attack photos 0.005–0.18.
- In an end-to-end test, a phone screen showing the badge holder **matched
  their face (0.54) and was denied only because liveness scored it 0.002.**
- A real face pasted onto a background, so that it had a visible rectangular
  edge, scored around 0.6–0.8. The model reacts to that edge on purpose,
  which is also why recording real sessions at your own doors matters.

To use a commercial liveness product (ideally one with iBeta ISO/IEC 30107-3
results), wrap it in a class with a `model_id` attribute and a
`score(frame, bbox) -> probability` method, and return it from
`liveness.load_liveness()`.

`PLANARITY_MODE` (experimental, default `"monitor"`) watches whether a face
moves like a flat surface. A tilted photo does; a real head turning doesn't.
It can't catch video replays. Only switch it to `"enforce"` if the
evaluation shows it never flags real people.

## Measuring accuracy

Do this before relying on the system, and again after changing cameras,
lighting, models or thresholds.

1. **Record sessions at the real doors** with `eval/capture.py`. Use
   pseudonymous IDs, and record demographic groups only with consent.

   ```bash
   uv run eval/capture.py --zone entrance --kind enroll   --person P01 --seconds 5
   uv run eval/capture.py --zone entrance --kind genuine  --person P01 --lighting day
   uv run eval/capture.py --zone entrance --kind impostor --person V07 --claimed P01
   uv run eval/capture.py --zone entrance --kind attack   --person P01 --attack phone_video
   ```

   - **More people matters more than more clips per person.** Record each
     person several times, in each lighting condition you'll operate in.
   - **Attacks:** include matte and glossy prints, photos and video on phones
     and tablets, and prints with the eyes cut out.
2. **Run the evaluation:**

   ```bash
   uv run eval/run_eval.py --data eval/data --out eval/report
   ```

   Open `eval/report/report.html`. Every clip goes through the same pipeline
   the doors run. The report shows:
   - **Badge + face:** staff wrongly rejected (FNMR) against impostors wrongly
     accepted (FMR), across thresholds.
   - **Unknown people:** staff missed (FNIR) and strangers matched (FPIR).
   - **Liveness:** attacks accepted, per attack type (APCER), and real people
     rejected (BPCER).
   - **Whole system:** how many attacks would have got in.
   - **Failures to acquire:** people who never gave a usable face.
   - **Breakdowns** by lighting and by group, and how long decisions take.
3. **Choose the thresholds with whoever owns security risk**, write them into
   `config.MATCH_THRESHOLDS` and `LIVENESS_THRESHOLD`, and run the
   evaluation again.

Error rates can only be measured down to about one in the number of
comparisons made, and the report says when a target isn't measurable with
your data. Scores take the best value reached while the door keeps checking.
That favours impostors and attacks, exactly as the real door would.

## Comparing or switching models

`config.DETECTOR` and `config.RECOGNIZER` choose the models by name:

- `"insightface:<pack>"` loads an InsightFace pack.
- `"onnx:/path/model.onnx"` loads any ArcFace-style recognizer, such as an
  ONNX export of AdaFace.

Embeddings record which model made them and are never compared across
models. To try a model:

```bash
uv run reembed.py --recognizer onnx:/models/candidate.onnx       # templates from stored enrollment photos
uv run eval/run_eval.py --data eval/data --recognizer onnx:/models/candidate.onnx --out eval/report-candidate
```

Then add its thresholds to `config.MATCH_THRESHOLDS` and set
`config.RECOGNIZER`. A commercial SDK plugs in the same way: a detector class
with `detect()`, or a recognizer with `model_id`, `align()` and
`embed_aligned()`, registered in `face_engine.py`. Compare candidates on the
same recorded data, and include licensing and NIST FRTE results in the
decision.

## Running it

**Dashboard + API** (two terminals):

```bash
uv run uvicorn server:app --host 0.0.0.0 --port 8000      # backend, from this directory
cd frontend && npm install && npm run dev          # dashboard
```

Open `http://localhost:5173` and sign in with the token. The dashboard shows:
- **Header:** whether each camera is online, the liveness state, and the
  number of open alarms (also shown in the browser tab title).
- **Live feeds:** each person's status drawn on the video, with a list of
  who's in view and the camera's frame rate and resolution under each feed.
  - **Hide video** stops the picture for that camera, and **Hide all video**
    in the header does it for every camera. Recognition, the in-view list
    and alarms keep running. While no dashboard is showing a camera's video,
    the server stops encoding its frames. Each browser remembers its choice.
  - The enlarge button widens a feed to the full width.
- **Alarms:** each with a snapshot. Filter to open or all; guards acknowledge
  alarms with a note. **Email** sends a status report to the addresses you
  enter; new alarms are also emailed automatically (see "Email alerts"). **Clear** permanently deletes the alarms on
  screen and their snapshots, open ones included (it asks first). An alarm
  that arrives while the confirmation is showing is kept.
- **Staff:** searchable. Each person has:
  - **Zones** (where they may be);
  - **Photos** (view, add or delete enrollment photos);
  - **Capture** (enroll at a door camera);
  - **More** (suspend or remove).
- **When the server can't be reached:** a red banner says so and the feeds
  show offline, so nobody mistakes a frozen screen for a quiet one.
- **With badge checks switched on:** a "Test badge" box under each camera,
  badge IDs on staff, and a list of badge decisions with the frame each was
  based on.

**Local window** (one camera, no dashboard):
`uv run recognize.py --zone entrance`. It runs the same pipeline and records
alarms the same way.

## Alarms and the audit trail

Alarms and badge decisions are stored in `staff.db` and survive restarts.
Each one keeps a snapshot of the frame it was based on. Snapshots show faces,
so they're saved owner-readable only in `snapshots/` and deleted after
`SNAPSHOT_RETENTION_DAYS` (30 by default). Every alarm except "info" also
plays a sound (`ALARM_SOUND_*`), is emailed if Resend is set up (see "Email
alerts" below) and, if `ALARM_WEBHOOK_URL` is set in `.env`, is POSTed to
that URL as JSON from a background thread, one request per alarm:

```json
{"reason": "Jane Doe is not authorized for Vault", "alarm_id": 42,
 "kind": "unauthorized", "title": "Unauthorized person", "severity": "high",
 "zone": "Vault", "time": 1791234567.8, "time_text": "11 Oct 2026, 14:32:05",
 "staff_id": 7, "staff_name": "Jane Doe", "track_id": 3,
 "details": {"reason": "Jane Doe has no access to Vault", "closest_match": "Jane Doe",
             "match_score": 0.71, "liveness": 0.98, "planarity": "inconclusive"}}
```

`staff_name` is set only when the person was identified. `details` depends on
the kind: monitoring alarms carry the match and liveness scores shown above,
and camera alarms carry none. Snapshots aren't sent. A failed call is logged
and not retried.

| Kind | When |
|---|---|
| `spoof_suspected` | A face failed liveness (monitoring, or at a badge check) |
| `face_mismatch` | A badge was used by someone whose face doesn't match |
| `unauthorized` | Someone is in, or badging into, a zone they aren't cleared for |
| `tailgating` | A second person stayed at the door after a grant |
| `multiple_faces` | More than one person was at the door during a badge check |
| `unknown_person` / `unverifiable` | Monitoring: no match, or never a usable face |
| `unknown_badge` / `suspended_badge` / `access_denied` / `liveness_unavailable` | Other badge denials |
| `camera_offline` / `camera_frozen` / `camera_obstructed` / `camera_moved` | Camera health (`camera_restored` when it clears) |
| `processing_error` | Face processing crashed on a camera; see the server log |

### Email alerts

Every alarm except "info" (the same ones that sound and call the webhook) is
emailed automatically through [Resend](https://resend.com). Alarms raised
within a few seconds of each other share one email, and emails go out at most
once a minute (`ALARM_EMAIL_MIN_INTERVAL_SECONDS`), so a burst of alarms
can't flood anyone's inbox: anything raised in between arrives in the next
email. Each email lists the new alarms, any that are still open from before,
and whether each camera is online. Snapshots aren't attached, since they
show people's faces.

If sending fails, temporary problems (network, Resend outage, rate limit) are
retried, and the dashboard shows a banner until an email gets through. The
dashboard's **Email** button sends a full status report on demand: every open
alarm, alarms acknowledged in the last 24 hours, and camera status.

Set it up in `.env` and restart the server:

```
RESEND_API_KEY=re_...                    # resend.com > API Keys
ALARM_EMAIL_TO=you@example.com           # comma-separate several
ALARM_EMAIL_FROM=Gatekeeper <alerts@your-domain.com>   # optional, see below
```

Until you verify a domain in Resend (Domains > Add Domain, then add the DNS
records it shows), leave `ALARM_EMAIL_FROM` unset. Emails then come from
Resend's test sender, `onboarding@resend.dev`, which only delivers to the
address your Resend account is registered with. To email anyone else, verify
a domain and set `ALARM_EMAIL_FROM` to an address on it. Set
`ALARM_EMAIL_AUTOMATIC = False` in `config.py` to keep only the Email button.

## Using the GPU on an Apple Silicon Mac

There's no CUDA on a Mac. The GPU path on Apple Silicon is ONNX Runtime's
**CoreML execution provider**, which runs models on the GPU and/or Neural
Engine. The standard macOS arm64 `onnxruntime` wheels include it, and
`config.py` requests it automatically on Apple Silicon (CPU elsewhere).
Check the startup log rather than assuming it's in use, because ONNX Runtime
silently falls back to CPU if CoreML fails to load:

```
[face_engine] detector insightface:buffalo_l running on: ['CoreMLExecutionProvider', 'CPUExecutionProvider']
[face_engine] recognizer insightface:buffalo_l running on: ['CoreMLExecutionProvider', 'CPUExecutionProvider']
```

In testing on an Apple Silicon Mac with two simulated 720p cameras, each
camera processed about 4 frames per second, enough for a decision in under a
second. Check the rate with `/api/status` on your hardware and camera count.

## Tests

```bash
uv run pytest tests
```

The tests use temporary databases, never `staff.db`. They cover:
- the decision pipeline: every badge outcome, tailgating, hidden faces, spoofs,
  and someone stepping into another person's tracked box;
- tracking, the quality gate, matching, the liveness checks and camera health;
- database upgrades;
- the API, including authentication;
- the worker running against a real video file;
- the evaluation harness.

## Before you rely on this operationally

- **Run the evaluation and set the thresholds** with your security officer
  (see [Measuring accuracy](#measuring-accuracy)).
- **Sort out model licensing** (see [Read this first](#read-this-first)).
- **Connect alarms to people:** set `ALARM_WEBHOOK_URL` or extend
  `alarms.notify()`, and make sure someone reviews and acknowledges alarms.
- **Keep the badge requirement.** Face plus badge is far stronger than either
  alone, and it's the backstop for liveness.
- **If attacks get through in your tests,** infrared and depth access cameras
  are the next step up: a photo or a screen can't fool them. 3D masks are
  outside what this system defends against.
- **Check local law on biometric data** (for example BIPA in Illinois, GDPR
  in the EU/UK) with whoever handles compliance.
- **Treat `staff.db*`, `enrollment_photos/`, `snapshots/` and `eval/data/` as
  sensitive biometric data:** restrict access, back them up securely, and keep
  them out of version control (`.gitignore` already does).
- **Put the backend behind HTTPS** if it's reachable beyond a trusted
  network, since the token and video travel with every request. Everyone
  shares one token today, so actions aren't attributed to individual
  operators.

## File overview

| File | Purpose |
|---|---|
| `config.py` | Every setting: cameras, models, thresholds, quality limits, liveness, alarms |
| `pipeline.py` | The decision pipeline shared by the server, the CLI and the evaluation |
| `face_engine.py` | Swappable detector and recognizer (InsightFace packs, ONNX models) |
| `quality.py` | Per-face quality gate: distance, sharpness, light, pose |
| `tracker.py` | Keeps one ID per person across frames |
| `matcher.py` | 1:1 verify and 1:N identify against the cached gallery |
| `liveness.py` | MiniFASNet liveness (hash-checked) and the experimental flat-face check |
| `camera.py` | Camera capture with reconnect, and camera-health alarms |
| `worker.py` | One background worker per door: camera, pipeline, badge checks, camera enrollment |
| `alarms.py` | Alarm outputs (sound, webhook), snapshots, audit records |
| `email_report.py` | Alarm emails through Resend: automatic alerts and on-demand reports |
| `enrollment.py` | Checked enrollment, shared by the dashboard, CLI and camera capture |
| `database.py` | SQLite storage and in-place upgrades |
| `server.py` | FastAPI backend: REST API, badge endpoint, live stream |
| `recognize.py` | One camera in a local window, same pipeline |
| `enroll.py`, `manage.py`, `reembed.py` | Command-line tools |
| `eval/capture.py`, `eval/run_eval.py` | Record test sessions; measure accuracy and suggest thresholds |
| `tools/convert_minifasnet.py` | How the liveness models were converted and verified |
| `models/liveness/` | Liveness models, manifest and license |
| `tests/` | Automated tests |
| `frontend/` | React dashboard (see `frontend/README.md`) |

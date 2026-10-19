"""
Configuration for the access-control face recognition system.

Most numbers below are starting points, not validated values. Before relying
on this operationally, record test sessions at your own doors and calibrate
them with eval/run_eval.py (see README "Measuring accuracy").
"""

import os
import platform

from dotenv import load_dotenv

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Pull secrets like GATEKEEPER_API_TOKEN from a .env file next to this one.
# Variables already set in the real environment take precedence.
load_dotenv(os.path.join(BASE_DIR, ".env"))


def _default_providers():
    """
    ONNX Runtime execution providers to try, in priority order. ORT falls
    back silently to the next entry (and ultimately CPU) if one isn't
    available, so listing CoreML here is harmless on non-Mac machines.

    On Apple Silicon (M1/M2/M3), CoreMLExecutionProvider routes inference
    to the GPU/Neural Engine via Apple's CoreML framework - this is the
    correct "use the GPU" path on macOS; there's no CUDA on Mac.
    """
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        return ["CoreMLExecutionProvider", "CPUExecutionProvider"]
    if platform.system() == "Windows":
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


EXECUTION_PROVIDERS = _default_providers()

# --- Cameras ---
# Zone name -> camera settings. "source" is a webcam's name as macOS lists it
# (System Information > Camera), a webcam index (0, 1, ...), a stream URL
# such as "rtsp://user:pass@10.0.0.20/stream1", or a video file (handy for
# testing; it loops). On a Mac, name webcams rather than numbering them:
# OpenCV numbers them by unique ID, so plugging a camera in or out (or an
# iPhone coming into Continuity Camera range) renumbers the others and a zone
# would silently show a different camera. Use a camera's unique ID only when
# two cameras share a name (same model): a USB camera's ID starts with the
# port it's plugged into, so it changes if the camera moves to another port.
# "width"/"height" request a capture size:
# webcams often default to a low resolution unless asked, so request the
# camera's full resolution. Stream cameras ignore them and send whatever
# they're configured to send.
#
# Faces need pixels. Aim for at least ~90 px between the eyes where the
# decision is made (60 px is the floor the quality gate enforces):
#   px between eyes ~= 0.063 * width_px / (2 * distance_m * tan(horizontal_fov / 2))
# A 1080p camera with a ~37 degree lens gives ~90 px at 2 m; a typical 78
# degree webcam gives ~37 px. See README "Camera and lighting".
CAMERAS = {
    "Entrance": {"source": "USB Camera", "width": 1920, "height": 1080},
    "Vault": {"source": "SamSwift Camera", "width": 1920, "height": 1080},
    "Lobby": {"source": "2M WebCam", "width": 1280, "height": 720},  # this camera's maximum
}

# The dashboard's live view is scaled down to this width to save bandwidth.
# Processing always uses the full-resolution frame.
STREAM_MAX_WIDTH = 1280
STREAM_JPEG_QUALITY = 70

# --- Recognition engines ---
# Chosen by name so models can be swapped and compared (eval/run_eval.py):
#   "insightface:<pack>"  an InsightFace model pack, e.g. buffalo_l
#   "onnx:<path>"         any ArcFace-style ONNX recognizer with a square
#                         112x112-or-larger input (recognizer only)
# Every stored embedding records which recognizer made it, and embeddings
# from different recognizers are never compared. After changing RECOGNIZER,
# run reembed.py to build templates for the new model.
#
# LICENSING: InsightFace's code is MIT-licensed, but its pretrained packs
# (buffalo_l and others) are for non-commercial research only. Deploying them
# needs a commercial license from InsightFace or a different model.
DETECTOR = "insightface:buffalo_l"
RECOGNIZER = "insightface:buffalo_l"
DETECTION_SIZE = (640, 640)
DETECTION_THRESHOLD = 0.5  # the quality gate below applies a stricter minimum

# Cosine-similarity thresholds, per recognizer (they don't transfer between
# models). UNCALIBRATED starting points: replace them with values measured on
# your own cameras by eval/run_eval.py.
#   verify_accept    1:1 (badge + face): the badge holder's evidence must reach this
#   identify_accept  1:N (watching for unknown people): the best match must reach this
#   identify_margin  1:N: the best person must beat the runner-up by this much
#   reject           below this, the face is treated as someone else / unknown;
#                    between reject and accept the system keeps collecting frames
#   duplicate        enrollment refuses a photo this similar to someone else
MATCH_THRESHOLDS = {
    "insightface:buffalo_l": {
        "verify_accept": 0.45,
        "identify_accept": 0.50,
        "identify_margin": 0.08,
        "reject": 0.30,
        "duplicate": 0.45,
    },
}
# Used (with a warning) for a recognizer that has no entry above.
DEFAULT_MATCH_THRESHOLDS = MATCH_THRESHOLDS["insightface:buffalo_l"]

# --- Face quality gate ---
# Faces failing any limit aren't used for any decision. This is what stops
# tiny, blurry, dark or turned-away faces from causing false alarms.
# Sharpness is fine detail relative to the face's contrast (sharp faces
# measured 0.15-1.4; heavy blur 0.005-0.05). Brightness is the mean grey level
# (0-255) of the face's centre. Pitch (tilt) from 5 landmarks is rough, so
# its limits are loose.
LIVE_QUALITY = {
    "min_ipd_px": 60,
    "min_det_score": 0.60,
    "max_yaw_deg": 30,
    "max_pitch_deg": 35,
    "min_sharpness": 0.03,
    "min_brightness": 45,
    "max_brightness": 220,
    "max_clipped_fraction": 0.10,
}
# Enrollment photos become the reference for every future decision, so the
# bar is higher.
ENROLL_QUALITY = {
    "min_ipd_px": 80,
    "min_det_score": 0.75,
    "max_yaw_deg": 20,
    "max_pitch_deg": 30,
    "min_sharpness": 0.06,
    "min_brightness": 60,
    "max_brightness": 210,
    "max_clipped_fraction": 0.05,
}

# --- Track decisions ---
# Decisions are made per tracked person from several good frames.
EVIDENCE_WINDOW_SECONDS = 4.0     # how far back a person's frames count
MIN_SAMPLES_FOR_DECISION = 3      # good frames needed before deciding (1:N)
UNVERIFIABLE_AFTER_SECONDS = 6.0  # present this long without enough good frames -> alarm
DECISION_TIMEOUT_SECONDS = 8.0    # still between reject and accept -> treated as unknown
TRACK_MAX_AGE_SECONDS = 1.0       # forget a face not seen for this long

# --- Badge + face (1:1) ---
# Your access control system calls POST /api/access/badge when a badge is
# read; the door should open only on "grant". Switched off until a badge
# reader is connected: the endpoint refuses requests and the dashboard hides
# the badge features. Monitoring (unknown people, liveness, zones, camera
# health) runs either way.
BADGE_CHECKS_ENABLED = False
BADGE_DECISION_TIMEOUT_SECONDS = 5.0
BADGE_MIN_SAMPLES = 2
DOOR_MIN_IPD_PX = 60              # faces at least this big count as "at the door"
TAILGATE_WINDOW_SECONDS = 8.0     # after a grant, watch for a second person this long...
TAILGATE_GRACE_SECONDS = 3.0      # ...who stays at the door this long without badging
DENY_WHEN_MULTIPLE_FACES = False  # True: refuse entry if anyone else is at the door

# --- Liveness (presentation attack detection) ---
#   "enforce"  a failed or unavailable check denies access and raises an alarm
#   "monitor"  scores are recorded and shown but don't change decisions
#   "off"      not run
# With ordinary RGB cameras this is one layer of protection, not a guarantee:
# measure it on your own attack set (eval/run_eval.py) before relying on it.
LIVENESS_MODE = "enforce"
LIVENESS_BACKEND = "minifasnet"
LIVENESS_MODEL_DIR = os.path.join(BASE_DIR, "models", "liveness")
LIVENESS_THRESHOLD = 0.70   # mean live-probability over a person's good frames
LIVENESS_MIN_SAMPLES = 2
# Experimental flat-face check from the 5 landmarks (see liveness.py). Keep it
# on "monitor" until eval/run_eval.py shows it helps at your doors.
PLANARITY_MODE = "monitor"

# --- Camera health (fail closed) ---
CAMERA_OFFLINE_ALARM_SECONDS = 10
CAMERA_FROZEN_SECONDS = 10
CAMERA_OBSTRUCTED_SECONDS = 5
CAMERA_OBSTRUCTED_MAX_STDDEV = 8.0   # an almost uniform frame: lens covered or blinded
CAMERA_SCENE_CHANGE_SECONDS = 20
CAMERA_SCENE_CHANGE_THRESHOLD = 40.0  # mean grey-level change vs the learned background

# --- API access ---
# Shared secret the dashboard must send on every API request and live stream.
# Set GATEKEEPER_API_TOKEN (in .env or the environment) to keep it stable
# across restarts; if it's unset, server.py generates a random one at
# startup and prints it.
API_TOKEN = os.environ.get("GATEKEEPER_API_TOKEN") or None

# --- Storage ---
DB_PATH = os.path.join(BASE_DIR, "staff.db")
ENROLLMENT_DIR = os.path.join(BASE_DIR, "enrollment_photos")
# Frames saved with alarms and access decisions, for review. They show
# people's faces, so they're deleted after SNAPSHOT_RETENTION_DAYS.
SNAPSHOT_DIR = os.path.join(BASE_DIR, "snapshots")
SNAPSHOT_RETENTION_DAYS = 30

# --- Alarm outputs ---
# Every alarm except "info" is POSTed as JSON to this URL (README "Alarms and
# the audit trail" lists the fields), e.g. an n8n workflow that forwards it to
# WhatsApp. Set it in .env, not here:
#   ALARM_WEBHOOK_URL=https://n8n.example.com/webhook/<id>
ALARM_WEBHOOK_URL = os.environ.get("ALARM_WEBHOOK_URL") or None
# Attach the alarm's snapshot (JPEG, base64) to the webhook call. Off unless
# ALARM_WEBHOOK_SNAPSHOT=true in .env: snapshots show faces, and whatever
# receives the webhook (and any chat app it forwards to) keeps its own copy,
# beyond SNAPSHOT_RETENTION_DAYS.
ALARM_WEBHOOK_SNAPSHOT = (os.environ.get("ALARM_WEBHOOK_SNAPSHOT") or "").strip().lower() in ("1", "true", "yes")

# Play an audible sound on every alarm trigger.
ALARM_SOUND_ENABLED = True
# Path to a sound file, or None to use a built-in system alert sound
# (macOS: /System/Library/Sounds/Sosumi.aiff). On Linux, a path is required -
# there's no universal built-in equivalent to fall back to.
ALARM_SOUND_PATH = None

# --- Email (Resend) ---
# Every alarm except "info" (the ones that also sound and call the webhook)
# is emailed automatically to ALARM_EMAIL_TO through Resend (resend.com).
# Alarms raised together share one email, and emails go out at most once per
# ALARM_EMAIL_MIN_INTERVAL_SECONDS; alarms in between wait for the next one.
# The dashboard's "Email" button also sends a full status report on demand.
# Set these in .env, not here (see README "Email alerts"):
#   RESEND_API_KEY=re_...
#   ALARM_EMAIL_TO=you@example.com,guard@example.com
#   ALARM_EMAIL_FROM=Gatekeeper <alerts@your-domain.com>   # an address on a domain verified in Resend
# Until you verify a domain, leave ALARM_EMAIL_FROM unset: Resend's test
# sender (onboarding@resend.dev) only delivers to your Resend account's address.
RESEND_API_KEY = os.environ.get("RESEND_API_KEY") or None
ALARM_EMAIL_FROM = os.environ.get("ALARM_EMAIL_FROM") or "Gatekeeper <onboarding@resend.dev>"
ALARM_EMAIL_TO = [a.strip() for a in (os.environ.get("ALARM_EMAIL_TO") or "").split(",") if a.strip()]
ALARM_EMAIL_AUTOMATIC = True
ALARM_EMAIL_MIN_INTERVAL_SECONDS = 60

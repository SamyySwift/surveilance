"""
FastAPI backend for the access-control dashboard.

Serves REST endpoints for staff enrollment and management, the badge + face
check your access control system calls, the alarm and access-decision audit
trail, and a WebSocket that streams live camera frames + face detections to
the React frontend.

Run with:
    uvicorn server:app --host 0.0.0.0 --port 8000
"""

import asyncio
import base64
import json
import secrets
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional

from fastapi import (
    APIRouter, Depends, FastAPI, File, Form, Header, HTTPException, UploadFile,
    WebSocket, WebSocketDisconnect,
)
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

import alarms
import config
import database
import email_report
import liveness as liveness_mod
from enrollment import decode_image, enroll_image
from face_engine import load_engine
from worker import ZoneWorker

API_TOKEN = config.API_TOKEN or secrets.token_urlsafe(24)
if not config.API_TOKEN:
    print(
        "[auth] GATEKEEPER_API_TOKEN is not set, so a token was generated for this run.\n"
        f"[auth] Enter it in the dashboard to sign in: {API_TOKEN}\n"
        "[auth] Set GATEKEEPER_API_TOKEN to keep the same token across restarts."
    )


class Runtime:
    """What the server loads at startup. Tests swap the build_* functions."""
    engine = None
    liveness = None
    workers: dict = {}


runtime = Runtime()


def build_engine():
    return load_engine()


def build_liveness():
    return liveness_mod.load_liveness()


def build_workers(engine, live):
    return {zone: ZoneWorker(zone, cfg, engine, live) for zone, cfg in config.CAMERAS.items()}


@asynccontextmanager
async def lifespan(app: FastAPI):
    database.init_db()
    Path(config.ENROLLMENT_DIR).mkdir(parents=True, exist_ok=True)
    alarms.purge_old_snapshots()
    runtime.engine = build_engine()
    runtime.liveness = build_liveness()
    runtime.workers = build_workers(runtime.engine, runtime.liveness)
    email_report.set_camera_status(lambda: {zone: w.camera_connected for zone, w in runtime.workers.items()})
    if email_report.automatic():
        print(f"[email] New alarms will be emailed to {', '.join(config.ALARM_EMAIL_TO)} through Resend.")
    else:
        print("[email] Automatic alarm emails are off: set RESEND_API_KEY and ALARM_EMAIL_TO in .env.")
    yield
    for w in runtime.workers.values():
        w.stop()


app = FastAPI(title="Access Control API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _token_ok(candidate: Optional[str]) -> bool:
    return candidate is not None and secrets.compare_digest(candidate.encode(), API_TOKEN.encode())


def require_token(authorization: Optional[str] = Header(None)):
    scheme, _, token = (authorization or "").partition(" ")
    if scheme.lower() != "bearer" or not _token_ok(token):
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API token",
            headers={"WWW-Authenticate": "Bearer"},
        )


# Every REST endpoint lives on this router, so anything added to it later is
# token-protected by default.
api = APIRouter(prefix="/api", dependencies=[Depends(require_token)])


def _worker(zone: str) -> ZoneWorker:
    worker = runtime.workers.get(zone)
    if worker is None:
        raise HTTPException(status_code=404, detail=f"Unknown zone {zone!r}")
    return worker


def _require_staff(staff_id: int):
    row = database.get_staff_by_id(staff_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Staff not found")
    return row


def _configured_zones(zones: List[str]) -> List[str]:
    """Zones as config.CAMERAS spells them; 422 for any zone that has no camera."""
    known, unknown = database.match_zones(zones, config.CAMERAS)
    if unknown:
        raise HTTPException(status_code=422, detail=f"Unknown zone(s): {', '.join(unknown)}. "
                            f"Configured zones: {', '.join(config.CAMERAS)}")
    return known


def _staff_to_dict(s: dict) -> dict:
    return {
        "id": s["id"],
        "name": s["name"],
        "employeeId": s["employee_id"],
        "badgeId": s["badge_id"],
        "active": s["active"],
        "numEmbeddings": s["num_embeddings"],
        "zones": s["zones"],
    }


def _photo_path(source_photo: Optional[str]) -> Optional[Path]:
    if not source_photo:
        return None
    # .name strips any directory parts, so a stored value can't point outside
    # the enrollment folder.
    return Path(config.ENROLLMENT_DIR) / Path(source_photo).name


def _remove_photo_file(source_photo: Optional[str]):
    """Delete an enrollment photo from disk once no embedding references it."""
    path = _photo_path(source_photo)
    if path is None or database.photo_in_use(source_photo):
        return
    path.unlink(missing_ok=True)


def _get_owned_embedding(staff_id: int, embedding_id: int):
    row = database.get_embedding(embedding_id)
    if row is None or row["staff_id"] != staff_id:
        raise HTTPException(status_code=404, detail="Photo not found")
    return row


async def _enroll_uploads(staff_id: int, photos: List[UploadFile]):
    enrolled, errors = 0, []
    for photo in photos:
        data = await photo.read()
        result = await run_in_threadpool(enroll_image, runtime.engine, staff_id, decode_image(data),
                                         photo.filename, data)
        if result.ok:
            enrolled += 1
        else:
            errors.append(result.message)
    return enrolled, errors


# ---------- Staff management ----------

@api.get("/staff")
def api_list_staff():
    return [_staff_to_dict(s) for s in database.list_staff(runtime.engine.model_id)]


@api.post("/staff")
async def api_enroll_staff(
    name: str = Form(...),
    employee_id: Optional[str] = Form(None),
    badge_id: Optional[str] = Form(None),
    zones: str = Form(""),
    photos: List[UploadFile] = File(...),
):
    badge_id = (badge_id or "").strip() or None
    existing = database.get_staff_by_name(name)
    if badge_id:
        holder = database.get_staff_by_badge(badge_id)
        if holder is not None and (existing is None or holder["id"] != existing["id"]):
            raise HTTPException(status_code=409, detail=f"Badge {badge_id} is already assigned to {holder['name']}")

    created_new = existing is None
    zone_list = _configured_zones([z for z in zones.split(",") if z.strip()])
    staff_id = existing["id"] if existing else database.add_staff(name, employee_id, zone_list, badge_id)
    if existing:
        database.set_staff_zones(staff_id, zone_list)
        if badge_id:
            database.set_staff_badge(staff_id, badge_id)

    enrolled, errors = await _enroll_uploads(staff_id, photos)
    if enrolled == 0:
        # Don't leave a ghost "0 photos" staff record behind if every photo
        # in this request failed and we were the one who created the row.
        if created_new:
            database.delete_staff(staff_id)
        raise HTTPException(status_code=400, detail={"message": "No photos could be enrolled", "errors": errors})

    return {"staffId": staff_id, "enrolled": enrolled, "errors": errors}


@api.post("/staff/{staff_id}/photos")
async def api_add_photos(staff_id: int, photos: List[UploadFile] = File(...)):
    """Add more enrollment photos to an existing staff member (retraining)."""
    _require_staff(staff_id)
    enrolled, errors = await _enroll_uploads(staff_id, photos)
    return {"enrolled": enrolled, "errors": errors}


class CameraEnrollRequest(BaseModel):
    zone: str
    seconds: float = Field(6.0, ge=1.0, le=30.0)
    maxPhotos: int = Field(3, ge=1, le=10)


@api.post("/staff/{staff_id}/enroll-from-camera")
async def api_enroll_from_camera(staff_id: int, req: CameraEnrollRequest):
    """
    Capture enrollment photos at a door camera, in the conditions the system
    will see every day. The person should stand alone in front of the camera
    for the capture.
    """
    _require_staff(staff_id)
    worker = _worker(req.zone)
    crops, rejected = await run_in_threadpool(worker.capture_enrollment_crops, req.seconds, req.maxPhotos)
    if not crops:
        summary = ", ".join(f"{reason} ({n})" for reason, n in sorted(rejected.items(), key=lambda kv: -kv[1]))
        raise HTTPException(status_code=422, detail={
            "message": "No usable frames captured",
            "errors": [summary or "no frames from the camera"],
        })
    enrolled, errors = 0, []
    for crop in crops:
        result = await run_in_threadpool(enroll_image, runtime.engine, staff_id, crop, f"{req.zone}_camera.jpg")
        if result.ok:
            enrolled += 1
        else:
            errors.append(result.message)
    return {"enrolled": enrolled, "errors": errors, "rejectedFrames": rejected}


@api.get("/staff/{staff_id}/photos")
def api_list_photos(staff_id: int):
    _require_staff(staff_id)
    rows = database.list_embeddings_for_staff(staff_id, runtime.engine.model_id)
    return [{"id": r["id"], "sourcePhoto": r["source_photo"], "createdAt": r["created_at"]} for r in rows]


@api.get("/staff/{staff_id}/photos/{embedding_id}/image")
def api_photo_image(staff_id: int, embedding_id: int):
    """Serve an enrolled photo. These are biometric data, so they go through
    the same token check as everything else rather than a public static mount."""
    row = _get_owned_embedding(staff_id, embedding_id)
    path = _photo_path(row["source_photo"])
    if path is None or not path.is_file():
        raise HTTPException(status_code=404, detail="Photo file missing")
    return FileResponse(path)


@api.delete("/staff/{staff_id}/photos/{embedding_id}")
def api_delete_photo(staff_id: int, embedding_id: int):
    """Remove an enrolled photo (every model's embedding of it, and the file)."""
    row = _get_owned_embedding(staff_id, embedding_id)
    if row["source_photo"]:
        database.delete_photo_embeddings(staff_id, row["source_photo"])
    else:
        database.delete_embedding(embedding_id)
    _remove_photo_file(row["source_photo"])
    return {"ok": True}


class StaffUpdate(BaseModel):
    active: Optional[bool] = None
    zones: Optional[List[str]] = None
    badgeId: Optional[str] = None


@api.patch("/staff/{staff_id}")
def api_update_staff(staff_id: int, payload: StaffUpdate):
    _require_staff(staff_id)
    if payload.active is not None:
        database.set_staff_active(staff_id, payload.active)
    if payload.zones is not None:
        database.set_staff_zones(staff_id, _configured_zones(payload.zones))
    if "badgeId" in payload.model_fields_set:
        try:
            database.set_staff_badge(staff_id, (payload.badgeId or "").strip() or None)
        except ValueError as e:
            raise HTTPException(status_code=409, detail=str(e))
    return {"ok": True}


@api.delete("/staff/{staff_id}")
def api_delete_staff(staff_id: int):
    _require_staff(staff_id)
    photos = database.list_source_photos(staff_id)
    database.delete_staff(staff_id)
    for photo in photos:
        _remove_photo_file(photo)
    return {"ok": True}


# ---------- Badge + face ----------

class BadgeRequest(BaseModel):
    zone: str
    badgeId: str = Field(..., min_length=1)


@api.post("/access/badge")
async def api_badge(req: BadgeRequest):
    """
    Call this when a badge is read at a door. Waits up to
    BADGE_DECISION_TIMEOUT_SECONDS for a good view of the person, then returns
    {"decision": "grant" | "deny", "reason": ...}. Open the door only on "grant".
    """
    if not config.BADGE_CHECKS_ENABLED:
        raise HTTPException(status_code=503,
                            detail="Badge checks are switched off (BADGE_CHECKS_ENABLED in config.py)")
    worker = _worker(req.zone)
    return await run_in_threadpool(worker.badge_check, req.badgeId.strip())


# ---------- Status, alarms, audit ----------

@api.get("/zones")
def api_zones():
    return list(config.CAMERAS.keys())


@api.get("/status")
def api_status():
    live = runtime.liveness
    return {
        "zones": {zone: w.status() for zone, w in runtime.workers.items()},
        "recognizer": runtime.engine.model_id,
        "liveness": {
            "mode": config.LIVENESS_MODE,
            "model": getattr(live, "model_id", None),
            "available": live is not None and not isinstance(live, liveness_mod.LivenessUnavailable),
            "problem": getattr(live, "reason", None),
        },
        "planarityMode": config.PLANARITY_MODE,
        "badgeChecks": config.BADGE_CHECKS_ENABLED,
        "email": {
            "configured": email_report.configured(),
            "automatic": email_report.automatic(),
            "defaultTo": config.ALARM_EMAIL_TO,
            "problem": email_report.delivery_problem(),
        },
    }


def _alarm_to_dict(r) -> dict:
    return {
        "id": r["id"],
        "time": r["created_at"],
        "zone": r["zone"],
        "kind": r["kind"],
        "severity": r["severity"],
        "message": r["message"],
        "staffId": r["staff_id"],
        "snapshot": r["snapshot"],
        "details": json.loads(r["details_json"]) if r["details_json"] else {},
        "acknowledgedAt": r["acknowledged_at"],
        "acknowledgedNote": r["acknowledged_note"],
    }


@api.get("/alarms")
def api_alarms(limit: int = 100, unacknowledged: bool = False):
    return [_alarm_to_dict(r) for r in database.list_alarms(min(max(limit, 1), 500), unacknowledged)]


@api.delete("/alarms")
def api_clear_alarms(up_to_id: int):
    """
    Clear the alarm log, with its snapshots. Only alarms up to the newest one
    the dashboard was showing are deleted, so an alarm raised while someone
    was confirming isn't wiped before anyone sees it.
    """
    return {"cleared": alarms.clear_alarms(up_to_id)}


class EmailReportRequest(BaseModel):
    to: List[str]


@api.post("/alarms/email")
def api_email_alarms(req: EmailReportRequest):
    """Email the alarm status report (open alarms, recently acknowledged ones, camera status) now."""
    if not email_report.configured():
        raise HTTPException(status_code=503, detail="Email isn't set up on the server yet. Add RESEND_API_KEY "
                            "to .env and restart the server (README \"Email alerts\").")
    try:
        recipients = email_report.check_recipients(req.to)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    try:
        subject = email_report.send_report(recipients)
    except email_report.EmailError as e:
        raise HTTPException(status_code=502, detail=str(e))
    return {"sent": True, "to": recipients, "subject": subject}


class AckRequest(BaseModel):
    note: Optional[str] = Field(None, max_length=500)


@api.post("/alarms/{alarm_id}/ack")
def api_ack_alarm(alarm_id: int, req: AckRequest):
    if not database.acknowledge_alarm(alarm_id, req.note):
        raise HTTPException(status_code=404, detail="Alarm not found or already acknowledged")
    return {"ok": True}


@api.get("/access-events")
def api_access_events(limit: int = 100):
    rows = database.list_access_events(min(max(limit, 1), 500))
    return [{
        "id": r["id"],
        "time": r["created_at"],
        "zone": r["zone"],
        "badgeId": r["badge_id"],
        "staffId": r["staff_id"],
        "name": r["staff_name"],
        "decision": r["decision"],
        "reason": r["reason"],
        "matchScore": r["match_score"],
        "livenessScore": r["liveness_score"],
        "planarity": r["planarity"],
        "samples": r["samples"],
        "modelId": r["model_id"],
        "livenessModelId": r["liveness_model_id"],
        "snapshot": r["snapshot"],
    } for r in rows]


@api.get("/snapshots/{name}")
def api_snapshot(name: str):
    path = alarms.snapshot_path(name)
    if path is None or not database.snapshot_referenced(name):
        raise HTTPException(status_code=404, detail="Snapshot not found")
    return FileResponse(path, media_type="image/jpeg")


app.include_router(api)


# ---------- Live stream ----------

WS_AUTH_TIMEOUT_SECONDS = 5


@app.websocket("/ws/stream/{zone}")
async def ws_stream(websocket: WebSocket, zone: str):
    await websocket.accept()

    # Browsers can't set headers on a WebSocket, and a token in the URL would
    # end up in access logs, so the client sends it as the first message.
    # CORS doesn't apply to WebSockets either, so without this any web page
    # the operator visits could open the feed.
    try:
        message = await asyncio.wait_for(websocket.receive(), timeout=WS_AUTH_TIMEOUT_SECONDS)
    except asyncio.TimeoutError:
        message = {}
    if message.get("type") == "websocket.disconnect":
        return
    if not _token_ok(message.get("text")):
        await websocket.close(code=1008, reason="Invalid or missing API token")
        return

    worker = runtime.workers.get(zone)
    if not worker:
        await websocket.close(reason="Unknown zone")
        return

    # After the token the client may send {"video": false} (or true) at any
    # time. With video off, no frames are sent - or even encoded, if no one
    # else is watching - but camera status and who's in view keep flowing,
    # and recognition carries on regardless.
    state = {"video": True, "open": True}
    worker.add_video_viewer()

    async def listen():
        try:
            while True:
                message = await websocket.receive()
                if message.get("type") == "websocket.disconnect":
                    break
                try:
                    wanted = json.loads(message.get("text") or "null").get("video")
                except (ValueError, AttributeError):
                    continue
                if isinstance(wanted, bool) and wanted != state["video"]:
                    state["video"] = wanted
                    (worker.add_video_viewer if wanted else worker.remove_video_viewer)()
        finally:
            state["open"] = False

    listener = asyncio.create_task(listen())
    try:
        while state["open"]:
            jpeg, detections = worker.snapshot()
            payload = {
                "connected": worker.camera_connected,
                "video": state["video"],
                "frame": base64.b64encode(jpeg).decode("ascii") if (jpeg and state["video"]) else None,
                "detections": detections,
            }
            await websocket.send_text(json.dumps(payload))
            # Outgoing rate to the browser, independent of the processing rate.
            await asyncio.sleep(1 / 15 if state["video"] else 1 / 4)
    except (WebSocketDisconnect, RuntimeError):  # RuntimeError: sending after the client left
        pass
    finally:
        listener.cancel()
        if state["video"]:
            worker.remove_video_viewer()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)

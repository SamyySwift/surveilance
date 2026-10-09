"""
SQLite storage for enrolled staff, their face embeddings, alarms and the
access-decision audit trail.

Each staff member can have several embeddings (one per enrollment photo),
and each embedding records which recognizer model produced it, because
embeddings from different models can't be compared with each other.

Rows come back as sqlite3.Row, which supports both row["name"] and row[1].
"""

import json
import os
import shutil
import sqlite3
import time
from contextlib import contextmanager
from typing import Optional

import numpy as np

import config

# Embeddings created before model tracking existed were all made by this pack.
LEGACY_MODEL_ID = "insightface:buffalo_l"

SCHEMA = """
CREATE TABLE IF NOT EXISTS staff (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    employee_id TEXT UNIQUE,
    badge_id TEXT,
    active INTEGER NOT NULL DEFAULT 1,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS embeddings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    staff_id INTEGER NOT NULL,
    vector BLOB NOT NULL,
    source_photo TEXT,
    model_id TEXT,
    quality_json TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (staff_id) REFERENCES staff(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS staff_zones (
    staff_id INTEGER NOT NULL,
    zone TEXT NOT NULL,
    FOREIGN KEY (staff_id) REFERENCES staff(id) ON DELETE CASCADE,
    UNIQUE(staff_id, zone)
);

-- Small key/value store; holds gallery_version, which every change to
-- enrolled faces bumps so caches in other processes know to reload.
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT
);

CREATE TABLE IF NOT EXISTS alarms (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at REAL NOT NULL,
    zone TEXT,
    kind TEXT NOT NULL,
    severity TEXT NOT NULL,
    message TEXT NOT NULL,
    staff_id INTEGER,
    track_id INTEGER,
    snapshot TEXT,
    details_json TEXT,
    acknowledged_at REAL,
    acknowledged_note TEXT
);

CREATE TABLE IF NOT EXISTS access_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at REAL NOT NULL,
    zone TEXT NOT NULL,
    badge_id TEXT,
    staff_id INTEGER,
    staff_name TEXT,
    decision TEXT NOT NULL,
    reason TEXT,
    match_score REAL,
    liveness_score REAL,
    planarity TEXT,
    samples INTEGER,
    model_id TEXT,
    liveness_model_id TEXT,
    snapshot TEXT,
    details_json TEXT
);
"""

# Columns added after the first release, applied to existing databases by
# init_db(). (table, column, definition)
_ADDED_COLUMNS = [
    ("staff", "badge_id", "TEXT"),
    ("embeddings", "model_id", "TEXT"),
    ("embeddings", "quality_json", "TEXT"),
]

_INDEXES = """
CREATE UNIQUE INDEX IF NOT EXISTS idx_staff_badge_id ON staff(badge_id);
CREATE INDEX IF NOT EXISTS idx_embeddings_model ON embeddings(model_id, staff_id);
CREATE INDEX IF NOT EXISTS idx_alarms_created ON alarms(created_at);
CREATE INDEX IF NOT EXISTS idx_access_events_created ON access_events(created_at);
"""


@contextmanager
def get_conn():
    # timeout: the server's threads and the CLI tools can write at the same time.
    conn = sqlite3.connect(config.DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def _needs_upgrade() -> bool:
    if not os.path.exists(config.DB_PATH) or os.path.getsize(config.DB_PATH) == 0:
        return False
    with get_conn() as conn:
        tables = {r["name"] for r in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
        for table, column, _ in _ADDED_COLUMNS:
            if table in tables and column not in {r["name"] for r in conn.execute(f"PRAGMA table_info({table})")}:
                return True
    return False


def init_db():
    """Create tables, and upgrade a database made by an older version
    (keeping a copy of it first, next to the original)."""
    if _needs_upgrade():
        backup = f"{config.DB_PATH}.before-upgrade-{time.strftime('%Y%m%d-%H%M%S')}"
        shutil.copy2(config.DB_PATH, backup)
        os.chmod(backup, 0o600)
        print(f"[database] Upgrading {config.DB_PATH}; the previous version is saved as {backup}")
    with get_conn() as conn:
        # WAL lets readers carry on while another process writes.
        conn.execute("PRAGMA journal_mode = WAL")
        conn.executescript(SCHEMA)
        for table, column, definition in _ADDED_COLUMNS:
            existing = {r["name"] for r in conn.execute(f"PRAGMA table_info({table})")}
            if column not in existing:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")
        conn.execute("UPDATE embeddings SET model_id = ? WHERE model_id IS NULL", (LEGACY_MODEL_ID,))
        conn.executescript(_INDEXES)
    for name, old, new in normalize_staff_zones(config.CAMERAS):
        print(f'[database] {name}: zone "{old}" is now "{new}", as config.CAMERAS spells it')


def _bump_gallery_version(conn):
    conn.execute(
        """
        INSERT INTO meta (key, value) VALUES ('gallery_version', '1')
        ON CONFLICT(key) DO UPDATE SET value = CAST(value AS INTEGER) + 1
        """
    )


def get_gallery_version() -> int:
    with get_conn() as conn:
        row = conn.execute("SELECT value FROM meta WHERE key = 'gallery_version'").fetchone()
        return int(row["value"]) if row else 0


# ---------- Staff ----------

def add_staff(name: str, employee_id: Optional[str] = None, zones: Optional[list[str]] = None,
              badge_id: Optional[str] = None) -> int:
    with get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO staff (name, employee_id, badge_id) VALUES (?, ?, ?)",
            (name, employee_id, badge_id),
        )
        staff_id = cur.lastrowid
        for z in zones or []:
            conn.execute("INSERT INTO staff_zones (staff_id, zone) VALUES (?, ?)", (staff_id, z))
        _bump_gallery_version(conn)
        return staff_id


def set_staff_zones(staff_id: int, zones: list[str]):
    with get_conn() as conn:
        conn.execute("DELETE FROM staff_zones WHERE staff_id = ?", (staff_id,))
        for z in zones:
            conn.execute("INSERT INTO staff_zones (staff_id, zone) VALUES (?, ?)", (staff_id, z))


def match_zones(zones: list[str], configured) -> tuple[list[str], list[str]]:
    """
    (zones spelled as in `configured`, zones matching none of them), ignoring
    case and spaces. Zone checks compare names exactly, so a "vault" stored
    for a camera zone called "Vault" would silently deny access there.
    """
    by_key = {z.casefold(): z for z in configured}
    known, unknown = [], []
    for z in zones:
        name = by_key.get(z.strip().casefold())
        if name is None:
            unknown.append(z)
        elif name not in known:
            known.append(name)
    return known, unknown


def normalize_staff_zones(configured) -> list[tuple[str, str, str]]:
    """
    Respell stored zones the way `configured` spells them (an old "vault"
    becomes "Vault"), merging duplicates. Zones with no configured camera are
    left alone. Returns (staff name, old spelling, new spelling) per change.
    """
    by_key = {z.casefold(): z for z in configured}
    changes = []
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT sz.staff_id, sz.zone, s.name FROM staff_zones sz JOIN staff s ON s.id = sz.staff_id"
            " ORDER BY s.name, sz.zone"
        ).fetchall()
        for r in rows:
            new = by_key.get(r["zone"].strip().casefold())
            if new is None or new == r["zone"]:
                continue
            conn.execute("DELETE FROM staff_zones WHERE staff_id = ? AND zone = ?", (r["staff_id"], r["zone"]))
            conn.execute("INSERT OR IGNORE INTO staff_zones (staff_id, zone) VALUES (?, ?)", (r["staff_id"], new))
            changes.append((r["name"], r["zone"], new))
    return changes


def get_staff_zones(staff_id: Optional[int]) -> list[str]:
    if staff_id is None:
        return []
    with get_conn() as conn:
        cur = conn.execute("SELECT zone FROM staff_zones WHERE staff_id = ?", (staff_id,))
        return [r["zone"] for r in cur.fetchall()]


def set_staff_badge(staff_id: int, badge_id: Optional[str]):
    """Raises ValueError if another staff member already has this badge."""
    try:
        with get_conn() as conn:
            conn.execute("UPDATE staff SET badge_id = ? WHERE id = ?", (badge_id or None, staff_id))
    except sqlite3.IntegrityError:
        raise ValueError(f"Badge {badge_id} is already assigned to someone else") from None


_STAFF_COLUMNS = "id, name, employee_id, badge_id, active"


def get_staff_by_name(name: str):
    with get_conn() as conn:
        return conn.execute(f"SELECT {_STAFF_COLUMNS} FROM staff WHERE name = ?", (name,)).fetchone()


def get_staff_by_id(staff_id: int):
    with get_conn() as conn:
        return conn.execute(f"SELECT {_STAFF_COLUMNS} FROM staff WHERE id = ?", (staff_id,)).fetchone()


def get_staff_by_badge(badge_id: str):
    with get_conn() as conn:
        return conn.execute(f"SELECT {_STAFF_COLUMNS} FROM staff WHERE badge_id = ?", (badge_id,)).fetchone()


def list_staff(model_id: Optional[str] = None) -> list[dict]:
    """
    Everyone enrolled, as dicts. num_embeddings counts only embeddings made by
    model_id when given (the ones the live system can actually use).
    """
    with get_conn() as conn:
        cur = conn.execute(
            """
            SELECT s.id, s.name, s.employee_id, s.badge_id, s.active,
                   (SELECT COUNT(*) FROM embeddings e
                     WHERE e.staff_id = s.id AND (? IS NULL OR e.model_id = ?)) AS num_embeddings,
                   (SELECT GROUP_CONCAT(sz.zone) FROM staff_zones sz
                     WHERE sz.staff_id = s.id) AS zones
            FROM staff s
            ORDER BY s.name
            """,
            (model_id, model_id),
        )
        rows = cur.fetchall()
    return [
        {
            "id": r["id"],
            "name": r["name"],
            "employee_id": r["employee_id"],
            "badge_id": r["badge_id"],
            "active": bool(r["active"]),
            "num_embeddings": r["num_embeddings"],
            "zones": sorted(r["zones"].split(",")) if r["zones"] else [],
        }
        for r in rows
    ]


def set_staff_active(staff_id: int, active: bool):
    """Toggle access without deleting history (e.g. suspension, leave)."""
    with get_conn() as conn:
        conn.execute("UPDATE staff SET active = ? WHERE id = ?", (1 if active else 0, staff_id))
        _bump_gallery_version(conn)


def delete_staff(staff_id: int):
    """Permanently remove a staff member and all their enrolled embeddings."""
    with get_conn() as conn:
        conn.execute("DELETE FROM staff WHERE id = ?", (staff_id,))
        _bump_gallery_version(conn)


# ---------- Embeddings ----------

def add_embedding(staff_id: int, vector: np.ndarray, source_photo: Optional[str] = None, *,
                  model_id: str, quality: Optional[dict] = None) -> int:
    vector = np.asarray(vector, dtype=np.float32)
    with get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO embeddings (staff_id, vector, source_photo, model_id, quality_json)"
            " VALUES (?, ?, ?, ?, ?)",
            (staff_id, vector.tobytes(), source_photo, model_id,
             json.dumps(quality) if quality is not None else None),
        )
        _bump_gallery_version(conn)
        return cur.lastrowid


def load_all_embeddings(model_id: str, include_inactive: bool = False):
    """
    Returns (staff_ids, names, vectors) for one recognizer model, where
    vectors is an (N, D) float32 array ready for a batched similarity
    computation. Suspended staff are left out unless include_inactive.
    """
    with get_conn() as conn:
        cur = conn.execute(
            """
            SELECT e.staff_id, s.name, e.vector
            FROM embeddings e
            JOIN staff s ON s.id = e.staff_id
            WHERE e.model_id = ? AND (s.active = 1 OR ?)
            ORDER BY e.id
            """,
            (model_id, 1 if include_inactive else 0),
        )
        rows = cur.fetchall()

    if not rows:
        return [], [], np.zeros((0, 0), dtype=np.float32)

    staff_ids = [r["staff_id"] for r in rows]
    names = [r["name"] for r in rows]
    vectors = np.stack([np.frombuffer(r["vector"], dtype=np.float32) for r in rows])
    return staff_ids, names, vectors


def list_embeddings_for_staff(staff_id: int, model_id: Optional[str] = None):
    """Rows of (id, source_photo, created_at, model_id), oldest first."""
    with get_conn() as conn:
        cur = conn.execute(
            """
            SELECT id, source_photo, created_at, model_id FROM embeddings
            WHERE staff_id = ? AND (? IS NULL OR model_id = ?)
            ORDER BY created_at, id
            """,
            (staff_id, model_id, model_id),
        )
        return cur.fetchall()


def list_source_photos(staff_id: int) -> list[str]:
    """Distinct enrollment photo files for a person, across all models."""
    with get_conn() as conn:
        cur = conn.execute(
            "SELECT DISTINCT source_photo FROM embeddings"
            " WHERE staff_id = ? AND source_photo IS NOT NULL ORDER BY source_photo",
            (staff_id,),
        )
        return [r["source_photo"] for r in cur.fetchall()]


def has_embedding(staff_id: int, source_photo: str, model_id: str) -> bool:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT 1 FROM embeddings WHERE staff_id = ? AND source_photo = ? AND model_id = ? LIMIT 1",
            (staff_id, source_photo, model_id),
        ).fetchone()
        return row is not None


def get_embedding(embedding_id: int):
    """Returns (id, staff_id, source_photo, model_id), or None if it doesn't exist."""
    with get_conn() as conn:
        cur = conn.execute(
            "SELECT id, staff_id, source_photo, model_id FROM embeddings WHERE id = ?",
            (embedding_id,),
        )
        return cur.fetchone()


def photo_in_use(source_photo: str) -> bool:
    """Whether any remaining embedding still references this photo file."""
    with get_conn() as conn:
        cur = conn.execute(
            "SELECT 1 FROM embeddings WHERE source_photo = ? LIMIT 1",
            (source_photo,),
        )
        return cur.fetchone() is not None


def delete_embedding(embedding_id: int):
    """Remove a single embedding."""
    with get_conn() as conn:
        conn.execute("DELETE FROM embeddings WHERE id = ?", (embedding_id,))
        _bump_gallery_version(conn)


def delete_photo_embeddings(staff_id: int, source_photo: str) -> int:
    """Remove every model's embedding of one enrollment photo. Returns the count."""
    with get_conn() as conn:
        cur = conn.execute(
            "DELETE FROM embeddings WHERE staff_id = ? AND source_photo = ?",
            (staff_id, source_photo),
        )
        _bump_gallery_version(conn)
        return cur.rowcount


# ---------- Alarms ----------

def add_alarm(kind: str, severity: str, message: str, zone: Optional[str] = None,
              staff_id: Optional[int] = None, track_id: Optional[int] = None,
              snapshot: Optional[str] = None, details: Optional[dict] = None,
              created_at: Optional[float] = None) -> int:
    with get_conn() as conn:
        cur = conn.execute(
            """
            INSERT INTO alarms (created_at, zone, kind, severity, message, staff_id,
                                track_id, snapshot, details_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (created_at or time.time(), zone, kind, severity, message, staff_id, track_id,
             snapshot, json.dumps(details) if details else None),
        )
        return cur.lastrowid


def list_alarms(limit: int = 100, unacknowledged_only: bool = False):
    with get_conn() as conn:
        cur = conn.execute(
            f"""
            SELECT * FROM alarms
            {"WHERE acknowledged_at IS NULL" if unacknowledged_only else ""}
            ORDER BY created_at DESC, id DESC LIMIT ?
            """,
            (limit,),
        )
        return cur.fetchall()


def acknowledge_alarm(alarm_id: int, note: Optional[str] = None) -> bool:
    with get_conn() as conn:
        cur = conn.execute(
            "UPDATE alarms SET acknowledged_at = ?, acknowledged_note = ?"
            " WHERE id = ? AND acknowledged_at IS NULL",
            (time.time(), note, alarm_id),
        )
        return cur.rowcount > 0


def get_alarms(alarm_ids: list[int]):
    """These alarms (those that still exist), newest first."""
    if not alarm_ids:
        return []
    with get_conn() as conn:
        return conn.execute(
            f"SELECT * FROM alarms WHERE id IN ({','.join('?' * len(alarm_ids))})"
            " ORDER BY created_at DESC, id DESC",
            list(alarm_ids),
        ).fetchall()


def list_acknowledged_alarms(since: float, limit: int = 500):
    """Alarms acknowledged at or after `since`, most recently acknowledged first."""
    with get_conn() as conn:
        return conn.execute(
            "SELECT * FROM alarms WHERE acknowledged_at >= ? ORDER BY acknowledged_at DESC, id DESC LIMIT ?",
            (since, limit),
        ).fetchall()


def delete_alarms(up_to_id: int) -> tuple[int, list[str]]:
    """
    Delete every alarm with id <= up_to_id. Returns (how many, snapshot names
    no longer referenced by any record, so their files can go too).
    """
    with get_conn() as conn:
        snapshots = [r[0] for r in conn.execute(
            "SELECT DISTINCT snapshot FROM alarms WHERE id <= ? AND snapshot IS NOT NULL", (up_to_id,))]
        count = conn.execute("DELETE FROM alarms WHERE id <= ?", (up_to_id,)).rowcount
        orphaned = [name for name in snapshots if conn.execute(
            "SELECT 1 FROM alarms WHERE snapshot = ? UNION SELECT 1 FROM access_events WHERE snapshot = ?",
            (name, name)).fetchone() is None]
        return count, orphaned


# ---------- Access decisions (audit trail) ----------

def add_access_event(zone: str, decision: str, reason: str, badge_id: Optional[str] = None,
                     staff_id: Optional[int] = None, staff_name: Optional[str] = None,
                     match_score: Optional[float] = None, liveness_score: Optional[float] = None,
                     planarity: Optional[str] = None, samples: Optional[int] = None,
                     model_id: Optional[str] = None, liveness_model_id: Optional[str] = None,
                     snapshot: Optional[str] = None, details: Optional[dict] = None,
                     created_at: Optional[float] = None) -> int:
    with get_conn() as conn:
        cur = conn.execute(
            """
            INSERT INTO access_events (created_at, zone, badge_id, staff_id, staff_name, decision,
                                       reason, match_score, liveness_score, planarity, samples,
                                       model_id, liveness_model_id, snapshot, details_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (created_at or time.time(), zone, badge_id, staff_id, staff_name, decision, reason,
             match_score, liveness_score, planarity, samples, model_id, liveness_model_id,
             snapshot, json.dumps(details) if details else None),
        )
        return cur.lastrowid


def list_access_events(limit: int = 100):
    with get_conn() as conn:
        cur = conn.execute(
            "SELECT * FROM access_events ORDER BY created_at DESC, id DESC LIMIT ?", (limit,)
        )
        return cur.fetchall()


def snapshot_referenced(name: str) -> bool:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT 1 FROM alarms WHERE snapshot = ? UNION SELECT 1 FROM access_events WHERE snapshot = ?",
            (name, name),
        ).fetchone()
        return row is not None


def clear_snapshot_references(names: list[str]):
    """Called when old snapshot files are purged, so records don't point at missing files."""
    if not names:
        return
    with get_conn() as conn:
        conn.executemany("UPDATE alarms SET snapshot = NULL WHERE snapshot = ?", [(n,) for n in names])
        conn.executemany("UPDATE access_events SET snapshot = NULL WHERE snapshot = ?", [(n,) for n in names])

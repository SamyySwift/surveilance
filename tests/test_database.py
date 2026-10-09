import sqlite3

import numpy as np
import pytest

import config
import database

# The schema as it was before badges, model tracking, alarms and the audit trail.
OLD_SCHEMA = """
CREATE TABLE staff (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL, employee_id TEXT UNIQUE,
                    active INTEGER NOT NULL DEFAULT 1, created_at TEXT DEFAULT CURRENT_TIMESTAMP);
CREATE TABLE embeddings (id INTEGER PRIMARY KEY AUTOINCREMENT, staff_id INTEGER NOT NULL, vector BLOB NOT NULL,
                         source_photo TEXT, created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                         FOREIGN KEY (staff_id) REFERENCES staff(id) ON DELETE CASCADE);
CREATE TABLE staff_zones (staff_id INTEGER NOT NULL, zone TEXT NOT NULL,
                          FOREIGN KEY (staff_id) REFERENCES staff(id) ON DELETE CASCADE, UNIQUE(staff_id, zone));
"""


def test_existing_database_is_upgraded_in_place(tmp_path, monkeypatch):
    path = tmp_path / "old.db"
    conn = sqlite3.connect(path)
    conn.executescript(OLD_SCHEMA)
    conn.execute("INSERT INTO staff (name, employee_id) VALUES ('Samuel', '123')")
    conn.execute("INSERT INTO embeddings (staff_id, vector, source_photo) VALUES (1, ?, 'p1.jpg')",
                 (np.ones(4, dtype=np.float32).tobytes(),))
    conn.execute("INSERT INTO staff_zones VALUES (1, 'entrance')")
    conn.commit()
    conn.close()

    monkeypatch.setattr(config, "DB_PATH", str(path))
    database.init_db()
    database.init_db()  # running it again must be harmless

    # The pre-upgrade database was kept, once, untouched.
    [backup] = list(tmp_path.glob("old.db.before-upgrade-*"))
    old = sqlite3.connect(backup)
    assert "badge_id" not in {r[1] for r in old.execute("PRAGMA table_info(staff)")}
    assert old.execute("SELECT name FROM staff").fetchall() == [("Samuel",)]
    old.close()

    staff = database.list_staff()
    assert staff == [{"id": 1, "name": "Samuel", "employee_id": "123", "badge_id": None, "active": True,
                      "num_embeddings": 1, "zones": ["entrance"]}]
    assert database.get_embedding(1)["model_id"] == database.LEGACY_MODEL_ID
    ids, names, vecs = database.load_all_embeddings(database.LEGACY_MODEL_ID)
    assert ids == [1] and vecs.shape == (1, 4)
    with database.get_conn() as c:
        assert c.execute("PRAGMA journal_mode").fetchone()[0] == "wal"


def test_badges_are_unique():
    a = database.add_staff("Alice", badge_id="1001")
    b = database.add_staff("Bob")
    with pytest.raises(ValueError):
        database.set_staff_badge(b, "1001")
    database.set_staff_badge(a, None)
    database.set_staff_badge(b, "1001")
    assert database.get_staff_by_badge("1001")["name"] == "Bob"


def test_every_enrollment_change_bumps_the_gallery_version():
    v0 = database.get_gallery_version()
    sid = database.add_staff("Alice")
    eid = database.add_embedding(sid, np.ones(4), model_id="m")
    database.set_staff_active(sid, False)
    database.delete_embedding(eid)
    database.delete_staff(sid)
    assert database.get_gallery_version() == v0 + 5


def test_deleting_a_photo_removes_every_models_embedding_of_it():
    sid = database.add_staff("Alice")
    database.add_embedding(sid, np.ones(4), source_photo="a.jpg", model_id="m1")
    database.add_embedding(sid, np.ones(4), source_photo="a.jpg", model_id="m2")
    database.add_embedding(sid, np.ones(4), source_photo="b.jpg", model_id="m1")
    assert database.delete_photo_embeddings(sid, "a.jpg") == 2
    assert database.list_source_photos(sid) == ["b.jpg"]


def test_alarms_and_acknowledgement():
    first = database.add_alarm("unknown_person", "medium", "Stranger at door", zone="entrance",
                               details={"match_score": 0.21})
    database.add_alarm("tailgating", "high", "Tailgating", zone="entrance")
    assert [r["kind"] for r in database.list_alarms()] == ["tailgating", "unknown_person"]
    assert database.acknowledge_alarm(first, "checked, visitor")
    assert not database.acknowledge_alarm(first, "again")          # only once
    assert [r["kind"] for r in database.list_alarms(unacknowledged_only=True)] == ["tailgating"]


def test_snapshot_references_are_cleared_on_purge():
    database.add_alarm("x", "high", "m", snapshot="s1.jpg")
    database.add_access_event("entrance", "grant", "ok", snapshot="s1.jpg")
    assert database.snapshot_referenced("s1.jpg")
    database.clear_snapshot_references(["s1.jpg"])
    assert not database.snapshot_referenced("s1.jpg")


def test_zones_are_matched_to_configured_cameras_ignoring_case():
    assert database.match_zones([" vault", "ENTRANCE", "Vault", "lobby"], ["Entrance", "Vault"]) == (
        ["Vault", "Entrance"], ["lobby"])


def test_stored_zones_are_respelled_to_match_config(monkeypatch):
    # Zones saved when the cameras were called "entrance"/"vault", then renamed.
    sam = database.add_staff("Samuel", zones=["Lobby", "Vault", "entrance", "vault", "garage"])
    david = database.add_staff("David", zones=["entrance", "vault"])
    changes = database.normalize_staff_zones(["Entrance", "Vault", "Lobby"])
    assert sorted(changes) == [("David", "entrance", "Entrance"), ("David", "vault", "Vault"),
                               ("Samuel", "entrance", "Entrance"), ("Samuel", "vault", "Vault")]
    assert sorted(database.get_staff_zones(sam)) == ["Entrance", "Lobby", "Vault", "garage"]
    assert sorted(database.get_staff_zones(david)) == ["Entrance", "Vault"]
    assert database.normalize_staff_zones(["Entrance", "Vault", "Lobby"]) == []

    # And it happens on startup.
    database.set_staff_zones(david, ["vault"])
    monkeypatch.setattr(config, "CAMERAS", {"Vault": 0})
    database.init_db()
    assert database.get_staff_zones(david) == ["Vault"]

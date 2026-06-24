"""SQLite persistence for HoopTracker sessions and shots."""
from pathlib import Path
import sqlite3
import datetime
import json

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "hooptracker.db"


def _conn():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(str(DB_PATH))
    c.row_factory = sqlite3.Row
    return c


def init_db():
    with _conn() as c:
        c.executescript(
            """
            CREATE TABLE IF NOT EXISTS sessions(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                started_at TEXT, ended_at TEXT,
                mode TEXT, source TEXT,
                makes INTEGER DEFAULT 0, attempts INTEGER DEFAULT 0, fg_pct REAL DEFAULT 0
            );
            CREATE TABLE IF NOT EXISTS shots(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER, t REAL, result TEXT, made INTEGER,
                zone TEXT, x REAL, y REAL,
                elbow_angle REAL, knee_angle REAL, lean_deg REAL, follow_through INTEGER,
                FOREIGN KEY(session_id) REFERENCES sessions(id)
            );
            CREATE INDEX IF NOT EXISTS idx_shots_session ON shots(session_id);
            CREATE TABLE IF NOT EXISTS goals(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT, metric TEXT, target REAL, label TEXT, achieved_at TEXT
            );
            CREATE TABLE IF NOT EXISTS practice_log(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                logged_at TEXT, focus_metric TEXT, drill TEXT, note TEXT, baseline_std REAL
            );
            CREATE TABLE IF NOT EXISTS corrections(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                logged_at TEXT, session_id INTEGER, shot_id INTEGER,
                kind TEXT, old_result TEXT, new_result TEXT, t REAL
            );
            """
        )
        # migrations: additive columns (safe to run repeatedly)
        cols = [r[1] for r in c.execute("PRAGMA table_info(shots)").fetchall()]
        if "form_json" not in cols:
            c.execute("ALTER TABLE shots ADD COLUMN form_json TEXT")
        if "arc_json" not in cols:
            c.execute("ALTER TABLE shots ADD COLUMN arc_json TEXT")
        if "form_image" not in cols:
            c.execute("ALTER TABLE shots ADD COLUMN form_image TEXT")
        if "manual" not in cols:
            c.execute("ALTER TABLE shots ADD COLUMN manual INTEGER DEFAULT 0")
        if "miss_json" not in cols:
            c.execute("ALTER TABLE shots ADD COLUMN miss_json TEXT")
        scols = [r[1] for r in c.execute("PRAGMA table_info(sessions)").fetchall()]
        if "video_path" not in scols:
            c.execute("ALTER TABLE sessions ADD COLUMN video_path TEXT")
        if "annotated_path" not in scols:
            c.execute("ALTER TABLE sessions ADD COLUMN annotated_path TEXT")


def _now():
    return datetime.datetime.now().isoformat(sep=" ", timespec="seconds")


def _attach_form(d):
    """Add parsed 'form' (and 'arc') dicts to a shot row (form_json, else legacy columns)."""
    fj = d.get("form_json")
    form = None
    if fj:
        try:
            form = json.loads(fj)
        except Exception:
            form = None
    if form is None:
        form = {}
        for k in ("elbow_angle", "knee_angle", "lean_deg"):
            if d.get(k) is not None:
                form[k] = d[k]
        if d.get("follow_through") is not None:
            form["follow_through"] = bool(d["follow_through"])
    d["form"] = form
    aj = d.get("arc_json")
    if aj:
        try:
            d["arc"] = json.loads(aj)
        except Exception:
            pass
    mj = d.get("miss_json")
    if mj:
        try:
            d["miss"] = json.loads(mj)
        except Exception:
            pass
    return d


def create_session(mode, source=""):
    with _conn() as c:
        cur = c.execute(
            "INSERT INTO sessions(started_at, mode, source) VALUES(?,?,?)",
            (_now(), mode, source),
        )
        return cur.lastrowid


def add_shot(session_id, result, t=None, zone=None, x=None, y=None, form=None,
             arc=None, form_image=None, miss=None):
    form = form or {}
    with _conn() as c:
        cur = c.execute(
            """INSERT INTO shots(session_id,t,result,made,zone,x,y,
                                 elbow_angle,knee_angle,lean_deg,follow_through,
                                 form_json,arc_json,form_image,miss_json)
               VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (session_id, t, result, 1 if result == "make" else 0, zone, x, y,
             form.get("elbow_angle"), form.get("knee_angle"), form.get("lean_deg"),
             1 if form.get("follow_through") else 0, json.dumps(form),
             json.dumps(arc) if arc else None, form_image,
             json.dumps(miss) if miss else None),
        )
        return cur.lastrowid


def finalize_session(session_id):
    with _conn() as c:
        row = c.execute(
            "SELECT COUNT(*) a, COALESCE(SUM(made),0) m FROM shots WHERE session_id=?",
            (session_id,),
        ).fetchone()
        a, m = row["a"] or 0, row["m"] or 0
        fg = round(100.0 * m / a, 1) if a else 0.0
        c.execute(
            "UPDATE sessions SET ended_at=?, makes=?, attempts=?, fg_pct=? WHERE id=?",
            (_now(), m, a, fg, session_id),
        )
    return {"session_id": session_id, "makes": m, "attempts": a, "fg_pct": fg}


def _duration_s(row):
    try:
        s = datetime.datetime.fromisoformat(row["started_at"])
        e = datetime.datetime.fromisoformat(row["ended_at"])
        return int((e - s).total_seconds())
    except Exception:
        return 0


def get_session(session_id):
    with _conn() as c:
        s = c.execute("SELECT * FROM sessions WHERE id=?", (session_id,)).fetchone()
        if not s:
            return None
        shots = c.execute(
            "SELECT * FROM shots WHERE session_id=? ORDER BY id", (session_id,)
        ).fetchall()
    s = dict(s)
    s["duration_s"] = _duration_s(s)
    s["date"] = s.get("started_at")
    return {"session": s, "shots": [_attach_form(dict(x)) for x in shots]}


def list_sessions(limit=200):
    with _conn() as c:
        rows = c.execute(
            "SELECT * FROM sessions ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        d["duration_s"] = _duration_s(d)
        d["date"] = d.get("started_at")
        out.append(d)
    return out


def all_shots():
    with _conn() as c:
        rows = c.execute(
            """SELECT sh.*, se.started_at FROM shots sh
               JOIN sessions se ON se.id = sh.session_id
               ORDER BY sh.id""").fetchall()
    return [_attach_form(dict(r)) for r in rows]


def delete_session(session_id):
    with _conn() as c:
        c.execute("DELETE FROM shots WHERE session_id=?", (session_id,))
        c.execute("DELETE FROM sessions WHERE id=?", (session_id,))


def set_form_image(shot_id, filename):
    """Record the saved release-frame snapshot filename for a shot."""
    with _conn() as c:
        c.execute("UPDATE shots SET form_image=? WHERE id=?", (filename, shot_id))


def forms_gallery(limit=300):
    """Newest-first shots that have a saved form snapshot (for the form gallery)."""
    with _conn() as c:
        rows = c.execute(
            """SELECT sh.*, se.started_at, se.mode FROM shots sh
               JOIN sessions se ON se.id = sh.session_id
               WHERE sh.form_image IS NOT NULL AND sh.form_image <> ''
               ORDER BY sh.id DESC LIMIT ?""", (limit,)).fetchall()
    return [_attach_form(dict(r)) for r in rows]


# ----------------------- manual correction + capture -----------------------
def get_shot(shot_id):
    with _conn() as c:
        r = c.execute("SELECT * FROM shots WHERE id=?", (shot_id,)).fetchone()
    return _attach_form(dict(r)) if r else None


def update_shot_result(shot_id, result):
    """Flip a shot's result (make<->miss); returns its session_id (or None)."""
    made = 1 if result == "make" else 0
    with _conn() as c:
        r = c.execute("SELECT session_id FROM shots WHERE id=?", (shot_id,)).fetchone()
        if not r:
            return None
        c.execute("UPDATE shots SET result=?, made=?, manual=1 WHERE id=?", (result, made, shot_id))
        return r["session_id"]


def delete_shot(shot_id):
    """Delete a (phantom) shot; returns its session_id (or None)."""
    with _conn() as c:
        r = c.execute("SELECT session_id FROM shots WHERE id=?", (shot_id,)).fetchone()
        if not r:
            return None
        c.execute("DELETE FROM shots WHERE id=?", (shot_id,))
        return r["session_id"]


def add_manual_shot(session_id, result, t=None, zone=None):
    """Insert a shot the tracker missed; returns the new shot id."""
    with _conn() as c:
        cur = c.execute(
            """INSERT INTO shots(session_id,t,result,made,zone,manual,form_json)
               VALUES(?,?,?,?,?,1,?)""",
            (session_id, t, result, 1 if result == "make" else 0, zone, json.dumps({})))
        return cur.lastrowid


def recompute_session(session_id):
    """Recompute makes/attempts/fg_pct from shot rows (preserves timestamps)."""
    with _conn() as c:
        row = c.execute("SELECT COUNT(*) a, COALESCE(SUM(made),0) m FROM shots WHERE session_id=?",
                        (session_id,)).fetchone()
        a, m = row["a"] or 0, row["m"] or 0
        fg = round(100.0 * m / a, 1) if a else 0.0
        c.execute("UPDATE sessions SET makes=?, attempts=?, fg_pct=? WHERE id=?", (m, a, fg, session_id))
    return {"makes": m, "attempts": a, "fg_pct": fg}


def set_session_video(session_id, path):
    with _conn() as c:
        c.execute("UPDATE sessions SET video_path=? WHERE id=?", (str(path), session_id))


def set_session_annotated(session_id, path):
    with _conn() as c:
        c.execute("UPDATE sessions SET annotated_path=? WHERE id=?", (str(path), session_id))


def log_correction(session_id, shot_id, kind, old_result=None, new_result=None, t=None):
    with _conn() as c:
        c.execute(
            """INSERT INTO corrections(logged_at,session_id,shot_id,kind,old_result,new_result,t)
               VALUES(?,?,?,?,?,?,?)""",
            (_now(), session_id, shot_id, kind, old_result, new_result, t))


def corrections_count():
    with _conn() as c:
        return c.execute("SELECT COUNT(*) n FROM corrections").fetchone()["n"]


# ----------------------------- goals -----------------------------
def create_goal(metric, target, label=""):
    with _conn() as c:
        cur = c.execute("INSERT INTO goals(created_at, metric, target, label) VALUES(?,?,?,?)",
                        (_now(), metric, target, label))
        return cur.lastrowid


def list_goals():
    with _conn() as c:
        return [dict(r) for r in c.execute("SELECT * FROM goals ORDER BY id DESC").fetchall()]


def set_goal_achieved(goal_id, when=None):
    with _conn() as c:
        c.execute("UPDATE goals SET achieved_at=? WHERE id=? AND achieved_at IS NULL",
                  (when or _now(), goal_id))


def delete_goal(goal_id):
    with _conn() as c:
        c.execute("DELETE FROM goals WHERE id=?", (goal_id,))


# ----------------------------- practice log -----------------------------
def log_practice(focus_metric, drill, note="", baseline_std=None):
    with _conn() as c:
        cur = c.execute(
            "INSERT INTO practice_log(logged_at, focus_metric, drill, note, baseline_std) VALUES(?,?,?,?,?)",
            (_now(), focus_metric, drill, note, baseline_std))
        return cur.lastrowid


def list_practice(limit=100):
    with _conn() as c:
        return [dict(r) for r in
                c.execute("SELECT * FROM practice_log ORDER BY id DESC LIMIT ?", (limit,)).fetchall()]


# Ensure the schema exists whenever this module is imported (any entry point).
init_db()

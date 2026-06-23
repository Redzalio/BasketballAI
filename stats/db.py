"""SQLite persistence for HoopTracker sessions and shots."""
from pathlib import Path
import sqlite3
import datetime

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
            """
        )


def _now():
    return datetime.datetime.now().isoformat(sep=" ", timespec="seconds")


def create_session(mode, source=""):
    with _conn() as c:
        cur = c.execute(
            "INSERT INTO sessions(started_at, mode, source) VALUES(?,?,?)",
            (_now(), mode, source),
        )
        return cur.lastrowid


def add_shot(session_id, result, t=None, zone=None, x=None, y=None, form=None):
    form = form or {}
    with _conn() as c:
        c.execute(
            """INSERT INTO shots(session_id,t,result,made,zone,x,y,
                                 elbow_angle,knee_angle,lean_deg,follow_through)
               VALUES(?,?,?,?,?,?,?,?,?,?,?)""",
            (session_id, t, result, 1 if result == "make" else 0, zone, x, y,
             form.get("elbow_angle"), form.get("knee_angle"), form.get("lean_deg"),
             1 if form.get("follow_through") else 0),
        )


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
    return {"session": s, "shots": [dict(x) for x in shots]}


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
    return [dict(r) for r in rows]


def delete_session(session_id):
    with _conn() as c:
        c.execute("DELETE FROM shots WHERE session_id=?", (session_id,))
        c.execute("DELETE FROM sessions WHERE id=?", (session_id,))


# Ensure the schema exists whenever this module is imported (any entry point).
init_db()

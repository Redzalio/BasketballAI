"""Motivation / accountability layer for HoopTracker.

A "last 7 days vs previous 7 days" digest, lifetime + this-week volume totals, a
current calendar-day streak, and a couple of personal bests that are NOT already
computed by stats.insights.overview_insights (which owns best_fg_session,
longest_make_streak, most_makes_session). This module adds:
  * most makes in a single calendar DAY
  * biggest single-session volume (attempts)

It NEVER queries the DB itself -- the caller passes the same shapes the rest of
the stats layer uses:
  * sessions: list of dicts (NEWEST FIRST) from db.list_sessions(): each has
    `id`, `started_at` (ISO, e.g. "2026-06-24 18:30:05"), `date` (same), `makes`,
    `attempts`, `fg_pct`.
  * shots: list of dicts from db.all_shots(): each carries `started_at` (the
    parent session's start, ISO) and `made` (0/1) or `result` ("make"/"miss").

Defensive throughout (mirrors stats.arc / stats.insights): every public function
returns a dict and never raises on bad input -- malformed/missing dates are
skipped, empty windows yield zeros + {"enough": False} where relevant.

Python stdlib only (datetime, collections).
"""
import datetime
from collections import defaultdict

WEEK = datetime.timedelta(days=7)
DAY = datetime.timedelta(days=1)


# --------------------------------------------------------------------------- #
# parsing helpers -- never raise
# --------------------------------------------------------------------------- #
def _now(now):
    """Coerce `now` to a datetime; fall back to wall-clock on bad input."""
    if isinstance(now, datetime.datetime):
        return now
    return datetime.datetime.now()


def _parse_dt(value):
    """Parse an ISO datetime string (or pass a datetime through). Returns a
    datetime or None for anything malformed/missing -- never raises."""
    if isinstance(value, datetime.datetime):
        return value
    if isinstance(value, datetime.date):
        return datetime.datetime(value.year, value.month, value.day)
    if not isinstance(value, str):
        return None
    s = value.strip()
    if not s:
        return None
    try:
        return datetime.datetime.fromisoformat(s)
    except (ValueError, TypeError):
        # Tolerate a trailing 'Z' or stray fractional/format noise.
        try:
            return datetime.datetime.fromisoformat(s.replace("Z", "").strip())
        except (ValueError, TypeError):
            return None


def _session_dt(s):
    """Best-effort datetime for a session dict (prefers started_at, then date)."""
    if not isinstance(s, dict):
        return None
    return _parse_dt(s.get("started_at")) or _parse_dt(s.get("date"))


def _int(v):
    """Coerce to a non-negative-safe int; 0 on anything weird (incl. bool-as-0/1
    is fine since bool is an int)."""
    try:
        if isinstance(v, bool):
            return int(v)
        return int(v)
    except (TypeError, ValueError):
        return 0


def _made(shot):
    """True if a shot dict represents a make (handles `made` 0/1 or `result`)."""
    if not isinstance(shot, dict):
        return False
    m = shot.get("made")
    if m is not None:
        try:
            return bool(int(m))
        except (TypeError, ValueError):
            return bool(m)
    return shot.get("result") == "make"


def _iter_sessions(sessions):
    """Yield (datetime, session_dict) for every session with a usable date."""
    if not isinstance(sessions, (list, tuple)):
        return
    for s in sessions:
        dt = _session_dt(s)
        if dt is not None:
            yield dt, s


def _window_totals(sessions, lo, hi, hi_inclusive=True):
    """Sum makes/attempts/sessions for sessions with lo <= dt <= hi (or < hi)."""
    makes = attempts = count = 0
    for dt, s in _iter_sessions(sessions):
        if dt < lo:
            continue
        if hi_inclusive:
            if dt > hi:
                continue
        else:
            if dt >= hi:
                continue
        makes += _int(s.get("makes"))
        attempts += _int(s.get("attempts"))
        count += 1
    fg = round(100.0 * makes / attempts, 1) if attempts else 0.0
    return {"fg_pct": fg, "makes": makes, "attempts": attempts, "sessions": count}


# --------------------------------------------------------------------------- #
# 1) weekly digest -- this 7 days vs the previous 7 days
# --------------------------------------------------------------------------- #
def weekly_digest(sessions, now=None):
    """Rolling 7-day comparison.

    "this_week" = sessions in [now-7d, now]; "last_week" = [now-14d, now-7d).
    fg_pct is computed from SUMMED makes/attempts in each window (not an average
    of per-session percentages).

    Returns:
      {"enough": bool,
       "this_week": {"fg_pct","makes","attempts","sessions"},
       "last_week": {same},
       "deltas":    {"fg_pct","makes","attempts","sessions"},  # this - last
       "summary": str}

    enough is False (and summary explains it) when BOTH windows are empty.
    """
    now = _now(now)
    this_lo = now - WEEK
    prev_lo = now - 2 * WEEK
    prev_hi = now - WEEK  # exclusive upper bound for "last week"

    this_week = _window_totals(sessions, this_lo, now, hi_inclusive=True)
    last_week = _window_totals(sessions, prev_lo, prev_hi, hi_inclusive=False)

    deltas = {
        "fg_pct": round(this_week["fg_pct"] - last_week["fg_pct"], 1),
        "makes": this_week["makes"] - last_week["makes"],
        "attempts": this_week["attempts"] - last_week["attempts"],
        "sessions": this_week["sessions"] - last_week["sessions"],
    }

    enough = this_week["sessions"] > 0 or last_week["sessions"] > 0
    summary = _digest_summary(this_week, last_week, deltas, enough)

    return {
        "enough": enough,
        "this_week": this_week,
        "last_week": last_week,
        "deltas": deltas,
        "summary": summary,
    }


def _digest_summary(this_week, last_week, deltas, enough):
    """Short human sentence describing the week-over-week change."""
    if not enough:
        return "No sessions logged in the last 14 days -- get some reps in to start your weekly trend."

    # No baseline to compare against (only this week has data).
    if last_week["sessions"] == 0:
        if this_week["sessions"] == 0:
            return "No sessions this week -- log a session to keep your streak alive."
        return (f"{this_week['makes']} makes on {this_week['attempts']} attempts across "
                f"{this_week['sessions']} session(s) this week at {this_week['fg_pct']:.0f}% "
                f"-- your first week of data, no prior week to compare yet.")

    # This week is empty but last week had data.
    if this_week["sessions"] == 0:
        return (f"No sessions yet this week -- you logged {last_week['makes']} makes over "
                f"{last_week['sessions']} session(s) the previous 7 days. Time to get back out there.")

    # Both windows have data: describe the deltas.
    fg_d = deltas["fg_pct"]
    mk_d = deltas["makes"]
    if fg_d > 0.5:
        fg_part = f"Up {fg_d:.0f}%"
    elif fg_d < -0.5:
        fg_part = f"Down {abs(fg_d):.0f}%"
    else:
        fg_part = "About even on FG%"

    if mk_d > 0:
        mk_part = f"{mk_d} more makes"
    elif mk_d < 0:
        mk_part = f"{abs(mk_d)} fewer makes"
    else:
        mk_part = "the same makes"

    return f"{fg_part} and {mk_part} than the previous 7 days."


# --------------------------------------------------------------------------- #
# 2) volume -- lifetime + this-week totals, active days, current day streak
# --------------------------------------------------------------------------- #
def volume(sessions, now=None):
    """Volume totals + engagement.

    Returns:
      {"lifetime":  {"makes","attempts","sessions"},
       "this_week": {"makes","attempts","sessions"},
       "active_days": int,            # distinct calendar days with >=1 session
       "current_day_streak": int}     # consecutive days ending today/yesterday

    current_day_streak: count back from today. If a session exists today OR
    yesterday the streak starts there, then walks backward one calendar day at a
    time while each prior day has >=1 session; the first gap day ends it. No
    session today or yesterday -> streak 0.
    """
    now = _now(now)

    lifetime = {"makes": 0, "attempts": 0, "sessions": 0}
    this = {"makes": 0, "attempts": 0, "sessions": 0}
    this_lo = now - WEEK
    days = set()

    for dt, s in _iter_sessions(sessions):
        mk = _int(s.get("makes"))
        at = _int(s.get("attempts"))
        lifetime["makes"] += mk
        lifetime["attempts"] += at
        lifetime["sessions"] += 1
        days.add(dt.date())
        if this_lo <= dt <= now:
            this["makes"] += mk
            this["attempts"] += at
            this["sessions"] += 1

    return {
        "lifetime": lifetime,
        "this_week": this,
        "active_days": len(days),
        "current_day_streak": _current_day_streak(days, now.date()),
    }


def _current_day_streak(day_set, today):
    """Consecutive calendar days with a session, ending today or yesterday.

    day_set: set of datetime.date with >=1 session. Anchor on today if present,
    else yesterday; if neither, the streak is 0. Then walk backward while each
    earlier day is in the set.
    """
    if not day_set:
        return 0
    if today in day_set:
        cursor = today
    elif (today - DAY) in day_set:
        cursor = today - DAY
    else:
        return 0

    streak = 0
    while cursor in day_set:
        streak += 1
        cursor -= DAY
    return streak


# --------------------------------------------------------------------------- #
# 3) personal bests NOT computed by insights.overview_insights
# --------------------------------------------------------------------------- #
def personal_bests_extra(sessions, shots):
    """Personal bests that overview_insights does NOT already surface.

    Returns:
      {"most_makes_day": {"date": "YYYY-MM-DD", "makes": int} | None,
       "biggest_session": {"id": int, "attempts": int} | None}

    most_makes_day: group `shots` by calendar day (from each shot's started_at),
    sum the makes per day, take the day with the most makes. Ties resolve to the
    most recent day.
    biggest_session: the single session with the most ATTEMPTS (volume PB), from
    `sessions`. (Best FG%, longest make streak and most makes in a single SESSION
    are owned by insights.overview_insights and are intentionally not repeated.)
    """
    return {
        "most_makes_day": _most_makes_day(shots),
        "biggest_session": _biggest_session(sessions),
    }


def _most_makes_day(shots):
    by_day = defaultdict(int)
    if isinstance(shots, (list, tuple)):
        for sh in shots:
            if not isinstance(sh, dict):
                continue
            dt = _parse_dt(sh.get("started_at"))
            if dt is None:
                continue
            if _made(sh):
                by_day[dt.date()] += 1
    if not by_day:
        return None
    # Max makes; tie -> most recent day. Sort key picks (makes, date) descending.
    best_day = max(by_day.items(), key=lambda kv: (kv[1], kv[0]))
    day, makes = best_day
    if makes <= 0:
        return None
    return {"date": day.isoformat(), "makes": int(makes)}


def _biggest_session(sessions):
    best = None
    best_attempts = 0
    if isinstance(sessions, (list, tuple)):
        for s in sessions:
            if not isinstance(s, dict):
                continue
            at = _int(s.get("attempts"))
            if at > best_attempts:
                best_attempts = at
                best = s
    if best is None or best_attempts <= 0:
        return None
    return {"id": _int(best.get("id")), "attempts": best_attempts}


# --------------------------------------------------------------------------- #
# 4) bundle
# --------------------------------------------------------------------------- #
def progress_report(sessions, shots, now=None):
    """Bundle the three sections for a single /api/progress payload.

    Returns:
      {"weekly": weekly_digest(sessions, now),
       "volume": volume(sessions, now),
       "pbs":    personal_bests_extra(sessions, shots)}
    """
    return {
        "weekly": weekly_digest(sessions, now=now),
        "volume": volume(sessions, now=now),
        "pbs": personal_bests_extra(sessions, shots),
    }

"""Tests for the motivation / accountability layer: stats.progress.

The module compares the last 7 days to the previous 7 days, totals lifetime +
this-week volume, computes a current calendar-day streak, and surfaces two
personal bests that stats.insights does NOT already own (most makes in a single
DAY, biggest single-session attempt volume).

It never queries the DB -- the caller passes the same shapes the stats layer
uses:
  * sessions (NEWEST FIRST): {id, started_at (ISO "YYYY-MM-DD HH:MM:SS"), date,
    makes, attempts, fg_pct}
  * shots: {started_at (parent session's start, ISO), made 0/1 OR result}

Everything is deterministic: a FIXED `now` is passed in, and timestamps are
built relative to it, so window membership and the day streak are stable
run-to-run.

Locked-in behavior:
  * weekly_digest: this-week vs last-week sums (fg from summed makes/attempts)
    and the deltas between them; boundary at exactly now-7d belongs to THIS week.
  * volume: lifetime + this-week totals, distinct active days, and a
    current_day_streak that counts consecutive days and BREAKS on a gap.
  * personal_bests_extra: most_makes_day picks the right calendar day; biggest
    session is the most-attempts one.
  * empty / garbage input -> enough False, zeros, never raises.
"""
import datetime

from stats import progress

# A fixed "now" so every window/streak assertion is deterministic.
NOW = datetime.datetime(2026, 6, 24, 12, 0, 0)
FMT = "%Y-%m-%d %H:%M:%S"  # matches db's started_at: "2026-06-24 18:30:05"


# --------------------------------------------------------------------------- #
# builders
# --------------------------------------------------------------------------- #
def _iso(dt):
    return dt.isoformat(sep=" ", timespec="seconds")


def _session(sid, dt, makes, attempts):
    """A session dict shaped like db.list_sessions() output."""
    iso = _iso(dt)
    fg = round(100.0 * makes / attempts, 1) if attempts else 0.0
    return {"id": sid, "started_at": iso, "date": iso,
            "makes": makes, "attempts": attempts, "fg_pct": fg}


def _shots_for_day(dt, makes, misses):
    """A bundle of shot dicts (db.all_shots() shape) parented to one day."""
    iso = _iso(dt)
    out = [{"started_at": iso, "made": 1, "result": "make"} for _ in range(makes)]
    out += [{"started_at": iso, "made": 0, "result": "miss"} for _ in range(misses)]
    return out


def _days_ago(n, hour=10):
    # hour defaults to 10:00 so a "today" (n=0) timestamp is at/before NOW (12:00).
    return NOW.replace(hour=hour, minute=0, second=0, microsecond=0) - datetime.timedelta(days=n)


# --------------------------------------------------------------------------- #
# weekly_digest: this week vs last week
# --------------------------------------------------------------------------- #
def _two_week_sessions():
    """Sessions in both the this-week and last-week windows (newest first).

    this week (days 0..6):  makes 30 / attempts 50  over 2 sessions
    last week (days 7..13): makes 22 / attempts 50  over 2 sessions
    """
    return [
        _session(5, _days_ago(1), makes=18, attempts=30),   # this week
        _session(4, _days_ago(6), makes=12, attempts=20),   # this week (inside 7d)
        _session(3, _days_ago(8), makes=10, attempts=25),   # last week
        _session(2, _days_ago(13), makes=12, attempts=25),  # last week (inside 14d)
        _session(1, _days_ago(40), makes=99, attempts=99),  # older: ignored by digest
    ]


def test_weekly_digest_sums_each_window():
    res = progress.weekly_digest(_two_week_sessions(), now=NOW)
    assert res["enough"] is True

    tw = res["this_week"]
    assert tw["makes"] == 30
    assert tw["attempts"] == 50
    assert tw["sessions"] == 2
    # fg from SUMMED makes/attempts: 30/50 = 60.0
    assert tw["fg_pct"] == 60.0

    lw = res["last_week"]
    assert lw["makes"] == 22
    assert lw["attempts"] == 50
    assert lw["sessions"] == 2
    assert lw["fg_pct"] == 44.0


def test_weekly_digest_deltas_are_this_minus_last():
    res = progress.weekly_digest(_two_week_sessions(), now=NOW)
    d = res["deltas"]
    assert d["makes"] == 8          # 30 - 22
    assert d["attempts"] == 0       # 50 - 50
    assert d["sessions"] == 0       # 2 - 2
    assert d["fg_pct"] == 16.0      # 60.0 - 44.0
    # summary reflects the improvement.
    assert "more makes" in res["summary"].lower()


def test_weekly_digest_boundary_now_minus_7d_is_this_week():
    # A session EXACTLY 7 days before now is the inclusive lower edge of "this
    # week" -> counted in this_week, not last_week.
    s = _session(1, NOW - datetime.timedelta(days=7), makes=5, attempts=10)
    res = progress.weekly_digest([s], now=NOW)
    assert res["this_week"]["sessions"] == 1
    assert res["last_week"]["sessions"] == 0


def test_weekly_digest_only_this_week_has_no_baseline():
    s = _session(1, _days_ago(2), makes=9, attempts=15)
    res = progress.weekly_digest([s], now=NOW)
    assert res["enough"] is True
    assert res["this_week"]["sessions"] == 1
    assert res["last_week"]["sessions"] == 0
    # delta == this week's own numbers when there is no prior week.
    assert res["deltas"]["makes"] == 9


def test_weekly_digest_empty_not_enough_and_zeros():
    res = progress.weekly_digest([], now=NOW)
    assert res["enough"] is False
    for win in ("this_week", "last_week"):
        assert res[win] == {"fg_pct": 0.0, "makes": 0, "attempts": 0, "sessions": 0}
    assert res["deltas"] == {"fg_pct": 0.0, "makes": 0, "attempts": 0, "sessions": 0}
    assert isinstance(res["summary"], str) and res["summary"]


# --------------------------------------------------------------------------- #
# volume: totals, active days, current day streak
# --------------------------------------------------------------------------- #
def test_volume_lifetime_and_this_week_totals():
    sessions = [
        _session(3, _days_ago(0), makes=10, attempts=20),   # this week
        _session(2, _days_ago(3), makes=8, attempts=16),    # this week
        _session(1, _days_ago(30), makes=5, attempts=25),   # older
    ]
    res = progress.volume(sessions, now=NOW)
    assert res["lifetime"] == {"makes": 23, "attempts": 61, "sessions": 3}
    assert res["this_week"] == {"makes": 18, "attempts": 36, "sessions": 2}
    assert res["active_days"] == 3  # three distinct calendar days


def test_volume_active_days_collapses_same_day_sessions():
    # Two sessions on the SAME calendar day count as one active day.
    d = _days_ago(2)
    sessions = [
        _session(2, d.replace(hour=9), makes=4, attempts=10),
        _session(1, d.replace(hour=19), makes=6, attempts=10),
    ]
    res = progress.volume(sessions, now=NOW)
    assert res["active_days"] == 1
    assert res["lifetime"]["sessions"] == 2


def test_current_day_streak_counts_consecutive_days_from_today():
    # Sessions today, yesterday, and the day before -> streak of 3.
    sessions = [
        _session(3, _days_ago(0), 5, 10),
        _session(2, _days_ago(1), 5, 10),
        _session(1, _days_ago(2), 5, 10),
    ]
    res = progress.volume(sessions, now=NOW)
    assert res["current_day_streak"] == 3


def test_current_day_streak_anchors_on_yesterday_when_no_session_today():
    # Nothing today, but yesterday + the two days before -> streak of 3.
    sessions = [
        _session(3, _days_ago(1), 5, 10),
        _session(2, _days_ago(2), 5, 10),
        _session(1, _days_ago(3), 5, 10),
    ]
    res = progress.volume(sessions, now=NOW)
    assert res["current_day_streak"] == 3


def test_current_day_streak_breaks_on_a_gap():
    # today + yesterday, then a GAP at day 2, then more sessions before it.
    sessions = [
        _session(4, _days_ago(0), 5, 10),
        _session(3, _days_ago(1), 5, 10),
        # day 2 missing -> streak stops at 2
        _session(2, _days_ago(3), 5, 10),
        _session(1, _days_ago(4), 5, 10),
    ]
    res = progress.volume(sessions, now=NOW)
    assert res["current_day_streak"] == 2


def test_current_day_streak_zero_when_stale():
    # Most recent session was 3 days ago (not today/yesterday) -> streak 0.
    sessions = [_session(1, _days_ago(3), 5, 10)]
    res = progress.volume(sessions, now=NOW)
    assert res["current_day_streak"] == 0


def test_volume_empty_is_zeros_no_raise():
    res = progress.volume([], now=NOW)
    assert res["lifetime"] == {"makes": 0, "attempts": 0, "sessions": 0}
    assert res["this_week"] == {"makes": 0, "attempts": 0, "sessions": 0}
    assert res["active_days"] == 0
    assert res["current_day_streak"] == 0


# --------------------------------------------------------------------------- #
# personal_bests_extra
# --------------------------------------------------------------------------- #
def test_most_makes_day_picks_the_right_day():
    shots = []
    # Day A (4 days ago): 7 makes, 3 misses
    shots += _shots_for_day(_days_ago(4), makes=7, misses=3)
    # Day B (2 days ago): 12 makes, 4 misses  <- the winner
    shots += _shots_for_day(_days_ago(2), makes=12, misses=4)
    # Day C (today): 5 makes
    shots += _shots_for_day(_days_ago(0), makes=5, misses=5)

    res = progress.personal_bests_extra([], shots)
    pb = res["most_makes_day"]
    assert pb is not None
    assert pb["makes"] == 12
    assert pb["date"] == _days_ago(2).date().isoformat()


def test_most_makes_day_groups_shots_across_sessions_same_day():
    # Two separate sessions on the SAME day should have their makes summed.
    d = _days_ago(3)
    shots = (_shots_for_day(d.replace(hour=9), makes=6, misses=0)
             + _shots_for_day(d.replace(hour=20), makes=5, misses=2))
    res = progress.personal_bests_extra([], shots)
    assert res["most_makes_day"]["makes"] == 11
    assert res["most_makes_day"]["date"] == d.date().isoformat()


def test_biggest_session_is_most_attempts():
    sessions = [
        _session(3, _days_ago(1), makes=20, attempts=40),
        _session(2, _days_ago(5), makes=30, attempts=90),   # most attempts
        _session(1, _days_ago(9), makes=10, attempts=15),
    ]
    res = progress.personal_bests_extra(sessions, [])
    big = res["biggest_session"]
    assert big == {"id": 2, "attempts": 90}


def test_pbs_use_result_field_when_made_absent():
    iso = _iso(_days_ago(1))
    shots = [{"started_at": iso, "result": "make"} for _ in range(4)]
    shots += [{"started_at": iso, "result": "miss"} for _ in range(2)]
    res = progress.personal_bests_extra([], shots)
    assert res["most_makes_day"]["makes"] == 4


def test_pbs_empty_inputs_are_none():
    res = progress.personal_bests_extra([], [])
    assert res["most_makes_day"] is None
    assert res["biggest_session"] is None


# --------------------------------------------------------------------------- #
# malformed input never raises
# --------------------------------------------------------------------------- #
def test_garbage_never_raises():
    garbage_sessions = [
        None, 5, "nope", {},
        {"started_at": "not-a-date", "makes": 3, "attempts": 5},
        {"started_at": None, "makes": 1, "attempts": 2},
        {"started_at": _iso(_days_ago(1)), "makes": "x", "attempts": None},  # bad ints -> 0
    ]
    garbage_shots = [None, 7, "x", {}, {"started_at": "bad", "made": 1},
                     {"made": 1}]  # missing started_at -> skipped

    # None of these should raise.
    w = progress.weekly_digest(garbage_sessions, now=NOW)
    v = progress.volume(garbage_sessions, now=NOW)
    p = progress.personal_bests_extra(garbage_sessions, garbage_shots)
    r = progress.progress_report(garbage_sessions, garbage_shots, now=NOW)

    assert isinstance(w, dict) and isinstance(v, dict)
    assert isinstance(p, dict) and isinstance(r, dict)
    # The one session with a valid date but bad ints contributes a counted
    # session with zero makes/attempts -> no crash, attempts stay 0.
    assert v["this_week"]["sessions"] == 1
    assert v["this_week"]["attempts"] == 0


def test_non_list_inputs_never_raise():
    for bad in (None, "nope", 5, {"a": 1}):
        assert isinstance(progress.weekly_digest(bad, now=NOW), dict)
        assert isinstance(progress.volume(bad, now=NOW), dict)
        assert isinstance(progress.personal_bests_extra(bad, bad), dict)
        assert isinstance(progress.progress_report(bad, bad, now=NOW), dict)


def test_now_defaults_to_wallclock_without_raising():
    # Omitting `now` must use datetime.now() internally and still return a dict.
    res = progress.weekly_digest([_session(1, datetime.datetime.now(), 5, 10)])
    assert isinstance(res, dict) and "this_week" in res


# --------------------------------------------------------------------------- #
# progress_report bundle
# --------------------------------------------------------------------------- #
def test_progress_report_bundles_all_three():
    sessions = _two_week_sessions()
    shots = _shots_for_day(_days_ago(1), makes=8, misses=2)
    res = progress.progress_report(sessions, shots, now=NOW)
    assert set(res.keys()) == {"weekly", "volume", "pbs"}
    assert res["weekly"]["this_week"]["makes"] == 30
    assert res["volume"]["lifetime"]["sessions"] == 5
    assert res["pbs"]["most_makes_day"]["makes"] == 8


# pytest is imported lazily so the module can also run as a plain script.
import pytest  # noqa: E402,F401

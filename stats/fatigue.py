"""Fatigue flag — pinpoint where shooting-form consistency starts to slide.

Consumes ONE session's shots in order and slides a fixed window across them,
scoring each window with the same engine the rest of the app uses
(consistency.metric_stats -> consistency.consistency_score). A baseline is taken
from the EARLY windows; the first window that drops >= DROP points below it AND
stays down on the next window is flagged as the fatigue onset.

Why rolling (not just first-half vs second-half like consistency.in_session_drift):
the half-split tells you *whether* form fell off; the rolling pass tells you the
*shot number* where it began — the actionable bit ("rest/stop around shot K").

Consistency = shot-to-shot variance, so a fixed single-camera angle's systematic
error cancels out and only the *change* in spread across the session is read.
"""
from stats import consistency

# Points a window's consistency score must fall below the baseline to count as a
# real drop (not just shot-to-shot noise). One window of confirmation is required.
DROP = 12


def _score_window(window_shots):
    """Rolling consistency 0..100 for one window, or None if the window can't
    yield any metric (needs >=3 values of a metric — see consistency.metric_stats)."""
    mstats = consistency.metric_stats(window_shots)
    if not mstats:
        return None
    return consistency.consistency_score(mstats)


def fatigue_analysis(shots, window=8):
    """Detect the onset shot where form consistency starts to drop off.

    shots:  chronological list of shot dicts for ONE session.
    window: number of consecutive shots per rolling window.

    Returns a dict (never raises). On insufficient data: {"enough": False, "n": n}.
    Otherwise:
      {"enough": True, "n", "window",
       "baseline", "late", "drop",            # late = final window's score; drop = baseline - late
       "onset_shot": int|None,                # 1-based shot # where drift begins
       "rolling": [{"shot": <1-based end index>, "score": <0..100>}, ...],
       "verdict": "holds" | "drifts",
       "advice": str}
    """
    shots = list(shots) if shots else []
    n = len(shots)

    # Need a sane window and enough shots to form >= 2 windows.
    if not isinstance(window, int) or window < 3:
        window = 8
    if n < 10 or n < window + 1:
        return {"enough": False, "n": n}

    # Rolling score per window, keyed by the window's LAST shot (1-based).
    rolling = []
    for end in range(window, n + 1):
        score = _score_window(shots[end - window:end])
        if score is not None:
            rolling.append({"shot": end, "score": score})

    # Need at least two scored windows to establish baseline + a comparison.
    if len(rolling) < 2:
        return {"enough": False, "n": n}

    # Baseline = best of the first couple of windows (forgiving early jitter so a
    # single rough opening window doesn't understate how good fresh form was).
    baseline = max(r["score"] for r in rolling[:2])
    late = rolling[-1]["score"]

    # Walk forward: first window that drops >= DROP below baseline AND the dip
    # persists into the next window (so a one-window blip doesn't trip the flag).
    onset_shot = None
    for i in range(len(rolling) - 1):
        cur, nxt = rolling[i], rolling[i + 1]
        if cur["score"] <= baseline - DROP and nxt["score"] <= baseline - DROP:
            onset_shot = cur["shot"]
            break

    drift = onset_shot is not None
    drop = baseline - late

    if drift:
        lost = max(drop, DROP)
        advice = (f"Consistency dropped ~{lost} pts after shot {onset_shot} — "
                  f"that's a good point to rest or end the session.")
    else:
        advice = "Form held all session - good conditioning."

    return {
        "enough": True,
        "n": n,
        "window": window,
        "baseline": baseline,
        "late": late,
        "drop": drop,
        "onset_shot": onset_shot,
        "rolling": rolling,
        "verdict": "drifts" if drift else "holds",
        "advice": advice,
    }


if __name__ == "__main__":
    import sys
    from pathlib import Path as _P
    sys.path.insert(0, str(_P(__file__).resolve().parent.parent))
    from stats import db
    import json as _json
    sess = db.list_sessions(1)
    if not sess:
        print("no sessions in DB"); sys.exit()
    obj = db.get_session(sess[0]["id"])
    print(f"session {sess[0]['id']}: {len(obj['shots'])} shots")
    print(_json.dumps(fatigue_analysis(obj["shots"]), indent=2, default=str))

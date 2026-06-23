"""Tests for the fatigue flag: stats.fatigue.

fatigue_analysis(shots, window=8) slides a window across ONE session, scores each
window with the consistency engine, takes a baseline from the early windows, and
flags the first window that drops >= DROP points below baseline AND stays down on
the next window. That window's last shot (1-based) is the onset.

Locked-in behavior:
  * STABLE session (form jittered only slightly) -> verdict "holds", onset None.
  * DRIFTING session (tight first half, noisy second half) -> verdict "drifts",
    onset is set and lands in the later half of the session.
  * < 10 shots -> {"enough": False}.

Values are fully deterministic (small fixed cycles, no randomness) so the rolling
scores — and therefore the verdict and onset — are stable run-to-run. Every shot
carries the SAME metric keys, so each window has >=3 values per metric and
consistency.metric_stats always returns a score.
"""
from stats import fatigue

# The seven form metrics consistency.METRIC_META knows about and that the pose
# layer emits. Each shot includes all of them so every window is fully scored.
FORM_KEYS = ("elbow_angle", "release_angle", "knee_bend", "lean_deg",
             "symmetry_deg", "release_height_ratio", "follow_through_deg")

# Centre (well within each metric's "good"/PRO range) and a TINY deterministic
# wobble cycle per metric — std stays far below tol so a stable window scores high.
CENTER = {
    "elbow_angle": 172.0,
    "release_angle": 52.0,
    "knee_bend": 120.0,
    "lean_deg": 3.0,
    "symmetry_deg": 3.0,
    "release_height_ratio": 1.30,
    "follow_through_deg": 170.0,
}
TIGHT_WOBBLE = {
    "elbow_angle": (-1.0, 0.0, 1.0, 0.0),
    "release_angle": (-1.0, 0.0, 1.0, 0.0),
    "knee_bend": (-1.0, 0.0, 1.0, 0.0),
    "lean_deg": (-1.0, 0.0, 1.0, 0.0),
    "symmetry_deg": (-1.0, 0.0, 1.0, 0.0),
    "release_height_ratio": (-0.03, 0.0, 0.03, 0.0),
    "follow_through_deg": (-1.0, 0.0, 1.0, 0.0),
}
# Big deterministic swings for the fatigued back half — well beyond each tol, so
# those windows score low and the drop clears DROP.
LOOSE_WOBBLE = {
    "elbow_angle": (-26.0, 24.0, -22.0, 28.0),
    "release_angle": (-20.0, 22.0, -18.0, 20.0),
    "knee_bend": (-34.0, 30.0, -28.0, 32.0),
    "lean_deg": (-1.0, 0.0, 1.0, 0.0),
    "symmetry_deg": (-1.0, 0.0, 1.0, 0.0),
    "release_height_ratio": (-0.03, 0.0, 0.03, 0.0),
    "follow_through_deg": (-1.0, 0.0, 1.0, 0.0),
}


def _form(i, wobble):
    """Deterministic form dict for shot index i using a per-metric wobble cycle."""
    return {k: round(CENTER[k] + wobble[k][i % 4], 3) for k in FORM_KEYS}


def _shot(i, wobble, made):
    return {
        "result": "make" if made else "miss",
        "made": 1 if made else 0,
        "form": _form(i, wobble),
    }


def _stable_session(n=16):
    """n shots, every metric tightly clustered the whole way through."""
    return [_shot(i, TIGHT_WOBBLE, made=(i % 2 == 0)) for i in range(n)]


def _drifting_session(n=16):
    """First half tight, second half with large jitter in elbow/release/knee."""
    half = n // 2
    out = []
    for i in range(n):
        wobble = TIGHT_WOBBLE if i < half else LOOSE_WOBBLE
        out.append(_shot(i, wobble, made=(i % 2 == 0)))
    return out


# --------------------------------------------------------------------------- #
# stable session
# --------------------------------------------------------------------------- #
def test_stable_session_holds():
    out = fatigue.fatigue_analysis(_stable_session(16))
    assert out["enough"] is True
    assert out["n"] == 16
    assert out["window"] == 8
    assert out["verdict"] == "holds"
    assert out["onset_shot"] is None
    # late form is essentially as consistent as the baseline (no real drop).
    assert out["drop"] <= fatigue.DROP
    assert "held" in out["advice"].lower()


def test_stable_session_rolling_scores_high_and_complete():
    out = fatigue.fatigue_analysis(_stable_session(16))
    # one window per end-index from `window`..n  ->  16 - 8 + 1 = 9 windows
    assert len(out["rolling"]) == 9
    assert out["rolling"][0]["shot"] == 8
    assert out["rolling"][-1]["shot"] == 16
    assert all(0 <= r["score"] <= 100 for r in out["rolling"])
    assert out["baseline"] >= 80  # tight clusters -> high consistency


# --------------------------------------------------------------------------- #
# drifting session
# --------------------------------------------------------------------------- #
def test_drifting_session_drifts():
    out = fatigue.fatigue_analysis(_drifting_session(16))
    assert out["enough"] is True
    assert out["verdict"] == "drifts"
    assert out["onset_shot"] is not None


def test_drifting_onset_lands_in_later_half():
    n = 16
    out = fatigue.fatigue_analysis(_drifting_session(n))
    # onset is a 1-based shot number in the back half of the session.
    assert out["onset_shot"] > n // 2
    assert out["onset_shot"] <= n
    # baseline (fresh form) clearly higher than the final, fatigued window.
    assert out["baseline"] - out["late"] >= fatigue.DROP
    assert "rest" in out["advice"].lower() or "end" in out["advice"].lower()


def test_drifting_advice_names_the_onset_shot():
    out = fatigue.fatigue_analysis(_drifting_session(16))
    assert str(out["onset_shot"]) in out["advice"]


# --------------------------------------------------------------------------- #
# insufficient data
# --------------------------------------------------------------------------- #
def test_too_few_shots_not_enough():
    out = fatigue.fatigue_analysis(_stable_session(9))
    assert out["enough"] is False
    assert out["n"] == 9


def test_empty_session_not_enough():
    out = fatigue.fatigue_analysis([])
    assert out["enough"] is False
    assert out["n"] == 0


def test_fewer_than_two_windows_not_enough():
    # exactly `window` shots -> only one window can be formed -> not enough.
    out = fatigue.fatigue_analysis(_stable_session(11), window=11)
    assert out["enough"] is False

"""Tests for stats.miss_diagnosis -- synthetic ball tracks (deterministic).

Tracks are built in IMAGE coords (y grows DOWN). A shot is an up-then-down
parabola in y; we place the ball's x at the rim plane to the left/right of the
rim to force a side miss, and vary knee_bend across makes/misses to force a cause.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from stats import miss_diagnosis as md  # noqa: E402


# Rim sits mid-frame; a 1280px-wide frame -> rim-width proxy rw = 160px.
FRAME = (720, 1280, 3)
RIM = (640.0, 300.0)  # rim center (x, y)
RW = FRAME[1] / 8.0   # 160px


def _arc_track(end_x, apex_y=120.0, end_y=360.0, n=9):
    """A descending-into-frame ball arc that ENDS at (end_x, end_y).

    Rises from a start point to an apex (min y), then falls to (end_x, end_y)
    which is below the rim plane (rim_y=300). x drifts linearly to end_x so the
    sample nearest the rim plane carries end_x. Frame indices are contiguous.
    """
    pts = []
    start_x = end_x - 40.0
    start_y = end_y - 20.0
    half = n // 2
    for i in range(n):
        f = i
        if i <= half:  # rising: y goes start_y -> apex_y
            t = i / half if half else 0.0
            y = start_y + (apex_y - start_y) * t
        else:          # falling: y goes apex_y -> end_y
            t = (i - half) / (n - 1 - half)
            y = apex_y + (end_y - apex_y) * t
        x = start_x + (end_x - start_x) * (i / (n - 1))
        pts.append((f, float(x), float(y)))
    return pts


def test_make_returns_dir_none():
    res = md.classify_miss(_arc_track(RIM[0]), RIM, FRAME, result="make")
    assert res["ok"] is True
    assert res["dir"] is None


def test_clear_left_miss():
    # Ball ends well LEFT of the rim (x = rim_x - 1.0*rw).
    track = _arc_track(RIM[0] - RW * 1.0)
    res = md.classify_miss(track, RIM, FRAME, result="miss")
    assert res["ok"] is True
    assert res["dir"] == "left"
    assert res["dx_norm"] is not None and res["dx_norm"] < 0
    assert res["confidence"] >= 0.6  # reliable axis -> high band


def test_clear_right_miss():
    track = _arc_track(RIM[0] + RW * 1.0)
    res = md.classify_miss(track, RIM, FRAME, result="miss")
    assert res["ok"] is True
    assert res["dir"] == "right"
    assert res["dx_norm"] is not None and res["dx_norm"] > 0
    assert res["confidence"] >= 0.6


def test_side_miss_outranks_depth_confidence():
    """Left/right confidence must exceed any short/long confidence (the honesty
    contract: depth is lower-confidence than the side axis)."""
    left = md.classify_miss(_arc_track(RIM[0] - RW * 1.2), RIM, FRAME, result="miss")
    # A short miss: ball never reaches the rim plane (stays above rim_y) and on-line.
    short_track = [(0, 640.0, 200.0), (1, 642.0, 170.0), (2, 644.0, 150.0),
                   (3, 645.0, 165.0), (4, 646.0, 190.0)]  # apex up high, never near y=300
    short = md.classify_miss(short_track, RIM, FRAME, result="miss")
    assert left["confidence"] > short["confidence"]


def test_short_miss_low_confidence():
    # On-line but the ball never gets down to the rim plane -> short, low conf.
    track = [(0, 640.0, 210.0), (1, 641.0, 175.0), (2, 643.0, 150.0),
             (3, 644.0, 170.0), (4, 645.0, 205.0)]
    res = md.classify_miss(track, RIM, FRAME, result="miss")
    assert res["ok"] is True
    if res["dir"] is not None:
        assert res["dir"] in ("short", "long", "in_out")
    # Whatever depth read we got, it must be low confidence.
    assert res["confidence"] <= 0.5


def test_no_rim_center_returns_not_ok_no_raise():
    res = md.classify_miss(_arc_track(500.0), None, FRAME, result="miss")
    assert res["ok"] is False
    assert res["dir"] is None


def test_empty_points_not_ok_no_raise():
    res = md.classify_miss([], RIM, FRAME, result="miss")
    assert res["ok"] is False


def test_garbage_input_never_raises():
    # None, wrong types, ragged tuples, NaNs, strings -- none may raise.
    for bad in (None, 42, "nope", [None, 1, (1, 2)], [(1, "a", "b")],
                [(float("nan"), 1.0, 2.0)], [(0, 1.0)], {}):
        res = md.classify_miss(bad, RIM, FRAME, result="miss")
        assert res["ok"] in (True, False)
        assert isinstance(res, dict)
    # Bad rim_center shapes too.
    for bad_rim in (("x", "y"), (1.0,), object()):
        res = md.classify_miss(_arc_track(500.0), bad_rim, FRAME, result="miss")
        assert isinstance(res, dict) and res["ok"] is False


def test_no_frame_shape_still_classifies():
    # rim-width proxy must fall back to the track's own spread; left stays left.
    track = _arc_track(RIM[0] - 200.0)
    res = md.classify_miss(track, RIM, frame_shape=None, result="miss")
    assert res["ok"] is True
    assert res["dir"] == "left"


# --- miss_breakdown ---------------------------------------------------------- #
def _miss_shot(direction):
    return {"result": "miss", "miss": {"ok": True, "dir": direction,
                                       "dx_norm": -0.9, "confidence": 0.7, "note": ""}}


def test_miss_breakdown_small_list():
    shots = ([_miss_shot("left")] * 4 + [_miss_shot("right")] * 2 +
             [_miss_shot("short")] * 1 +
             [{"result": "make", "miss": {"ok": True, "dir": None}}] * 3)
    bd = md.miss_breakdown(shots)
    assert bd["enough"] is True          # 7 classified misses >= 4
    assert bd["n_misses"] == 7
    assert bd["dist"]["left"] == 4
    assert bd["dist"]["right"] == 2
    assert bd["dist"]["short"] == 1
    assert bd["dominant"] == "left"
    assert 0 <= bd["pct"]["left"] <= 100
    assert sum(bd["dist"].values()) == bd["n_misses"]


def test_miss_breakdown_not_enough():
    bd = md.miss_breakdown([_miss_shot("left")] * 2)
    assert bd["enough"] is False
    assert bd["n_misses"] == 2


def test_miss_breakdown_garbage_no_raise():
    for bad in (None, 5, "x", [None, 1, {}], [{"miss": "bad"}]):
        bd = md.miss_breakdown(bad)
        assert isinstance(bd, dict) and bd["enough"] is False


# --- miss_cause -------------------------------------------------------------- #
def test_miss_cause_picks_knee_bend():
    """Left misses systematically have LOWER knee_bend than makes; the cause
    metric should come back as knee_bend with a fix string."""
    makes, misses = [], []
    for i in range(6):
        # Makes: high knee bend (~125), other metrics steady.
        makes.append({
            "result": "make",
            "form": {"knee_bend": 125.0 + (i % 2), "elbow_angle": 170.0,
                     "release_angle": 52.0, "lean_deg": 2.0},
        })
        # Left misses: low knee bend (~95), SAME other metrics (no separation).
        misses.append({
            "result": "miss",
            "miss": {"ok": True, "dir": "left", "dx_norm": -0.9,
                     "confidence": 0.7, "note": ""},
            "form": {"knee_bend": 95.0 + (i % 2), "elbow_angle": 170.0,
                     "release_angle": 52.0, "lean_deg": 2.0},
        })
    res = md.miss_cause(makes + misses)
    assert res["enough"] is True
    assert res["findings"], "expected at least one finding"
    assert res["findings"][0]["metric"] == "knee_bend"
    assert res["findings"][0]["dir"] == "left"
    assert res["findings"][0]["mean_make"] > res["findings"][0]["mean_miss"]
    # Fix is the knee-bend cue (FIX lookup or the per-metric fallback, which
    # talks about the "leg dip"); either way it must reference the legs.
    fix = res["findings"][0]["fix"].lower()
    assert "knee" in fix or "leg" in fix
    assert "knee" in res["headline"].lower()


def test_miss_cause_not_enough():
    # Too few of everything -> enough False, no raise.
    res = md.miss_cause([_miss_shot("left")] * 2)
    assert res["enough"] is False
    assert res["findings"] == []


def test_miss_cause_garbage_no_raise():
    for bad in (None, 7, "x", [], [None, {"form": "bad"}]):
        res = md.miss_cause(bad)
        assert isinstance(res, dict) and res["enough"] is False


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))

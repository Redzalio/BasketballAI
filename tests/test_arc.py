"""Tests for the ball-flight arc analytics: stats.arc.

A shot's arc is a list of (frame_idx, x, y) ball centers in IMAGE pixels, where
y grows DOWNWARD. A real make rises (y DECREASES) then falls (y INCREASES) --
an upward-opening parabola (a>0) whose vertex is the highest point (min image-y).

Behaviors locked in here:
  * analyze_arc fits y = a f^2 + b f + c, finds the apex, and measures an
    image-space entry angle on the descending branch.
  * It survives GAPS (the ball leaving frame -> missing/zeroed apex region):
    the apex is flagged estimated / off_frame_top, but an angle still comes out.
  * It survives a single OUTLIER (rebound / second ball) via residual rejection.
  * <3 points -> ok False.
  * arc_consistency scores shot-to-shot repeatability: tight arcs -> high
    overall, scattered arcs -> low, <3 usable arcs -> enough False.
"""
import math

import numpy as np

from stats import arc


# --------------------------------------------------------------------------- #
# helpers: synthesize an image-space arc (y down then up) with horizontal drift
# --------------------------------------------------------------------------- #
def _make_arc(n=21, f0=0, x0=100.0, dx=12.0, apex_y=40.0, drop=300.0):
    """Build a clean upward-opening (in image-y) parabola.

    The vertex sits at the middle frame with y=apex_y (small = high on screen);
    both ends reach apex_y+drop (low on screen). x drifts linearly so dx/df != 0
    and the entry angle is well defined.
    """
    pts = []
    mid = (n - 1) / 2.0
    # a chosen so that at the ends (|f-mid| = mid) y rises by `drop`.
    a = drop / (mid ** 2)
    for i in range(n):
        f = f0 + i
        y = apex_y + a * (i - mid) ** 2
        x = x0 + dx * i
        pts.append((f, round(x, 2), round(y, 2)))
    return pts


# --------------------------------------------------------------------------- #
# clean make-style arc
# --------------------------------------------------------------------------- #
def test_clean_arc_fits_and_finds_apex():
    pts = _make_arc()
    res = arc.analyze_arc(pts, rim_center=(340.0, 150.0), frame_shape=(720, 1280))
    assert res["ok"] is True
    assert res["apex_y"] is not None
    # apex is the highest point => smallest image-y, near our planted apex_y=40.
    assert res["apex_y"] == pytest.approx(40.0, abs=8.0)
    # entry angle is a real image-space descent angle, strictly inside (0, 90).
    assert res["entry_angle_deg"] is not None
    assert 0.0 < res["entry_angle_deg"] < 90.0
    # rim at y=150, apex at ~40 => apex sits ~110px above the rim.
    assert res["peak_height_px"] == pytest.approx(110.0, abs=10.0)
    assert res["peak_height_ratio"] is not None
    assert res["quality"] > 0.5


def test_clean_arc_apex_is_observed_not_estimated():
    pts = _make_arc()
    res = arc.analyze_arc(pts, rim_center=(340.0, 150.0), frame_shape=(720, 1280))
    # the apex frame is mid-flight and inside the tracked range -> observed.
    assert res["estimated"] is False
    assert res["off_frame_top"] is False
    assert res["gap_frames"] == 1  # contiguous frames


# --------------------------------------------------------------------------- #
# gapped arc: the ball leaves frame around the apex
# --------------------------------------------------------------------------- #
def test_gapped_arc_estimates_apex_but_still_gives_angle():
    pts = _make_arc(n=21)
    # Drop the middle third (the apex region) -> a wide frame-index gap and no
    # directly-observed peak, exactly like the ball going over the backboard.
    kept = [p for p in pts if not (6 <= p[0] <= 14)]
    res = arc.analyze_arc(kept, rim_center=(340.0, 150.0), frame_shape=(720, 1280))
    assert res["ok"] is True
    # apex is reconstructed from the two visible branches, so flag it.
    assert (res["estimated"] is True) or (res["off_frame_top"] is True)
    assert res["gap_frames"] >= 8           # the hole we punched
    assert res["entry_angle_deg"] is not None
    assert 0.0 < res["entry_angle_deg"] < 90.0


def test_apex_above_top_of_frame_is_off_frame_top():
    # Plant the apex above the top edge (negative image-y when extrapolated):
    # only the lower/descending samples are visible.
    pts = _make_arc(n=21, apex_y=-60.0, drop=300.0)
    visible = [p for p in pts if p[2] >= 0]  # camera can't see y<0
    res = arc.analyze_arc(visible, rim_center=(340.0, 150.0), frame_shape=(720, 1280))
    assert res["ok"] is True
    assert res["off_frame_top"] is True
    assert res["estimated"] is True
    assert res["entry_angle_deg"] is not None


# --------------------------------------------------------------------------- #
# too few points
# --------------------------------------------------------------------------- #
def test_two_points_is_not_ok():
    res = arc.analyze_arc([(0, 100.0, 300.0), (1, 110.0, 250.0)])
    assert res["ok"] is False
    assert res["n_points"] == 2
    assert res["apex_y"] is None
    assert res["entry_angle_deg"] is None


def test_empty_and_garbage_never_raise():
    for bad in (None, [], "nope", 5, [(0, 1)], [("a", "b", "c")]):
        res = arc.analyze_arc(bad)
        assert res["ok"] is False
        assert res["quality"] == 0.0


# --------------------------------------------------------------------------- #
# outlier rejection: a rebound / second-ball point
# --------------------------------------------------------------------------- #
def test_single_outlier_is_rejected_and_fit_survives():
    pts = _make_arc(n=21)
    clean = arc.analyze_arc(pts, rim_center=(340.0, 150.0), frame_shape=(720, 1280))

    # Inject one wild point (a rebound bouncing the OTHER way: huge y jump).
    noisy = list(pts)
    noisy[10] = (10, 600.0, 5000.0)
    res = arc.analyze_arc(noisy, rim_center=(340.0, 150.0), frame_shape=(720, 1280))

    assert res["ok"] is True
    assert "outlier" in res["note"]                  # it was rejected
    assert res["quality"] > 0.4                       # still a decent fit
    # apex recovered to roughly the clean value despite the spike.
    assert res["apex_y"] == pytest.approx(clean["apex_y"], abs=20.0)


# --------------------------------------------------------------------------- #
# arc_consistency
# --------------------------------------------------------------------------- #
def _shot_with_arc(entry_angle, peak_ratio):
    """A minimal shot dict carrying a forced-ok arc result."""
    return {"arc": {"ok": True, "entry_angle_deg": entry_angle,
                    "peak_height_ratio": peak_ratio, "peak_height_px": peak_ratio * 80.0}}


def test_arc_consistency_tight_arcs_score_high():
    shots = [_shot_with_arc(a, r) for a, r in
             [(46.0, 1.40), (47.0, 1.42), (45.0, 1.38), (46.5, 1.41), (46.0, 1.39)]]
    res = arc.arc_consistency(shots)
    assert res["enough"] is True
    assert res["n"] == 5
    assert res["entry_angle"]["consistency"] >= 85
    assert res["peak_height"]["consistency"] >= 85
    assert res["overall"] >= 85


def test_arc_consistency_scattered_arcs_score_low():
    shots = [_shot_with_arc(a, r) for a, r in
             [(20.0, 0.4), (70.0, 2.2), (35.0, 1.0), (62.0, 1.9), (28.0, 0.6)]]
    res = arc.arc_consistency(shots)
    assert res["enough"] is True
    assert res["overall"] <= 40
    assert res["entry_angle"]["consistency"] < 50


def test_arc_consistency_needs_three_usable_arcs():
    shots = [
        _shot_with_arc(46.0, 1.4),
        {"arc": {"ok": False}},          # not ok -> ignored
        {"no_arc": True},                # no arc at all -> ignored
    ]
    res = arc.arc_consistency(shots)
    assert res["enough"] is False
    assert res["n"] == 1


def test_arc_consistency_falls_back_to_px_when_ratio_absent():
    # arcs with no peak_height_ratio but a present peak_height_px still score.
    shots = [{"arc": {"ok": True, "entry_angle_deg": 46.0 + i * 0.2,
                      "peak_height_ratio": None, "peak_height_px": 110.0 + i}}
             for i in range(4)]
    res = arc.arc_consistency(shots)
    assert res["enough"] is True
    assert res["peak_height"] is not None
    assert res["peak_height"]["consistency"] >= 85


def test_arc_consistency_empty_is_safe():
    res = arc.arc_consistency([])
    assert res == {"enough": False, "n": 0}


# pytest is imported lazily so the module also runs as a plain script.
import pytest  # noqa: E402

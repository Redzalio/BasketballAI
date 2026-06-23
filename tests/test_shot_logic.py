"""Tests for the make/miss engine: detection.shot_logic.ShotTracker.

The tracker consumes per-frame detections and emits a {"result": ...} event
when a shot completes: the ball must pass from an 'up' region (above the rim)
through the rim plane to a 'down' region (below the rim). A linear fit of the
ball's path through the rim plane decides make vs miss -- a make requires the
predicted crossing x to land within +-0.4*rim_width of the rim center.

Detection geometry (from shot_logic.py), for a rim at center (cx, cy) with
width w and height h:
  * rim/scoring plane:  rim_y = cy - 0.5*h
  * 'up' region:        x in (cx - 4w, cx + 4w),  y in (cy - 2h, cy - 0.5h)
  * 'down' region:      y > cy + 0.5h
  * make x-window:      (cx - 0.4w, cx + 0.4w)

We synthesize descending ball arcs through a fixed rim and assert the outcome.
"""
import warnings

import pytest

from detection import shot_logic
from detection.shot_logic import ShotTracker

# Fixed rim used by every test. Round, generous dimensions so the synthetic
# ball boxes are comfortably inside the detection windows.
RIM_CX, RIM_CY = 300.0, 200.0
RIM_W, RIM_H = 40.0, 20.0
RIM_DET = ("rim", (RIM_CX, RIM_CY), RIM_W, RIM_H, 0.9)

# Derived reference values (mirror the module's geometry):
RIM_PLANE_Y = RIM_CY - 0.5 * RIM_H          # = 190.0
MAKE_X_LO = RIM_CX - 0.4 * RIM_W            # = 284.0
MAKE_X_HI = RIM_CX + 0.4 * RIM_W            # = 316.0

# A "roughly square" ball box that won't be discarded by _clean_ball
# (which drops boxes whose w/h differ by >1.4x or that jump >4x their diagonal).
BALL_W = BALL_H = 14.0


def _ball(x, y, conf=0.9):
    return ("ball", (float(x), float(y)), BALL_W, BALL_H, conf)


def _feed_arc(tracker, points, conf=0.9):
    """Feed one ball detection (plus the rim) per frame; return the last event.

    `points` is an iterable of (x, y) ball centers, oldest first. The tracker
    sees a fresh frame_idx for each point so the up/down state machine can
    advance frame-over-frame.
    """
    event = None
    # The straight-line MAKE arc produces a numpy "poorly conditioned" RankWarning
    # from polyfit; it is benign and not what we are testing.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for i, (x, y) in enumerate(points):
            ev = tracker.update([RIM_DET, _ball(x, y, conf)], frame_idx=i)
            if ev is not None:
                event = ev
    return event


# Descends straight down the middle (x ~ rim center) from above to below the rim.
MAKE_ARC = [(296, 150), (297, 165), (298, 180), (300, 195),
            (302, 210), (303, 225), (304, 240)]

# Descends on a diagonal that crosses the rim plane (y=190) well to the side of
# the rim center: at y=190 the line is near x~342, outside the 284..316 window.
MISS_ARC = [(360, 150), (356, 165), (352, 180), (348, 195),
            (344, 210), (340, 225), (336, 240)]


# --------------------------------------------------------------------------- #
# MAKE
# --------------------------------------------------------------------------- #
def test_make_arc_through_rim_center_counts_a_make():
    t = ShotTracker()
    event = _feed_arc(t, MAKE_ARC)

    assert event is not None, "a completed shot should emit an event"
    assert event["result"] == "make"
    assert t.makes == 1
    assert t.attempts == 1
    assert t.fg_pct == 100.0
    # event payload sanity
    assert event["makes"] == 1
    assert event["attempt"] == 1
    assert event["up_frame"] < event["frame"]


def test_make_arc_crosses_plane_inside_make_window():
    """Guard the geometry assumption: the make arc really does cross y=190
    inside the +-0.4*rim_width window (documents WHY it's a make)."""
    # Two points straddling the plane: (298,180) -> (300,195)
    x0, y0, x1, y1 = 298, 180, 300, 195
    m = (y1 - y0) / (x1 - x0)
    pred_x = x0 + (RIM_PLANE_Y - y0) / m
    assert MAKE_X_LO < pred_x < MAKE_X_HI


# --------------------------------------------------------------------------- #
# MISS
# --------------------------------------------------------------------------- #
def test_miss_arc_to_the_side_counts_a_miss():
    t = ShotTracker()
    event = _feed_arc(t, MISS_ARC)

    assert event is not None, "the shot still completes (up -> down)"
    assert event["result"] == "miss"
    assert t.makes == 0
    assert t.attempts == 1
    assert t.fg_pct == 0.0


def test_miss_arc_crosses_plane_outside_make_window():
    """Guard: the miss arc crosses y=190 OUTSIDE the make window."""
    x0, y0, x1, y1 = 352, 180, 348, 195
    m = (y1 - y0) / (x1 - x0)
    pred_x = x0 + (RIM_PLANE_Y - y0) / m
    assert not (MAKE_X_LO < pred_x < MAKE_X_HI)


# --------------------------------------------------------------------------- #
# No attempt
# --------------------------------------------------------------------------- #
def test_rim_only_no_ball_no_attempt():
    t = ShotTracker()
    event = None
    for i in range(12):
        ev = t.update([RIM_DET], frame_idx=i)
        if ev is not None:
            event = ev

    assert event is None
    assert t.attempts == 0
    assert t.makes == 0
    assert t.fg_pct == 0.0
    # the rim is still tracked even with no ball
    assert t.rim_center == (RIM_CX, RIM_CY)


def test_no_detections_at_all_is_inert():
    t = ShotTracker()
    for i in range(5):
        assert t.update([], frame_idx=i) is None
    assert t.attempts == 0 and t.makes == 0
    assert t.rim_center is None


def test_ball_passing_through_without_going_up_first_no_attempt():
    """A ball that only ever appears below the rim never enters the 'up' state,
    so no attempt is recorded."""
    t = ShotTracker()
    event = None
    for i, y in enumerate([215, 225, 235, 245, 255]):  # all below down-line (210)
        ev = t.update([RIM_DET, _ball(300, y)], frame_idx=i)
        if ev is not None:
            event = ev
    assert event is None
    assert t.attempts == 0


# --------------------------------------------------------------------------- #
# Confidence gating
# --------------------------------------------------------------------------- #
def test_low_confidence_ball_far_from_rim_is_ignored():
    """A ball below the ball_conf threshold and not in the hoop region is not
    tracked, so the arc never completes -> no attempt."""
    t = ShotTracker(ball_conf=0.35)
    event = None
    # conf 0.2 < 0.35, and these points (x=300, high above) are not within the
    # tight _in_hoop_region box, so they're dropped.
    for i, (x, y) in enumerate(MAKE_ARC):
        ev = t.update([RIM_DET, _ball(x, y, conf=0.20)], frame_idx=i)
        if ev is not None:
            event = ev
    assert event is None
    assert t.attempts == 0


def test_low_confidence_rim_is_ignored():
    """Rim below rim_conf is not tracked, so there is no scoring plane and no
    attempt can be detected even with a clean ball arc."""
    t = ShotTracker(rim_conf=0.45)
    weak_rim = ("rim", (RIM_CX, RIM_CY), RIM_W, RIM_H, 0.30)  # < 0.45
    event = None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for i, (x, y) in enumerate(MAKE_ARC):
            ev = t.update([weak_rim, _ball(x, y)], frame_idx=i)
            if ev is not None:
                event = ev
    assert event is None
    assert t.attempts == 0
    assert t.rim_center is None


# --------------------------------------------------------------------------- #
# Multiple shots accumulate
# --------------------------------------------------------------------------- #
def test_two_shots_accumulate_makes_attempts_and_fg_pct():
    t = ShotTracker()
    _feed_arc(t, MAKE_ARC)
    _feed_arc(t, MISS_ARC)

    assert t.attempts == 2
    assert t.makes == 1
    assert t.fg_pct == 50.0


# --------------------------------------------------------------------------- #
# fg_pct edge case
# --------------------------------------------------------------------------- #
def test_fg_pct_is_zero_before_any_attempt():
    assert ShotTracker().fg_pct == 0.0


def test_score_helper_make_vs_miss_directly():
    """Exercise the internal _score() linear-fit decision directly, so a change
    to the +-0.4*rim_width gate is caught even if the state machine changes."""
    # hoop_pos entry shape: (center, frame_idx, w, h, conf)
    hoop = [((RIM_CX, RIM_CY), 0, RIM_W, RIM_H, 0.9)]

    # ball_pos must contain a point ABOVE the plane (y < 190) followed by the
    # next point, so _score builds a 2-point line. Straight down the middle:
    make_balls = [((300, 180), 1, BALL_W, BALL_H, 0.9),
                  ((300, 200), 2, BALL_W, BALL_H, 0.9)]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # NOTE: _score returns a numpy bool (np.True_/np.False_), so assert on
        # truthiness, not Python `is True`.
        assert bool(shot_logic._score(make_balls, hoop)) is True

    # Far to the side: crosses the plane well outside the make window.
    miss_balls = [((360, 180), 1, BALL_W, BALL_H, 0.9),
                  ((360, 200), 2, BALL_W, BALL_H, 0.9)]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert bool(shot_logic._score(miss_balls, hoop)) is False

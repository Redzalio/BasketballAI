"""Tests for the shooting-form consistency engine: stats.consistency.

A shot is {"made": bool, "zone": str, "form": {metric: value}}.

Key behaviors locked in here:
  * metric_stats: per-metric mean/std + a 0-100 consistency sub-score where
    std=0 -> 100 and std>=tol -> 0 (tol is per-metric in METRIC_META).
    Needs >=3 values for a metric to appear.
  * consistency_score: rounded average of the sub-scores.
  * biggest_inconsistency: the metric with the LOWEST sub-score.
  * makes_vs_misses: per-metric make/miss means + delta; needs >=3 makes AND
    >=3 misses, else {"enough": False}.
  * what_to_work_on: a focus dict carrying a drill string.
"""
import pytest

from stats import consistency as C

# A metric known to METRIC_META, with a generous tol so a tight cluster scores
# near 100. elbow_angle: tol=22.
TIGHT_METRIC = "elbow_angle"
# A metric we will make swing wildly. lean_deg: tol=14 -> a ~13 std => ~0 score.
NOISY_METRIC = "lean_deg"


def _shots():
    """8 shots: elbow_angle tightly clustered, lean_deg swinging hugely,
    alternating make/miss so makes_vs_misses has >=3 of each."""
    elbow = [170, 171, 169, 170, 171, 169, 170, 171]   # std ~0.8
    lean = [2, 25, 1, 28, 0, 30, 3, 27]                # std ~13 (>= tol 14-ish)
    made = [True, False, True, False, True, False, True, False]
    return [
        {"made": m, "zone": "top", "form": {TIGHT_METRIC: e, NOISY_METRIC: l}}
        for e, l, m in zip(elbow, lean, made)
    ]


# --------------------------------------------------------------------------- #
# metric_stats
# --------------------------------------------------------------------------- #
def test_metric_stats_reports_both_metrics():
    ms = C.metric_stats(_shots())
    assert set(ms) == {TIGHT_METRIC, NOISY_METRIC}
    assert ms[TIGHT_METRIC]["n"] == 8
    assert ms[TIGHT_METRIC]["label"]  # human label carried through


def test_tight_metric_scores_high_noisy_scores_low():
    ms = C.metric_stats(_shots())
    assert ms[TIGHT_METRIC]["consistency"] >= 85
    assert ms[NOISY_METRIC]["consistency"] <= 20
    # and the tight metric's mean lands in the coaching "good" range
    assert ms[TIGHT_METRIC]["in_range"] is True


def test_metric_needs_at_least_three_values():
    # only 2 readings for elbow_angle -> excluded
    shots = [
        {"made": True, "form": {TIGHT_METRIC: 170}},
        {"made": False, "form": {TIGHT_METRIC: 171}},
    ]
    assert C.metric_stats(shots) == {}


def test_zero_variance_metric_scores_100():
    shots = [{"made": True, "form": {TIGHT_METRIC: 170}} for _ in range(5)]
    ms = C.metric_stats(shots)
    assert ms[TIGHT_METRIC]["std"] == 0
    assert ms[TIGHT_METRIC]["consistency"] == 100


def test_huge_variance_metric_clamps_to_zero():
    # spread far beyond tol on lean_deg (tol=14) -> clamped at 0, never negative
    shots = [
        {"made": True, "form": {NOISY_METRIC: v}}
        for v in [0, 90, 0, 90, 0, 90]
    ]
    ms = C.metric_stats(shots)
    assert ms[NOISY_METRIC]["consistency"] == 0


def test_non_numeric_and_bool_values_are_skipped():
    # bool is excluded explicitly by _vals; strings ignored. Only 3 real floats.
    shots = [
        {"made": True, "form": {TIGHT_METRIC: 170}},
        {"made": True, "form": {TIGHT_METRIC: True}},     # bool -> skipped
        {"made": True, "form": {TIGHT_METRIC: "x"}},      # str  -> skipped
        {"made": True, "form": {TIGHT_METRIC: 171}},
        {"made": True, "form": {TIGHT_METRIC: 169}},
    ]
    ms = C.metric_stats(shots)
    assert ms[TIGHT_METRIC]["n"] == 3


# --------------------------------------------------------------------------- #
# consistency_score
# --------------------------------------------------------------------------- #
def test_consistency_score_is_average_of_subscores():
    ms = C.metric_stats(_shots())
    expected = round(sum(m["consistency"] for m in ms.values()) / len(ms))
    assert C.consistency_score(ms) == expected


def test_consistency_score_empty_is_zero():
    assert C.consistency_score({}) == 0


# --------------------------------------------------------------------------- #
# biggest_inconsistency
# --------------------------------------------------------------------------- #
def test_biggest_inconsistency_flags_the_noisy_metric():
    ms = C.metric_stats(_shots())
    bi = C.biggest_inconsistency(ms)
    assert bi is not None
    assert bi["metric"] == NOISY_METRIC
    assert bi["consistency"] == ms[NOISY_METRIC]["consistency"]


def test_biggest_inconsistency_none_when_empty():
    assert C.biggest_inconsistency({}) is None


# --------------------------------------------------------------------------- #
# makes_vs_misses
# --------------------------------------------------------------------------- #
def test_makes_vs_misses_reports_means_and_delta():
    mvm = C.makes_vs_misses(_shots())
    assert mvm["enough"] is True
    rows = {r["metric"]: r for r in mvm["rows"]}
    assert NOISY_METRIC in rows
    row = rows[NOISY_METRIC]
    # makes had lean in {2,1,0,3} (mean ~1.5), misses {25,28,30,27} (mean ~27.5)
    assert row["make"] == pytest.approx(1.5, abs=0.6)
    assert row["miss"] == pytest.approx(27.5, abs=0.6)
    assert row["delta"] == pytest.approx(row["miss"] - row["make"], abs=0.1)
    # rows are sorted by normalized separation, biggest first
    assert mvm["top"]["metric"] == NOISY_METRIC


def test_makes_vs_misses_needs_three_makes():
    shots = (
        [{"made": True, "form": {TIGHT_METRIC: 170}}] * 2          # only 2 makes
        + [{"made": False, "form": {TIGHT_METRIC: 150 + i}} for i in range(4)]
    )
    assert C.makes_vs_misses(shots) == {"enough": False}


def test_makes_vs_misses_needs_three_misses():
    shots = (
        [{"made": True, "form": {TIGHT_METRIC: 170 + i}} for i in range(4)]
        + [{"made": False, "form": {TIGHT_METRIC: 150}}] * 2       # only 2 misses
    )
    assert C.makes_vs_misses(shots) == {"enough": False}


def test_made_accepts_result_string_form():
    """_made() honors either {"made": True} or {"result": "make"}."""
    shots = (
        [{"result": "make", "form": {TIGHT_METRIC: 170 + i}} for i in range(3)]
        + [{"result": "miss", "form": {TIGHT_METRIC: 150 + i}} for i in range(3)]
    )
    mvm = C.makes_vs_misses(shots)
    assert mvm["enough"] is True


# --------------------------------------------------------------------------- #
# what_to_work_on
# --------------------------------------------------------------------------- #
def test_what_to_work_on_returns_focus_with_drill():
    ms = C.metric_stats(_shots())
    mvm = C.makes_vs_misses(_shots())
    focus = C.what_to_work_on(ms, mvm)
    assert focus is not None
    assert focus["focus"] == NOISY_METRIC          # highest-leverage metric
    assert isinstance(focus["drill"], str) and focus["drill"]
    assert focus["drill"] == C.DRILLS[NOISY_METRIC]
    assert focus["label"]
    assert focus["why"]


def test_what_to_work_on_none_when_no_metrics():
    assert C.what_to_work_on({}, {"enough": False}) is None


def test_what_to_work_on_uses_generic_drill_for_unknown_metric():
    """A metric absent from DRILLS falls back to GENERIC_DRILL. We fake an
    mstats dict for a key without a specific drill (release_height_ratio HAS
    one, so use a synthetic key not in DRILLS)."""
    fake_ms = {"mystery_metric": {"label": "Mystery", "unit": "", "std": 9,
                                  "consistency": 10, "mean": 1, "good": [0, 1],
                                  "in_range": False, "n": 5}}
    focus = C.what_to_work_on(fake_ms, {"enough": False})
    assert focus["drill"] == C.GENERIC_DRILL


# --------------------------------------------------------------------------- #
# session_consistency (integration of the above)
# --------------------------------------------------------------------------- #
def test_session_consistency_bundles_everything():
    out = C.session_consistency(_shots())
    assert "consistency_score" in out
    assert out["metrics"].keys() == {TIGHT_METRIC, NOISY_METRIC}
    assert out["biggest_inconsistency"]["metric"] == NOISY_METRIC
    assert out["makes_vs_misses"]["enough"] is True
    assert out["focus"]["focus"] == NOISY_METRIC
    assert out["shots_analyzed"] == 8


def test_session_consistency_empty_is_safe():
    out = C.session_consistency([])
    assert out["consistency_score"] == 0
    assert out["metrics"] == {}
    assert out["biggest_inconsistency"] is None
    assert out["makes_vs_misses"] == {"enough": False}
    assert out["focus"] is None
    assert out["shots_analyzed"] == 0

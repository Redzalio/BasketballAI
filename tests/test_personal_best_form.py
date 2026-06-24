"""Tests for the personalized-form engine: stats.personal_best_form.

A shot is {"made": bool/result, "form": {metric: value}}. The module builds a
form TEMPLATE from the player's own makes (median per metric, MAD spread), scores
each shot's per-metric deviation from that template, and reports a session match
+ the biggest gap from the player's best form.

Key behaviors locked in here:
  * build_template: per-metric MEDIAN over makes; needs >= 5 makes-with-form, and
    >= 5 readings for a given metric to template it. Reuses consistency.METRIC_META
    labels/tols (not redefined).
  * score_shot: per-metric delta/z/off + a 0..100 closeness; a shot far from the
    template scores low, flags off=True, and names the worst metric.
  * analyze: builds the template from a broader pool (or the session itself),
    scores the session, and surfaces the systematically-off metric as biggest_gap
    with the correct direction.
  * garbage / empty input -> enough False, never raises.
"""
import pytest

from stats import personal_best_form as P
from stats.consistency import METRIC_META

# Metrics we drive in the synthetic data. Both are in METRIC_META.
ELBOW = "elbow_angle"      # good (160,180), tol 22
RELEASE = "release_angle"  # good (48,58),  tol 18
LEAN = "lean_deg"          # good (0,8),    tol 14


def _make(elbow, release, lean, made=True):
    return {"made": made, "form": {ELBOW: elbow, RELEASE: release, LEAN: lean}}


def _clean_makes(n=10):
    """n makes tightly clustered around elbow=172, release=52, lean=3.
    Small alternating jitter keeps a real (but small) spread."""
    out = []
    for i in range(n):
        j = (i % 5) - 2  # -2..+2
        out.append(_make(172 + j, 52 + j * 0.5, 3 + (j * 0.4)))
    return out


# --------------------------------------------------------------------------- #
# build_template
# --------------------------------------------------------------------------- #
def test_build_template_medians_match_expected():
    tmpl = P.build_template(_clean_makes(10))
    assert tmpl["enough"] is True
    assert tmpl["n_makes"] == 10
    # Median of a symmetric -2..+2 jitter sits on the center value.
    assert tmpl["template"][ELBOW] == pytest.approx(172, abs=1.0)
    assert tmpl["template"][RELEASE] == pytest.approx(52, abs=1.0)
    assert tmpl["template"][LEAN] == pytest.approx(3, abs=1.0)
    # Spread is present and positive for each templated metric.
    for m in (ELBOW, RELEASE, LEAN):
        assert tmpl["spread"][m] > 0


def test_build_template_reuses_metric_meta_keys_only():
    """A non-METRIC_META numeric form key (and the follow_through bool / hand)
    must never enter the template."""
    makes = _clean_makes(8)
    for m in makes:
        m["form"]["mystery_metric"] = 999.0   # unknown numeric -> ignored
        m["form"]["follow_through"] = True     # bool -> ignored
        m["form"]["hand"] = "right"            # str  -> ignored
    tmpl = P.build_template(makes)
    assert tmpl["enough"] is True
    assert "mystery_metric" not in tmpl["template"]
    assert "follow_through" not in tmpl["template"]
    assert "hand" not in tmpl["template"]
    assert set(tmpl["template"]).issubset(set(METRIC_META))


def test_build_template_only_uses_makes_not_misses():
    """Misses (even with wild form) must not move the template off the makes."""
    makes = _clean_makes(8)
    misses = [_make(120, 30, 40, made=False) for _ in range(8)]
    tmpl = P.build_template(makes + misses)
    assert tmpl["enough"] is True
    assert tmpl["n_makes"] == 8                       # only makes counted
    assert tmpl["template"][ELBOW] == pytest.approx(172, abs=1.5)
    assert tmpl["template"][LEAN] == pytest.approx(3, abs=1.5)


def test_build_template_needs_five_makes():
    tmpl = P.build_template(_clean_makes(4))
    assert tmpl["enough"] is False
    assert tmpl["n_makes"] == 4
    assert tmpl["template"] == {}
    assert tmpl["spread"] == {}


def test_build_template_metric_below_five_readings_excluded():
    """elbow present in all 6 makes; release present in only 3 -> release excluded
    while elbow is templated."""
    makes = []
    for i in range(6):
        form = {ELBOW: 172 + (i % 3 - 1)}
        if i < 3:
            form[RELEASE] = 52
        makes.append({"made": True, "form": form})
    tmpl = P.build_template(makes)
    assert tmpl["enough"] is True
    assert ELBOW in tmpl["template"]
    assert RELEASE not in tmpl["template"]   # only 3 readings, < 5


def test_build_template_bool_not_counted_as_number():
    """A True in a numeric metric slot must be skipped, not read as 1.0."""
    makes = [{"made": True, "form": {ELBOW: True}} for _ in range(6)]
    tmpl = P.build_template(makes)
    # No numeric readings at all -> not enough usable makes.
    assert tmpl["enough"] is False


# --------------------------------------------------------------------------- #
# score_shot
# --------------------------------------------------------------------------- #
def test_score_shot_on_template_scores_high():
    tmpl = P.build_template(_clean_makes(10))
    on = {ELBOW: 172, RELEASE: 52, LEAN: 3}
    res = P.score_shot(on, tmpl["template"], tmpl["spread"])
    assert res["score"] >= 90
    # nothing flagged off when sitting on the template
    assert all(not d["off"] for d in res["deltas"].values())


def test_score_shot_far_from_template_flags_worst():
    """One metric (elbow) wildly off, the others on template: the worst metric is
    elbow, it's flagged off, and the score is well below a perfect 100 (but not
    rock-bottom, since 2 of 3 metrics still match)."""
    tmpl = P.build_template(_clean_makes(10))
    far = {ELBOW: 130, RELEASE: 52, LEAN: 3}
    res = P.score_shot(far, tmpl["template"], tmpl["spread"])
    assert res["score"] < 80                      # dragged down by the off metric
    assert res["worst"] == ELBOW
    assert res["deltas"][ELBOW]["off"] is True
    assert res["deltas"][ELBOW]["delta"] == pytest.approx(130 - tmpl["template"][ELBOW], abs=0.1)
    # z is negative (shot below template) and large in magnitude
    assert res["deltas"][ELBOW]["z"] < -1.0
    # the on-template metrics are NOT flagged off
    assert res["deltas"][RELEASE]["off"] is False
    assert res["deltas"][LEAN]["off"] is False
    # label/unit are carried straight from METRIC_META
    assert res["deltas"][ELBOW]["label"] == METRIC_META[ELBOW]["label"]
    assert res["deltas"][ELBOW]["unit"] == METRIC_META[ELBOW]["unit"]


def test_score_shot_off_on_every_metric_scores_low():
    """When EVERY metric is >= 2 spreads off, closeness bottoms out near 0."""
    tmpl = P.build_template(_clean_makes(10))
    t = tmpl["template"]
    # push each metric far past 2 spreads in some direction
    far = {ELBOW: t[ELBOW] - 40, RELEASE: t[RELEASE] + 25, LEAN: t[LEAN] + 35}
    res = P.score_shot(far, tmpl["template"], tmpl["spread"])
    assert res["score"] <= 10
    assert all(d["off"] for d in res["deltas"].values())


def test_score_shot_falls_back_to_tol_without_spread():
    tmpl = P.build_template(_clean_makes(10))
    # No spread dict -> METRIC_META tol used; an on-template shot still scores high.
    res = P.score_shot({ELBOW: tmpl["template"][ELBOW]}, tmpl["template"])
    assert res["score"] == 100
    assert res["deltas"][ELBOW]["z"] == 0.0


def test_score_shot_only_scores_overlapping_metrics():
    template = {ELBOW: 172, RELEASE: 52}
    # form has elbow (on) + an unknown key; release absent -> only elbow scored.
    res = P.score_shot({ELBOW: 172, "mystery_metric": 5}, template)
    assert set(res["deltas"]) == {ELBOW}


def test_score_shot_no_overlap_is_safe():
    res = P.score_shot({LEAN: 3}, {ELBOW: 172})
    assert res == {"score": 0, "deltas": {}, "worst": None}


# --------------------------------------------------------------------------- #
# analyze
# --------------------------------------------------------------------------- #
def test_analyze_systematic_gap_is_detected_with_direction():
    """Template from a clean broad pool; the session shoots with elbow
    SYSTEMATICALLY ~20deg lower. biggest_gap must be elbow, direction 'lower'."""
    pool = _clean_makes(12)
    session = [
        {"made": False, "form": {ELBOW: 152, RELEASE: 52, LEAN: 3}}
        for _ in range(8)
    ]
    out = P.analyze(session, all_shots=pool)
    assert out["enough"] is True
    assert out["n_makes"] == 12
    assert out["biggest_gap"]["metric"] == ELBOW
    assert out["biggest_gap"]["direction"] == "lower"
    # mean delta is negative (~ -20) and the fix talks about building it back up
    assert out["biggest_gap"]["mean_delta"] < 0
    assert "lower" in out["biggest_gap"]["fix"]
    # per-shot scoring present, one entry per session shot with form
    assert len(out["per_shot"]) == 8
    assert all("score" in p and "i" in p for p in out["per_shot"])
    # session_match is low-ish since every shot drifts on elbow
    assert 0 <= out["session_match"] <= 100


def test_analyze_higher_direction():
    """Session elbow systematically HIGHER than the makes -> direction 'higher'."""
    pool = _clean_makes(12)
    session = [
        {"made": False, "form": {ELBOW: 195, RELEASE: 52, LEAN: 3}}
        for _ in range(6)
    ]
    out = P.analyze(session, all_shots=pool)
    assert out["biggest_gap"]["metric"] == ELBOW
    assert out["biggest_gap"]["direction"] == "higher"
    assert out["biggest_gap"]["mean_delta"] > 0
    assert "higher" in out["biggest_gap"]["fix"]


def test_analyze_matched_session_scores_high():
    """A session that matches the template scores a high session_match."""
    pool = _clean_makes(12)
    session = _clean_makes(6)               # same form as the makes
    out = P.analyze(session, all_shots=pool)
    assert out["enough"] is True
    assert out["session_match"] >= 85


def test_analyze_defaults_template_to_session_when_no_pool():
    """With no all_shots, the template is built from the session's own makes."""
    session = _clean_makes(8)
    out = P.analyze(session)
    assert out["enough"] is True
    assert out["n_makes"] == 8
    assert ELBOW in out["template"]


def test_analyze_per_shot_indices_track_session_positions():
    """per_shot 'i' indexes into session_shots; formless shots are skipped but
    indices of scored shots stay correct."""
    pool = _clean_makes(10)
    session = [
        {"made": True, "form": {}},                                   # i=0 skipped
        {"made": False, "form": {ELBOW: 150, RELEASE: 52, LEAN: 3}},  # i=1 scored
        {"made": False},                                              # i=2 skipped
        {"made": False, "form": {ELBOW: 151, RELEASE: 52, LEAN: 3}},  # i=3 scored
    ]
    out = P.analyze(session, all_shots=pool)
    idxs = [p["i"] for p in out["per_shot"]]
    assert idxs == [1, 3]


# --------------------------------------------------------------------------- #
# garbage / empty input -> enough False, never raises
# --------------------------------------------------------------------------- #
def test_build_template_garbage_is_safe():
    assert P.build_template(None)["enough"] is False
    assert P.build_template([])["enough"] is False
    assert P.build_template("nope")["enough"] is False
    # shots that aren't dicts / have non-dict form
    assert P.build_template([None, 5, {"made": True, "form": "x"}])["enough"] is False


def test_score_shot_garbage_is_safe():
    assert P.score_shot(None, None) == {"score": 0, "deltas": {}, "worst": None}
    assert P.score_shot({}, {}) == {"score": 0, "deltas": {}, "worst": None}
    assert P.score_shot("x", {ELBOW: 172}) == {"score": 0, "deltas": {}, "worst": None}


def test_analyze_garbage_is_safe():
    for bad in (None, [], "nope", [None, 5], 42):
        out = P.analyze(bad)
        assert out["enough"] is False
        assert out["session_match"] == 0
        assert out["biggest_gap"] is None
        assert out["per_shot"] == []


def test_analyze_empty_session_with_pool_is_safe():
    """Template can build from the pool, but an empty session yields no scores."""
    out = P.analyze([], all_shots=_clean_makes(10))
    assert out["enough"] is False
    assert out["session_match"] == 0
    assert out["per_shot"] == []

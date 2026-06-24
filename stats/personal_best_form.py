"""Personalized form target: compare a shot to the player's OWN best form.

Generic pro ranges (consistency.PRO) say what an elite shooter looks like. This
module says what YOU look like when you make shots -- it builds a *personalized*
form template from the player's own makes, then scores every shot by how far it
drifts from that template, per metric. The point is self-relative coaching: "you
made shots with your elbow at ~172deg; this miss was at 158deg" beats "the pros
are at 168-180".

Why makes, why median, why MAD: a player's makes are their de-facto "good" form,
and the MEDIAN of those makes is a robust center that ignores the odd lucky make
with broken mechanics. Spread is the MAD (median absolute deviation, scaled to a
std-equivalent) so a couple of outlier makes don't inflate the tolerance. We also
drop the worst-quartile makes (by overall deviation from the rough center) before
fitting, so the template reflects the player's *cleanest repeatable* form, not
their average make.

This sits on top of stats.consistency: it IMPORTS and reuses that module's
METRIC_META (labels / units / tols) verbatim -- it does not redefine them. Only
the numeric metrics that appear in METRIC_META are templated; everything else on
a form dict (hand, the follow_through bool, unknown keys) is ignored.

Pure numpy + stdlib. No torch/cv2. Every public function returns a dict and never
raises; bad input yields an {"enough": False, ...} dict (mirrors arc.py /
consistency.py defensive style).
"""
import math

import numpy as np

# Reuse the per-metric metadata (label / unit / good range / tol) from the
# consistency engine -- do NOT redefine it here. New pose metrics light up
# automatically once they're in METRIC_META and present in the form dicts.
from stats.consistency import METRIC_META

# A metric must appear in at least this many makes to be templated. Below this
# the median/spread aren't trustworthy. (Matches the ">= ~5 makes" spec.)
MIN_MAKES_PER_METRIC = 5
# A session needs at least this many makes-with-form before a template is built.
MIN_MAKES_FOR_TEMPLATE = 5
# Scale factor turning a MAD into a std-equivalent for a normal distribution.
MAD_TO_STD = 1.4826
# A spread floor (as a fraction of the metric's tol) so a freakishly tight set of
# makes can't produce a near-zero spread that makes every other shot look "off".
MIN_SPREAD_FRAC_OF_TOL = 0.25
# Closeness mapping: |z| of this much -> 0 contribution; |z|==0 -> full credit.
# z is in spread-units, so 2 spreads off = no credit for that metric.
Z_ZERO_CREDIT = 2.0


def _made(s):
    """Made-shot test honoring either {"made": ...} or {"result": "make"}.

    Mirrors consistency._made. Never raises on a non-dict shot."""
    if not isinstance(s, dict):
        return False
    return bool(s.get("made")) or s.get("result") == "make"


def _is_num(v):
    """A real, finite number -- bools are explicitly NOT numbers here (a stray
    True must never be treated as 1.0). Mirrors consistency._vals' guard."""
    return (
        isinstance(v, (int, float))
        and not isinstance(v, bool)
        and math.isfinite(v)
    )


def _form(s):
    """The shot's form dict, or {} for anything malformed. Never raises."""
    if not isinstance(s, dict):
        return {}
    f = s.get("form")
    return f if isinstance(f, dict) else {}


def _template_metrics(form):
    """The METRIC_META keys carrying a usable numeric value on this form dict."""
    return [k for k in METRIC_META if k in form and _is_num(form.get(k))]


def _mad_std(vals, med):
    """Std-equivalent spread from the median absolute deviation, falling back to
    the population std when the MAD collapses to ~0 (e.g. a tied cluster)."""
    arr = np.asarray(vals, dtype=float)
    mad = float(np.median(np.abs(arr - med)))
    spread = mad * MAD_TO_STD
    if spread <= 1e-9:
        spread = float(np.std(arr))  # population std, like consistency.pstdev
    return spread


def build_template(shots):
    """Build a personalized form target from the player's MAKES.

    For each METRIC_META metric present in >= MIN_MAKES_PER_METRIC makes, the
    template value is the MEDIAN of those makes (robust center) and the spread is
    the MAD scaled to a std-equivalent (floored at a fraction of the metric's
    tol). Before fitting we drop the worst-quartile makes by overall deviation
    from a rough median center, so the template reflects the player's cleanest
    repeatable form rather than their average make.

    shots: list of shot dicts; each may carry shot["form"] (a dict) and a
           made-flag (shot["made"] truthy or shot["result"] == "make").

    Returns:
      {"enough": bool,
       "n_makes": int,                 # makes-with-form considered
       "template": {metric: float},    # median target per metric
       "spread":   {metric: float},    # std-equivalent spread per metric
       "note": str}
    enough is False (with empty template/spread) when fewer than
    MIN_MAKES_FOR_TEMPLATE makes carry form.
    """
    base = {"enough": False, "n_makes": 0, "template": {}, "spread": {}, "note": ""}

    if not isinstance(shots, (list, tuple)):
        base["note"] = "no shots provided"
        return base

    # Makes that actually carry at least one templatable metric.
    makes = []
    for s in shots:
        if not _made(s):
            continue
        f = _form(s)
        if _template_metrics(f):
            makes.append(f)

    n_makes = len(makes)
    base["n_makes"] = n_makes
    if n_makes < MIN_MAKES_FOR_TEMPLATE:
        base["note"] = (
            f"need >= {MIN_MAKES_FOR_TEMPLATE} makes with form, have {n_makes}"
        )
        return base

    # --- rough per-metric center, used only to rank makes by cleanliness ----- #
    rough_center = {}
    for key in METRIC_META:
        vals = [f[key] for f in makes if _is_num(f.get(key))]
        if vals:
            rough_center[key] = float(np.median(np.asarray(vals, dtype=float)))

    def _overall_dev(f):
        """Mean normalized |deviation| of this make from the rough center, over
        whatever templatable metrics it has (tol-normalized so metrics on
        different scales are comparable). Used to drop the worst-quartile makes.
        """
        zs = []
        for key, c in rough_center.items():
            v = f.get(key)
            if _is_num(v):
                tol = METRIC_META[key]["tol"] or 1.0
                zs.append(abs(v - c) / tol)
        return float(np.mean(zs)) if zs else float("inf")

    # Keep the cleanest makes: drop the worst quartile by overall deviation, but
    # only when we have enough makes left to still satisfy the per-metric floor.
    kept = makes
    if n_makes >= MIN_MAKES_FOR_TEMPLATE + 2:
        ranked = sorted(makes, key=_overall_dev)
        keep_n = max(MIN_MAKES_FOR_TEMPLATE, int(round(n_makes * 0.75)))
        kept = ranked[:keep_n]
    dropped = n_makes - len(kept)

    # --- build template + spread per metric over the kept makes -------------- #
    template = {}
    spread = {}
    for key, meta in METRIC_META.items():
        vals = [f[key] for f in kept if _is_num(f.get(key))]
        if len(vals) < MIN_MAKES_PER_METRIC:
            continue
        arr = np.asarray(vals, dtype=float)
        med = float(np.median(arr))
        sp = _mad_std(vals, med)
        tol = meta["tol"] or 1.0
        sp = max(sp, MIN_SPREAD_FRAC_OF_TOL * tol)  # floor so spread is meaningful
        template[key] = round(med, 2)
        spread[key] = round(sp, 3)

    if not template:
        base["note"] = (
            f"{n_makes} makes but no single metric reached "
            f"{MIN_MAKES_PER_METRIC} readings"
        )
        return base

    note = f"template from {len(kept)} of {n_makes} makes"
    if dropped:
        note += f" (dropped {dropped} least-consistent)"
    return {
        "enough": True,
        "n_makes": n_makes,
        "template": template,
        "spread": spread,
        "note": note,
    }


def score_shot(form, template, spread=None):
    """Compare ONE shot's form against a personalized template.

    For each metric in `template` that's also present (numeric) in `form`:
      delta = value - template[metric]
      z     = delta / (spread[metric] if given else METRIC_META[metric]["tol"])
      off   = |z| > 1            (more than one spread-unit away)
    A closeness score 0..100 = round(100 * mean over those metrics of
    max(0, 1 - |z| / Z_ZERO_CREDIT)) -- i.e. exactly on template -> 100, two
    spread-units off on every metric -> 0, clamped to [0, 100].

    form:     a shot's form dict (may be empty / malformed).
    template: {metric: target_value} from build_template (or any dict-like).
    spread:   optional {metric: spread} from build_template; when a metric's
              spread is missing/<=0 the metric's METRIC_META tol is used.

    Returns:
      {"score": 0..100,
       "deltas": {metric: {"value","template","delta","z","off","label","unit"}},
       "worst": metric|None}      # the metric with the largest |z|, or None
    On no comparable metrics: {"score": 0, "deltas": {}, "worst": None}.
    """
    out = {"score": 0, "deltas": {}, "worst": None}

    if not isinstance(form, dict) or not isinstance(template, dict):
        return out
    if not isinstance(spread, dict):
        spread = {}

    deltas = {}
    credits = []
    worst_key = None
    worst_abs_z = -1.0

    for metric, t in template.items():
        if metric not in METRIC_META:
            continue
        if not _is_num(t):
            continue
        v = form.get(metric)
        if not _is_num(v):
            continue

        meta = METRIC_META[metric]
        sp = spread.get(metric)
        if not _is_num(sp) or sp <= 0:
            sp = meta["tol"]
        if not sp or sp <= 0:  # final guard against a zero/None tol
            sp = 1.0

        delta = float(v) - float(t)
        z = delta / sp
        abs_z = abs(z)

        deltas[metric] = {
            "value": round(float(v), 2),
            "template": round(float(t), 2),
            "delta": round(delta, 2),
            "z": round(z, 2),
            "off": abs_z > 1.0,
            "label": meta["label"],
            "unit": meta["unit"],
        }
        credits.append(max(0.0, 1.0 - abs_z / Z_ZERO_CREDIT))
        if abs_z > worst_abs_z:
            worst_abs_z = abs_z
            worst_key = metric

    if not credits:
        return out

    score = round(100.0 * (sum(credits) / len(credits)))
    score = int(max(0, min(100, score)))
    return {"score": score, "deltas": deltas, "worst": worst_key}


def _direction_word(mean_delta):
    """'higher'/'lower'/'on target' for a signed mean delta (shot minus template)."""
    if mean_delta > 0:
        return "higher"
    if mean_delta < 0:
        return "lower"
    return "on target"


def _coaching_fix(label, mean_delta, unit):
    """Short self-relative cue: nudge the metric back DOWN/UP toward template."""
    mag = abs(round(mean_delta, 1))
    low = label.lower()
    if mean_delta > 0:
        return (
            f"Your {low} runs about {mag}{unit} higher than your makes -- "
            f"bring it back down toward your own best form."
        )
    if mean_delta < 0:
        return (
            f"Your {low} runs about {mag}{unit} lower than your makes -- "
            f"build it back up toward your own best form."
        )
    return f"Your {low} matches your makes -- keep grooving it."


def analyze(session_shots, all_shots=None):
    """End-to-end personalized-form read for a SESSION.

    The template is built from a BROADER make set (`all_shots`) when provided --
    e.g. every shot the player has ever logged -- so a single weak session is
    judged against the player's true best form. When `all_shots` is None the
    template is built from `session_shots` themselves (all there is to go on).

    Each session shot that carries form is scored against the template. The
    session_match is the mean closeness over those scored shots. The biggest_gap
    is the metric whose mean |z| across the session is largest -- the form trait
    that drifts most from the player's makes -- carried with its label, mean
    signed delta, a direction word ("higher"/"lower"), and a short coaching fix.

    session_shots: the session's shot dicts (each may carry shot["form"]).
    all_shots:     optional broader pool to build the template from; defaults to
                   session_shots.

    Returns:
      {"enough": bool,
       "n_makes": int,                 # makes that fed the template
       "template": {metric: float},
       "spread":   {metric: float},
       "session_match": 0..100,        # mean closeness over scored session shots
       "biggest_gap": {"metric","label","mean_delta","direction","fix"}|None,
       "per_shot": [{"i": idx, "score": 0..100, "worst": metric|None}, ...],
       "note": str}
    When no template can be built: enough=False, session_match=0,
    biggest_gap=None, per_shot=[] (never raises).
    """
    base = {
        "enough": False, "n_makes": 0, "template": {}, "spread": {},
        "session_match": 0, "biggest_gap": None, "per_shot": [], "note": "",
    }

    if not isinstance(session_shots, (list, tuple)):
        session_shots = []
    pool = all_shots if isinstance(all_shots, (list, tuple)) else session_shots

    tmpl = build_template(pool)
    base["n_makes"] = tmpl["n_makes"]
    if not tmpl["enough"]:
        base["note"] = tmpl["note"]
        return base

    template = tmpl["template"]
    spread = tmpl["spread"]
    base.update({"template": template, "spread": spread})

    per_shot = []
    scores = []
    # Accumulate per-metric signed deltas + |z| across scored session shots.
    z_abs_acc = {}     # metric -> [|z|, ...]
    delta_acc = {}     # metric -> [signed delta, ...]

    for i, s in enumerate(session_shots):
        f = _form(s)
        if not f:
            continue
        res = score_shot(f, template, spread)
        if not res["deltas"]:
            continue
        per_shot.append({"i": i, "score": res["score"], "worst": res["worst"]})
        scores.append(res["score"])
        for metric, d in res["deltas"].items():
            z_abs_acc.setdefault(metric, []).append(abs(d["z"]))
            delta_acc.setdefault(metric, []).append(d["delta"])

    if not scores:
        base["note"] = "template built, but no session shots carried form to score"
        return base

    session_match = int(max(0, min(100, round(sum(scores) / len(scores)))))

    # Biggest gap = metric with the largest MEAN |z| across the session.
    biggest_gap = None
    if z_abs_acc:
        gap_metric = max(z_abs_acc, key=lambda k: float(np.mean(z_abs_acc[k])))
        mean_delta = float(np.mean(delta_acc[gap_metric]))
        meta = METRIC_META[gap_metric]
        biggest_gap = {
            "metric": gap_metric,
            "label": meta["label"],
            "mean_delta": round(mean_delta, 2),
            "direction": _direction_word(mean_delta),
            "fix": _coaching_fix(meta["label"], mean_delta, meta["unit"]),
        }

    return {
        "enough": True,
        "n_makes": tmpl["n_makes"],
        "template": template,
        "spread": spread,
        "session_match": session_match,
        "biggest_gap": biggest_gap,
        "per_shot": per_shot,
        "note": tmpl["note"],
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
    print(_json.dumps(analyze(obj["shots"]), indent=2, default=str))

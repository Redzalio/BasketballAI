"""Miss diagnosis: WHERE a miss went (left / right / short / long / in-and-out)
and WHY (which form metric the dominant miss bias lines up with).

This turns "you missed" into "you miss short -> bend your knees". It is the
companion to arc.py (ball-flight shape) and consistency.py (form variance):

  * classify_miss(...)  -- one shot: which side/depth the ball missed on.
  * miss_breakdown(...)  -- session: tally misses by direction, find the dominant.
  * miss_cause(...)      -- session: correlate the dominant miss with a form metric
                            and hand back a concrete fix.

It consumes the SAME per-shot ball track arc.py uses -- a list of
(frame_idx, x, y) in IMAGE PIXEL coordinates (y grows DOWNWARD) -- plus the rim
center. The descending-branch ball point nearest the rim plane is compared to
the rim x; the horizontal miss is normalized by the same rim-width proxy arc.py
uses (frame_width / 8, since we never get a true rim radius).

HONESTY ABOUT THE AXES (same spirit as arc.py's entry-angle caveat):

  * LEFT / RIGHT is the RELIABLE axis. Horizontal offset of the ball from the
    rim at the rim plane reads well from almost any camera that faces the hoop;
    it barely depends on depth. These get the higher confidence.

  * SHORT / LONG is BEST-EFFORT and lower confidence. True depth is the one
    thing a single 2D camera cannot see. We use a defensible image-space proxy
    (does the descending ball reach the rim's horizontal band, fall in front of
    it, or sail past it) and, when an arc is supplied, its descent steepness.
    These ALWAYS carry a lower confidence than left/right and say so in `note`.

  * IN_OUT (rattle) is inferred, never certain: a miss whose ball arrives
    essentially on-line (small |dx|) and is still tracked dropping through the
    rim band. Flagged with modest confidence.

Pure numpy + stdlib. No torch/cv2. Every public function returns a dict and
never raises; bad input yields an {"ok": False} / {"enough": False} dict.
"""
import math

import numpy as np

# Reuse the form-metric labels/units/tolerances -- do NOT redefine them here.
try:  # package import (normal: `from stats import miss_diagnosis`)
    from .consistency import METRIC_META
except ImportError:  # flat import / running as a script from inside stats/
    from consistency import METRIC_META


# --- thresholds (in rim-width-proxy units; rw = frame_w / 8) ----------------- #
# Horizontal offset (|dx_norm|) at/over which a miss is called to a side. ~0.5
# proxy-widths off-center is a clear left/right miss.
SIDE_T = 0.5
# Below this |dx_norm| the ball arrived essentially on-line -- the miss was depth
# (short/long) or a rattle (in_out), not a side miss.
ONLINE_T = 0.4

# Aggregate gates.
MIN_MISSES = 4          # miss_breakdown needs >=4 classified misses to be "enough"
MIN_GROUP = 4           # miss_cause needs >=4 makes AND >=4 dominant-dir misses

DIRECTIONS = ("left", "right", "short", "long", "in_out")


def _clean_points(points):
    """Coerce `points` to a sorted list of (f, x, y) floats, dropping anything
    non-finite or malformed. Returns [] on bad input -- never raises. (Same
    cleaning contract as arc._clean_points so the two stay interchangeable.)"""
    out = []
    if not isinstance(points, (list, tuple)):
        return out
    for p in points:
        try:
            f, x, y = p[0], p[1], p[2]
        except (TypeError, IndexError, KeyError):
            continue
        try:
            f = float(f); x = float(x); y = float(y)
        except (TypeError, ValueError):
            continue
        if not (math.isfinite(f) and math.isfinite(x) and math.isfinite(y)):
            continue
        out.append((f, x, y))
    out.sort(key=lambda t: t[0])
    return out


def _rim_width(frame_shape, xs=None):
    """Pixel scale proxy used to normalize the horizontal miss. Mirrors
    arc._rim_width: a regulation rim is ~1/8 of a typical phone/broadcast frame
    width. When frame_shape is missing, fall back to a robust scale from the
    horizontal spread of the track itself, so dx_norm stays meaningful; final
    fallback is 1.0 (then dx_norm is just raw px, never a divide-by-zero)."""
    if frame_shape is not None:
        try:
            w = float(frame_shape[1])
            if math.isfinite(w) and w > 0:
                return w / 8.0
        except (TypeError, IndexError, ValueError):
            pass
    # No frame width: approximate from the track's own horizontal extent. The
    # ball sweeps on the order of a rim-width-plus near the hoop, so half the
    # observed x-range is a defensible, scale-free proxy.
    if xs is not None:
        a = np.asarray(xs, dtype=float)
        if a.size >= 2:
            spread = float(np.max(a) - np.min(a))
            if math.isfinite(spread) and spread > 1e-6:
                return max(spread / 2.0, 1.0)
    return 1.0


def _descending_near_rim(pts, rim_y):
    """Index (into pts) of the ball sample on the DESCENDING branch closest to
    the rim plane.

    The apex is the minimum image-y (highest point). We look at samples at/after
    that apex -- where the ball is falling, y increasing -- and pick the one
    whose y is nearest rim_y. Falls back to the globally-nearest-y sample if the
    post-apex slice is empty (e.g. only two points). Returns None if pts is empty.
    """
    n = len(pts)
    if n == 0:
        return None
    ys = np.asarray([p[2] for p in pts], dtype=float)
    apex_i = int(np.argmin(ys))  # highest point = smallest image-y
    # Candidate samples: apex onward (the descending branch). Keep apex itself so
    # a 2-point track still yields a candidate.
    cand = list(range(apex_i, n))
    if len(cand) < 1:
        cand = list(range(n))
    cand_arr = np.asarray(cand, dtype=int)
    diffs = np.abs(ys[cand_arr] - float(rim_y))
    return int(cand_arr[int(np.argmin(diffs))])


def _reached_rim_band(pts, near_i, rim_y):
    """Best-effort depth read in IMAGE space (low confidence by construction).

    Did the descending ball actually get DOWN to the rim plane?
      * If the nearest descending sample sits clearly ABOVE the rim line (its y
        is well less than rim_y) the shot fell short of even reaching the plane
        in-frame -> lean "short".
      * If samples continue well BELOW the rim line, the ball carried past the
        plane -> lean "long".
      * Otherwise it arrived about at the plane -> inconclusive depth.

    Returns one of "short" / "long" / None. This is the one axis a single 2D
    camera genuinely cannot resolve, so callers must keep its confidence low.
    """
    ys = np.asarray([p[2] for p in pts], dtype=float)
    near_y = float(ys[near_i])
    # Band tolerance scaled to the vertical spread of the track (perspective-safe-ish).
    vspread = float(np.max(ys) - np.min(ys)) if ys.size >= 2 else 0.0
    band = max(vspread * 0.10, 8.0)  # ~10% of flight height, min 8px

    below = ys[ys > float(rim_y) + band]
    if near_y < float(rim_y) - band and below.size == 0:
        # Closest we ever got was above the rim and nothing went below it.
        return "short"
    if below.size >= 1 and float(np.max(below)) > float(rim_y) + 2.0 * band:
        # The ball clearly continued past the rim plane.
        return "long"
    return None


def classify_miss(points, rim_center, frame_shape=None, result="miss"):
    """Diagnose WHERE one shot missed.

    points: list of (frame_idx:int, x:float, y:float) -- the tracked ball
            centers for ONE shot, chronological, gaps allowed (image coords, y
            grows DOWN). Same format arc.analyze_arc consumes.
    rim_center: (x, y) pixel of the rim center, or None.
    frame_shape: (h, w) or (h, w, c), or None.
    result: "make" or "miss".

    Returns:
      {
        "ok": bool,             # a direction COULD be assessed (or it's a make)
        "dir": "left"|"right"|"short"|"long"|"in_out"|None,
        "dx_norm": float|None,  # (ball_x - rim_x) / rim-width proxy at rim plane
                                #   -ve = left of rim, +ve = right
        "confidence": float,    # 0..1; LOWER for short/long than for left/right
        "note": str,            # short human note (carries the 2D caveat)
      }

    Method: take the descending-branch ball sample nearest the rim plane, form
    dx = ball_x - rim_x, normalize by rw = frame_w/8 (arc's rim-width proxy).
    |dx_norm| >= ~0.5 -> left/right (reliable). |dx_norm| < ~0.4 with the ball
    still dropping through the rim band -> in_out (rattle). Otherwise fall back to
    a low-confidence 2D depth proxy for short/long. A make returns dir=None, ok=True.

    HONESTY: left/right is camera-robust; short/long is a 2D guess and always
    carries lower confidence (single-camera depth is unrecoverable) -- see module
    docstring.
    """
    base = {"ok": False, "dir": None, "dx_norm": None,
            "confidence": 0.0, "note": ""}

    # A make has no miss direction -- report cleanly and bail.
    if result == "make":
        base.update({"ok": True, "note": "made shot - no miss direction"})
        return base

    pts = _clean_points(points)
    if rim_center is None:
        base["note"] = "no rim center - cannot locate the miss"
        return base
    if len(pts) < 2:
        base["note"] = "not enough tracked points to locate the miss"
        return base

    try:
        rim_x = float(rim_center[0]); rim_y = float(rim_center[1])
    except (TypeError, IndexError, ValueError):
        base["note"] = "malformed rim center"
        return base
    if not (math.isfinite(rim_x) and math.isfinite(rim_y)):
        base["note"] = "non-finite rim center"
        return base

    xs = [p[1] for p in pts]
    rw = _rim_width(frame_shape, xs)

    near_i = _descending_near_rim(pts, rim_y)
    if near_i is None:
        base["note"] = "could not find a descending ball sample"
        return base

    ball_x = float(pts[near_i][1])
    dx = ball_x - rim_x
    dx_norm = dx / rw
    base["dx_norm"] = round(dx_norm, 3)

    adx = abs(dx_norm)

    # --- RELIABLE axis: left / right ---------------------------------------- #
    if adx >= SIDE_T:
        side = "left" if dx_norm < 0 else "right"
        # Confidence grows with how far off-line it is, capped; this axis is
        # camera-robust so it earns the high band.
        conf = round(min(0.9, 0.6 + 0.3 * min(1.0, (adx - SIDE_T) / SIDE_T)), 3)
        base.update({
            "ok": True, "dir": side, "confidence": conf,
            "note": f"missed {side}: ball {adx:.2f} rim-widths off-line "
                    f"(horizontal offset reads reliably from a fixed camera).",
        })
        return base

    # --- on-line: rattle (in_out) or a depth (short/long) miss -------------- #
    depth = _reached_rim_band(pts, near_i, rim_y)

    if adx < ONLINE_T:
        ys = np.asarray([p[2] for p in pts], dtype=float)
        vspread = float(np.max(ys) - np.min(ys)) if ys.size >= 2 else 0.0
        band = max(vspread * 0.10, 8.0)
        near_y = float(ys[near_i])
        through_band = abs(near_y - rim_y) <= band
        continued = bool(np.any(ys > rim_y + band))  # tracked past the plane
        if through_band and continued and depth != "long":
            # On-line, reached the rim band, kept dropping -> rattled in and out.
            base.update({
                "ok": True, "dir": "in_out", "confidence": 0.45,
                "note": "in-and-out: arrived on-line and rattled through the rim "
                        "band (rattle inferred from the 2D track, not certain).",
            })
            return base

    # --- BEST-EFFORT depth: short / long (LOW confidence) ------------------- #
    if depth is not None:
        base.update({
            "ok": True, "dir": depth, "confidence": 0.3,
            "note": f"likely {depth} (LOW confidence: depth is estimated from a "
                    f"2D image proxy -- a single camera can't truly see depth).",
        })
        return base

    # On-line but depth inconclusive: report the offset, no direction.
    base.update({
        "ok": True, "dir": None, "confidence": 0.2,
        "note": "miss arrived near on-line; direction inconclusive in 2D "
                "(no clear side or depth signal).",
    })
    return base


# --- aggregate helpers ------------------------------------------------------- #
def _made(s):
    """make? -- mirrors consistency._made (made flag OR result=='make')."""
    if not isinstance(s, dict):
        return False
    return bool(s.get("made")) or s.get("result") == "make"


def _miss_dir(s):
    """The classified miss direction for a shot, or None. Only trusts a miss
    dict that is present, ok, and carries a real direction."""
    if not isinstance(s, dict):
        return None
    m = s.get("miss")
    if not (isinstance(m, dict) and m.get("ok")):
        return None
    d = m.get("dir")
    return d if d in DIRECTIONS else None


def miss_breakdown(shots):
    """Tally a session's misses by direction and surface the dominant bias.

    shots: list of shot dicts; a shot MAY carry shot["miss"] = a classify_miss()
           result, and has shot["result"]/"made". Only shots with an ok miss dict
           whose dir is a real direction are counted.

    Returns:
      {
        "enough": bool,                  # >=4 classified misses
        "n_misses": int,                 # classified misses counted
        "dist": {left,right,short,long,in_out: int},
        "pct":  {same keys: 0..100},     # share of classified misses
        "dominant": dir|None,            # most common direction (None if tie-empty)
        "note": str,
      }
    enough=False (with the partial tally still filled in) if <4 classified misses.
    """
    dist = {d: 0 for d in DIRECTIONS}
    shot_list = shots if isinstance(shots, (list, tuple)) else []
    for s in shot_list:
        d = _miss_dir(s)
        if d is not None:
            dist[d] += 1

    n = sum(dist.values())
    pct = {d: (round(100.0 * dist[d] / n) if n else 0) for d in DIRECTIONS}

    if n == 0:
        dominant = None
    else:
        # Highest count; deterministic tie-break by DIRECTIONS order.
        dominant = max(DIRECTIONS, key=lambda d: (dist[d], -DIRECTIONS.index(d)))
        if dist[dominant] == 0:
            dominant = None

    if n < MIN_MISSES:
        note = f"only {n} classified miss(es) - need {MIN_MISSES} for a reliable read"
        return {"enough": False, "n_misses": n, "dist": dist, "pct": pct,
                "dominant": dominant, "note": note}

    if dominant is not None:
        share = pct[dominant]
        note = (f"{dominant} is your most common miss ({dist[dominant]}/{n}, "
                f"{share}%).")
        # Side misses are the trustworthy ones; flag when depth dominates.
        if dominant in ("short", "long"):
            note += " Depth is a 2D estimate, so treat short/long as a softer signal."
    else:
        note = "no dominant miss direction"

    return {"enough": True, "n_misses": n, "dist": dist, "pct": pct,
            "dominant": dominant, "note": note}


def _form_val(s, key):
    """A finite numeric form metric for one shot, or None. Skips bools (so the
    bool `follow_through` never sneaks in as 0/1)."""
    if not isinstance(s, dict):
        return None
    f = s.get("form")
    if not isinstance(f, dict):
        return None
    v = f.get(key)
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)) and math.isfinite(v):
        return float(v)
    return None


# (direction, metric) -> concrete fix. Falls back to GENERIC_FIX[metric] then a
# bland default. Phrased as a coach would: name the link, give the cue.
FIX = {
    ("short", "knee_bend"):
        "Your short misses line up with less knee bend - load your legs more and "
        "drive up through the shot so the ball isn't all arm.",
    ("short", "release_angle"):
        "Short misses with a flatter release - lift the arc; aim to drop the ball "
        "INTO the rim, not at the front of it.",
    ("short", "follow_through_deg"):
        "Short misses track with a short follow-through - snap the wrist and hold "
        "it; let the legs finish the shot.",
    ("short", "release_height_ratio"):
        "Short misses with a low release - get fully up and release at the top so "
        "you're not pushing the ball flat.",
    ("long", "knee_bend"):
        "Your long misses come with extra leg dip - you're overpowering it; ease "
        "the dip and let the touch carry it.",
    ("long", "release_angle"):
        "Long misses with a steeper release - you're heaving it; settle the arc "
        "and trust a softer touch.",
    ("long", "follow_through_deg"):
        "Long misses with an exaggerated follow-through - dial back the push and "
        "keep the same easy snap every time.",
    ("left", "elbow_angle"):
        "Your left misses line up with a flying elbow - tuck it under the ball so "
        "the shot pushes straight, not across.",
    ("left", "lean_deg"):
        "Left misses track with leaning off-balance - square your feet and stay "
        "vertical so you're not drifting the ball left.",
    ("left", "symmetry_deg"):
        "Left misses with off shoulders - square them to the rim on the catch so "
        "the release goes straight.",
    ("right", "elbow_angle"):
        "Your right misses line up with a flying elbow - tuck it under the ball so "
        "the shot pushes straight, not across.",
    ("right", "lean_deg"):
        "Right misses track with leaning off-balance - square your feet and stay "
        "vertical so you're not drifting the ball right.",
    ("right", "symmetry_deg"):
        "Right misses with off shoulders - square them to the rim on the catch so "
        "the release goes straight.",
    ("in_out", "release_angle"):
        "In-and-outs with an inconsistent release angle - groove one arc so the "
        "ball drops soft instead of rattling.",
    ("in_out", "follow_through_deg"):
        "In-and-outs track with a clipped follow-through - hold the snap so the "
        "ball carries soft over the front rim.",
}

# Per-metric fallback fix (direction-agnostic), keyed off METRIC_META.
GENERIC_FIX = {
    "elbow_angle": "Tuck the elbow under the ball and snap it fully so the shot pushes straight.",
    "release_angle": "Groove a consistent release arc (aim ~50-55 deg) so the ball drops in soft.",
    "knee_bend": "Match the same leg dip every rep and drive up through the shot.",
    "knee_angle": "Shoot from the legs - same knee drive every time, not all arm.",
    "lean_deg": "Stay balanced and vertical on the release so you don't drift the ball off-line.",
    "release_height_ratio": "Get fully up and release at a consistent high point.",
    "set_point_ratio": "Bring the ball to the same set point on every catch.",
    "symmetry_deg": "Square your shoulders to the rim before you go up.",
    "follow_through_deg": "Hold a full follow-through - cookie-jar finish - until the ball hits.",
}
DEFAULT_FIX = ("Groove this one cue with slow form shots until it repeats, then "
               "take it to game speed.")


def _fix_for(direction, metric):
    return FIX.get((direction, metric)) or GENERIC_FIX.get(metric) or DEFAULT_FIX


def miss_cause(shots):
    """Correlate the DOMINANT miss direction with the form metric that most
    separates those misses from makes, and hand back a concrete fix.

    shots: list of shot dicts; each may carry shot["miss"] (a classify_miss
           result), shot["form"] (metric dict), and shot["result"]/"made".

    Method: pick the dominant directional miss (via miss_breakdown). Split each
    METRIC_META metric's values into {makes} vs {misses in that direction}; for
    each metric present in BOTH groups compute mean per group and a normalized
    gap = |mean_miss - mean_make| / tol (tol from METRIC_META). The metric with
    the largest normalized gap is the likely cause; (direction, metric) maps to a
    fix string.

    Returns:
      {
        "enough": bool,
        "headline": str,
        "findings": [ {"dir","metric","label","mean_make","mean_miss",
                       "norm_gap","fix"}, ... ],   # sorted, biggest gap first
        "note": str,
      }
    enough=False if there aren't >=4 makes AND >=4 dominant-direction misses that
    carry form data.
    """
    shot_list = shots if isinstance(shots, (list, tuple)) else []

    bd = miss_breakdown(shot_list)
    dominant = bd.get("dominant")
    if not bd.get("enough") or dominant is None:
        return {"enough": False, "headline": "", "findings": [],
                "note": bd.get("note", "not enough classified misses to find a cause")}

    makes = [s for s in shot_list if _made(s)]
    dir_misses = [s for s in shot_list if _miss_dir(s) == dominant]

    findings = []
    n_make_form = 0
    n_miss_form = 0
    for key, meta in METRIC_META.items():
        make_vals = [v for v in (_form_val(s, key) for s in makes) if v is not None]
        miss_vals = [v for v in (_form_val(s, key) for s in dir_misses) if v is not None]
        n_make_form = max(n_make_form, len(make_vals))
        n_miss_form = max(n_miss_form, len(miss_vals))
        if len(make_vals) < 2 or len(miss_vals) < 2:
            continue
        mean_make = float(np.mean(make_vals))
        mean_miss = float(np.mean(miss_vals))
        tol = float(meta.get("tol", 1.0)) or 1.0
        norm_gap = abs(mean_miss - mean_make) / tol
        findings.append({
            "dir": dominant,
            "metric": key,
            "label": meta["label"],
            "mean_make": round(mean_make, 2),
            "mean_miss": round(mean_miss, 2),
            "norm_gap": round(norm_gap, 3),
            "fix": _fix_for(dominant, key),
        })

    # Need a real sample on BOTH sides AND at least one comparable metric.
    if len(makes) < MIN_GROUP or len(dir_misses) < MIN_GROUP or not findings \
            or n_make_form < 2 or n_miss_form < 2:
        return {"enough": False, "headline": "", "findings": [],
                "note": (f"need >={MIN_GROUP} makes and >={MIN_GROUP} '{dominant}' "
                         f"misses with form data to pin a cause")}

    findings.sort(key=lambda r: r["norm_gap"], reverse=True)
    top = findings[0]
    headline = (f"Your {dominant} misses line up most with "
                f"{top['label'].lower()} (makes ~{top['mean_make']} vs "
                f"misses ~{top['mean_miss']}). {top['fix']}")
    note = (f"Compared {len(makes)} makes against {len(dir_misses)} '{dominant}' "
            f"misses across {len(findings)} form metric(s).")
    if dominant in ("short", "long"):
        note += " Depth direction is a 2D estimate - treat as a lead, not proof."

    return {"enough": True, "headline": headline, "findings": findings, "note": note}


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
    shots = obj["shots"]
    print(f"session {sess[0]['id']}: {len(shots)} shots")
    print(_json.dumps(miss_breakdown(shots), indent=2, default=str))
    print(_json.dumps(miss_cause(shots), indent=2, default=str))

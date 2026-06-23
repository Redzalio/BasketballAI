"""Ball-flight arc analytics: apex height + entry angle from a tracked shot.

Consumes the per-frame ball centers of ONE shot -- a list of
(frame_idx, x, y) in IMAGE PIXEL coordinates (y grows DOWNWARD) -- and fits a
parabola y = a*f^2 + b*f + c over (frame_idx, y) to recover the apex (the
highest point of the flight, i.e. the MINIMUM image-y) and the descent angle as
the ball reaches the rim.

Two real-world messes are handled on purpose:
  * GAPS -- the ball routinely leaves frame (over the backboard, behind a head),
    so frame indices are not contiguous. The largest gap is reported, the fit
    still runs over whatever points remain, and an apex that lands in a gap /
    above the top of frame is flagged `estimated` / `off_frame_top` rather than
    trusted as observed.
  * OUTLIERS -- a rebound, a second ball, or a blown detection drops a stray
    point far off the arc. We fit, then iteratively reject points whose residual
    exceeds ~2.5*std and refit, so one bad point can't drag the parabola.

HONESTY ABOUT ENTRY ANGLE: the entry angle is measured in IMAGE space
(degrees(atan2(|dy/df|, |dx/df|)) on the descending branch). It only equals the
true real-world entry angle for a clean SIDE-ON camera; any other angle bakes in
perspective + foreshortening. So the robust, camera-agnostic use of this number
is the SAME thing consistency.py leans on everywhere else: shot-to-shot
CONSISTENCY of the value from a FIXED camera, not its absolute degrees.

Pure numpy + stdlib. No torch/cv2. Every function returns a dict and never
raises; bad input yields an {"ok": False} / {"enough": False} dict.
"""
import math

import numpy as np

# Residual rejection: drop points whose |residual| > REJECT_SIGMA * std, refit.
REJECT_SIGMA = 2.5
MAX_REJECT_PASSES = 5
MIN_POINTS = 3

# Consistency tolerances (the std that maps to a 0 sub-score), mirroring
# consistency.METRIC_META. Entry angle is in degrees; peak height ratio is
# unitless (apex height / rim-width proxy); the px fallback is raw pixels.
ENTRY_ANGLE_TOL = 8.0
PEAK_RATIO_TOL = 0.5
PEAK_PX_TOL = 40.0


def _clean_points(points):
    """Coerce `points` to a sorted list of (f, x, y) floats, dropping anything
    non-finite or malformed. Returns [] on bad input -- never raises."""
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


def _largest_gap(fs):
    """Largest jump between consecutive (sorted) frame indices. A contiguous
    run gives 1 (frame i -> i+1); a 5-frame hole gives 5."""
    if len(fs) < 2:
        return 0
    diffs = np.diff(np.asarray(fs, dtype=float))
    return int(round(float(np.max(diffs))))


def _frame_observed(sorted_fs, f_star, window=2.0):
    """Was the apex frame actually tracked, or does it sit in a gap?

    True when some kept frame lies within `window` of f_star (the peak was
    seen); False when the nearest kept frame is farther away (the ball left
    frame over the apex, so the peak is interpolated)."""
    fs = np.asarray(sorted_fs, dtype=float)
    if fs.size == 0:
        return False
    return bool(np.min(np.abs(fs - f_star)) <= window)


def _rim_width(rim_center, frame_shape):
    """Best-effort pixel scale proxy so peak height can be normalized. We do not
    get a rim radius here, so approximate rim width as a fraction of frame width
    (a regulation rim is ~1/8 of a typical broadcast/phone frame). Returns None
    when there's nothing to anchor to."""
    if frame_shape is None:
        return None
    try:
        w = float(frame_shape[1])
    except (TypeError, IndexError, ValueError):
        return None
    if not math.isfinite(w) or w <= 0:
        return None
    return w / 8.0


def _fit_with_rejection(fs, ys):
    """Fit y = a f^2 + b f + c, iteratively rejecting residual outliers.

    Returns (coeffs, keep_mask, rms) where coeffs is the final degree-2
    polynomial (numpy poly, highest power first) or None if a fit was
    impossible. keep_mask marks the inlier points used by the final fit.
    """
    f = np.asarray(fs, dtype=float)
    y = np.asarray(ys, dtype=float)
    keep = np.ones(f.shape, dtype=bool)

    coeffs = None
    rms = 0.0
    for _ in range(MAX_REJECT_PASSES):
        if int(keep.sum()) < MIN_POINTS:
            break
        # A degree-2 fit needs >=3 distinct frame indices; guard it.
        if np.unique(f[keep]).size < 3:
            deg = 1 if np.unique(f[keep]).size >= 2 else 0
        else:
            deg = 2
        try:
            with np.errstate(all="ignore"):
                c = np.polyfit(f[keep], y[keep], deg)
        except (np.linalg.LinAlgError, ValueError, TypeError):
            break
        # Promote to a length-3 (degree-2) coefficient vector for a stable apex.
        c = np.atleast_1d(c).astype(float)
        if c.size < 3:
            c = np.concatenate([np.zeros(3 - c.size), c])
        resid = y - np.polyval(c, f)
        in_resid = resid[keep]
        rms = float(np.sqrt(np.mean(in_resid ** 2))) if in_resid.size else 0.0
        std = float(np.std(in_resid)) if in_resid.size else 0.0
        coeffs = c
        if std <= 1e-9:
            break
        thresh = REJECT_SIGMA * std
        new_keep = keep & (np.abs(resid) <= thresh)
        # Stop if nothing new is rejected or we'd fall below the minimum.
        if int(new_keep.sum()) == int(keep.sum()) or int(new_keep.sum()) < MIN_POINTS:
            break
        keep = new_keep

    return coeffs, keep, rms


def _entry_angle_deg(fs, xs, ys, coeffs, rim_center):
    """Descent angle (image space, 0..90) on the DOWNWARD branch near the rim.

    Prefer the analytic parabola slope dy/df = 2 a f + b at the frame closest to
    the rim (or the last frame); pair it with a local dx/df from the observed
    points. atan2(|dy|, |dx|): ~90 = dropping almost straight down, ~0 = skimming
    in nearly flat.
    """
    f = np.asarray(fs, dtype=float)
    x = np.asarray(xs, dtype=float)
    if f.size < 2:
        return None

    a, b = float(coeffs[0]), float(coeffs[1])

    # Pick the reference frame: nearest the rim plane if we know it, else the
    # last observed frame (the ball is descending toward the hoop by then).
    ref_idx = int(np.argmax(f))
    if rim_center is not None:
        try:
            rim_x = float(rim_center[0])
            ref_idx = int(np.argmin(np.abs(x - rim_x)))
        except (TypeError, IndexError, ValueError):
            ref_idx = int(np.argmax(f))
    f_ref = f[ref_idx]

    # dy/df from the fitted parabola (robust to per-frame jitter).
    dydf = 2.0 * a * f_ref + b

    # dx/df from a local finite difference across neighboring observed points.
    order = np.argsort(f)
    fo, xo = f[order], x[order]
    pos = int(np.searchsorted(fo, f_ref))
    lo = max(0, min(pos - 1, fo.size - 2))
    hi = lo + 1
    df = fo[hi] - fo[lo]
    dxdf = (xo[hi] - xo[lo]) / df if df != 0 else 0.0

    ang = math.degrees(math.atan2(abs(dydf), abs(dxdf)))
    # Numerical clamp into the reportable 0..90 band.
    return round(max(0.0, min(90.0, ang)), 1)


def analyze_arc(points, rim_center=None, frame_shape=None):
    """Fit a ball-flight arc and derive apex height + entry angle.

    points: list of (frame_idx:int, x:float, y:float) -- tracked ball centers
            during ONE shot, chronological, possibly with gaps (missing frame
            indices) where the ball left frame.
    rim_center: (x, y) pixel of the rim center, or None.
    frame_shape: (h, w) or (h, w, c), or None.

    Returns a dict:
      {
        "ok": bool,                 # a usable arc was fit
        "n_points": int,
        "gap_frames": int,          # largest frame-index gap between points
        "off_frame_top": bool,      # apex extrapolated above frame top, or
                                    #   the points hug the top edge
        "estimated": bool,          # apex/height extrapolated, not observed
        "apex_y": float|None,       # image y of the fitted apex (highest point)
        "peak_height_px": float|None,    # apex above rim (rim_y - apex_y), px
        "peak_height_ratio": float|None, # peak_height_px / rim-width proxy
        "entry_angle_deg": float|None,   # image-space descent angle, 0..90
        "quality": float,           # 0..1 confidence
        "note": str,                # short human note
      }

    Method: numpy.polyfit(deg=2) over (frame_idx, y), iteratively rejecting
    residual outliers (>~2.5*std) so a rebound / second ball doesn't corrupt the
    fit. Apex frame f* = -b/(2a); apex_y = parabola(f*). In image coords y points
    DOWN, so a>0 (opens upward) is the valid arc -- the vertex is then the
    minimum-y, i.e. the highest point. If f* falls outside the observed frames or
    apex_y < 0 the apex is flagged `estimated` / `off_frame_top`.

    NOTE ON ENTRY ANGLE: it is an IMAGE-space angle and is only the true
    real-world entry angle from a clean side-on view; otherwise perspective
    distorts it. Its dependable use is shot-to-shot CONSISTENCY from a fixed
    camera (see arc_consistency), not the absolute degrees.
    """
    pts = _clean_points(points)
    n = len(pts)
    base = {
        "ok": False, "n_points": n, "gap_frames": 0,
        "off_frame_top": False, "estimated": False,
        "apex_y": None, "peak_height_px": None, "peak_height_ratio": None,
        "entry_angle_deg": None, "quality": 0.0, "note": "",
    }

    if n < MIN_POINTS:
        base["note"] = "not enough tracked points for an arc"
        return base

    fs = [p[0] for p in pts]
    xs = [p[1] for p in pts]
    ys = [p[2] for p in pts]
    gap = _largest_gap(fs)
    base["gap_frames"] = gap

    coeffs, keep, rms = _fit_with_rejection(fs, ys)
    if coeffs is None or int(np.sum(keep)) < MIN_POINTS:
        base["note"] = "could not fit a parabola to the points"
        return base

    a, b, c = float(coeffs[0]), float(coeffs[1]), float(coeffs[2])
    f_keep = np.sort(np.asarray(fs, dtype=float)[keep])
    f_min, f_max = float(f_keep[0]), float(f_keep[-1])

    estimated = False
    off_frame_top = False
    notes = []

    # --- apex --------------------------------------------------------------- #
    if a > 1e-9:
        # Valid upward-opening arc: vertex is the highest point (min image-y).
        f_star = -b / (2.0 * a)
        apex_y = a * f_star * f_star + b * f_star + c
        if f_star < f_min or f_star > f_max:
            estimated = True
            notes.append("apex falls outside the tracked frames - height estimated")
        elif not _frame_observed(f_keep, f_star):
            # f* lands inside a GAP between kept frames (ball left frame over the
            # apex): the peak is interpolated, not seen -> flag it.
            estimated = True
            notes.append("apex falls in a tracking gap - height estimated")
    else:
        # a<=0: not a clean upward arc (too few/flat points, or noise). Fall back
        # to the highest OBSERVED point and flag the apex as estimated.
        f_star = f_min
        apex_y = float(np.min(np.asarray(ys, dtype=float)[keep]))
        estimated = True
        notes.append("no clean upward arc - apex taken from highest tracked point")

    # Apex extrapolated above the top of frame, or points hugging the top edge.
    if apex_y < 0:
        off_frame_top = True
        estimated = True
        notes.append("apex left the frame (above the top) - height estimated")
    else:
        top_y = float(np.min(np.asarray(ys, dtype=float)[keep]))
        if top_y <= 2.0:
            off_frame_top = True
            notes.append("ball hugs the top edge of frame - apex may be clipped")

    apex_y_r = round(float(apex_y), 1)

    # --- peak height vs rim ------------------------------------------------- #
    peak_px = None
    peak_ratio = None
    if rim_center is not None:
        try:
            rim_y = float(rim_center[1])
        except (TypeError, IndexError, ValueError):
            rim_y = None
        if rim_y is not None and math.isfinite(rim_y):
            peak_px = round(rim_y - apex_y, 1)  # +ve: apex sits above the rim
            scale = _rim_width(rim_center, frame_shape)
            if scale:
                peak_ratio = round((rim_y - apex_y) / scale, 3)

    # --- entry angle -------------------------------------------------------- #
    entry = _entry_angle_deg(fs, xs, ys, coeffs, rim_center)

    # --- quality ------------------------------------------------------------ #
    # More points help; large gaps and high residual RMS hurt. Each factor is a
    # 0..1 multiplier (mirrors the "shrinks with X" spec).
    n_factor = min(1.0, n / 10.0)                       # ~10 pts -> saturated
    gap_factor = 1.0 / (1.0 + max(0, gap - 1) / 6.0)    # gap 1 -> 1.0, 7 -> 0.5
    rms_factor = 1.0 / (1.0 + rms / 8.0)                # rms 0 -> 1.0, 8px -> 0.5
    rejected = int(np.sum(~keep))
    rej_factor = 1.0 / (1.0 + rejected / 4.0)           # a couple outliers ok
    quality = n_factor * gap_factor * rms_factor * rej_factor
    if estimated:
        quality *= 0.85
    quality = round(max(0.0, min(1.0, quality)), 3)

    if rejected:
        notes.append(f"{rejected} outlier point(s) rejected")
    if gap >= 4:
        notes.append(f"largest frame gap {gap}")
    if not notes:
        notes.append("clean arc")

    base.update({
        "ok": True,
        "off_frame_top": off_frame_top,
        "estimated": estimated,
        "apex_y": apex_y_r,
        "peak_height_px": peak_px,
        "peak_height_ratio": peak_ratio,
        "entry_angle_deg": entry,
        "quality": quality,
        "note": "; ".join(notes),
    })
    return base


def _arc_vals(shots, key):
    """Pull a finite numeric `key` out of each shot's ok arc dict."""
    out = []
    for s in shots:
        if not isinstance(s, dict):
            continue
        arc = s.get("arc")
        if not (isinstance(arc, dict) and arc.get("ok")):
            continue
        v = arc.get(key)
        if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v):
            out.append(float(v))
    return out


def _sub_score(vals, tol):
    """mean/std + a 0..100 consistency sub-score: round(100*(1 - std/tol)),
    clamped to [0,100]. Mirrors consistency.metric_stats."""
    mean = float(np.mean(vals))
    sd = float(np.std(vals))  # population std, like statistics.pstdev
    score = max(0.0, min(100.0, 100.0 * (1.0 - sd / tol)))
    return {"mean": round(mean, 2), "std": round(sd, 2), "consistency": round(score)}


def arc_consistency(shots):
    """Score how repeatable a player's arc is across shots.

    shots: list of shot dicts; a shot may carry shot["arc"] = an analyze_arc()
           result. Only shots whose arc dict is present and "ok" are used.

    Returns:
      {"enough": bool, "n": int,
       "entry_angle": {"mean":..,"std":..,"consistency":0..100}|None,
       "peak_height": {"mean":..,"std":..,"consistency":0..100}|None,
       "overall": 0..100}

    Each consistency sub-score = round(100*(1 - std/tol)) clamped to [0,100],
    with tol~8.0 for entry_angle_deg and ~0.5 for peak_height_ratio (falling back
    to peak_height_px with a px tol when the ratio is absent). <3 usable arcs ->
    {"enough": False, "n": n}.
    """
    usable = [s for s in shots if isinstance(s, dict)
              and isinstance(s.get("arc"), dict) and s["arc"].get("ok")] \
        if isinstance(shots, (list, tuple)) else []
    n = len(usable)
    if n < 3:
        return {"enough": False, "n": n}

    entry = None
    angle_vals = _arc_vals(usable, "entry_angle_deg")
    if len(angle_vals) >= 3:
        entry = _sub_score(angle_vals, ENTRY_ANGLE_TOL)

    peak = None
    ratio_vals = _arc_vals(usable, "peak_height_ratio")
    if len(ratio_vals) >= 3:
        peak = _sub_score(ratio_vals, PEAK_RATIO_TOL)
    else:
        px_vals = _arc_vals(usable, "peak_height_px")
        if len(px_vals) >= 3:
            peak = _sub_score(px_vals, PEAK_PX_TOL)

    subs = [d["consistency"] for d in (entry, peak) if d is not None]
    overall = round(sum(subs) / len(subs)) if subs else 0

    return {
        "enough": True, "n": n,
        "entry_angle": entry,
        "peak_height": peak,
        "overall": overall,
    }

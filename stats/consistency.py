"""Shooting-form consistency + improvement engine.

Consumes a session's shots (each: {made:bool/result, zone:str, t:float, form:{metric:value}})
and produces: per-metric stats + consistency sub-scores, an overall consistency
score (0-100), the biggest inconsistency, makes-vs-misses form deltas, in-session
drift, and a prioritized "what to work on next" with a drill.

Consistency = shot-to-shot variance, which is robust to the 2D single-camera
limitation: a fixed angle's systematic error cancels out of the variation.

The engine is generic over metric keys via METRIC_META, so new metrics added to
the pose layer (release_angle, knee_bend, set_point_ratio, symmetry_deg, ...)
light up automatically once they're present in the stored form dicts.
"""
import statistics

# Per-metric metadata. tol = the std (in the metric's unit) that maps to a 0
# consistency sub-score (std=0 -> 100, std>=tol -> 0). good = coaching target range.
METRIC_META = {
    "elbow_angle":          {"label": "Elbow extension",  "unit": "°", "good": (160, 180), "tol": 22},
    "release_angle":        {"label": "Release angle",     "unit": "°", "good": (48, 58),   "tol": 18},
    "knee_bend":            {"label": "Knee bend (dip)",   "unit": "°", "good": (105, 140), "tol": 28},
    "knee_angle":           {"label": "Leg drive (knee)",  "unit": "°", "good": (120, 170), "tol": 28},
    "lean_deg":             {"label": "Balance (lean)",    "unit": "°", "good": (0, 8),     "tol": 14},
    "release_height_ratio": {"label": "Release height",    "unit": "",       "good": (0.9, 1.8), "tol": 0.45},
    "set_point_ratio":      {"label": "Set point",         "unit": "",       "good": (0.6, 1.1), "tol": 0.40},
    "symmetry_deg":         {"label": "Shoulder symmetry", "unit": "°", "good": (0, 6),     "tol": 12},
    "follow_through_deg":   {"label": "Follow-through",     "unit": "°", "good": (150, 180), "tol": 22},
}

# Elite/pro reference ranges per metric (tighter than 'good') — hardcoded coaching targets.
PRO = {
    "elbow_angle": (168, 180),
    "release_angle": (47, 54),
    "knee_bend": (108, 132),
    "knee_angle": (120, 160),
    "lean_deg": (0, 4),
    "release_height_ratio": (1.1, 1.7),
    "set_point_ratio": (0.7, 1.05),
    "symmetry_deg": (0, 3),
    "follow_through_deg": (165, 180),
}

DRILLS = {
    "elbow_angle":   "Form shooting: 2x25 from 5 ft, freeze a fully snapped elbow + follow-through 2s each rep.",
    "release_angle": "Arc work: shoot over a chair or a partner's reach from the elbow; groove a 50-55° release, 50 makes.",
    "knee_bend":     "Slow dip-and-rise: 30 reps matching the SAME knee bend, then 20 live at game speed.",
    "knee_angle":    "Power-dribble into shot: same leg dip every rep, shot comes from the legs - 3x20.",
    "lean_deg":      "Balance shooting: back foot on a floor seam, stay vertical, 40 makes without drifting off-line.",
    "release_height_ratio": "Set-and-hold: pause 1s at your release point before every shot, 3x15, lock the height.",
    "set_point_ratio": "1-motion reps: catch and bring the ball to the SAME set point every time, 50 catch-to-sets.",
    "symmetry_deg":  "Square-up drill: feet & shoulders to the rim on the catch, mirror-check, 40 reps.",
    "follow_through_deg": "Cookie-jar finish: hold the wrist snap until the ball hits the rim, 50 reps.",
}
GENERIC_DRILL = "Pick one cue and groove it: 50 slow form shots holding that exact position, then 25 at game speed."


def _made(s):
    return bool(s.get("made")) or s.get("result") == "make"


def _vals(shots, key):
    out = []
    for s in shots:
        f = s.get("form")
        if isinstance(f, dict) and isinstance(f.get(key), (int, float)) and not isinstance(f.get(key), bool):
            out.append(f[key])
    return out


def metric_stats(shots):
    out = {}
    for key, meta in METRIC_META.items():
        vals = _vals(shots, key)
        if len(vals) < 3:
            continue
        mean = statistics.mean(vals)
        sd = statistics.pstdev(vals)
        score = max(0.0, min(100.0, 100.0 * (1.0 - sd / meta["tol"])))
        pro = PRO.get(key)
        vs_pro = None
        if pro:
            vs_pro = "in" if pro[0] <= mean <= pro[1] else ("low" if mean < pro[0] else "high")
        out[key] = {
            "label": meta["label"], "unit": meta["unit"], "good": list(meta["good"]),
            "mean": round(mean, 2), "std": round(sd, 2), "n": len(vals),
            "consistency": round(score),
            "in_range": meta["good"][0] <= mean <= meta["good"][1],
            "pro": list(pro) if pro else None, "vs_pro": vs_pro,
        }
    return out


def consistency_score(mstats):
    if not mstats:
        return 0
    return round(sum(m["consistency"] for m in mstats.values()) / len(mstats))


def biggest_inconsistency(mstats):
    if not mstats:
        return None
    key = min(mstats, key=lambda k: mstats[k]["consistency"])
    m = mstats[key]
    return {"metric": key, "label": m["label"], "consistency": m["consistency"],
            "std": m["std"], "unit": m["unit"]}


def makes_vs_misses(shots):
    makes = [s for s in shots if _made(s)]
    misses = [s for s in shots if not _made(s)]
    if len(makes) < 3 or len(misses) < 3:
        return {"enough": False}
    rows = []
    for key, meta in METRIC_META.items():
        mv, xv = _vals(makes, key), _vals(misses, key)
        if len(mv) < 2 or len(xv) < 2:
            continue
        mm, xm = statistics.mean(mv), statistics.mean(xv)
        rows.append({"metric": key, "label": meta["label"], "unit": meta["unit"],
                     "make": round(mm, 1), "miss": round(xm, 1),
                     "delta": round(xm - mm, 1), "norm": abs(xm - mm) / meta["tol"]})
    rows.sort(key=lambda r: r["norm"], reverse=True)
    return {"enough": True, "rows": rows, "top": rows[0] if rows else None}


def in_session_drift(shots):
    if len(shots) < 8:
        return {"enough": False}
    half = len(shots) // 2
    early = consistency_score(metric_stats(shots[:half]))
    late = consistency_score(metric_stats(shots[half:]))
    return {"enough": True, "early": early, "late": late, "delta": late - early,
            "verdict": "holds up" if late >= early - 5 else "drops off"}


def what_to_work_on(mstats, mvm):
    """Single highest-leverage focus: a metric that's inconsistent AND (ideally)
    separates makes from misses. Returns {focus, label, why, drill}."""
    if not mstats:
        return None
    mvm_norm = {r["metric"]: r["norm"] for r in mvm.get("rows", [])} if mvm.get("enough") else {}

    def leverage(k):
        inconsistency = (100 - mstats[k]["consistency"]) / 100.0
        return inconsistency + 0.8 * mvm_norm.get(k, 0)

    key = max(mstats, key=leverage)
    m = mstats[key]
    why = (f"Your {m['label'].lower()} swings ±{m['std']}{m['unit']} shot-to-shot "
           f"(consistency {m['consistency']}/100)")
    if mvm.get("top") and mvm["top"]["metric"] == key:
        t = mvm["top"]
        why += (f", and it's {abs(t['delta'])}{t['unit']} off on your misses vs makes "
                f"— it's directly costing you points")
    why += "."
    return {"focus": key, "label": m["label"], "why": why, "drill": DRILLS.get(key, GENERIC_DRILL)}


def session_consistency(shots):
    ms = metric_stats(shots)
    mvm = makes_vs_misses(shots)
    return {
        "consistency_score": consistency_score(ms),
        "metrics": ms,
        "biggest_inconsistency": biggest_inconsistency(ms),
        "makes_vs_misses": mvm,
        "drift": in_session_drift(shots),
        "focus": what_to_work_on(ms, mvm),
        "shots_analyzed": sum(1 for s in shots if isinstance(s.get("form"), dict) and s["form"]),
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
    print(_json.dumps(session_consistency(obj["shots"]), indent=2, default=str))

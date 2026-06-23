"""Coaching analytics: per-session insights + lifetime overview."""
from collections import defaultdict

try:
    from . import db
except ImportError:
    import db


def _streaks(shots):
    longest_make = longest_miss = cur = 0
    cur_kind = None
    for s in shots:
        made = s.get("made") if s.get("made") is not None else (s.get("result") == "make")
        kind = "make" if made else "miss"
        cur = cur + 1 if kind == cur_kind else 1
        cur_kind = kind
        if kind == "make":
            longest_make = max(longest_make, cur)
        else:
            longest_miss = max(longest_miss, cur)
    return {"longest_make": longest_make, "longest_miss": longest_miss}


def _form_summary(shots):
    vals = defaultdict(list)
    ft = []
    for s in shots:
        for k in ("elbow_angle", "knee_angle", "lean_deg"):
            if s.get(k) is not None:
                vals[k].append(s[k])
        if s.get("follow_through") is not None:
            ft.append(1 if s["follow_through"] else 0)
    out = {k: round(sum(v) / len(v), 1) for k, v in vals.items() if v}
    if ft:
        out["follow_through_pct"] = round(100 * sum(ft) / len(ft))
    return out


def _zones(shots):
    z = defaultdict(lambda: {"makes": 0, "attempts": 0})
    for s in shots:
        zone = s.get("zone") or "unknown"
        z[zone]["attempts"] += 1
        if s.get("made") or s.get("result") == "make":
            z[zone]["makes"] += 1
    return {k: {**v, "pct": round(100 * v["makes"] / v["attempts"], 1) if v["attempts"] else 0.0}
            for k, v in z.items()}


def _form_tips(fs):
    tips = []
    if fs.get("elbow_angle") is not None and fs["elbow_angle"] < 150:
        tips.append(f"Elbow averages {fs['elbow_angle']}° at release — extend fully (aim 155–180°) for a truer, softer shot.")
    if fs.get("knee_angle") is not None and fs["knee_angle"] > 150:
        tips.append(f"Knees average {fs['knee_angle']}° — bend more into the dip (110–140°) so power comes from your legs, not your arm.")
    if fs.get("lean_deg") is not None and fs["lean_deg"] > 12:
        tips.append(f"You lean ~{fs['lean_deg']}° — stay vertical and balanced through the release.")
    if fs.get("follow_through_pct", 100) < 70:
        tips.append(f"Follow-through held only {fs.get('follow_through_pct', 0)}% of shots — finish high every time ('reach into the cookie jar').")
    return tips


def session_insights(session_obj):
    shots = session_obj["shots"]
    fs = _form_summary(shots)
    zones = _zones(shots)
    tips = _form_tips(fs)
    qualified = {k: v for k, v in zones.items() if v["attempts"] >= 3}
    if len(qualified) >= 2:
        best = max(qualified.items(), key=lambda kv: kv[1]["pct"])
        worst = min(qualified.items(), key=lambda kv: kv[1]["pct"])
        if best[0] != worst[0]:
            tips.append(f"Strongest from {best[0]} ({best[1]['pct']:.0f}%), weakest from {worst[0]} ({worst[1]['pct']:.0f}%) — add reps from {worst[0]}.")
    if not tips:
        tips.append("Solid, consistent session — keep the same routine and rhythm.")
    return {"form_summary": fs, "zones": zones, "streaks": _streaks(shots), "tips": tips}


def overview_insights(sessions):
    shots = db.all_shots()
    by_zone = _zones(shots)
    fs = _form_summary(shots)
    trend = [{"date": s["date"], "fg_pct": s["fg_pct"], "attempts": s["attempts"]}
             for s in reversed(sessions) if s["attempts"]]

    qualified = [s for s in sessions if s["attempts"] >= 5]
    best_fg = max(qualified, key=lambda s: s["fg_pct"], default=None)
    most_makes = max(sessions, key=lambda s: s["makes"], default=None)
    longest = max((_streaks(db.get_session(s["id"])["shots"])["longest_make"] for s in sessions), default=0)
    pbs = {
        "best_fg_session": ({"id": best_fg["id"], "fg_pct": best_fg["fg_pct"], "attempts": best_fg["attempts"]}
                            if best_fg else None),
        "longest_make_streak": longest,
        "most_makes_session": ({"id": most_makes["id"], "makes": most_makes["makes"]}
                               if most_makes and most_makes["makes"] else None),
    }

    fgs = [s["fg_pct"] for s in sessions if s["attempts"] >= 3]  # newest-first
    hot_cold = ""
    if len(fgs) >= 4:
        overall = sum(fgs) / len(fgs)
        recent = sum(fgs[:3]) / 3
        if recent >= overall + 4:
            hot_cold = f"You're hot — last 3 sessions ({recent:.0f}%) are above your {overall:.0f}% average."
        elif recent <= overall - 4:
            hot_cold = f"Cooling off — last 3 sessions ({recent:.0f}%) are below your {overall:.0f}% average. Back to fundamentals."
        else:
            hot_cold = f"Steady — recent form ({recent:.0f}%) is right around your {overall:.0f}% average."

    tips = _form_tips(fs)
    weak = [k for k, v in by_zone.items() if v["attempts"] >= 5 and v["pct"] < 40]
    if weak:
        tips.append("Lowest-percentage zones: " + ", ".join(weak) + " — target these in practice.")
    if not tips and shots:
        tips.append("Mechanics look consistent — keep logging sessions to surface trends.")

    return {"trend": trend, "by_zone": by_zone, "form_avg": fs,
            "personal_bests": pbs, "tips": tips, "hot_cold": hot_cold}

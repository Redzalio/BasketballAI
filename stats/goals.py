"""Goals + progress: compute progress toward user-set targets."""
try:
    from . import db, insights
except ImportError:
    import db
    import insights

METRIC_LABEL = {
    "fg_pct": "Shooting %",
    "consistency": "Consistency",
    "makes_total": "Total makes",
    "longest_streak": "Longest make streak",
}
SUFFIX = {"fg_pct": "%", "consistency": "/100", "makes_total": "", "longest_streak": ""}


def _current():
    sessions = db.list_sessions()
    makes = sum(s["makes"] for s in sessions)
    attempts = sum(s["attempts"] for s in sessions)
    fg = round(100 * makes / attempts, 1) if attempts else 0.0
    ov = insights.overview_insights(sessions)
    return {
        "fg_pct": fg,
        "makes_total": makes,
        "consistency": ov.get("consistency_score", 0),
        "longest_streak": (ov.get("personal_bests") or {}).get("longest_make_streak", 0),
    }


def goals_with_progress():
    cur = _current()
    out = []
    for g in db.list_goals():
        c = cur.get(g["metric"], 0)
        target = g["target"] or 0
        pct = max(0, min(100, round(100 * c / target))) if target else 0
        achieved = bool(g.get("achieved_at")) or (target and c >= target)
        if achieved and not g.get("achieved_at"):
            db.set_goal_achieved(g["id"])
        out.append({
            "id": g["id"], "metric": g["metric"], "target": g["target"],
            "label": g["label"] or METRIC_LABEL.get(g["metric"], g["metric"]),
            "metric_label": METRIC_LABEL.get(g["metric"], g["metric"]),
            "suffix": SUFFIX.get(g["metric"], ""),
            "current": c, "pct": pct, "achieved": bool(achieved),
        })
    return {"goals": out, "current": cur, "metric_options": METRIC_LABEL}

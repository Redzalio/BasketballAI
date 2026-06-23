"""Practice loop: log a drill, then measure whether the focus metric's
shot-to-shot variance actually tightened in later sessions."""
try:
    from . import db, consistency
except ImportError:
    import db
    import consistency


def _latest_metric_std(metric):
    for s in db.list_sessions():        # newest first
        if s["attempts"]:
            ms = consistency.metric_stats(db.get_session(s["id"])["shots"])
            return ms[metric]["std"] if metric in ms else None
    return None


def log_drill(focus_metric, drill, note=""):
    """Log a drill done, snapshotting the focus metric's current variance as the baseline."""
    baseline = _latest_metric_std(focus_metric)
    return db.log_practice(focus_metric, drill, note, baseline)


def practice_with_progress():
    sessions = db.list_sessions()       # newest first
    out = []
    for e in db.list_practice():
        metric, baseline, logged = e["focus_metric"], e["baseline_std"], e["logged_at"]
        after = []
        for s in sessions:
            if s.get("date") and logged and s["date"] > logged and s["attempts"]:
                ms = consistency.metric_stats(db.get_session(s["id"])["shots"])
                if metric in ms:
                    after.append(ms[metric]["std"])
        after_std = round(sum(after) / len(after), 1) if after else None
        improved = after_std is not None and baseline is not None and after_std < baseline
        out.append({
            "id": e["id"], "logged_at": logged, "focus_metric": metric,
            "label": consistency.METRIC_META.get(metric, {}).get("label", metric),
            "drill": e["drill"], "note": e["note"],
            "baseline_std": baseline, "after_std": after_std,
            "delta": (round(after_std - baseline, 1) if (after_std is not None and baseline is not None) else None),
            "improved": improved, "sessions_since": len(after),
        })
    return {"practice": out}

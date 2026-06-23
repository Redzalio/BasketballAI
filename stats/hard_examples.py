"""Hard-example capture.

When the user corrects a shot (flip / delete / add), bank the video frames around
that moment from the session's source clip into training/hard_examples/. Those are
exactly the moments the detector/tracker got wrong (or nearly missed), recorded from
the user's OWN court — the raw material for a later, OFFLINE detector fine-tune.

This is pure data collection. It NEVER touches live detection or make/miss, and it
never raises: on any problem (live session with no video, missing file, no OpenCV)
it simply returns None and the correction still goes through.
"""
from pathlib import Path
import json

try:
    import cv2
except Exception:
    cv2 = None

ROOT = Path(__file__).resolve().parent.parent
HARD_DIR = ROOT / "training" / "hard_examples"
WINDOW = 6     # frames captured on each side of the shot moment
STEP = 2       # sample every STEP-th frame


def capture_for_correction(session, shot_id, shot_t, kind, old_result=None, new_result=None):
    """session: the session dict (needs 'video_path' + 'id'); shot_t: seconds into the clip.
    Extracts frames around shot_t into training/hard_examples/s<sid>_shot<id>/ + meta.json.
    Returns the output dir (str) or None (live session / no video / cv2 missing / failure)."""
    if cv2 is None or not session:
        return None
    vp = session.get("video_path")
    if not vp or shot_t is None or not Path(vp).exists():
        return None
    try:
        cap = cv2.VideoCapture(str(vp))
        if not cap.isOpened():
            return None
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        center = int(float(shot_t) * fps)
        out = HARD_DIR / ("s%s_shot%s" % (session.get("id"), shot_id))
        out.mkdir(parents=True, exist_ok=True)
        saved = []
        start = max(0, center - WINDOW * STEP)
        for f in range(start, center + WINDOW * STEP + 1, STEP):
            cap.set(cv2.CAP_PROP_POS_FRAMES, f)
            ok, frame = cap.read()
            if ok:
                fn = out / ("frame_%06d.jpg" % f)
                cv2.imwrite(str(fn), frame)
                saved.append(fn.name)
        cap.release()
        (out / "meta.json").write_text(json.dumps({
            "session_id": session.get("id"), "shot_id": shot_id, "t": shot_t,
            "kind": kind, "old_result": old_result, "new_result": new_result,
            "frames": saved}, indent=2))
        return str(out) if saved else None
    except Exception:
        return None


def manifest():
    """Summary of what's banked so far (Part C readiness)."""
    if not HARD_DIR.exists():
        return {"examples": 0, "frames": 0}
    dirs = [d for d in HARD_DIR.iterdir() if d.is_dir()]
    frames = sum(len(list(d.glob("*.jpg"))) for d in dirs)
    return {"examples": len(dirs), "frames": frames}

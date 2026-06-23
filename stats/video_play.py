"""In-browser playback transcode manager for annotated session videos.

OpenCV on this machine cannot write browser-playable H.264 (the openh264 DLL is
broken), but it *can* write valid VP8 WebM. So to make an annotated mp4v file
playable in a <video> tag we transcode it once to a cached `_play.webm` sibling.

Design rules:
  * Thread-safe (best-effort) via a module-level lock around the job registry.
  * Never raises into a request: prepare()/status() always return a plain dict,
    and the background transcode thread wraps its whole body so a crash is always
    recorded as an "error" state rather than vanishing silently.
  * Idempotent: prepare() short-circuits if the webm already exists or a job is
    already in flight for that session.
"""
import os
import threading
from pathlib import Path

import cv2

# session_id -> {"state": "transcoding"|"error"|"ready", "pct": int, ["message": str]}
_JOBS = {}
_LOCK = threading.Lock()


def webm_path_for(annotated_path):
    """Deterministic cache path next to the annotated file.

    `.../<fid>_annotated.mp4` -> `.../<fid>_play.webm`. A trailing `_annotated`
    on the stem is stripped before appending `_play.webm`.
    """
    p = Path(annotated_path)
    stem = p.stem
    if stem.endswith("_annotated"):
        stem = stem[: -len("_annotated")]
    return p.with_name(f"{stem}_play.webm")


def _exists_nonempty(path):
    try:
        return os.path.exists(path) and os.path.getsize(path) > 0
    except OSError:
        return False


def status(session_id, annotated_path):
    """Current playback state for a session (cache first, then job registry)."""
    if annotated_path:
        webm = webm_path_for(annotated_path)
        if _exists_nonempty(webm):
            return {"state": "ready", "pct": 100}
    with _LOCK:
        job = _JOBS.get(session_id)
        if job is not None:
            return dict(job)
    return {"state": "none", "pct": 0}


def prepare(session_id, annotated_path):
    """Idempotently ensure a playable webm exists; kick off a transcode if needed."""
    if annotated_path:
        webm = webm_path_for(annotated_path)
        if _exists_nonempty(webm):
            return {"state": "ready", "pct": 100}
    if not annotated_path or not os.path.exists(annotated_path):
        return {"state": "error", "pct": 0, "message": "no annotated video on disk"}

    with _LOCK:
        job = _JOBS.get(session_id)
        if job is not None and job.get("state") == "transcoding":
            return dict(job)
        started = {"state": "transcoding", "pct": 0}
        _JOBS[session_id] = started
        snapshot = dict(started)

    dst = webm_path_for(annotated_path)
    t = threading.Thread(target=_transcode,
                         args=(session_id, annotated_path, dst),
                         daemon=True)
    t.start()
    return snapshot


def _set_job(session_id, state):
    with _LOCK:
        _JOBS[session_id] = state


def _transcode(session_id, src, dst):
    """Read `src` (mp4v) frame-by-frame and re-encode to VP8 WebM at `dst`."""
    cap = None
    writer = None
    try:
        cap = cv2.VideoCapture(str(src))
        if not cap.isOpened():
            _set_job(session_id, {"state": "error", "pct": 0,
                                  "message": "could not open annotated video"})
            return
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps != fps or fps <= 0:  # 0 / NaN guard
            fps = 30.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

        writer = cv2.VideoWriter(str(dst), cv2.VideoWriter_fourcc(*"VP80"), fps, (w, h))
        if not writer.isOpened():
            _set_job(session_id, {"state": "error", "pct": 0,
                                  "message": "could not open webm writer"})
            return

        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            writer.write(frame)
            idx += 1
            if idx % 15 == 0:
                pct = min(99, max(1, int(100 * idx / total))) if total else 1
                _set_job(session_id, {"state": "transcoding", "pct": pct})

        cap.release()
        cap = None
        writer.release()
        writer = None

        if _exists_nonempty(dst):
            _set_job(session_id, {"state": "ready", "pct": 100})
        else:
            _set_job(session_id, {"state": "error", "pct": 0,
                                  "message": "transcode produced no output"})
    except Exception as e:  # never let the thread die without recording it
        _set_job(session_id, {"state": "error", "pct": 0, "message": str(e)})
        # release handles BEFORE deleting the partial file (Windows keeps it locked)
        try:
            if cap is not None:
                cap.release()
                cap = None
        except Exception:
            pass
        try:
            if writer is not None:
                writer.release()
                writer = None
        except Exception:
            pass
        try:
            if os.path.exists(dst):
                os.remove(dst)
        except OSError:
            pass
    finally:
        try:
            if cap is not None:
                cap.release()
        except Exception:
            pass
        try:
            if writer is not None:
                writer.release()
        except Exception:
            pass

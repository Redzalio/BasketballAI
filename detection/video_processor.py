"""Process an uploaded video: detect -> score -> log shots -> annotated output.

Runs in a background thread; progress is published in PROGRESS[file_id].
Form metrics are produced by the engine at the ball's release frame.
"""
from pathlib import Path
import threading
import cv2

try:
    from detection.engine import HoopEngine
    from detection.zones import derive_zone
    from detection import form_capture
except ImportError:
    from engine import HoopEngine
    from zones import derive_zone
    import form_capture

from stats import db

PROGRESS = {}  # file_id -> {status, percentage, stats, session_id, [message]}
FORMS_DIR = Path(__file__).resolve().parent.parent / "processed" / "forms"


def run_processing(file_id, input_path, output_path, mode="full_tracking",
                   detector_path=None, with_pose=True):
    PROGRESS[file_id] = {"status": "processing", "percentage": 0, "stats": {}, "session_id": None}
    try:
        eng = HoopEngine(detector_path=detector_path, with_pose=with_pose, with_court=True)
        cap = cv2.VideoCapture(str(input_path))
        if not cap.isOpened():
            raise RuntimeError("could not open video")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

        sid = db.create_session("video", Path(input_path).name)
        db.set_session_video(sid, input_path)   # so corrections can re-extract hard-example frames
        db.set_session_annotated(sid, output_path)
        PROGRESS[file_id]["session_id"] = sid
        draw = (mode != "stats_only")
        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            annotated, event, stats = eng.process_frame(frame, draw=draw)
            if event:
                form = event.get("form")
                zone = event.get("zone") or derive_zone(eng.tracker.rim_center, form, frame, event)
                bp = event.get("ball_path") or []
                shot_id = db.add_shot(sid, event["result"], t=round(idx / fps, 2), zone=zone,
                                      x=(bp[-1][0] if bp else None), y=(bp[-1][1] if bp else None),
                                      form=form, arc=event.get("arc"))
                rimg = event.get("release_img")
                if rimg is not None and shot_id:
                    fn = "%d.jpg" % shot_id
                    if form_capture.save(str(FORMS_DIR / fn), rimg,
                                         event.get("release_kp"), event.get("release_hand"), form):
                        db.set_form_image(shot_id, fn)
            writer.write(annotated)
            idx += 1
            if idx % 15 == 0:
                PROGRESS[file_id]["percentage"] = min(99, int(100 * idx / total)) if total else 0
                PROGRESS[file_id]["stats"] = stats

        cap.release()
        writer.release()
        final = db.finalize_session(sid)
        PROGRESS[file_id] = {"status": "completed", "percentage": 100, "session_id": sid,
                             "stats": {"makes": final["makes"], "attempts": final["attempts"],
                                       "fg_pct": final["fg_pct"]}}
    except Exception as e:
        PROGRESS[file_id] = {"status": "error", "percentage": 0, "message": str(e),
                             "stats": {}, "session_id": PROGRESS.get(file_id, {}).get("session_id")}


def start_job(file_id, input_path, output_path, mode="full_tracking", detector_path=None):
    t = threading.Thread(target=run_processing,
                         args=(file_id, input_path, output_path, mode, detector_path),
                         daemon=True)
    t.start()
    return t

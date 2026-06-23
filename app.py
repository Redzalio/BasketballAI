"""HoopTracker — Flask app: live webcam + video import + dashboard API."""
from pathlib import Path
import threading
import time
import uuid

import cv2
from flask import Flask, render_template, request, jsonify, Response, send_file, abort

import config
from stats import db
from stats.insights import session_insights, overview_insights
from detection.engine import HoopEngine
from detection.pose import PoseAnalyzer
from detection.court import CourtMapper
from detection.zones import derive_zone
from detection import video_processor as vp

app = Flask(__name__)
db.init_db()

_POSE = None
_COURT = None


def get_pose():
    global _POSE
    if _POSE is None:
        _POSE = PoseAnalyzer()
    return _POSE


def get_court():
    global _COURT
    if _COURT is None:
        _COURT = CourtMapper()
    return _COURT if _COURT.available else None


# ----------------------------- LIVE STATE -----------------------------
LIVE = {
    "running": False, "thread": None, "session_id": None,
    "frame_jpeg": None, "lock": threading.Lock(),
    "stats": {"makes": 0, "attempts": 0, "fg_pct": 0.0},
    "shots": [], "streak": 0, "last_result": None,
}


def _live_loop(camera):
    cap = cv2.VideoCapture(camera, cv2.CAP_DSHOW)
    eng = HoopEngine(pose_analyzer=get_pose(), court_mapper=get_court())
    streak = 0
    while LIVE["running"]:
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.02)
            continue
        annotated, event, stats = eng.process_frame(frame)
        if event:
            form = event.get("form")
            zone = event.get("zone") or derive_zone(eng.tracker.rim_center, form, frame, event)
            db.add_shot(LIVE["session_id"], event["result"], t=round(time.time(), 2),
                        zone=zone, form=form)
            streak = streak + 1 if event["result"] == "make" else 0
            LIVE["streak"] = streak
            LIVE["last_result"] = event["result"]
            LIVE["shots"].insert(0, {"i": event["attempt"], "result": event["result"],
                                     "zone": zone, "form": form})
            LIVE["shots"] = LIVE["shots"][:50]
        LIVE["stats"] = stats
        ok2, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ok2:
            with LIVE["lock"]:
                LIVE["frame_jpeg"] = buf.tobytes()
    cap.release()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/cameras")
def cameras():
    found = []
    for i in range(3):
        c = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if c.isOpened() and c.read()[0]:
            found.append(i)
        c.release()
    return jsonify({"cameras": found or [0]})


@app.route("/api/live/start", methods=["POST"])
def live_start():
    if LIVE["running"]:
        return jsonify({"session_id": LIVE["session_id"]})
    camera = int((request.get_json(silent=True) or {}).get("camera", 0))
    LIVE.update({"running": True, "shots": [], "streak": 0, "last_result": None,
                 "frame_jpeg": None, "stats": {"makes": 0, "attempts": 0, "fg_pct": 0.0}})
    LIVE["session_id"] = db.create_session("live", f"camera {camera}")
    t = threading.Thread(target=_live_loop, args=(camera,), daemon=True)
    LIVE["thread"] = t
    t.start()
    return jsonify({"session_id": LIVE["session_id"]})


@app.route("/api/live/stop", methods=["POST"])
def live_stop():
    LIVE["running"] = False
    if LIVE["thread"]:
        LIVE["thread"].join(timeout=3)
    sid = LIVE["session_id"]
    final = db.finalize_session(sid) if sid else {"makes": 0, "attempts": 0, "fg_pct": 0.0}
    return jsonify({"session_id": sid, "stats": {"makes": final["makes"],
                    "attempts": final["attempts"], "fg_pct": final["fg_pct"]}})


@app.route("/api/live/stats")
def live_stats():
    return jsonify({"active": LIVE["running"], **LIVE["stats"], "streak": LIVE["streak"],
                    "last_result": LIVE["last_result"], "shots": LIVE["shots"]})


@app.route("/video_feed")
def video_feed():
    def gen():
        while True:
            with LIVE["lock"]:
                f = LIVE["frame_jpeg"]
            if f:
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + f + b"\r\n"
            time.sleep(0.04)
    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/api/upload", methods=["POST"])
def upload():
    f = request.files.get("file")
    if not f:
        abort(400, "no file")
    fid = uuid.uuid4().hex[:12]
    ext = Path(f.filename).suffix or ".mp4"
    f.save(str(config.UPLOAD_DIR / f"{fid}{ext}"))
    return jsonify({"file_id": fid, "filename": f.filename})


@app.route("/api/process/<file_id>", methods=["POST"])
def process(file_id):
    matches = list(config.UPLOAD_DIR.glob(f"{file_id}.*"))
    if not matches:
        abort(404, "upload not found")
    mode = (request.get_json(silent=True) or {}).get("mode", "full_tracking")
    out = config.PROCESSED_DIR / f"{file_id}_annotated.mp4"
    vp.start_job(file_id, matches[0], out, mode=mode)
    return jsonify({"status": "started"})


@app.route("/api/process/<file_id>/status")
def process_status(file_id):
    return jsonify(vp.PROGRESS.get(file_id, {"status": "not_found"}))


@app.route("/api/download/<file_id>")
def download(file_id):
    p = config.PROCESSED_DIR / f"{file_id}_annotated.mp4"
    if not p.exists():
        abort(404)
    return send_file(str(p), mimetype="video/mp4", as_attachment=True,
                     download_name=f"hooptracker_{file_id}.mp4")


@app.route("/api/sessions")
def sessions():
    return jsonify({"sessions": db.list_sessions()})


@app.route("/api/session/<int:sid>")
def session_detail(sid):
    obj = db.get_session(sid)
    if not obj:
        abort(404)
    shots = [{"i": i + 1, "result": s["result"], "zone": s["zone"], "t": s["t"],
              "form": s.get("form", {})}
             for i, s in enumerate(obj["shots"])]
    return jsonify({"session": obj["session"], "shots": shots,
                    "insights": session_insights(obj)})


@app.route("/api/session/<int:sid>", methods=["DELETE"])
def session_delete(sid):
    db.delete_session(sid)
    return jsonify({"ok": True})


@app.route("/api/overview")
def overview():
    sess = db.list_sessions()
    makes = sum(s["makes"] for s in sess)
    attempts = sum(s["attempts"] for s in sess)
    lifetime = {"makes": makes, "attempts": attempts, "shots": attempts, "sessions": len(sess),
                "fg_pct": round(100 * makes / attempts, 1) if attempts else 0.0}
    return jsonify({"lifetime": lifetime, **overview_insights(sess)})


if __name__ == "__main__":
    print(f"HoopTracker -> http://{config.HOST}:{config.PORT}")
    app.run(host=config.HOST, port=config.PORT, threaded=True, debug=False)

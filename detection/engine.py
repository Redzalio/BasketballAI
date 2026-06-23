"""HoopEngine: per-frame ball/rim detection + trajectory make/miss + drawing.

Works with the custom detector (ball/rim/person) or the fallback model
(Basketball/Basketball Hoop) — class names are normalized either way, so
swapping models/detector.pt in is zero-code.

If a pose analyzer is attached, form metrics are sampled at the ball's RELEASE
frame and attached to the event. If a court mapper is attached, the shot is
also labeled with a real court zone (left/center/right + 2PT/3PT).

Standalone self-test:
    python detection/engine.py [video]
"""
from pathlib import Path
from collections import deque
import cv2
import numpy as np
import torch
from ultralytics import YOLO

try:
    from detection.shot_logic import ShotTracker
    from detection.pose import PoseAnalyzer
    from detection.court import CourtMapper
except ImportError:
    from shot_logic import ShotTracker
    from pose import PoseAnalyzer
    from court import CourtMapper

try:
    from stats import arc as arc_mod          # arc / entry-angle analytics (numpy-only)
except Exception:
    arc_mod = None

ROOT = Path(__file__).resolve().parent.parent
FALLBACK = ROOT / "models" / "best_fallback.pt"
DETECTOR = ROOT / "models" / "detector.pt"

COLORS = {"ball": (0, 165, 255), "rim": (0, 0, 255), "person": (0, 255, 0)}


def _canon(name):
    n = str(name).strip().lower()
    if n in ("ball", "basketball"):
        return "ball"
    if n in ("rim", "hoop", "basket", "basketball hoop", "basketball rim"):
        return "rim"
    if n in ("person", "people", "player", "players"):
        return "person"
    return None


class HoopEngine:
    def __init__(self, detector_path=None, ball_conf=0.35, rim_conf=0.45, device=None,
                 with_pose=False, pose_analyzer=None, with_court=False, court_mapper=None):
        path = Path(detector_path) if detector_path else (DETECTOR if DETECTOR.exists() else FALLBACK)
        self.model = YOLO(str(path))
        self.model_path = path
        self.device = device if device is not None else (0 if torch.cuda.is_available() else "cpu")
        self.tracker = ShotTracker(ball_conf=ball_conf, rim_conf=rim_conf)
        self.pose = pose_analyzer if pose_analyzer is not None else (PoseAnalyzer() if with_pose else None)
        self.court = court_mapper if court_mapper is not None else (CourtMapper() if with_court else None)
        if self.court is not None and not self.court.available:
            self.court = None
        self.recent = deque(maxlen=45)   # (frame_idx, frame copy) for release-time pose
        self.arc_buf = deque(maxlen=180) # (frame_idx, cx, cy) ball track for arc analytics
        self.frame_idx = 0
        self.flash = 0
        self.flash_color = (0, 0, 0)
        self.last_result = None

    def process_frame(self, frame, draw=True):
        if self.pose is not None:
            self.recent.append((self.frame_idx, frame.copy()))
        if self.court is not None:
            self.court.maybe_refresh(frame, self.frame_idx)

        res = self.model.predict(frame, imgsz=640, device=self.device, conf=0.12,
                                 half=(self.device != "cpu"), verbose=False)[0]
        dets, boxes_draw = [], []
        for box in res.boxes:
            canon = _canon(self.model.names[int(box.cls[0])])
            if canon is None:
                continue
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            w, h = x2 - x1, y2 - y1
            center = (x1 + w // 2, y1 + h // 2)
            if canon in ("ball", "rim"):
                dets.append((canon, center, w, h, conf))
            boxes_draw.append((canon, x1, y1, x2, y2, conf))

        balls = [(c, cf) for (k, c, w, h, cf) in dets if k == "ball"]
        if balls:
            bc = max(balls, key=lambda t: t[1])[0]
            self.arc_buf.append((self.frame_idx, int(bc[0]), int(bc[1])))

        event = self.tracker.update(dets, self.frame_idx)
        if event:
            self.last_result = event["result"]
            self.flash = 18
            self.flash_color = (0, 200, 0) if event["result"] == "make" else (0, 0, 220)
            if arc_mod is not None:
                lo = event["frame"] - 55
                pts = [(f, x, y) for (f, x, y) in self.arc_buf if lo <= f <= event["frame"] + 2]
                try:
                    event["arc"] = arc_mod.analyze_arc(pts, self.tracker.rim_center, frame.shape)
                except Exception:
                    pass
            if self.pose is not None:
                up = event.get("up_frame", event["frame"])
                # window across the shot: dip (~up-12) -> release -> follow-through (down)
                window = [(i, im) for (i, im) in self.recent if up - 12 <= i <= event["frame"]]
                if len(window) > 22:
                    window = window[::2]
                if not window and self.recent:
                    window = [min(self.recent, key=lambda fr: abs(fr[0] - up))]
                try:
                    res_p = self.pose.analyze_motion(window)
                except Exception:
                    res_p = None
                if res_p:
                    event["form"] = res_p["form"]
                    if self.court is not None and res_p.get("feet"):
                        z = self.court.zone(res_p["feet"], self.tracker.rim_center, window[-1][1].shape)
                        if z:
                            event["zone"] = z
                    rel = res_p.get("release")
                    if rel:
                        img = next((im for (i, im) in window if i == rel.get("src_idx")), None)
                        if img is None and window:
                            img = window[-1][1]
                        if img is not None:
                            event["release_img"] = img
                            event["release_kp"] = rel.get("xy")
                            event["release_hand"] = rel.get("hand")

        if draw:
            frame = self._draw(frame, boxes_draw)
        self.frame_idx += 1
        return frame, event, self.stats()

    def _draw(self, frame, boxes):
        for canon, x1, y1, x2, y2, conf in boxes:
            c = COLORS.get(canon, (200, 200, 200))
            cv2.rectangle(frame, (x1, y1), (x2, y2), c, 2)
            cv2.putText(frame, f"{canon} {conf:.2f}", (x1, max(12, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 2)
        for p in self.tracker.ball_pos:           # ball path
            cv2.circle(frame, tuple(map(int, p[0])), 3, (0, 165, 255), -1)
        rc = self.tracker.rim_center
        if rc:
            cv2.circle(frame, tuple(map(int, rc)), 4, (255, 128, 0), -1)
        self._hud(frame)
        if self.flash > 0:
            a = 0.25 * self.flash / 18
            cv2.addWeighted(np.full_like(frame, self.flash_color), a, frame, 1 - a, 0, frame)
            txt = "MAKE" if self.last_result == "make" else "MISS"
            cv2.putText(frame, txt, (frame.shape[1] - 230, 72),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, self.flash_color, 4)
            self.flash -= 1
        return frame

    def _hud(self, frame):
        t = self.tracker
        txt = f"{t.makes} / {t.attempts}   {t.fg_pct:.0f}%"
        cv2.putText(frame, txt, (24, 56), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 6)
        cv2.putText(frame, txt, (24, 56), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 255), 2)

    def stats(self):
        t = self.tracker
        return {"makes": t.makes, "attempts": t.attempts, "fg_pct": round(t.fg_pct, 1)}


def _selftest(video):
    eng = HoopEngine()
    print("model:", eng.model_path.name, "| names:", eng.model.names, "| device:", eng.device)
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        print("cannot open", video)
        return
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_path = ROOT / "processed" / "selftest_annotated.mp4"
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    n = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame, event, stats = eng.process_frame(frame)
        if event:
            print(f"  frame {event['frame']}: {event['result'].upper()}  -> {stats}")
        writer.write(frame)
        n += 1
    cap.release()
    writer.release()
    print(f"processed {n} frames -> {out_path}")
    print("FINAL:", eng.stats())


if __name__ == "__main__":
    import sys
    vid = sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "data" / "sample_clip.mp4")
    _selftest(vid)

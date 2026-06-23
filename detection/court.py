"""Court-geometry model: detect court lines + rim to map shots to real zones.

The court is static, so detections are refreshed every REFRESH_EVERY frames
(not every frame). zone() combines the shooter's feet (from pose) with the
detected 3-point line to label side (left/center/right) and range (2PT/3PT).

Heuristic, and assumes the camera is roughly behind/beside the shooter facing
the hoop (the natural way you film yourself). Falls back gracefully when the
court model or lines aren't available.
"""
from pathlib import Path
import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
COURT_WEIGHTS = ROOT / "models" / "court.pt"
REFRESH_EVERY = 90


def _canon_court(name):
    n = str(name).strip().lower()
    if "three" in n or "3 point" in n or "3pt" in n:
        return "three"
    if "free throw" in n:
        return "ft"
    if "rim" in n:
        return "rim"
    return None


class CourtMapper:
    def __init__(self, weights=None, device=None):
        p = Path(weights) if weights else COURT_WEIGHTS
        self.model = YOLO(str(p)) if p.exists() else None
        self.device = device if device is not None else (0 if torch.cuda.is_available() else "cpu")
        self.lines = {}        # canon -> (cx, cy, x1, y1, x2, y2, conf)
        self._last = -10 ** 9

    @property
    def available(self):
        return self.model is not None

    def maybe_refresh(self, frame, frame_idx, conf=0.30):
        if self.model is None or (frame_idx - self._last) < REFRESH_EVERY:
            return
        self._last = frame_idx
        try:
            res = self.model.predict(frame, imgsz=640, device=self.device,
                                     half=(self.device != "cpu"), conf=conf, verbose=False)[0]
        except Exception:
            return
        best = {}
        for box in res.boxes:
            c = _canon_court(self.model.names[int(box.cls[0])])
            if not c:
                continue
            cf = float(box.conf[0])
            if c in best and best[c][6] >= cf:
                continue
            x1, y1, x2, y2 = map(float, box.xyxy[0])
            best[c] = ((x1 + x2) / 2, (y1 + y2) / 2, x1, y1, x2, y2, cf)
        if best:
            self.lines.update(best)

    def zone(self, feet_xy, rim_center, frame_shape):
        if feet_xy is None:
            return None
        fx, fy = feet_xy
        fw = frame_shape[1]
        ref_x = rim_center[0] if rim_center else fw / 2.0
        margin = 0.10 * fw
        side = "left" if fx < ref_x - margin else "right" if fx > ref_x + margin else "center"

        three = self.lines.get("three")
        if three:
            # hoop is up-court (small y); beyond the 3pt line = farther from rim = larger y
            return f"{side}-{'3' if fy > three[1] else '2'}"
        return side

"""Court-geometry model: detect court lines + rim to map shots to real zones.

The court is static, so detections are refreshed every REFRESH_EVERY frames
(not every frame). zone() combines the shooter's feet (from pose) with the
detected 3-point line, free-throw line and rim to label a real basketball
zone: a horizontal lane (left corner / left wing / top of key / right wing /
right corner) crossed with a range (2 vs 3), plus the special "paint" and
"free throw" spots near the basket.

Heuristic, and assumes the camera is roughly behind/beside the shooter facing
the hoop (the natural way you film yourself): the hoop sits up-court (small y)
and the shooter stands down-court (larger y), so "farther from the rim" reads
as "larger y". Falls back gracefully when the court model or lines aren't
available (feet/ball x vs rim -> left / center / right).

--------------------------------------------------------------------------
NOTE on accuracy / future upgrade
--------------------------------------------------------------------------
models/court.pt is BOUNDING-BOX annotated (Basketball, Backboard, Net, Rim,
Free Throw Line, Three Point Line), NOT polygon/segmentation annotated. That
means we only know the axis-aligned box each court line lives in, so the zone
mapping below is a geometric heuristic built from those boxes + the shooter's
feet. It cannot know the true curved arc or paint polygon.

Precise per-shot zones (true arc-aware 2/3, exact corner vs wing splits, real
paint outline) would need one of:
  * a court model re-annotated with INSTANCE SEGMENTATION polygons for the
    lines/keys, or
  * a HOMOGRAPHY: solve a perspective transform from detected court-line
    intersections (e.g. corners of the key, where the 3pt line meets the
    baseline) to a top-down court template, then classify the shooter's feet
    in real court coordinates.
Both are future work and would replace the heuristics in zone() — the public
signature (zone(feet_xy, rim_center, frame_shape)) is meant to stay stable so
that upgrade is drop-in.
"""
from pathlib import Path
import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
COURT_WEIGHTS = ROOT / "models" / "court.pt"
REFRESH_EVERY = 90

# --- horizontal-lane tuning (fraction of frame width, distance from rim-x) ---
# |dx| below CENTER_BAND -> central column (top of key / paint / free throw);
# between CENTER_BAND and WING_BAND -> a wing; beyond WING_BAND -> a corner.
CENTER_BAND = 0.11
WING_BAND = 0.27


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

    # ------------------------------------------------------------------ #
    # zone mapping
    # ------------------------------------------------------------------ #
    def _lane(self, fx, ref_x, fw):
        """5-way horizontal lane from feet-x vs rim-x (fraction of width)."""
        dx = (fx - ref_x) / float(fw)        # <0 = left of rim, >0 = right
        a = abs(dx)
        if a <= CENTER_BAND:
            return "center"
        side = "left" if dx < 0 else "right"
        return f"{side} wing" if a <= WING_BAND else f"{side} corner"

    def _is_three(self, fy, fx, ref_x, fw):
        """True if the feet are beyond the detected 3-point line.

        Camera faces the hoop, so beyond the arc == farther from the rim ==
        larger y than the 3pt line's near (down-court) edge. The 3pt box is
        annotated as the whole arc region; its bottom edge (y2) is the part
        nearest the shooter, so crossing it = a three. A small margin avoids
        flapping right on the line. Corners are shallow, so out there we lean
        on the box center instead of its far bottom edge.
        """
        three = self.lines.get("three")
        if not three:
            return None                       # unknown -> caller decides
        _cx, cy, _x1, _y1, _x2, y2, _cf = three
        a = abs((fx - ref_x) / float(fw))
        # near the corners the arc hugs the baseline; use the box center there,
        # and its near edge out toward the top of the key.
        thresh = cy if a > WING_BAND else (cy + y2) / 2.0
        return fy >= thresh - 0.01 * float(fw)

    def zone(self, feet_xy, rim_center, frame_shape):
        if feet_xy is None:
            return None
        fx, fy = feet_xy
        fw = frame_shape[1]

        rim = self.lines.get("rim")
        # prefer the live detector rim_center; fall back to the court-model rim.
        if rim_center:
            ref_x, rim_y = rim_center[0], rim_center[1]
        elif rim:
            ref_x, rim_y = rim[0], rim[1]
        else:
            ref_x, rim_y = fw / 2.0, None

        lane = self._lane(fx, ref_x, fw)
        three = self._is_three(fy, fx, ref_x, fw)

        # ---- close-range special spots (only when central and not a three) ----
        if lane == "center" and three is not True:
            ft = self.lines.get("ft")
            if ft is not None:
                ft_y = ft[1]
                band = 0.035 * float(fw)       # tight: only on/right-at the line
                if abs(fy - ft_y) <= band:
                    return "free throw"        # standing on/near the FT line
                # between the rim and the FT line (closer than FT) -> in the key
                if (rim_y is None or rim_y <= fy <= ft_y):
                    return "paint"
            elif rim_y is not None:
                # no FT-line box: the central area just below the rim is the key.
                if 0 <= (fy - rim_y) <= 0.22 * float(fw):
                    return "paint"

        # ---- general lane + range ----
        if three is None:
            # 3pt line not detected this pass: emit the lane alone (graceful).
            return "top of key" if lane == "center" else lane
        rng = "3" if three else "2"
        if lane == "center":
            return f"top {rng}"                # "top 2" / "top 3"
        return f"{lane} {rng}"                 # e.g. "left wing 3", "right corner 2"

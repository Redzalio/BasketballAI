"""Draw the shooter's pose skeleton + angle labels onto a release frame.

Pure drawing layer for the "form gallery": given the frame captured at the
shot-release moment and the shooter's COCO-17 keypoints (the format YOLO-pose
outputs), it overlays the skeleton, highlights the shooting arm, and labels the
key angles, then hands back a new annotated image (or writes it to disk).

torch-free: cv2 + numpy + stdlib only, no model/GPU stack. Every drawing op is
wrapped so a bad/missing keypoint can never crash the caller -- worst case the
frame comes back unchanged.

Standalone:  python detection/form_capture.py <image>  (draws a demo pose)
"""
from pathlib import Path
import os
import numpy as np
import cv2

# COCO-17 keypoint indices (same map as detection.pose)
NOSE = 0
L_EYE, R_EYE, L_EAR, R_EAR = 1, 2, 3, 4
L_SH, R_SH, L_EL, R_EL, L_WR, R_WR = 5, 6, 7, 8, 9, 10
L_HIP, R_HIP, L_KNEE, R_KNEE, L_ANK, R_ANK = 11, 12, 13, 14, 15, 16

# Bones to draw: (a, b) -> draw a line only if BOTH endpoints are valid.
SKELETON = [
    (5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16), (0, 5), (0, 6),
]

# Bones belonging to each arm -- highlighted when that side is shooting.
RIGHT_ARM = {(6, 8), (8, 10)}
LEFT_ARM = {(5, 7), (7, 9)}

# Colours are BGR (OpenCV order).
BASE_COLOR = (255, 255, 0)        # cyan          -- the body
HIGHLIGHT_COLOR = (0, 165, 255)   # orange        -- the shooting arm
JOINT_COLOR = (255, 255, 255)     # white         -- joint dots
TEXT_COLOR = (255, 255, 255)      # white         -- label text
TEXT_OUTLINE = (0, 0, 0)          # black         -- text outline for contrast

# Human-readable labels + units for the caption block / joint tags.
_METRIC_LABELS = {
    "elbow_angle": "Elbow",
    "knee_bend": "Knee",
    "knee_angle": "Knee",
    "release_angle": "Release",
    "lean_deg": "Lean",
    "follow_through_deg": "Follow-thru",
}
_CAPTION_ORDER = [
    "elbow_angle", "knee_bend", "knee_angle",
    "release_angle", "lean_deg", "follow_through_deg",
]


def _valid(pt, w, h):
    """A keypoint is usable if it's non-zero (0,0 == missing) and on-screen."""
    try:
        x, y = float(pt[0]), float(pt[1])
    except (TypeError, ValueError, IndexError):
        return None
    if not (np.isfinite(x) and np.isfinite(y)):
        return None
    if x <= 0 and y <= 0:          # (0,0) sentinel for an undetected joint
        return None
    if x < 0 or y < 0 or x >= w or y >= h:
        return None
    return (int(round(x)), int(round(y)))


def _coerce_xy(xy):
    """Best-effort cast of the keypoints to a float [17,2] array, else None."""
    try:
        arr = np.asarray(xy, dtype=float)
    except (TypeError, ValueError):
        return None
    if arr.ndim != 2 or arr.shape[0] < 17 or arr.shape[1] < 2:
        return None
    return arr[:17, :2]


def _put_label(img, text, org, scale=0.5, thick=1):
    """Draw text with a dark outline so it reads on any background."""
    try:
        x, y = int(org[0]), int(org[1])
        cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale,
                    TEXT_OUTLINE, thick + 2, cv2.LINE_AA)
        cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale,
                    TEXT_COLOR, thick, cv2.LINE_AA)
    except Exception:
        pass


def _fmt(value, degrees=True):
    """Format a metric value for display (drop a trailing .0; add a degree mark)."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(v):
        return None
    txt = f"{v:.0f}" if abs(v - round(v)) < 0.05 else f"{v:.1f}"
    return txt + chr(176) if degrees else txt   # chr(176) == degree sign


def annotate(img_bgr, xy, hand=None, form=None):
    """Overlay the pose skeleton + angle labels and return a NEW image.

    img_bgr: HxWx3 uint8 BGR image. xy: array-like [17,2] COCO-17 keypoints in
    pixel coords ((0,0) == missing -> skipped). hand: "left"|"right"|None -- the
    shooting arm, drawn in the highlight colour. form: optional metrics dict
    (elbow_angle, knee_bend/knee_angle, release_angle, lean_deg,
    follow_through_deg) labelled near the relevant joints + a top-left caption.

    Never raises: on any failure or unusable input the original frame is
    returned (as a copy). The caller's array is never mutated.
    """
    # Validate the image first; if it's not a drawable BGR array, hand it back.
    if img_bgr is None or not isinstance(img_bgr, np.ndarray):
        return img_bgr
    if img_bgr.ndim != 3 or img_bgr.shape[2] != 3:
        return img_bgr.copy() if isinstance(img_bgr, np.ndarray) else img_bgr

    try:
        out = img_bgr.copy()
        if out.dtype != np.uint8:
            out = np.clip(out, 0, 255).astype(np.uint8)
    except Exception:
        return img_bgr

    h, w = out.shape[:2]
    pts = _coerce_xy(xy)
    if pts is None:
        # Unusable keypoints -> nothing to draw; return the (copied) frame.
        return out

    form = form if isinstance(form, dict) else None
    hand = hand if hand in ("left", "right") else None

    # --- skeleton bones -----------------------------------------------------
    for a, b in SKELETON:
        pa = _valid(pts[a], w, h)
        pb = _valid(pts[b], w, h)
        if pa is None or pb is None:
            continue
        edge = (a, b)
        if (hand == "right" and edge in RIGHT_ARM) or (hand == "left" and edge in LEFT_ARM):
            color, thick = HIGHLIGHT_COLOR, 4
        else:
            color, thick = BASE_COLOR, 2
        try:
            cv2.line(out, pa, pb, color, thick, cv2.LINE_AA)
        except Exception:
            pass

    # --- joints -------------------------------------------------------------
    for i in range(pts.shape[0]):
        p = _valid(pts[i], w, h)
        if p is None:
            continue
        on_arm = (
            (hand == "right" and i in (R_SH, R_EL, R_WR)) or
            (hand == "left" and i in (L_SH, L_EL, L_WR))
        )
        try:
            cv2.circle(out, p, 4, JOINT_COLOR, -1, cv2.LINE_AA)
            if on_arm:
                cv2.circle(out, p, 5, HIGHLIGHT_COLOR, 1, cv2.LINE_AA)
        except Exception:
            pass

    # --- per-joint angle tags ----------------------------------------------
    if form:
        elbow_idx = R_EL if hand == "right" else (L_EL if hand == "left" else None)
        knee_idx = R_KNEE if hand == "right" else (L_KNEE if hand == "left" else None)

        if elbow_idx is not None and "elbow_angle" in form:
            p = _valid(pts[elbow_idx], w, h)
            txt = _fmt(form.get("elbow_angle"))
            if p is not None and txt is not None:
                _put_label(out, txt, (p[0] + 8, p[1] - 6), scale=0.55, thick=1)

        knee_val = form.get("knee_bend", form.get("knee_angle"))
        if knee_idx is not None and knee_val is not None:
            p = _valid(pts[knee_idx], w, h)
            txt = _fmt(knee_val)
            if p is not None and txt is not None:
                _put_label(out, txt, (p[0] + 8, p[1] - 6), scale=0.55, thick=1)

    # --- top-left caption block --------------------------------------------
    if form:
        y = 26
        if hand:
            _put_label(out, f"Shooting hand: {hand}", (10, y), scale=0.6, thick=1)
            y += 24
        for key in _CAPTION_ORDER:
            if key not in form:
                continue
            txt = _fmt(form.get(key))
            if txt is None:
                continue
            label = _METRIC_LABELS.get(key, key)
            _put_label(out, f"{label}: {txt}", (10, y), scale=0.55, thick=1)
            y += 22
            if y > h - 6:
                break

    return out


def save(path, img_bgr, xy, hand=None, form=None):
    """annotate() then write to `path` (creating parent dirs).

    Returns the path string on success, or None on any failure. Never raises.
    """
    try:
        out = annotate(img_bgr, xy, hand=hand, form=form)
        if out is None or not isinstance(out, np.ndarray):
            return None
        p = Path(path)
        if p.parent and not p.parent.exists():
            os.makedirs(str(p.parent), exist_ok=True)
        ok = cv2.imwrite(str(p), out)
        return str(p) if ok else None
    except Exception:
        return None


if __name__ == "__main__":
    import sys

    # A plausible standing right-handed shooter at release, for a quick demo.
    demo = np.array([
        [320, 70],    # nose
        [332, 62], [308, 62],            # eyes
        [344, 66], [296, 66],            # ears
        [360, 130], [280, 130],          # shoulders
        [392, 96], [256, 188],           # elbows  (R high = shooting)
        [404, 52], [240, 250],           # wrists  (R up = release)
        [346, 240], [294, 240],          # hips
        [352, 330], [288, 330],          # knees
        [356, 420], [284, 420],          # ankles
    ], dtype=float)
    metrics = {"elbow_angle": 168.0, "knee_bend": 138.0,
               "release_angle": 52.0, "lean_deg": 4.0, "follow_through_deg": 171.0}

    if len(sys.argv) > 1 and Path(sys.argv[1]).exists():
        base = cv2.imread(sys.argv[1])
    else:
        base = np.full((480, 640, 3), 40, np.uint8)

    out_path = Path(__file__).resolve().parent.parent / "processed" / "form_demo.png"
    dst = save(str(out_path), base, demo, hand="right", form=metrics)
    print("wrote:", dst)

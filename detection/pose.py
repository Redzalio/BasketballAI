"""Shooting-form analysis via YOLO-pose (COCO-17 keypoints).

Runs on the same ultralytics/CUDA stack as the detector (no MediaPipe).
Computes elbow extension, knee bend, balance/lean, release height and
follow-through for the shooter, and grades them into coaching tips.

Form is best measured at the release moment; the engine snapshots a frame
near release and calls form_metrics() on it.

Standalone:  python detection/pose.py <image>
"""
from pathlib import Path
import math
import numpy as np
import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
POSE_WEIGHTS = ROOT / "models" / "yolo11s-pose.pt"

# COCO-17 keypoint indices
NOSE = 0
L_SH, R_SH, L_EL, R_EL, L_WR, R_WR = 5, 6, 7, 8, 9, 10
L_HIP, R_HIP, L_KNEE, R_KNEE, L_ANK, R_ANK = 11, 12, 13, 14, 15, 16


def _angle(a, b, c):
    """Angle ABC in degrees (vertex at b)."""
    a, b, c = np.asarray(a), np.asarray(b), np.asarray(c)
    ba, bc = a - b, c - b
    cosang = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-9)
    return float(np.degrees(np.arccos(np.clip(cosang, -1, 1))))


class PoseAnalyzer:
    def __init__(self, weights=None, device=None):
        p = Path(weights) if weights else POSE_WEIGHTS
        self.model = YOLO(str(p) if p.exists() else "yolo11s-pose.pt")
        self.device = device if device is not None else (0 if torch.cuda.is_available() else "cpu")

    def keypoints(self, frame, conf=0.4):
        """(xy[17,2], conf[17]) for the largest person, or None."""
        res = self.model.predict(frame, imgsz=640, device=self.device,
                                 half=(self.device != "cpu"), conf=conf, verbose=False)[0]
        if res.keypoints is None or len(res.keypoints) == 0:
            return None
        if res.boxes is not None and len(res.boxes):
            xyxy = res.boxes.xyxy.cpu().numpy()
            idx = int(np.argmax((xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])))
        else:
            idx = 0
        xy = res.keypoints.xy[idx].cpu().numpy()
        kc = (res.keypoints.conf[idx].cpu().numpy()
              if res.keypoints.conf is not None else np.ones(len(xy)))
        return xy, kc

    def _metrics(self, xy):
        # shooting side = the higher wrist (more extended upward)
        if xy[R_WR][1] <= xy[L_WR][1]:
            sh, el, wr, hip, kn, ank, hand = R_SH, R_EL, R_WR, R_HIP, R_KNEE, R_ANK, "right"
        else:
            sh, el, wr, hip, kn, ank, hand = L_SH, L_EL, L_WR, L_HIP, L_KNEE, L_ANK, "left"
        elbow = _angle(xy[sh], xy[el], xy[wr])
        knee = _angle(xy[hip], xy[kn], xy[ank])
        torso = xy[sh] - xy[hip]
        lean = abs(math.degrees(math.atan2(torso[0], -torso[1])))
        rel_height = float((xy[sh][1] - xy[wr][1]) / (abs(xy[sh][1] - xy[hip][1]) + 1e-6))
        return {
            "hand": hand,
            "elbow_angle": round(elbow, 1),
            "knee_angle": round(knee, 1),
            "lean_deg": round(lean, 1),
            "release_height_ratio": round(rel_height, 2),
            "follow_through": bool(xy[wr][1] < xy[NOSE][1]),
        }

    @staticmethod
    def _feet(xy):
        # midpoint of the two ankles -> shooter's court position proxy
        ax = (xy[L_ANK][0] + xy[R_ANK][0]) / 2.0
        ay = (xy[L_ANK][1] + xy[R_ANK][1]) / 2.0
        if ax <= 0 and ay <= 0:
            return None
        return (float(ax), float(ay))

    def analyze(self, frame):
        """One inference -> {'form': {...}, 'feet': (x, y)|None} for the shooter."""
        kp = self.keypoints(frame)
        if kp is None:
            return None
        xy, _ = kp
        return {"form": self._metrics(xy), "feet": self._feet(xy)}

    def analyze_motion(self, frames):
        """frames: list of (idx, img) around the shot, chronological. Runs pose on
        each, picks the release (highest-wrist) frame, and computes the full
        motion metric set: release angle, knee bend at the dip, shoulder symmetry,
        follow-through hold, etc. Returns {'form': {...}, 'feet': (x,y)} or None."""
        seq = []
        for _idx, img in frames:
            kp = self.keypoints(img)
            if kp is not None:
                seq.append(kp[0])
        if not seq:
            return None
        if len(seq) == 1:
            return {"form": self._metrics(seq[0]), "feet": self._feet(seq[0])}

        right = min(xy[R_WR][1] for xy in seq) <= min(xy[L_WR][1] for xy in seq)
        if right:
            SH, EL, WR, HIP, KN, ANK, hand = R_SH, R_EL, R_WR, R_HIP, R_KNEE, R_ANK, "right"
        else:
            SH, EL, WR, HIP, KN, ANK, hand = L_SH, L_EL, L_WR, L_HIP, L_KNEE, L_ANK, "left"

        rel_i = min(range(len(seq)), key=lambda i: seq[i][WR][1])   # highest wrist = release
        r = seq[rel_i]
        elbow = _angle(r[SH], r[EL], r[WR])
        release_angle = abs(math.degrees(math.atan2(r[EL][1] - r[WR][1], abs(r[WR][0] - r[EL][0]) + 1e-6)))
        torso = r[SH] - r[HIP]
        lean = abs(math.degrees(math.atan2(torso[0], -torso[1] - 1e-6)))
        symmetry = abs(math.degrees(math.atan2(r[R_SH][1] - r[L_SH][1], abs(r[R_SH][0] - r[L_SH][0]) + 1e-6)))
        torso_len = abs(r[SH][1] - r[HIP][1]) + 1e-6
        rel_height = float((r[SH][1] - r[WR][1]) / torso_len)
        knee_bend = min(_angle(xy[HIP], xy[KN], xy[ANK]) for xy in seq)   # deepest dip
        after = seq[rel_i + 1: rel_i + 4]
        ft_deg = (sum(_angle(xy[SH], xy[EL], xy[WR]) for xy in after) / len(after)) if after else elbow

        return {
            "form": {
                "hand": hand,
                "elbow_angle": round(elbow, 1),
                "release_angle": round(release_angle, 1),
                "knee_bend": round(knee_bend, 1),
                "lean_deg": round(lean, 1),
                "symmetry_deg": round(symmetry, 1),
                "release_height_ratio": round(rel_height, 2),
                "follow_through": bool(r[WR][1] < r[NOSE][1]),
                "follow_through_deg": round(ft_deg, 1),
            },
            "feet": self._feet(r),
        }

    def form_metrics(self, frame):
        a = self.analyze(frame)
        return a["form"] if a else None

    @staticmethod
    def grade(m):
        tips = []
        if m["elbow_angle"] < 150:
            tips.append("Extend the shooting arm fully — snap the elbow straight on release.")
        if m["knee_angle"] > 165:
            tips.append("Use your legs — bend the knees into the dip for power and rhythm.")
        if m["lean_deg"] > 12:
            tips.append("Stay balanced — keep your torso vertical instead of leaning into the shot.")
        if not m["follow_through"]:
            tips.append("Hold your follow-through — finish with the wrist up, fingers down ('cookie jar').")
        if not tips:
            tips.append("Clean form — full extension, balanced base, good follow-through.")
        return tips


if __name__ == "__main__":
    import sys
    img = sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "processed" / "f_make.png")
    pa = PoseAnalyzer()
    m = pa.form_metrics(img)
    print("metrics:", m)
    if m:
        print("tips:", pa.grade(m))

"""Trajectory-based make/miss detection.

Ported and cleaned from avishah3/AI-Basketball-Shot-Detection-Tracker
(utils.py + the shot_detector state machine), decoupled from drawing/IO and
reduced to canonical 'ball'/'rim' detections.

Idea: track the ball's recent path. When the ball goes from an 'up' region
(above/around the rim) to a 'down' region (below the rim), count an attempt;
fit a line to the ball's path and see if it crosses the rim plane within the
rim width -> make, else miss.

Each ball/hoop entry is: (center(x, y), frame_idx, w, h, conf)
"""
import math
import numpy as np


def _score(ball_pos, hoop_pos):
    """True if the ball's path crosses the rim plane within the rim width."""
    xs, ys = [], []
    rim_y = hoop_pos[-1][0][1] - 0.5 * hoop_pos[-1][3]
    # first point above the rim + the following point -> a 2-point line
    for i in reversed(range(len(ball_pos))):
        if ball_pos[i][0][1] < rim_y:
            xs.append(ball_pos[i][0][0]); ys.append(ball_pos[i][0][1])
            if i + 1 < len(ball_pos):
                xs.append(ball_pos[i + 1][0][0]); ys.append(ball_pos[i + 1][0][1])
            break
    if len(xs) > 1:
        m, b = np.polyfit(xs, ys, 1)
        if m == 0:
            return False
        pred_x = (rim_y - b) / m
        rim_x1 = hoop_pos[-1][0][0] - 0.4 * hoop_pos[-1][2]
        rim_x2 = hoop_pos[-1][0][0] + 0.4 * hoop_pos[-1][2]
        return rim_x1 < pred_x < rim_x2
    return False


def _detect_down(ball_pos, hoop_pos):
    y = hoop_pos[-1][0][1] + 0.5 * hoop_pos[-1][3]
    return ball_pos[-1][0][1] > y


def _detect_up(ball_pos, hoop_pos):
    h = hoop_pos[-1]
    x1, x2 = h[0][0] - 4 * h[2], h[0][0] + 4 * h[2]
    y1, y2 = h[0][1] - 2 * h[3], h[0][1]
    bx, by = ball_pos[-1][0]
    return x1 < bx < x2 and y1 < by < y2 - 0.5 * h[3]


def _in_hoop_region(center, hoop_pos):
    if not hoop_pos:
        return False
    x, y = center
    h = hoop_pos[-1]
    return (h[0][0] - h[2] < x < h[0][0] + h[2]) and (h[0][1] - h[3] < y < h[0][1] + 0.5 * h[3])


def _clean_ball(ball_pos, frame_idx, max_frames):
    """Drop points that jump too far (wrong ball) or aren't roughly square."""
    if len(ball_pos) > 1:
        w1, h1 = ball_pos[-2][2], ball_pos[-2][3]
        (x1, y1), (x2, y2) = ball_pos[-2][0], ball_pos[-1][0]
        f_dif = ball_pos[-1][1] - ball_pos[-2][1]
        if math.hypot(x2 - x1, y2 - y1) > 4 * math.hypot(w1, h1) and f_dif < 5:
            ball_pos.pop()
        else:
            w2, h2 = ball_pos[-1][2], ball_pos[-1][3]
            if w2 * 1.4 < h2 or h2 * 1.4 < w2:
                ball_pos.pop()
    if ball_pos and frame_idx - ball_pos[0][1] > max_frames:
        ball_pos.pop(0)
    return ball_pos


def _clean_hoop(hoop_pos):
    if len(hoop_pos) > 1:
        (x1, y1), (x2, y2) = hoop_pos[-2][0], hoop_pos[-1][0]
        w1, h1 = hoop_pos[-2][2], hoop_pos[-2][3]
        f_dif = hoop_pos[-1][1] - hoop_pos[-2][1]
        if math.hypot(x2 - x1, y2 - y1) > 0.5 * math.hypot(w1, h1) and f_dif < 5:
            hoop_pos.pop()
        else:
            w2, h2 = hoop_pos[-1][2], hoop_pos[-1][3]
            if w2 * 1.3 < h2 or h2 * 1.3 < w2:
                hoop_pos.pop()
    if len(hoop_pos) > 25:
        hoop_pos.pop(0)
    return hoop_pos


class ShotTracker:
    """Feed canonical detections per frame; get a make/miss event on a completed shot."""

    def __init__(self, max_frames=30, ball_conf=0.35, rim_conf=0.45):
        self.ball_pos, self.hoop_pos = [], []
        self.max_frames = max_frames
        self.ball_conf, self.rim_conf = ball_conf, rim_conf
        self.up = self.down = False
        self.up_frame = self.down_frame = 0
        self.makes = self.attempts = 0
        self.frame = 0

    def update(self, dets, frame_idx=None):
        """dets: list of (kind, (cx, cy), w, h, conf) with kind in {'ball', 'rim'}.
        Returns an event dict on a completed attempt, else None."""
        frame_idx = self.frame if frame_idx is None else frame_idx
        self.frame = frame_idx
        for kind, center, w, h, conf in dets:
            if kind == "ball" and (conf > self.ball_conf or
                                   (_in_hoop_region(center, self.hoop_pos) and conf > 0.15)):
                self.ball_pos.append((center, frame_idx, w, h, conf))
            elif kind == "rim" and conf > self.rim_conf:
                self.hoop_pos.append((center, frame_idx, w, h, conf))

        self.ball_pos = _clean_ball(self.ball_pos, frame_idx, self.max_frames)
        if len(self.hoop_pos) > 1:
            self.hoop_pos = _clean_hoop(self.hoop_pos)
        return self._detect(frame_idx)

    def _detect(self, frame_idx):
        if not self.hoop_pos or not self.ball_pos:
            return None
        if not self.up:
            self.up = _detect_up(self.ball_pos, self.hoop_pos)
            if self.up:
                self.up_frame = self.ball_pos[-1][1]
        if self.up and not self.down:
            self.down = _detect_down(self.ball_pos, self.hoop_pos)
            if self.down:
                self.down_frame = self.ball_pos[-1][1]

        if self.up and self.down and self.up_frame < self.down_frame:
            self.attempts += 1
            made = _score(self.ball_pos, self.hoop_pos)
            if made:
                self.makes += 1
            self.up = self.down = False
            return {
                "result": "make" if made else "miss",
                "attempt": self.attempts,
                "makes": self.makes,
                "frame": frame_idx,
                "up_frame": self.up_frame,
                "ball_path": [tuple(map(int, p[0])) for p in self.ball_pos],
            }
        return None

    @property
    def fg_pct(self):
        return 100.0 * self.makes / self.attempts if self.attempts else 0.0

    @property
    def rim_center(self):
        return self.hoop_pos[-1][0] if self.hoop_pos else None

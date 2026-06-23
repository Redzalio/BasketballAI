/* HoopTracker mobile - trajectory make/miss.
   Faithful port of detection/shot_logic.py. Loaded as a plain <script>;
   exposes window.HoopShot.ShotTracker.
   Detection objects passed to update(): { kind:'ball'|'rim', c:[x,y], w, h, conf } */
(function (global) {
  function hypot(a, b) { return Math.sqrt(a * a + b * b); }

  function polyfit1(xs, ys) {
    const n = xs.length; let sx = 0, sy = 0, sxx = 0, sxy = 0;
    for (let i = 0; i < n; i++) { sx += xs[i]; sy += ys[i]; sxx += xs[i] * xs[i]; sxy += xs[i] * ys[i]; }
    const d = n * sxx - sx * sx; if (Math.abs(d) < 1e-9) return [0, 0];
    const m = (n * sxy - sx * sy) / d; return [m, (sy - m * sx) / n];
  }

  function score(ball, hoop) {
    const h = hoop[hoop.length - 1];
    const rimY = h.c[1] - 0.5 * h.h;
    const xs = [], ys = [];
    for (let i = ball.length - 1; i >= 0; i--) {
      if (ball[i].c[1] < rimY) {
        xs.push(ball[i].c[0]); ys.push(ball[i].c[1]);
        if (i + 1 < ball.length) { xs.push(ball[i + 1].c[0]); ys.push(ball[i + 1].c[1]); }
        break;
      }
    }
    if (xs.length > 1) {
      const fit = polyfit1(xs, ys); const m = fit[0], b = fit[1];
      if (m === 0) return false;
      const px = (rimY - b) / m;
      return (h.c[0] - 0.4 * h.w) < px && px < (h.c[0] + 0.4 * h.w);
    }
    return false;
  }

  function detectDown(ball, hoop) {
    const h = hoop[hoop.length - 1];
    return ball[ball.length - 1].c[1] > h.c[1] + 0.5 * h.h;
  }

  function detectUp(ball, hoop) {
    const h = hoop[hoop.length - 1];
    const x1 = h.c[0] - 4 * h.w, x2 = h.c[0] + 4 * h.w;
    const y1 = h.c[1] - 2 * h.h, y2 = h.c[1];
    const b = ball[ball.length - 1].c;
    return x1 < b[0] && b[0] < x2 && y1 < b[1] && b[1] < (y2 - 0.5 * h.h);
  }

  function inHoopRegion(c, hoop) {
    if (!hoop.length) return false;
    const h = hoop[hoop.length - 1];
    return (h.c[0] - h.w) < c[0] && c[0] < (h.c[0] + h.w) &&
           (h.c[1] - h.h) < c[1] && c[1] < (h.c[1] + 0.5 * h.h);
  }

  function cleanBall(ball, frameIdx, maxFrames) {
    if (ball.length > 1) {
      const a = ball[ball.length - 2], b = ball[ball.length - 1];
      const dist = hypot(b.c[0] - a.c[0], b.c[1] - a.c[1]);
      if (dist > 4 * hypot(a.w, a.h) && (b.f - a.f) < 5) ball.pop();
      else if (b.w * 1.4 < b.h || b.h * 1.4 < b.w) ball.pop();
    }
    if (ball.length && (frameIdx - ball[0].f) > maxFrames) ball.shift();
    return ball;
  }

  function cleanHoop(hoop) {
    if (hoop.length > 1) {
      const a = hoop[hoop.length - 2], b = hoop[hoop.length - 1];
      const dist = hypot(b.c[0] - a.c[0], b.c[1] - a.c[1]);
      if (dist > 0.5 * hypot(a.w, a.h) && (b.f - a.f) < 5) hoop.pop();
      else if (b.w * 1.3 < b.h || b.h * 1.3 < b.w) hoop.pop();
    }
    if (hoop.length > 25) hoop.shift();
    return hoop;
  }

  class ShotTracker {
    constructor(opts) {
      opts = opts || {};
      this.maxFrames = opts.maxFrames || 30;
      this.ballConf = opts.ballConf || 0.35;
      this.rimConf = opts.rimConf || 0.45;
      this.ball = []; this.hoop = [];
      this.up = false; this.down = false; this.upFrame = 0; this.downFrame = 0;
      this.makes = 0; this.attempts = 0; this.frame = 0;
    }
    get fgPct() { return this.attempts ? 100 * this.makes / this.attempts : 0; }
    get rimCenter() { return this.hoop.length ? this.hoop[this.hoop.length - 1].c : null; }

    update(dets, frameIdx) {
      this.frame = frameIdx;
      for (const d of dets) {
        if (d.kind === 'ball' && (d.conf > this.ballConf || (inHoopRegion(d.c, this.hoop) && d.conf > 0.15)))
          this.ball.push({ c: d.c, f: frameIdx, w: d.w, h: d.h, conf: d.conf });
        else if (d.kind === 'rim' && d.conf > this.rimConf)
          this.hoop.push({ c: d.c, f: frameIdx, w: d.w, h: d.h, conf: d.conf });
      }
      cleanBall(this.ball, frameIdx, this.maxFrames);
      if (this.hoop.length > 1) cleanHoop(this.hoop);
      return this._detect(frameIdx);
    }

    _detect(frameIdx) {
      if (!this.hoop.length || !this.ball.length) return null;
      if (!this.up) { this.up = detectUp(this.ball, this.hoop); if (this.up) this.upFrame = this.ball[this.ball.length - 1].f; }
      if (this.up && !this.down) { this.down = detectDown(this.ball, this.hoop); if (this.down) this.downFrame = this.ball[this.ball.length - 1].f; }
      if (this.up && this.down && this.upFrame < this.downFrame) {
        this.attempts++;
        const made = score(this.ball, this.hoop);
        if (made) this.makes++;
        this.up = false; this.down = false;
        return { result: made ? 'make' : 'miss', attempt: this.attempts, makes: this.makes, frame: frameIdx };
      }
      return null;
    }
  }

  global.HoopShot = { ShotTracker };
})(window);

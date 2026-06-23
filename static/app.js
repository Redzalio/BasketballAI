/* ============================================================
   HoopTracker frontend
   Vanilla JS. No frameworks, no CDNs. All charts hand-drawn (SVG).
   ============================================================ */
(function () {
  "use strict";

  /* ---------------- helpers ---------------- */
  const $  = (sel, root) => (root || document).querySelector(sel);
  const $$ = (sel, root) => Array.from((root || document).querySelectorAll(sel));
  const el = (tag, attrs, html) => {
    const n = document.createElement(tag);
    if (attrs) for (const k in attrs) {
      if (k === "class") n.className = attrs[k];
      else if (k === "style") n.style.cssText = attrs[k];
      else n.setAttribute(k, attrs[k]);
    }
    if (html != null) n.innerHTML = html;
    return n;
  };
  const esc = (s) => String(s == null ? "" : s)
    .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;").replace(/'/g, "&#39;");
  const num = (v, d) => (typeof v === "number" && isFinite(v)) ? v : (d == null ? 0 : d);
  const pct1 = (v) => num(v).toFixed(1);
  const cap = (s) => String(s || "").replace(/(^|[\s_-])([a-z])/g, (m, p, c) => p + c.toUpperCase());

  /* Robust fetch wrapper. Returns parsed JSON or throws. */
  async function api(path, opts) {
    const res = await fetch(path, opts);
    if (!res.ok) {
      let msg = "HTTP " + res.status;
      try { const j = await res.json(); if (j && j.error) msg = j.error; } catch (e) {}
      throw new Error(msg);
    }
    const ct = res.headers.get("content-type") || "";
    if (ct.indexOf("application/json") !== -1) return res.json();
    return res.text();
  }

  let toastTimer = null;
  function toast(msg, isErr) {
    const t = $("#toast");
    t.textContent = msg;
    t.className = "toast show" + (isErr ? " err" : "");
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => { t.className = "toast"; }, 3200);
  }

  function emptyState(icon, title, sub) {
    return '<div class="empty-state"><div class="big" aria-hidden="true">' + icon +
      '</div><div class="t">' + esc(title) + '</div>' +
      (sub ? '<div class="s">' + esc(sub) + '</div>' : '') + '</div>';
  }

  /* FG% -> bar quality class */
  function pctClass(p) {
    p = num(p);
    if (p >= 60) return "good";
    if (p >= 40) return "mid";
    return "bad";
  }

  /* ============================================================
     Form evaluation (ideal ranges from the spec)
     elbow release 155-180 good | knee dip 110-140 | lean <12 |
     follow-through% higher = better
     ============================================================ */
  const FORM = {
    elbow_angle:    { label: "Elbow angle",     unit: "°", min: 155, max: 180, lo: 120, hi: 190, type: "range",
                      tip: "Aim for a clean upward release; full extension around 160-180°." },
    knee_angle:     { label: "Knee bend",        unit: "°", min: 110, max: 140, lo: 90,  hi: 170, type: "range",
                      tip: "A loaded dip into 110-140° feeds power up through the shot." },
    lean_deg:       { label: "Body lean",        unit: "°", min: 0,   max: 12,  lo: 0,   hi: 30,  type: "lower",
                      tip: "Stay balanced — keep lean under 12° for a repeatable base." },
    follow_through: { label: "Follow-through",   unit: "%",      min: 80,  max: 100, lo: 0,   hi: 100, type: "higher",
                      tip: "Hold the follow-through every rep — higher is better." }
  };
  /* given a form metric + value, return quality 'good'|'mid'|'bad' */
  function formQuality(key, val) {
    const m = FORM[key];
    if (!m) return "mid";
    val = num(val);
    if (m.type === "range") {
      if (val >= m.min && val <= m.max) return "good";
      const slack = (m.max - m.min) * 0.4;
      if (val >= m.min - slack && val <= m.max + slack) return "mid";
      return "bad";
    }
    if (m.type === "lower") {
      if (val <= m.max) return "good";
      if (val <= m.max * 1.6) return "mid";
      return "bad";
    }
    if (m.type === "higher") {
      if (val >= 80) return "good";
      if (val >= 55) return "mid";
      return "bad";
    }
    return "mid";
  }
  /* Is a single shot's form "ok" overall? follow_through bool + elbow in range */
  function shotFormOk(form) {
    if (!form) return null;
    const elbowOk = formQuality("elbow_angle", form.elbow_angle) !== "bad";
    const ft = form.follow_through === true || form.follow_through === "true";
    const leanOk = formQuality("lean_deg", form.lean_deg) !== "bad";
    return elbowOk && ft && leanOk;
  }
  /* Rough 0-100 form grade -> letter, from a form_summary-ish object */
  function formGrade(fs) {
    if (!fs) return null;
    let score = 0, n = 0;
    const add = (q) => { score += (q === "good" ? 100 : q === "mid" ? 65 : 30); n++; };
    if (fs.elbow_angle != null) add(formQuality("elbow_angle", fs.elbow_angle));
    if (fs.knee_angle  != null) add(formQuality("knee_angle",  fs.knee_angle));
    if (fs.lean_deg    != null) add(formQuality("lean_deg",    fs.lean_deg));
    const ftp = fs.follow_through_pct;
    if (ftp != null) add(formQuality("follow_through", ftp));
    if (!n) return null;
    const avg = score / n;
    const letter = avg >= 90 ? "A" : avg >= 80 ? "A-" : avg >= 72 ? "B+" : avg >= 64 ? "B"
                  : avg >= 56 ? "B-" : avg >= 48 ? "C+" : avg >= 40 ? "C" : "D";
    return { score: Math.round(avg), letter: letter };
  }

  /* Pretty rows of a per-shot form object (raw form metrics, as returned by
     /api/forms and /api/session shots[i].form). Returns [{label,text}] for the
     keys that are present — used by the form gallery cards + lightbox. */
  function formMetricRows(form) {
    if (!form || typeof form !== "object") return [];
    const rows = [];
    const push = (label, v, unit) => {
      if (v == null) return;
      if (typeof v === "boolean") { rows.push({ label: label, text: v ? "Yes" : "No" }); return; }
      const nv = num(v, null);
      if (nv == null) return;
      rows.push({ label: label, text: Math.round(nv) + (unit || "") });
    };
    push("Elbow", form.elbow_angle, "°");
    push("Release", form.release_angle, "°");
    push("Knee bend", form.knee_bend, "°");
    push("Body lean", form.lean_deg, "°");
    push("Symmetry", form.symmetry_deg, "°");
    if (form.release_height_ratio != null) {
      const r = num(form.release_height_ratio, null);
      if (r != null) rows.push({ label: "Release height", text: fmtMetric(r) });
    }
    if (form.follow_through != null) {
      const ft = form.follow_through === true || form.follow_through === "true";
      rows.push({ label: "Follow-through", text: ft ? "Held" : "Dropped" });
    }
    if (form.hand) rows.push({ label: "Hand", text: cap(String(form.hand)) });
    return rows;
  }

  /* ============================================================
     Bars + gauges (reusable)
     ============================================================ */
  function zoneBars(zones) {
    if (!zones || !Object.keys(zones).length) {
      return '<div class="muted" style="font-size:13px">No zone data yet.</div>';
    }
    // dynamic zone keys — render whatever comes back
    return Object.keys(zones).map((k) => {
      const z = zones[k] || {};
      const p = num(z.pct);
      const makes = num(z.makes), att = num(z.attempts);
      return '<div class="bar-row">' +
        '<div class="bar-head"><span class="name">' + esc(cap(k)) + '</span>' +
        '<span class="val">' + makes + '/' + att + ' · ' + pct1(p) + '%</span></div>' +
        '<div class="bar-track"><div class="bar-fill ' + pctClass(p) + '" style="width:' +
        Math.max(0, Math.min(100, p)) + '%"></div></div></div>';
    }).join("");
  }

  function formGauge(key, val) {
    const m = FORM[key];
    if (!m) return "";
    val = num(val);
    const q = formQuality(key, val);
    const span = m.hi - m.lo;
    const clamp = (x) => Math.max(0, Math.min(100, ((x - m.lo) / span) * 100));
    const markerPct = clamp(val);
    const idealStart = clamp(m.min), idealEnd = clamp(m.max);
    const idealLabel = m.type === "higher" ? "≥ " + m.min + m.unit
                     : m.type === "lower" ? "< " + m.max + m.unit
                     : m.min + "–" + m.max + m.unit;
    return '<div class="gauge">' +
      '<div class="gauge-head"><span class="name">' + esc(m.label) + '</span>' +
      '<span class="ideal">ideal ' + esc(idealLabel) + '</span></div>' +
      '<div class="gauge-val ' + q + '">' + Math.round(val) + '<span class="gauge-unit">' + m.unit + '</span></div>' +
      '<div class="gauge-bar">' +
        '<div class="gauge-ideal" style="left:' + idealStart + '%;width:' + Math.max(0, idealEnd - idealStart) + '%"></div>' +
        '<div class="gauge-marker" style="left:' + markerPct + '%"></div>' +
      '</div></div>';
  }

  function followThroughGauge(pctVal) {
    // follow_through_pct is a 0-100 % so the gauge spans 0..100 directly
    const val = num(pctVal);
    const q = formQuality("follow_through", val);
    return '<div class="gauge">' +
      '<div class="gauge-head"><span class="name">Follow-through</span>' +
      '<span class="ideal">higher is better</span></div>' +
      '<div class="gauge-val ' + q + '">' + Math.round(val) + '<span class="gauge-unit">%</span></div>' +
      '<div class="gauge-bar">' +
        '<div class="gauge-ideal" style="left:80%;width:20%"></div>' +
        '<div class="gauge-marker" style="left:' + Math.max(0, Math.min(100, val)) + '%"></div>' +
      '</div></div>';
  }

  function tipsList(tips, withRank) {
    if (!tips || !tips.length) {
      return '<div class="muted" style="font-size:14px">No tips yet — track a few sessions and coaching pointers will show up here.</div>';
    }
    return tips.map((t, i) => {
      // tip may be a string, or an object {tip/text, note/why}
      let text = t, note = "";
      if (t && typeof t === "object") { text = t.tip || t.text || t.message || ""; note = t.note || t.why || t.detail || ""; }
      return '<div class="tip">' +
        (withRank ? '<span class="rank">' + (i + 1) + '</span>' : '') +
        '<div class="body">' + esc(text) +
        (note ? '<div class="note">' + esc(note) + '</div>' : '') +
        '</div></div>';
    }).join("");
  }

  /* ============================================================
     SVG: FG% over time trend line
     ============================================================ */
  function trendChart(trend) {
    const W = 640, H = 240, padL = 38, padR = 16, padT = 18, padB = 28;
    if (!trend || !trend.length) {
      return '<div class="chart-wrap"><svg viewBox="0 0 ' + W + ' ' + H + '" role="img" aria-label="FG% over time, no data">' +
        '<text class="svg-empty" x="' + (W/2) + '" y="' + (H/2) + '" text-anchor="middle">No trend data yet — track sessions to see your FG% over time.</text></svg></div>';
    }
    const data = trend.slice();
    const n = data.length;
    const innerW = W - padL - padR, innerH = H - padT - padB;
    const xFor = (i) => padL + (n === 1 ? innerW / 2 : (i / (n - 1)) * innerW);
    const yFor = (p) => padT + innerH - (Math.max(0, Math.min(100, num(p))) / 100) * innerH;

    let parts = ['<div class="chart-wrap"><svg viewBox="0 0 ' + W + ' ' + H +
      '" role="img" aria-label="Line chart of field goal percentage over time">'];

    // gridlines + y labels (0,25,50,75,100)
    [0, 25, 50, 75, 100].forEach((g) => {
      const y = yFor(g);
      parts.push('<line class="svg-grid" x1="' + padL + '" y1="' + y.toFixed(1) + '" x2="' + (W - padR) + '" y2="' + y.toFixed(1) + '"/>');
      parts.push('<text class="svg-axis" x="' + (padL - 6) + '" y="' + (y + 3).toFixed(1) + '" text-anchor="end">' + g + '</text>');
    });

    const pts = data.map((d, i) => [xFor(i), yFor(d.fg_pct)]);
    const linePath = pts.map((p, i) => (i ? "L" : "M") + p[0].toFixed(1) + " " + p[1].toFixed(1)).join(" ");
    // area
    if (n > 1) {
      const area = linePath + " L" + pts[n-1][0].toFixed(1) + " " + (padT + innerH) +
        " L" + pts[0][0].toFixed(1) + " " + (padT + innerH) + " Z";
      parts.push('<path class="svg-area" d="' + area + '"/>');
    }
    parts.push('<path class="svg-line" d="' + linePath + '"/>');

    // dots + x labels (thin them if many)
    const step = Math.ceil(n / 7);
    pts.forEach((p, i) => {
      parts.push('<circle class="svg-dot" cx="' + p[0].toFixed(1) + '" cy="' + p[1].toFixed(1) + '" r="3.5"><title>' +
        esc(data[i].date || "") + ": " + pct1(data[i].fg_pct) + "% (" + num(data[i].attempts) + ' att)</title></circle>');
      if (i % step === 0 || i === n - 1) {
        const lbl = String(data[i].date || "").slice(5); // MM-DD
        parts.push('<text class="svg-axis" x="' + p[0].toFixed(1) + '" y="' + (H - 8) + '" text-anchor="middle">' + esc(lbl) + '</text>');
      }
    });

    parts.push('</svg></div>');
    return parts.join("");
  }

  /* ============================================================
     SVG: half-court shot chart with DYNAMIC zones
     Hoop at top, key/paint, 3pt arc. Zones shaded by FG% and
     placed by simple heuristics on their key names; unknown
     keys get auto-placed in a fan so the set is never hardcoded.
     ============================================================ */
  function shadeFor(p, hasData) {
    if (!hasData) return "#1b1f26";
    p = num(p);
    // interpolate red(0%) -> amber(50%) -> green(100%)
    const c1 = [255, 90, 90], c2 = [255, 194, 74], c3 = [57, 217, 138];
    let a, b, t;
    if (p <= 50) { a = c1; b = c2; t = p / 50; } else { a = c2; b = c3; t = (p - 50) / 50; }
    const mix = a.map((v, i) => Math.round(v + (b[i] - v) * t));
    const alpha = 0.30 + 0.45 * Math.min(1, p / 100);
    return "rgba(" + mix[0] + "," + mix[1] + "," + mix[2] + "," + alpha.toFixed(2) + ")";
  }

  function courtChart(byZone) {
    const W = 500, H = 470;
    // court geometry: hoop near top center
    const cx = W / 2, hoopY = 56;
    let parts = ['<div class="chart-wrap"><svg viewBox="0 0 ' + W + ' ' + H +
      '" class="court" role="img" aria-label="Half court shot chart by zone">'];

    // court floor
    parts.push('<rect class="court-fill" x="8" y="8" width="' + (W-16) + '" height="' + (H-16) + '" rx="10"/>');

    const keys = byZone ? Object.keys(byZone) : [];

    // --- zone wedges (drawn first, under the court lines) ---
    // Place known names; everything else fans across remaining angles.
    const knownAngles = { // center angle in degrees measured from hoop, 90 = straight out (down)
      left: 145, "left corner": 168, "left wing": 122, "left baseline": 165,
      center: 90, middle: 90, top: 90, paint: 90, key: 90,
      right: 35, "right corner": 12, "right wing": 58, "right baseline": 15
    };
    const arcR = 250, innerR = 30;
    const usedAngles = [];
    const unknownKeys = [];
    keys.forEach((k) => {
      const lk = String(k).toLowerCase();
      if (knownAngles[lk] != null) usedAngles.push(knownAngles[lk]);
      else unknownKeys.push(k);
    });
    // distribute unknowns across 25..155 avoiding rough collisions
    let ui = 0;
    const unknownAngleFor = () => {
      const count = unknownKeys.length;
      const a = count === 1 ? 90 : 28 + (ui / (count - 1)) * 124;
      ui++;
      return a;
    };

    const toXY = (angleDeg, r) => {
      const rad = angleDeg * Math.PI / 180;
      // angle 90 => straight down (out from hoop); 0 => to the right, 180 => left
      return [cx + Math.cos(rad) * r, hoopY + Math.sin(rad) * r];
    };

    keys.forEach((k) => {
      const z = byZone[k] || {};
      const att = num(z.attempts);
      const p = num(z.pct);
      const lk = String(k).toLowerCase();
      const center = knownAngles[lk] != null ? knownAngles[lk] : unknownAngleFor();
      // wedge half-width depends on how many zones share the fan; keep readable
      const half = Math.max(14, Math.min(34, 150 / Math.max(3, keys.length)));
      const a0 = (center - half) * Math.PI / 180;
      const a1 = (center + half) * Math.PI / 180;
      const p0i = [cx + Math.cos(a0) * innerR, hoopY + Math.sin(a0) * innerR];
      const p1i = [cx + Math.cos(a1) * innerR, hoopY + Math.sin(a1) * innerR];
      const p0o = [cx + Math.cos(a0) * arcR, hoopY + Math.sin(a0) * arcR];
      const p1o = [cx + Math.cos(a1) * arcR, hoopY + Math.sin(a1) * arcR];
      const wedge = "M" + p0i[0].toFixed(1) + " " + p0i[1].toFixed(1) +
        " L" + p0o[0].toFixed(1) + " " + p0o[1].toFixed(1) +
        " A" + arcR + " " + arcR + " 0 0 1 " + p1o[0].toFixed(1) + " " + p1o[1].toFixed(1) +
        " L" + p1i[0].toFixed(1) + " " + p1i[1].toFixed(1) +
        " A" + innerR + " " + innerR + " 0 0 0 " + p0i[0].toFixed(1) + " " + p0i[1].toFixed(1) + " Z";
      parts.push('<path class="court-zone" d="' + wedge + '" fill="' + shadeFor(p, att > 0) + '"><title>' +
        esc(cap(k)) + ": " + num(z.makes) + "/" + att + (att ? " · " + pct1(p) + "%" : " · no attempts") + '</title></path>');
      // label at mid radius
      const lbl = toXY(center, 150);
      parts.push('<text class="court-zone-label" x="' + lbl[0].toFixed(1) + '" y="' + lbl[1].toFixed(1) + '">' + esc(cap(k)) + '</text>');
      if (att > 0) {
        parts.push('<text class="court-zone-sub" x="' + lbl[0].toFixed(1) + '" y="' + (lbl[1] + 15).toFixed(1) + '">' + pct1(p) + '%</text>');
      } else {
        parts.push('<text class="court-zone-sub" x="' + lbl[0].toFixed(1) + '" y="' + (lbl[1] + 15).toFixed(1) + '">—</text>');
      }
    });

    // --- court lines on top ---
    // baseline
    parts.push('<line class="court-stroke" x1="40" y1="28" x2="' + (W-40) + '" y2="28"/>');
    // backboard
    parts.push('<line class="court-stroke" x1="' + (cx-26) + '" y1="40" x2="' + (cx+26) + '" y2="40" stroke-width="3"/>');
    // hoop
    parts.push('<circle cx="' + cx + '" cy="' + (hoopY-2) + '" r="9" fill="none" stroke="var(--orange)" stroke-width="2.5"/>');
    // paint / key
    parts.push('<rect class="court-stroke" x="' + (cx-58) + '" y="28" width="116" height="150" rx="0"/>');
    // free-throw arc
    parts.push('<path class="court-stroke" d="M' + (cx-58) + ' 178 A 58 58 0 0 0 ' + (cx+58) + ' 178"/>');
    // 3pt arc
    parts.push('<path class="court-stroke" d="M40 64 L40 110 A 220 220 0 0 0 ' + (W-40) + ' 110 L' + (W-40) + ' 64"/>');

    parts.push('</svg></div>');
    return parts.join("");
  }

  function courtLegend() {
    return '<div class="row" style="gap:16px;margin-top:12px;font-size:12px;color:var(--text-dim)">' +
      '<span><span style="display:inline-block;width:11px;height:11px;border-radius:3px;background:var(--red);vertical-align:-1px;margin-right:5px"></span>Cold</span>' +
      '<span><span style="display:inline-block;width:11px;height:11px;border-radius:3px;background:var(--amber);vertical-align:-1px;margin-right:5px"></span>~50%</span>' +
      '<span><span style="display:inline-block;width:11px;height:11px;border-radius:3px;background:var(--green);vertical-align:-1px;margin-right:5px"></span>Hot</span>' +
      '<span style="margin-left:auto">Shade = FG% · hover a zone for makes/attempts</span></div>';
  }

  /* ============================================================
     CONSISTENCY LAYER (shared helpers)
     0-100 "consistency" sub-scores -> color band (red <40, amber
     40-70, green >70). Used by the Coaching view, the Dashboard
     KPI tile + mini-trend, and (optionally) the session modal.
     ============================================================ */
  function conBand(score) {
    score = num(score);
    if (score < 40) return "c-red";
    if (score <= 70) return "c-amber";
    return "c-green";
  }
  function conBandLabel(score) {
    score = num(score);
    if (score < 40) return "Inconsistent";
    if (score <= 70) return "Developing";
    return "Repeatable";
  }
  /* trim trailing zeros: 0.90 -> "0.9", 158.6 -> "158.6", 18 -> "18" */
  function fmtMetric(v) {
    if (typeof v !== "number" || !isFinite(v)) return "—";
    let s = (Math.abs(v) < 10 ? v.toFixed(2) : v.toFixed(1));
    if (s.indexOf(".") !== -1) s = s.replace(/0+$/, "").replace(/\.$/, "");
    return s;
  }

  /* Goal progress band: red <40, amber 40-79, green >=80 (or achieved).
     Distinct from conBand() — goals use the spec's 40/80 thresholds. */
  function goalBand(pct, achieved) {
    if (achieved) return "g-green";
    pct = num(pct);
    if (pct < 40) return "g-red";
    if (pct < 80) return "g-amber";
    return "g-green";
  }
  /* Drop a trailing ".0" so "80.0%" reads "80%", but keep "91.4%". */
  function fmtGoalNum(v) {
    if (typeof v !== "number" || !isFinite(v)) return "0";
    let s = v.toFixed(1);
    if (s.indexOf(".") !== -1) s = s.replace(/\.0$/, "");
    return s;
  }

  /* Radial arc gauge for the 0-100 consistency score (inline SVG). */
  function consistencyGauge(score, label) {
    score = Math.max(0, Math.min(100, num(score)));
    const band = conBand(score);
    const R = 64, C = 80, sw = 14;            // viewBox 160x160
    const circ = 2 * Math.PI * R;
    const dash = (score / 100) * circ;
    const offset = circ - dash;
    return '<div class="con-gauge">' +
      '<svg viewBox="0 0 160 160" role="img" aria-label="Consistency score ' + Math.round(score) + ' out of 100">' +
        '<circle class="track" cx="' + C + '" cy="' + C + '" r="' + R + '" stroke-width="' + sw + '"/>' +
        '<circle class="arc ' + band + '" cx="' + C + '" cy="' + C + '" r="' + R + '" stroke-width="' + sw + '" ' +
          'stroke-dasharray="' + circ.toFixed(1) + '" stroke-dashoffset="' + offset.toFixed(1) + '"/>' +
      '</svg>' +
      '<div class="center"><div class="big ' + band + '">' + Math.round(score) + '</div>' +
      '<div class="out-of">' + esc(label || "/ 100") + '</div></div></div>';
  }

  /* Mini consistency-over-time line (Dashboard). trend: [{date,score}] oldest->newest. */
  function consistencyTrendChart(trend) {
    const W = 560, H = 150, padL = 30, padR = 14, padT = 14, padB = 22;
    if (!trend || trend.length < 2) {
      const only = trend && trend.length === 1 ? trend[0] : null;
      const msg = only ? "One session so far (" + Math.round(num(only.score)) +
        "/100) — track more to see your consistency trend."
        : "No consistency history yet — track a few sessions.";
      return '<div class="chart-wrap"><svg viewBox="0 0 ' + W + ' ' + H + '" role="img" aria-label="Consistency over time, not enough data">' +
        '<text class="svg-empty" x="' + (W/2) + '" y="' + (H/2) + '" text-anchor="middle">' + esc(msg) + '</text></svg></div>';
    }
    const data = trend.slice();
    const n = data.length;
    const innerW = W - padL - padR, innerH = H - padT - padB;
    const xFor = (i) => padL + (i / (n - 1)) * innerW;
    const yFor = (p) => padT + innerH - (Math.max(0, Math.min(100, num(p))) / 100) * innerH;

    let parts = ['<div class="chart-wrap"><svg viewBox="0 0 ' + W + ' ' + H +
      '" role="img" aria-label="Line chart of shot-to-shot consistency over time">'];
    [0, 50, 100].forEach((g) => {
      const y = yFor(g);
      parts.push('<line class="svg-grid" x1="' + padL + '" y1="' + y.toFixed(1) + '" x2="' + (W - padR) + '" y2="' + y.toFixed(1) + '"/>');
      parts.push('<text class="svg-axis" x="' + (padL - 6) + '" y="' + (y + 3).toFixed(1) + '" text-anchor="end">' + g + '</text>');
    });
    const pts = data.map((d, i) => [xFor(i), yFor(d.score)]);
    const linePath = pts.map((p, i) => (i ? "L" : "M") + p[0].toFixed(1) + " " + p[1].toFixed(1)).join(" ");
    const area = linePath + " L" + pts[n-1][0].toFixed(1) + " " + (padT + innerH) +
      " L" + pts[0][0].toFixed(1) + " " + (padT + innerH) + " Z";
    parts.push('<path class="svg-area-c" d="' + area + '"/>');
    parts.push('<path class="svg-line-c" d="' + linePath + '"/>');
    const step = Math.ceil(n / 7);
    pts.forEach((p, i) => {
      parts.push('<circle class="svg-dot-c" cx="' + p[0].toFixed(1) + '" cy="' + p[1].toFixed(1) + '" r="3.2"><title>' +
        esc(data[i].date || "") + ": " + Math.round(num(data[i].score)) + '/100</title></circle>');
      if (i % step === 0 || i === n - 1) {
        const lbl = String(data[i].date || "").slice(5, 10); // MM-DD
        parts.push('<text class="svg-axis" x="' + p[0].toFixed(1) + '" y="' + (H - 6) + '" text-anchor="middle">' + esc(lbl) + '</text>');
      }
    });
    parts.push('</svg></div>');
    return parts.join("");
  }

  /* Rolling fatigue line (session modal). rolling: [{shot,score}] in order.
     Same visual language as consistencyTrendChart; x-axis is the shot number,
     and an optional onsetShot draws a vertical "drift starts" marker. */
  function fatigueChart(rolling, onsetShot) {
    const W = 560, H = 150, padL = 30, padR = 14, padT = 14, padB = 22;
    if (!rolling || rolling.length < 2) {
      return '<div class="chart-wrap"><svg viewBox="0 0 ' + W + ' ' + H + '" role="img" aria-label="Rolling consistency, not enough data">' +
        '<text class="svg-empty" x="' + (W/2) + '" y="' + (H/2) + '" text-anchor="middle">Not enough windows to chart yet.</text></svg></div>';
    }
    const data = rolling.slice();
    const n = data.length;
    const innerW = W - padL - padR, innerH = H - padT - padB;
    const shots = data.map((d) => num(d.shot));
    const minS = Math.min.apply(null, shots), maxS = Math.max.apply(null, shots);
    const spanS = (maxS - minS) || 1;
    const xFor = (sh) => padL + ((num(sh) - minS) / spanS) * innerW;
    const yFor = (p) => padT + innerH - (Math.max(0, Math.min(100, num(p))) / 100) * innerH;

    let parts = ['<div class="chart-wrap"><svg viewBox="0 0 ' + W + ' ' + H +
      '" role="img" aria-label="Line chart of rolling shot-to-shot consistency by shot number">'];
    [0, 50, 100].forEach((g) => {
      const y = yFor(g);
      parts.push('<line class="svg-grid" x1="' + padL + '" y1="' + y.toFixed(1) + '" x2="' + (W - padR) + '" y2="' + y.toFixed(1) + '"/>');
      parts.push('<text class="svg-axis" x="' + (padL - 6) + '" y="' + (y + 3).toFixed(1) + '" text-anchor="end">' + g + '</text>');
    });
    // onset marker
    if (onsetShot != null && num(onsetShot) >= minS && num(onsetShot) <= maxS) {
      const ox = xFor(onsetShot);
      parts.push('<line class="svg-onset" x1="' + ox.toFixed(1) + '" y1="' + padT + '" x2="' + ox.toFixed(1) + '" y2="' + (padT + innerH) + '"/>');
      parts.push('<text class="svg-onset-lbl" x="' + ox.toFixed(1) + '" y="' + (padT - 3) + '" text-anchor="middle">drift &#8595; #' + num(onsetShot) + '</text>');
    }
    const pts = data.map((d) => [xFor(d.shot), yFor(d.score)]);
    const linePath = pts.map((p, i) => (i ? "L" : "M") + p[0].toFixed(1) + " " + p[1].toFixed(1)).join(" ");
    const area = linePath + " L" + pts[n-1][0].toFixed(1) + " " + (padT + innerH) +
      " L" + pts[0][0].toFixed(1) + " " + (padT + innerH) + " Z";
    parts.push('<path class="svg-area-c" d="' + area + '"/>');
    parts.push('<path class="svg-line-c" d="' + linePath + '"/>');
    const step = Math.ceil(n / 7);
    pts.forEach((p, i) => {
      parts.push('<circle class="svg-dot-c" cx="' + p[0].toFixed(1) + '" cy="' + p[1].toFixed(1) + '" r="3.2"><title>through shot #' +
        num(data[i].shot) + ": " + Math.round(num(data[i].score)) + '/100</title></circle>');
      if (i % step === 0 || i === n - 1) {
        parts.push('<text class="svg-axis" x="' + p[0].toFixed(1) + '" y="' + (H - 6) + '" text-anchor="middle">#' + num(data[i].shot) + '</text>');
      }
    });
    parts.push('</svg></div>');
    return parts.join("");
  }

  /* "What to work on next" hero card from a focus object {label, why, drill}. */
  function focusHero(focus) {
    if (!focus || !focus.label) {
      return '<div class="focus-hero"><div class="eyebrow"><span class="ico" aria-hidden="true">&#127919;</span> What to work on next</div>' +
        '<h2>Keep shooting</h2>' +
        '<div class="why">Once a session has enough tracked shots, your single highest-leverage fix shows up right here.</div></div>';
    }
    let html = '<div class="focus-hero"><div class="eyebrow"><span class="ico" aria-hidden="true">&#127919;</span> What to work on next</div>' +
      '<h2>' + esc(focus.label) + '</h2>';
    if (focus.why) html += '<div class="why">' + esc(focus.why) + '</div>';
    if (focus.drill) {
      html += '<div class="drill"><span class="badge">Your drill</span>' +
        '<span class="drill-text">' + esc(focus.drill) + '</span></div>';
    }
    html += '</div>';
    return html;
  }

  /* Per-metric consistency card. m = {label,unit,good:[lo,hi],mean,std,n,consistency,in_range} */
  function consistencyMetricCard(m) {
    const band = conBand(m.consistency);
    const unit = m.unit || "";
    const inRange = !!m.in_range;
    const good = Array.isArray(m.good) ? m.good : null;
    const rangeTxt = good ? (fmtMetric(good[0]) + "–" + fmtMetric(good[1]) + unit) : "";
    const cons = Math.round(num(m.consistency));
    return '<div class="cmetric">' +
      '<div class="top"><span class="name">' + esc(m.label || "") + '</span>' +
        '<span class="mean">' + fmtMetric(m.mean) + unit +
          ' <span class="pm">± ' + fmtMetric(m.std) + unit + '</span></span></div>' +
      '<div class="sub"><span class="csub-label">Shot-to-shot consistency</span>' +
        '<span class="csub-val ' + band + '">' + cons + '/100</span></div>' +
      '<div class="ctrack"><div class="cfill ' + band + '" style="width:' + cons + '%"></div></div>' +
      (good ? '<span class="range ' + (inRange ? "in" : "out") + '"><span class="dot"></span>' +
        (inRange ? "In ideal range" : "Outside ideal") + ' (' + esc(rangeTxt) + ')</span>' : '') +
      (m.n ? ' <span class="muted" style="font-size:12px;margin-left:8px">' + num(m.n) + ' shots</span>' : '') +
      proBenchLine(m, unit) +
    '</div>';
  }

  /* Pro-benchmark line for a per-metric card. m.pro = [lo,hi] (or null),
     m.vs_pro = "in"|"low"|"high" (or null). Renders nothing without a range. */
  function proBenchLine(m, unit) {
    const pro = Array.isArray(m.pro) ? m.pro : null;
    if (!pro) return "";
    unit = unit != null ? unit : (m.unit || "");
    const rangeTxt = '<b>' + fmtMetric(pro[0]) + '–' + fmtMetric(pro[1]) + unit + '</b>';
    let tag = "";
    const vs = m.vs_pro;
    if (vs === "in") {
      tag = '<span class="pro-tag in"><span class="dot"></span>in range</span>';
    } else if (vs === "low") {
      tag = '<span class="pro-tag out"><span class="dot"></span>below pro</span>';
    } else if (vs === "high") {
      tag = '<span class="pro-tag out"><span class="dot"></span>above pro</span>';
    }
    return '<div class="pro"><span class="pro-range">Pro: ' + rangeTxt + '</span>' + tag + '</div>';
  }

  /* ============================================================
     ARC / ENTRY-ANGLE (shared)
     Renders the arc_consistency block: a small overall gauge + the
     entry-angle and arc-height consistency sub-cards (mean ± std,
     0..100). Graceful when !enough. Used by Coaching (overview.arc)
     and available to the session modal (insights.arc).
     ============================================================ */
  /* one arc sub-metric -> a cmetric-style card. sub = {mean,std,consistency}. */
  function arcSubCard(label, sub, unit) {
    if (!sub) return "";
    const band = conBand(sub.consistency);
    const cons = Math.round(num(sub.consistency));
    unit = unit || "";
    return '<div class="cmetric">' +
      '<div class="top"><span class="name">' + esc(label) + '</span>' +
        '<span class="mean">' + fmtMetric(num(sub.mean)) + unit +
          ' <span class="pm">± ' + fmtMetric(num(sub.std)) + unit + '</span></span></div>' +
      '<div class="sub"><span class="csub-label">Shot-to-shot consistency</span>' +
        '<span class="csub-val ' + band + '">' + cons + '/100</span></div>' +
      '<div class="ctrack"><div class="cfill ' + band + '" style="width:' + cons + '%"></div></div>' +
    '</div>';
  }

  /* Full arc block body (no card wrapper). */
  function arcBlock(arc) {
    if (!arc || !arc.enough) {
      const n = arc ? num(arc.n) : 0;
      return '<div class="muted" style="font-size:14px">Not enough arc data yet' +
        (n ? ' (' + n + ' tracked' + (n === 1 ? " arc" : " arcs") + ', need 3+)' : '') +
        ' — track shots with the ball in view and your arc consistency shows up here.</div>';
    }
    let out = '<div class="arc-block">';
    out += '<div class="arc-overall">' +
      '<div class="arc-overall-num ' + conBand(arc.overall) + '">' + Math.round(num(arc.overall)) + '<span class="o">/100</span></div>' +
      '<div class="arc-overall-lbl">Overall arc consistency<span class="sub">' + esc(conBandLabel(arc.overall)) + ' · ' + num(arc.n) + ' arcs</span></div>' +
    '</div>';
    const cards = arcSubCard("Entry angle", arc.entry_angle, "°") + arcSubCard("Arc height", arc.peak_height, "");
    out += cards
      ? '<div class="cmetric-grid" style="margin-top:14px">' + cards + '</div>'
      : '<div class="muted" style="font-size:13px;margin-top:10px">Tracked arcs, but not enough of a single measure to score yet.</div>';
    out += '<div class="arc-note">Entry angle is most accurate filmed from the side; shot-to-shot consistency is the reliable read from any fixed angle.</div>';
    out += '</div>';
    return out;
  }

  /* ============================================================
     VIEW ROUTER  (handles teardown so polling loops stop)
     ============================================================ */
  const Views = {};
  let currentView = null;

  function show(view) {
    if (view === currentView) return;
    if (currentView && Views[currentView] && Views[currentView].leave) {
      try { Views[currentView].leave(); } catch (e) {}
    }
    currentView = view;
    $$(".view").forEach((v) => v.classList.toggle("active", v.id === "view-" + view));
    $$("#nav button").forEach((b) => b.classList.toggle("active", b.dataset.view === view));
    if (Views[view] && Views[view].enter) {
      try { Views[view].enter(); } catch (e) { console.error(e); }
    }
  }

  /* ============================================================
     LIVE VIEW
     ============================================================ */
  Views.live = (function () {
    let pollTimer = null, active = false, lastResultShot = -1, started = false;

    async function loadCameras() {
      const sel = $("#cameraSelect");
      try {
        const data = await api("/api/cameras");
        const cams = (data && data.cameras) || [];
        if (!cams.length) {
          sel.innerHTML = '<option value="">No cameras found</option>';
          $("#liveStartBtn").disabled = true;
          return;
        }
        sel.innerHTML = cams.map((c) => '<option value="' + esc(c) + '">Camera ' + esc(c) + '</option>').join("");
        $("#liveStartBtn").disabled = false;
      } catch (e) {
        sel.innerHTML = '<option value="">Cameras unavailable</option>';
        toast("Could not load cameras: " + e.message, true);
      }
    }

    function setActiveUI(on) {
      active = on;
      $("#liveStartBtn").disabled = on;
      $("#liveStopBtn").disabled = !on;
      $("#cameraSelect").disabled = on;
      $("#liveBadge").style.display = on ? "" : "none";
      const panel = $("#videoPanel");
      let img = $("#liveVideo");
      if (on) {
        $("#videoPlaceholder").style.display = "none";
        if (!img) {
          img = el("img", { id: "liveVideo", alt: "Live camera feed" });
          img.onerror = () => { /* stream may not be ready instantly; leave placeholder logic alone */ };
          panel.insertBefore(img, $("#resultFlash"));
        }
        // cache-bust so the MJPEG stream (re)starts
        img.src = "/video_feed?t=" + Date.now();
      } else {
        $("#videoPlaceholder").style.display = "";
        if (img) { img.src = ""; img.remove(); }
      }
    }

    function flash(result) {
      const f = $("#resultFlash");
      const isMake = result === "make";
      f.textContent = isMake ? "MAKE" : "MISS";
      f.className = "result-flash show " + (isMake ? "make" : "miss");
      clearTimeout(f._t);
      f._t = setTimeout(() => { f.className = "result-flash " + (isMake ? "make" : "miss"); }, 900);
    }

    function renderShots(shots) {
      const log = $("#liveShotLog");
      if (!shots || !shots.length) {
        log.innerHTML = '<div class="empty">No shots yet — your makes and misses will appear here as you shoot.</div>';
        $("#liveLogCount").textContent = "";
        return;
      }
      $("#liveLogCount").textContent = shots.length + " shots";
      // newest on top
      const ordered = shots.slice().sort((a, b) => num(b.i) - num(a.i));
      log.innerHTML = ordered.map((s) => {
        const isMake = s.result === "make";
        const ok = shotFormOk(s.form);
        const chip = ok == null ? "" :
          '<span class="form-chip ' + (ok ? "ok" : "flag") + '">' + (ok ? "form ok" : "form flag") + '</span>';
        return '<div class="log-row">' +
          '<span class="log-num">#' + num(s.i) + '</span>' +
          '<span class="log-res ' + (isMake ? "make" : "miss") + '">' + (isMake ? "MAKE" : "MISS") + '</span>' +
          '<span class="log-zone">' + esc(cap(s.zone || "")) + '</span>' +
          chip + '</div>';
      }).join("");
    }

    async function poll() {
      try {
        const s = await api("/api/live/stats");
        if (!s) return;
        $("#liveMakes").textContent = num(s.makes);
        $("#liveAttempts").textContent = num(s.attempts);
        $("#liveFg").innerHTML = pct1(s.fg_pct) + "<small>%</small>";
        $("#liveStreak").textContent = num(s.streak);
        renderShots(s.shots);
        // flash on a newly-registered shot
        const shots = s.shots || [];
        const maxI = shots.reduce((m, x) => Math.max(m, num(x.i)), 0);
        if (maxI > lastResultShot && lastResultShot !== -1 && s.last_result) flash(s.last_result);
        lastResultShot = maxI;
        // server says session ended
        if (s.active === false && active) { setActiveUI(false); }
      } catch (e) {
        // transient: don't spam toasts on every poll
      }
    }

    function startPolling() {
      stopPolling();
      poll();
      pollTimer = setInterval(poll, 700);
    }
    function stopPolling() { if (pollTimer) { clearInterval(pollTimer); pollTimer = null; } }

    async function start() {
      const cam = $("#cameraSelect").value;
      $("#liveStartBtn").disabled = true;
      try {
        await api("/api/live/start", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ camera: cam === "" ? 0 : (isNaN(+cam) ? cam : +cam) })
        });
        started = true;
        lastResultShot = -1;
        setActiveUI(true);
        startPolling();
        toast("Tracking started");
      } catch (e) {
        $("#liveStartBtn").disabled = false;
        toast("Couldn't start: " + e.message, true);
      }
    }

    async function stop() {
      $("#liveStopBtn").disabled = true;
      stopPolling();
      try {
        const r = await api("/api/live/stop", { method: "POST" });
        setActiveUI(false);
        started = false;
        if (r && r.stats) {
          toast("Session saved — " + num(r.stats.makes) + "/" + num(r.stats.attempts) +
            " (" + pct1(r.stats.fg_pct) + "%)");
        } else {
          toast("Session stopped");
        }
      } catch (e) {
        setActiveUI(false);
        toast("Stop error: " + e.message, true);
      }
    }

    return {
      init() {
        $("#liveStartBtn").addEventListener("click", start);
        $("#liveStopBtn").addEventListener("click", stop);
      },
      enter() {
        loadCameras();
        if (started && active) startPolling(); // resume if we navigated away mid-session
      },
      leave() {
        // stop polling when view inactive, but DON'T stop the server session —
        // user may be checking another tab while still recording.
        stopPolling();
      }
    };
  })();

  /* ============================================================
     IMPORT VIEW
     ============================================================ */
  Views["import"] = (function () {
    let pollTimer = null, fileId = null, sessionId = null, busy = false;

    function reset() {
      $("#importIdle").style.display = "";
      $("#importActive").style.display = "none";
      $("#importPreview").style.display = "none";
      $("#importDone").style.display = "none";
      $("#importError").style.display = "none";
      $("#procFill").style.width = "0%";
      $("#procPct").textContent = "0%";
      fileId = null; sessionId = null; busy = false;
    }

    function setStage(html) { $("#procStage").innerHTML = html; }

    function stopPolling() { if (pollTimer) { clearInterval(pollTimer); pollTimer = null; } }

    function showError(msg) {
      stopPolling();
      busy = false;
      $("#importError").style.display = "";
      $("#importError").textContent = msg;
      setStage('<span style="color:var(--red)">Failed</span>');
    }

    async function handleFile(file) {
      if (!file || busy) return;
      if (file.type && file.type.indexOf("video") === -1) {
        toast("That doesn't look like a video file.", true);
        return;
      }
      reset();
      busy = true;
      $("#importIdle").style.display = "none";
      $("#importActive").style.display = "";
      $("#procFile").textContent = file.name;
      setStage('<span class="spinner"></span> Uploading…');

      // 1) upload
      let up;
      try {
        const fd = new FormData();
        fd.append("file", file);
        up = await api("/api/upload", { method: "POST", body: fd });
      } catch (e) { return showError("Upload failed: " + e.message); }
      fileId = up && up.file_id;
      if (!fileId) return showError("Server did not return a file id.");

      // 2) start processing
      try {
        await api("/api/process/" + encodeURIComponent(fileId), {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ mode: "full_tracking" })
        });
      } catch (e) { return showError("Could not start processing: " + e.message); }

      setStage('<span class="spinner"></span> Tracking shots…');
      // 3) poll status
      stopPolling();
      pollTimer = setInterval(pollStatus, 800);
      pollStatus();
    }

    async function pollStatus() {
      if (!fileId) return;
      let st;
      try { st = await api("/api/process/" + encodeURIComponent(fileId) + "/status"); }
      catch (e) { return; } // transient
      if (!st) return;
      const p = Math.max(0, Math.min(100, num(st.percentage)));
      $("#procFill").style.width = p + "%";
      $("#procPct").textContent = Math.round(p) + "%";

      if (st.stats) {
        $("#importPreview").style.display = "";
        $("#impMakes").textContent = num(st.stats.makes);
        $("#impAttempts").textContent = num(st.stats.attempts);
        $("#impFg").innerHTML = pct1(st.stats.fg_pct) + "<small>%</small>";
      }

      if (st.status === "completed") {
        stopPolling();
        busy = false;
        sessionId = st.session_id;
        $("#procFill").style.width = "100%";
        $("#procPct").textContent = "100%";
        setStage('<span style="color:var(--green)">Done — session saved</span>');
        $("#importDone").style.display = "";
        $("#downloadBtn").href = "/api/download/" + encodeURIComponent(fileId);
        $("#downloadBtn").setAttribute("download", "");
        toast("Video processed");
      } else if (st.status === "error") {
        showError("Processing failed on the server.");
      } else {
        setStage('<span class="spinner"></span> Tracking shots… ' + Math.round(p) + '%');
      }
    }

    return {
      init() {
        const dz = $("#dropzone"), input = $("#fileInput");
        dz.addEventListener("click", () => input.click());
        dz.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); input.click(); } });
        input.addEventListener("change", () => { if (input.files[0]) handleFile(input.files[0]); });
        ["dragenter", "dragover"].forEach((ev) => dz.addEventListener(ev, (e) => {
          e.preventDefault(); dz.classList.add("drag");
        }));
        ["dragleave", "drop"].forEach((ev) => dz.addEventListener(ev, (e) => {
          e.preventDefault(); dz.classList.remove("drag");
        }));
        dz.addEventListener("drop", (e) => {
          const f = e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0];
          if (f) handleFile(f);
        });
        $("#viewSessionBtn").addEventListener("click", () => {
          if (sessionId != null) { show("sessions"); openSession(sessionId); }
          else show("sessions");
        });
      },
      leave() { stopPolling(); }   // stop polling when leaving; processing continues server-side
    };
  })();

  /* ============================================================
     DASHBOARD VIEW
     ============================================================ */
  Views.dashboard = (function () {
    async function load() {
      const root = $("#dashContent");
      root.innerHTML = '<div class="empty-state"><div class="spinner"></div></div>';
      let d;
      try { d = await api("/api/overview"); }
      catch (e) { root.innerHTML = emptyState("&#9888;", "Couldn't load dashboard", e.message); return; }

      const life = (d && d.lifetime) || {};
      if (!num(life.attempts) && !num(life.sessions)) {
        root.innerHTML = emptyState("&#127936;", "No shooting data yet",
          "Track a live session or import a video, then your dashboard fills in here.");
        return;
      }

      const grade = formGrade(d.form_avg);
      const pb = d.personal_bests || {};
      const conScore = d.consistency_score;
      const conTrend = d.consistency_trend || [];
      const hasCon = typeof conScore === "number" && (conTrend.length > 0 || num(conScore) > 0);

      let html = "";
      // KPI tiles
      html += '<div class="tiles" style="margin-bottom:16px">' +
        '<div class="tile accent"><div class="label">Lifetime FG%</div><div class="value">' + pct1(life.fg_pct) + '<small>%</small></div></div>' +
        '<div class="tile"><div class="label">Makes / Attempts</div><div class="value">' + num(life.makes) + '<small> / ' + num(life.attempts) + '</small></div></div>' +
        '<div class="tile"><div class="label">Sessions</div><div class="value">' + num(life.sessions) + '</div><div class="delta">' + num(life.shots) + ' shots tracked</div></div>' +
        (hasCon
          ? '<div class="tile"><div class="label">Consistency</div><div class="value ' + conBand(conScore) + '">' + Math.round(num(conScore)) + '<small> / 100</small></div>' +
              '<div class="delta">' + esc(conBandLabel(conScore)) + ' · shot-to-shot</div></div>'
          : '<div class="tile good"><div class="label">Avg form grade</div><div class="value">' + (grade ? grade.letter : "—") + '</div>' +
              (grade ? '<div class="delta">' + grade.score + ' / 100</div>' : '') + '</div>') +
      '</div>';

      // goals (loaded async after the dashboard paints — keeps overview fast)
      html += '<div class="card pad" id="goalsCard" style="margin-bottom:16px">' +
        '<div class="card-title">Goals <span class="sub">track progress toward a target</span></div>' +
        '<div id="goalsBody"><div class="empty-state" style="padding:24px"><div class="spinner"></div></div></div>' +
      '</div>';

      // trend + court
      html += '<div class="grid cols-2" style="margin-bottom:16px">' +
        '<div class="card pad"><div class="card-title">FG% over time</div>' + trendChart(d.trend) + '</div>' +
        '<div class="card pad"><div class="card-title">Shot chart <span class="sub">by zone</span></div>' +
          courtChart(d.by_zone) + courtLegend() + '</div>' +
      '</div>';

      // consistency over time (full-width row; only when we have a score)
      if (hasCon) {
        html += '<div class="card pad" style="margin-bottom:16px"><div class="card-title">Consistency over time ' +
          '<span class="sub">shot-to-shot form, higher = more repeatable</span></div>' +
          consistencyTrendChart(conTrend) + '</div>';
      }

      // personal bests + tips
      html += '<div class="grid cols-2">';
      html += '<div class="card pad"><div class="card-title">Personal bests</div>' + pbCards(pb) + '</div>';
      html += '<div class="card pad"><div class="card-title">Top coaching tips</div>' + tipsList(d.tips, true) + '</div>';
      html += '</div>';

      root.innerHTML = html;

      // goals come from their own endpoint; load after the main paint
      loadGoals();
    }

    /* ---- Goals ---- */
    async function loadGoals() {
      const body = $("#goalsBody");
      if (!body) return;
      let g;
      try { g = await api("/api/goals"); }
      catch (e) {
        body.innerHTML = '<div class="muted" style="font-size:14px">Couldn\'t load goals: ' + esc(e.message) + '</div>';
        return;
      }
      body.innerHTML = goalsMarkup(g);
    }

    function goalsMarkup(g) {
      const goals = (g && g.goals) || [];
      let out = "";
      if (!goals.length) {
        out += emptyState("&#127919;", "No goals yet", "Set a goal to start tracking progress.");
      } else {
        out += '<div class="goal-list">' + goals.map(goalRow).join("") + '</div>';
      }
      out += addGoalForm(g && g.metric_options);
      return out;
    }

    function goalRow(goal) {
      const pct = Math.max(0, Math.min(100, num(goal.pct)));
      const achieved = !!goal.achieved;
      const band = goalBand(pct, achieved);
      const suffix = goal.suffix || "";
      const label = goal.label || goal.metric_label || "Goal";
      const cur = fmtGoalNum(num(goal.current)) + esc(suffix);
      const tgt = fmtGoalNum(num(goal.target)) + esc(suffix);
      return '<div class="goal" data-gid="' + num(goal.id) + '">' +
        '<button class="goal-del" data-gid="' + num(goal.id) + '" title="Delete goal" aria-label="Delete goal">&times;</button>' +
        '<div class="goal-head">' +
          '<span class="goal-label">' + esc(label) + '</span>' +
          '<span class="goal-vals"><span class="cur">' + cur + '</span> / ' + tgt + '</span>' +
        '</div>' +
        '<div class="goal-track"><div class="goal-fill ' + band + '" style="width:' + pct + '%"></div></div>' +
        '<div class="goal-foot">' +
          '<span class="goal-pct ' + band + '">' + Math.round(pct) + '%</span>' +
          (achieved ? '<span class="goal-badge">&#10003; Achieved</span>' : '') +
        '</div>' +
      '</div>';
    }

    function addGoalForm(metricOptions) {
      const opts = metricOptions || {};
      const keys = Object.keys(opts);
      if (!keys.length) {
        // no metrics offered by the server — skip the form rather than render an empty select
        return '';
      }
      const optHtml = keys.map((k) => '<option value="' + esc(k) + '">' + esc(opts[k]) + '</option>').join("");
      return '<div class="goal-add">' +
        '<div class="goal-add-title">Add a goal</div>' +
        '<div class="goal-add-row">' +
          '<select class="input goal-metric" id="goalMetric" aria-label="Goal metric">' + optHtml + '</select>' +
          '<input class="input goal-target" id="goalTarget" type="number" step="any" min="0" placeholder="Target" aria-label="Target value" />' +
          '<input class="input goal-optlabel" id="goalLabel" type="text" maxlength="60" placeholder="Label (optional)" aria-label="Goal label" />' +
          '<button class="btn primary sm" id="goalAddBtn">Add goal</button>' +
        '</div>' +
      '</div>';
    }

    async function addGoal() {
      const metricSel = $("#goalMetric"), targetInput = $("#goalTarget"), labelInput = $("#goalLabel");
      if (!metricSel || !targetInput) return;
      const metric = metricSel.value;
      const targetRaw = targetInput.value;
      if (targetRaw === "" || isNaN(+targetRaw)) { toast("Enter a target number for your goal.", true); targetInput.focus(); return; }
      const body = { metric: metric, target: +targetRaw };
      const lbl = (labelInput && labelInput.value || "").trim();
      if (lbl) body.label = lbl;
      const btn = $("#goalAddBtn");
      if (btn) btn.disabled = true;
      try {
        const g = await api("/api/goals", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body)
        });
        $("#goalsBody").innerHTML = goalsMarkup(g);
        toast("Goal added");
      } catch (e) {
        toast("Couldn't add goal: " + e.message, true);
        if (btn) btn.disabled = false;
      }
    }

    async function deleteGoal(id) {
      try {
        await api("/api/goals/" + encodeURIComponent(id), { method: "DELETE" });
        await loadGoals();
        toast("Goal removed");
      } catch (e) { toast("Couldn't remove goal: " + e.message, true); }
    }

    function pbCards(pb) {
      const items = [];
      if (pb.best_fg_session) {
        const b = pb.best_fg_session;
        items.push(['Best FG% session', pct1(b.fg_pct) + '%', num(b.attempts) + ' attempts', b.id]);
      }
      if (pb.longest_make_streak != null) {
        items.push(['Longest make streak', num(pb.longest_make_streak) + '', 'in a row', null]);
      }
      if (pb.most_makes_session) {
        const m = pb.most_makes_session;
        items.push(['Most makes in a session', num(m.makes) + '', 'session #' + num(m.id), m.id]);
      }
      if (!items.length) return '<div class="muted" style="font-size:14px">No records yet.</div>';
      return '<div class="grid" style="grid-template-columns:1fr;gap:10px">' + items.map((it) => {
        const click = it[3] != null ? ' data-sid="' + it[3] + '" style="cursor:pointer"' : '';
        return '<div class="tile"' + click + '><div class="label">' + esc(it[0]) + '</div>' +
          '<div class="value" style="font-size:24px">' + esc(it[1]) + '</div>' +
          '<div class="delta">' + esc(it[2]) + '</div></div>';
      }).join("") + '</div>';
    }

    return {
      init() {
        $("#dashContent").addEventListener("click", (e) => {
          // goal delete
          const del = e.target.closest(".goal-del");
          if (del) { deleteGoal(+del.dataset.gid); return; }
          // add goal
          if (e.target.closest("#goalAddBtn")) { addGoal(); return; }
          // personal-best tile -> open session
          const t = e.target.closest("[data-sid]");
          if (t) { show("sessions"); openSession(+t.dataset.sid); }
        });
        // Enter inside the goal target/label inputs submits the goal
        $("#dashContent").addEventListener("keydown", (e) => {
          if (e.key !== "Enter") return;
          const inp = e.target.closest("#goalTarget, #goalLabel");
          if (inp) { e.preventDefault(); addGoal(); }
        });
      },
      enter() { load(); }
    };
  })();

  /* ============================================================
     SESSIONS VIEW + detail modal
     ============================================================ */
  async function openSession(id) {
    const overlay = $("#sessionModal");
    $("#modalTitle").textContent = "Session #" + id;
    $("#modalMeta").textContent = "";
    $("#modalBody").innerHTML = '<div class="empty-state"><div class="spinner"></div></div>';
    overlay.classList.add("open");
    document.body.style.overflow = "hidden";

    let d;
    try { d = await api("/api/session/" + encodeURIComponent(id)); }
    catch (e) { $("#modalBody").innerHTML = emptyState("&#9888;", "Couldn't load session", e.message); return; }

    const s = (d && d.session) || {};
    const shots = (d && d.shots) || [];
    const ins = (d && d.insights) || {};
    const fs = ins.form_summary || {};
    const grade = formGrade(fs);
    const con = ins.consistency || null;
    const conScore = con && typeof con.consistency_score === "number" ? con.consistency_score : null;
    const arc = ins.arc || null;
    const fatigue = ins.fatigue || null;

    $("#modalTitle").textContent = "Session #" + num(s.id);
    const chip = '<span class="chip ' + (s.mode === "live" ? "live" : "video") + '">' + esc(s.mode || "") + '</span>';
    $("#modalMeta").innerHTML = esc(s.date || "") + " &nbsp;·&nbsp; " + chip +
      (s.source ? " &nbsp;·&nbsp; " + esc(s.source) : "") +
      (s.duration_s != null ? " &nbsp;·&nbsp; " + fmtDur(s.duration_s) : "");

    let body = "";
    // summary tiles
    body += '<div class="modal-section"><div class="tiles" style="grid-template-columns:repeat(4,1fr)">' +
      '<div class="tile good"><div class="label">Makes</div><div class="value">' + num(s.makes) + '</div></div>' +
      '<div class="tile"><div class="label">Attempts</div><div class="value">' + num(s.attempts) + '</div></div>' +
      '<div class="tile accent"><div class="label">FG%</div><div class="value">' + pct1(s.fg_pct) + '<small>%</small></div></div>' +
      (conScore != null
        ? '<div class="tile"><div class="label">Consistency</div><div class="value ' + conBand(conScore) + '">' + Math.round(num(conScore)) + '<small> / 100</small></div></div>'
        : '<div class="tile"><div class="label">Form grade</div><div class="value">' + (grade ? grade.letter : "—") + '</div></div>') +
    '</div></div>';

    // what to work on (focus) for this session
    if (con && con.focus && con.focus.label) {
      body += '<div class="modal-section">' + focusHero(con.focus) + '</div>';
    }

    // two columns: zones + form summary
    body += '<div class="grid cols-2"><div class="modal-section"><h3>Zone breakdown</h3>' + zoneBars(ins.zones) + '</div>';

    body += '<div class="modal-section"><h3>Form summary</h3>';
    if (fs && Object.keys(fs).length) {
      body += formGauge("elbow_angle", fs.elbow_angle);
      body += formGauge("knee_angle", fs.knee_angle);
      body += formGauge("lean_deg", fs.lean_deg);
      body += followThroughGauge(fs.follow_through_pct);
    } else {
      body += '<div class="muted" style="font-size:14px">No form data for this session.</div>';
    }
    body += '</div></div>';

    // shot chart for THIS session (same builder as the dashboard)
    body += '<div class="modal-section"><h3>Shot chart</h3>' +
      courtChart(ins.zones) + courtLegend() + '</div>';

    // arc & entry angle for this session
    body += '<div class="modal-section"><h3>Arc &amp; entry angle</h3>' + arcBlock(arc) + '</div>';

    // stamina / focus (fatigue)
    body += '<div class="modal-section"><h3>Stamina / focus</h3>' + fatigueBlock(fatigue) + '</div>';

    // streaks
    const stk = ins.streaks || {};
    if (stk.longest_make != null || stk.longest_miss != null) {
      body += '<div class="modal-section"><h3>Streaks</h3><div class="row" style="gap:12px">' +
        '<div class="tile good" style="flex:1"><div class="label">Longest make streak</div><div class="value">' + num(stk.longest_make) + '</div></div>' +
        '<div class="tile" style="flex:1"><div class="label">Longest miss streak</div><div class="value" style="color:var(--red)">' + num(stk.longest_miss) + '</div></div>' +
      '</div></div>';
    }

    // shot list
    body += '<div class="modal-section"><h3>Shots (' + shots.length + ')</h3>';
    if (shots.length) {
      body += '<div class="shotlist">' + shots.slice().sort((a,b)=>num(a.i)-num(b.i)).map((sh) => {
        const isMake = sh.result === "make";
        const ok = shotFormOk(sh.form);
        const chip2 = ok == null ? "" : '<span class="form-chip ' + (ok ? "ok" : "flag") + '">' + (ok ? "form ok" : "form flag") + '</span>';
        const t = sh.t != null ? '<span class="log-num" style="width:auto">@' + Number(sh.t).toFixed(1) + 's</span>' : '';
        // arc entry-angle (when present) + an "arc left frame" tag if estimated/clipped
        const a = sh.arc;
        let arcTag = "";
        if (a && a.ok) {
          if (num(a.entry_angle_deg, null) != null) {
            arcTag += '<span class="arc-pill" title="Image-space entry angle">' + Math.round(num(a.entry_angle_deg)) + '&deg; entry</span>';
          }
          if (a.off_frame_top || a.estimated) {
            arcTag += '<span class="arc-pill est" title="Arc apex left the frame or was estimated">arc left frame</span>';
          }
        }
        return '<div class="log-row">' +
          '<span class="log-num">#' + num(sh.i) + '</span>' +
          '<span class="log-res ' + (isMake ? "make" : "miss") + '">' + (isMake ? "MAKE" : "MISS") + '</span>' +
          '<span class="log-zone">' + esc(cap(sh.zone || "")) + '</span>' + t + arcTag + chip2 + '</div>';
      }).join("") + '</div>';
    } else {
      body += '<div class="muted" style="font-size:14px">No individual shots recorded.</div>';
    }
    body += '</div>';

    // tips
    if (ins.tips && ins.tips.length) {
      body += '<div class="modal-section"><h3>Tips</h3>' + tipsList(ins.tips, false) + '</div>';
    }

    // delete
    body += '<div class="modal-section" style="border-top:1px solid var(--border);padding-top:16px;margin-bottom:0">' +
      '<button class="btn danger" id="deleteSessionBtn" data-id="' + num(s.id) + '">Delete this session</button></div>';

    $("#modalBody").innerHTML = body;
  }

  function closeModal() {
    $("#sessionModal").classList.remove("open");
    document.body.style.overflow = "";
  }

  function fmtDur(sec) {
    sec = num(sec);
    const m = Math.floor(sec / 60), s = Math.round(sec % 60);
    return m + "m " + (s < 10 ? "0" : "") + s + "s";
  }

  /* Fatigue/stamina callout from insights.fatigue. Reuses the .drift block
     visual language (holds = green, drifts = amber) + a rolling line. */
  function fatigueBlock(f) {
    if (!f || !f.enough) {
      const n = f ? num(f.n) : 0;
      return '<div class="muted" style="font-size:14px">Track ~10+ shots in a session to see fatigue' +
        (n ? ' (this one has ' + n + ').' : '.') + '</div>';
    }
    const holds = f.verdict !== "drifts";
    const onset = f.onset_shot != null ? num(f.onset_shot) : null;
    let det = 'Rolling consistency went from ' + Math.round(num(f.baseline)) + ' early to ' +
      Math.round(num(f.late)) + ' late';
    if (onset != null) det += ' — form started sliding around shot #' + onset;
    det += '.';
    let html = '<div class="drift ' + (holds ? "holds" : "drops") + '">' +
      '<span class="ico" aria-hidden="true">' + (holds ? "&#128170;" : "&#128201;") + '</span>' +
      '<div class="txt"><div class="verdict ' + (holds ? "holds" : "drops") + '">' +
        (holds ? "Form held all session" : "Consistency dropped") + '</div>' +
        '<div class="det">' + esc(det) + (f.advice ? ' ' + esc(f.advice) : '') + '</div></div>' +
      '<div class="pair"><div class="leg"><div class="k">Early</div><div class="v ' + conBand(f.baseline) + '">' + Math.round(num(f.baseline)) + '</div></div>' +
        '<span class="arrow" aria-hidden="true">&rarr;</span>' +
        '<div class="leg"><div class="k">Late</div><div class="v ' + conBand(f.late) + '">' + Math.round(num(f.late)) + '</div></div></div>' +
    '</div>';
    // rolling line under the callout
    if (f.rolling && f.rolling.length >= 2) {
      html += '<div style="margin-top:14px">' + fatigueChart(f.rolling, onset) + '</div>';
    }
    return html;
  }

  Views.sessions = (function () {
    async function load() {
      const root = $("#sessionsContent");
      root.innerHTML = '<div class="empty-state"><div class="spinner"></div></div>';
      let d;
      try { d = await api("/api/sessions"); }
      catch (e) { root.innerHTML = emptyState("&#9888;", "Couldn't load sessions", e.message); return; }

      const sessions = (d && d.sessions) || [];
      if (!sessions.length) {
        root.innerHTML = emptyState("&#128203;", "No sessions yet",
          "Finish a live session or import a video and it'll show up here.");
        return;
      }

      let rows = sessions.map((s) => {
        const p = num(s.fg_pct);
        const chip = '<span class="chip ' + (s.mode === "live" ? "live" : "video") + '">' + esc(s.mode || "") + '</span>';
        return '<tr class="clickable" data-id="' + num(s.id) + '">' +
          '<td><span class="muted">#' + num(s.id) + '</span></td>' +
          '<td>' + esc(s.date || "") + '</td>' +
          '<td>' + chip + '</td>' +
          '<td class="ma"><span class="m">' + num(s.makes) + '</span><span class="sep"> / </span><span class="a">' + num(s.attempts) + '</span></td>' +
          '<td><div class="fgbar"><div class="track"><div class="fill" style="width:' + Math.max(0,Math.min(100,p)) + '%;background:' +
            (p>=60?'var(--green)':p>=40?'var(--amber)':'var(--red)') + '"></div></div><span class="num">' + pct1(p) + '%</span></div></td>' +
          '<td class="muted">' + fmtDur(s.duration_s) + '</td>' +
          '<td style="text-align:right"><button class="btn ghost sm del-btn" data-id="' + num(s.id) + '" title="Delete">Delete</button></td>' +
        '</tr>';
      }).join("");

      root.innerHTML = '<table class="sessions"><thead><tr>' +
        '<th>ID</th><th>Date</th><th>Mode</th><th>M / A</th><th>FG%</th><th>Duration</th><th></th>' +
        '</tr></thead><tbody>' + rows + '</tbody></table>';
    }

    async function del(id) {
      if (!confirm("Delete session #" + id + "? This can't be undone.")) return;
      try {
        await api("/api/session/" + encodeURIComponent(id), { method: "DELETE" });
        toast("Session #" + id + " deleted");
        if ($("#sessionModal").classList.contains("open")) closeModal();
        load();
      } catch (e) { toast("Delete failed: " + e.message, true); }
    }

    return {
      init() {
        $("#sessionsRefresh").addEventListener("click", load);
        $("#sessionsContent").addEventListener("click", (e) => {
          const delBtn = e.target.closest(".del-btn");
          if (delBtn) { e.stopPropagation(); del(+delBtn.dataset.id); return; }
          const row = e.target.closest("tr.clickable");
          if (row) openSession(+row.dataset.id);
        });
        // modal delete (delegated on body since modal content is dynamic)
        $("#modalBody").addEventListener("click", (e) => {
          const b = e.target.closest("#deleteSessionBtn");
          if (b) del(+b.dataset.id);
        });
      },
      enter() { load(); }
    };
  })();

  /* ============================================================
     COACHING VIEW  (consistency-first)
     Hero "what to work on next" -> consistency score gauge ->
     makes vs misses -> per-metric breakdown (worst first) ->
     in-session drift. Hero + score come from /api/overview; the
     detailed breakdown comes from the newest session's
     insights.consistency. Both are handled defensively.
     ============================================================ */
  Views.coaching = (function () {
    let currentFocus = null;  // focus object backing the "log this drill" button

    async function load() {
      const root = $("#coachContent");
      root.innerHTML = '<div class="empty-state"><div class="spinner"></div></div>';

      // overview drives the hero focus + headline score
      let d;
      try { d = await api("/api/overview"); }
      catch (e) { root.innerHTML = emptyState("&#9888;", "Couldn't load coaching", e.message); return; }

      const life = (d && d.lifetime) || {};
      if (!num(life.attempts) && !num(life.sessions)) {
        root.innerHTML = emptyState("&#127919;", "No form data yet",
          "Play a session or import a video, then your consistency breakdown and the single thing to work on next will appear here.");
        return;
      }

      // pull the newest session for the detailed consistency report
      let con = null, sessId = null, sessErr = false, latestFatigue = null;
      try {
        const sd = await api("/api/sessions");
        const list = (sd && sd.sessions) || [];
        if (list.length) {
          // sessions are newest-first; be defensive and pick the max id anyway
          sessId = list.reduce((best, s) => (num(s.id) > num(best.id) ? s : best), list[0]).id;
          const full = await api("/api/session/" + encodeURIComponent(sessId));
          con = full && full.insights && full.insights.consistency;
          latestFatigue = full && full.insights && full.insights.fatigue;
        }
      } catch (e) { sessErr = true; }

      // headline focus/score: prefer overview, fall back to the session report
      const focus = (d.focus && d.focus.label) ? d.focus : (con && con.focus) || null;
      const score = (typeof d.consistency_score === "number")
        ? d.consistency_score
        : (con ? con.consistency_score : null);

      const fa = d.form_avg || {};
      const hc = d.hot_cold || "";

      let html = "";

      // 1) HERO — what to work on next (+ a "log this drill" action when we
      //    have a concrete focus metric + drill to log against)
      html += focusHeroWithLog(focus);

      // 2) CONSISTENCY SCORE — radial gauge
      if (typeof score === "number") {
        html += '<div class="card pad" style="margin-bottom:16px"><div class="card-title">Consistency score</div>' +
          '<div class="con-score-card">' + consistencyGauge(score, "/ 100") +
          '<div class="con-score-meta"><div class="head">Shot-to-shot consistency</div>' +
            '<span class="band ' + conBand(score) + '">' + esc(conBandLabel(score)) + '</span>' +
            '<div class="sub">How repeatable your form is rep-to-rep. Higher means your mechanics barely change between shots — the foundation of a reliable jumper.' +
            (con && con.biggest_inconsistency
              ? ' Your most variable piece right now is <strong>' + esc(con.biggest_inconsistency.label) +
                '</strong> (±' + fmtMetric(con.biggest_inconsistency.std) + esc(con.biggest_inconsistency.unit || "") + ').'
              : '') +
            '</div></div></div></div>';
      }

      // 2b) ARC & ENTRY ANGLE (lifetime aggregate from /api/overview.arc)
      html += '<div class="card pad" style="margin-bottom:16px"><div class="card-title">Arc &amp; entry angle ' +
        '<span class="sub">how repeatable your arc is</span></div>' + arcBlock(d.arc) + '</div>';

      // 2c) latest-session fatigue one-liner (only when it actually drifted)
      if (latestFatigue && latestFatigue.enough && latestFatigue.verdict === "drifts") {
        html += '<div class="hotcold cold" style="margin-bottom:16px"><span class="ico" aria-hidden="true">&#128201;</span>' +
          '<span>Last session: ' + esc(latestFatigue.advice || "consistency dropped late in the session.") + '</span></div>';
      }

      // hot/cold banner (kept — useful context, below the headline)
      if (hc) {
        const low = hc.toLowerCase();
        const cls = low.indexOf("hot") !== -1 ? "hot" : (low.indexOf("cold") !== -1 ? "cold" : "neutral");
        const ico = cls === "hot" ? "&#128293;" : cls === "cold" ? "&#10052;" : "&#9889;";
        html += '<div class="hotcold ' + cls + '" style="margin-bottom:16px"><span class="ico" aria-hidden="true">' + ico + '</span><span>' + esc(hc) + '</span></div>';
      }

      // 3) MAKES VS MISSES
      html += '<div class="card pad" style="margin-bottom:16px"><div class="card-title">Makes vs misses ' +
        '<span class="sub">what changes when you miss</span></div>' + makesVsMissesBlock(con) + '</div>';

      // 4) PER-METRIC BREAKDOWN (worst/most-inconsistent first)
      html += '<div class="card pad" style="margin-bottom:16px"><div class="card-title">Per-metric consistency ' +
        '<span class="sub">most inconsistent first</span></div>' + metricBreakdown(con) + '</div>';

      // 5) IN-SESSION DRIFT
      const driftHtml = driftBlock(con);
      if (driftHtml) {
        html += '<div class="card pad" style="margin-bottom:16px"><div class="card-title">In-session drift ' +
          '<span class="sub">did your form hold up?</span></div>' + driftHtml + '</div>';
      }

      // 5b) PRACTICE LOG — drills you logged + whether they tightened up.
      //     Loaded from its own endpoint after paint.
      html += '<div class="card pad" id="practiceCard" style="margin-bottom:16px">' +
        '<div class="card-title">Practice log <span class="sub">did your drills tighten up your form?</span></div>' +
        '<div id="practiceBody"><div class="empty-state" style="padding:24px"><div class="spinner"></div></div></div>' +
      '</div>';

      // 6) Secondary — form vs ideal ranges + classic tips (kept from before)
      html += '<div class="grid cols-2">';
      html += '<div class="card pad"><div class="card-title">Form vs ideal ranges <span class="sub">lifetime averages</span></div>';
      if (fa && Object.keys(fa).length) {
        html += formGauge("elbow_angle", fa.elbow_angle);
        html += formGauge("knee_angle", fa.knee_angle);
        html += formGauge("lean_deg", fa.lean_deg);
        html += followThroughGauge(fa.follow_through_pct);
      } else {
        html += '<div class="muted" style="font-size:14px">No form averages yet.</div>';
      }
      html += '</div>';
      html += '<div class="card pad"><div class="card-title">More pointers <span class="sub">prioritized</span></div>' +
        tipsWithExplain(d.tips, fa) + '</div>';
      html += '</div>';

      root.innerHTML = html;

      // remember the current focus so the "log this drill" button can POST it
      currentFocus = (focus && focus.focus) ? focus : null;

      // practice log loads from its own endpoint
      loadPractice();
    }

    /* The shared focusHero() is also used by the session modal, so we don't
       bake the log button into it. Here we append a "log this drill" action
       when the focus has both a metric key and a drill to record. */
    function focusHeroWithLog(focus) {
      let html = focusHero(focus);
      if (focus && focus.focus && focus.drill) {
        const btn = '<div class="log-drill-row">' +
          '<button class="btn log-drill" id="logDrillBtn">&#10003; Log this drill</button></div>';
        // inject the button just before the hero's closing tag
        const close = html.lastIndexOf("</div>");
        if (close !== -1) html = html.slice(0, close) + btn + html.slice(close);
        else html += btn;
      }
      return html;
    }

    /* ---- Practice log ---- */
    async function loadPractice() {
      const body = $("#practiceBody");
      if (!body) return;
      let p;
      try { p = await api("/api/practice"); }
      catch (e) {
        body.innerHTML = '<div class="muted" style="font-size:14px">Couldn\'t load practice log: ' + esc(e.message) + '</div>';
        return;
      }
      body.innerHTML = practiceMarkup(p);
    }

    function practiceMarkup(p) {
      const list = (p && p.practice) || [];
      if (!list.length) {
        return emptyState("&#128221;", "No drills logged yet",
          "Log a drill from your focus above, then play a session — this will show if it tightened up.");
      }
      return '<div class="practice-list">' + list.map(practiceRow).join("") + '</div>';
    }

    function practiceRow(p) {
      const label = p.label || cap(p.focus_metric || "") || "Drill";
      const baseStd = num(p.baseline_std);
      const hasAfter = (typeof p.after_std === "number" && isFinite(p.after_std));
      let deltaLine = "";
      if (hasAfter) {
        const improved = !!p.improved;
        const variance = '<span class="p-var">±' + fmtMetric(baseStd) + '&deg; ' +
          '<span class="arrow">&rarr;</span> ±' + fmtMetric(num(p.after_std)) + '&deg;</span>';
        const badge = improved
          ? '<span class="p-improved">improved &darr;</span>'
          : '';
        let meta = "";
        if (typeof p.delta === "number" && isFinite(p.delta)) {
          const dv = num(p.delta);
          const dStr = (dv > 0 ? "+" : "") + fmtMetric(dv) + "&deg;";
          meta = '<span class="p-meta">change <span class="dlt-num' + (improved ? " good" : "") + '">' + dStr + '</span>';
          if (p.sessions_since != null) meta += ' · ' + num(p.sessions_since) + ' session' + (num(p.sessions_since) === 1 ? "" : "s") + ' since';
          meta += '</span>';
        } else if (p.sessions_since != null) {
          meta = '<span class="p-meta">' + num(p.sessions_since) + ' session' + (num(p.sessions_since) === 1 ? "" : "s") + ' since</span>';
        }
        deltaLine = '<div class="p-delta">' + variance + badge + meta + '</div>';
      } else {
        deltaLine = '<div class="p-delta"><span class="p-var">±' + fmtMetric(baseStd) + '&deg;</span>' +
          '<span class="p-pending">no session since</span></div>';
      }
      return '<div class="practice">' +
        '<div class="p-head"><span class="p-metric">' + esc(label) + '</span>' +
          (p.logged_at ? '<span class="p-when">' + esc(p.logged_at) + '</span>' : '') + '</div>' +
        (p.drill ? '<div class="p-drill">' + esc(p.drill) + '</div>' : '') +
        (p.note ? '<div class="p-note">' + esc(p.note) + '</div>' : '') +
        deltaLine +
      '</div>';
    }

    async function logDrill() {
      if (!currentFocus || !currentFocus.focus) { toast("No drill to log right now.", true); return; }
      const btn = $("#logDrillBtn");
      if (btn) { btn.disabled = true; btn.innerHTML = "Logging…"; }
      try {
        const p = await api("/api/practice", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ focus_metric: currentFocus.focus, drill: currentFocus.drill || "" })
        });
        const body = $("#practiceBody");
        if (body) body.innerHTML = practiceMarkup(p);
        toast("Drill logged — play a session to see if it tightens up");
        if (btn) { btn.innerHTML = "&#10003; Logged"; }
      } catch (e) {
        toast("Couldn't log drill: " + e.message, true);
        if (btn) { btn.disabled = false; btn.innerHTML = "&#10003; Log this drill"; }
      }
    }

    /* makes_vs_misses -> table of rows, highlighting the top leak */
    function makesVsMissesBlock(con) {
      const mvm = con && con.makes_vs_misses;
      if (!mvm || !mvm.enough) {
        return '<div class="mvm-note">Need more shots to compare — once a session has at least a few makes <em>and</em> a few misses, you\'ll see exactly which part of your form breaks down on misses.</div>';
      }
      const rows = mvm.rows || [];
      if (!rows.length) return '<div class="mvm-note">Not enough per-metric data to compare makes and misses yet.</div>';
      const topKey = mvm.top && mvm.top.metric;
      let out = '<table class="mvm"><thead><tr><th>Metric</th><th class="r">On makes</th><th class="r">On misses</th><th class="r">Difference</th></tr></thead><tbody>';
      out += rows.map((r) => {
        const unit = r.unit || "";
        const isTop = r.metric === topKey;
        const d = num(r.delta);
        const dStr = (d > 0 ? "+" : "") + fmtMetric(d) + unit;
        return '<tr' + (isTop ? ' class="top"' : '') + '>' +
          '<td class="metric">' + esc(r.label || "") + (isTop ? '<span class="top-flag">biggest</span>' : '') + '</td>' +
          '<td class="r mk">' + fmtMetric(r.make) + unit + '</td>' +
          '<td class="r ms">' + fmtMetric(r.miss) + unit + '</td>' +
          '<td class="r dlt">' + dStr + '</td></tr>';
      }).join("");
      out += '</tbody></table>';
      if (mvm.top) {
        out += '<div class="mvm-callout"><strong>This is what changes when you miss:</strong> your ' +
          esc((mvm.top.label || "").toLowerCase()) + ' is ' + fmtMetric(Math.abs(num(mvm.top.delta))) +
          esc(mvm.top.unit || "") + ' different on misses vs makes. Lock that down first.</div>';
      }
      return out;
    }

    /* metrics object -> per-metric cards, sorted by consistency ASC (leaks on top) */
    function metricBreakdown(con) {
      const metrics = con && con.metrics;
      if (!metrics || !Object.keys(metrics).length) {
        return '<div class="muted" style="font-size:14px">No per-metric form data yet — track a session (with pose detected) and each measured part of your shot will be scored here.</div>';
      }
      const entries = Object.keys(metrics).map((k) => metrics[k])
        .sort((a, b) => num(a.consistency) - num(b.consistency));
      return '<div class="cmetric-grid">' + entries.map(consistencyMetricCard).join("") + '</div>';
    }

    /* drift -> holds up / drops off block; null when not enough data */
    function driftBlock(con) {
      const dr = con && con.drift;
      if (!dr || !dr.enough) return null;
      const holds = dr.verdict !== "drops off";
      const delta = num(dr.delta);
      const dStr = (delta > 0 ? "+" : "") + Math.round(delta);
      return '<div class="drift ' + (holds ? "holds" : "drops") + '">' +
        '<span class="ico" aria-hidden="true">' + (holds ? "&#9989;" : "&#128201;") + '</span>' +
        '<div class="txt"><div class="verdict ' + (holds ? "holds" : "drops") + '">' +
          (holds ? "Your form holds up" : "Your form drops off") + '</div>' +
          '<div class="det">Shot-to-shot consistency went from ' + Math.round(num(dr.early)) +
          ' early to ' + Math.round(num(dr.late)) + ' late in the session (' + dStr + ' points).' +
          (holds ? ' Nice — fatigue isn\'t wrecking your mechanics.' : ' Tighten up your routine as you tire, or shorten reps.') +
          '</div></div>' +
        '<div class="pair"><div class="leg"><div class="k">Early</div><div class="v ' + conBand(dr.early) + '">' + Math.round(num(dr.early)) + '</div></div>' +
          '<span class="arrow" aria-hidden="true">&rarr;</span>' +
          '<div class="leg"><div class="k">Late</div><div class="v ' + conBand(dr.late) + '">' + Math.round(num(dr.late)) + '</div></div></div>' +
      '</div>';
    }

    // tips list with a one-line explanation each (use server note if present, else infer from form)
    function tipsWithExplain(tips, fa) {
      if (!tips || !tips.length) {
        return '<div class="muted" style="font-size:14px">Nothing flagged right now — keep shooting to build a fuller picture.</div>';
      }
      return tips.map((t, i) => {
        let text = t, note = "";
        if (t && typeof t === "object") { text = t.tip || t.text || t.message || ""; note = t.note || t.why || t.detail || ""; }
        if (!note) note = inferNote(text, fa);
        return '<div class="tip"><span class="rank">' + (i + 1) + '</span>' +
          '<div class="body">' + esc(text) +
          (note ? '<div class="note">' + esc(note) + '</div>' : '') + '</div></div>';
      }).join("");
    }

    function inferNote(text, fa) {
      const s = String(text || "").toLowerCase();
      if (s.indexOf("elbow") !== -1 && fa.elbow_angle != null)
        return "Your average release elbow is " + Math.round(num(fa.elbow_angle)) + "° (ideal 155–180°).";
      if ((s.indexOf("knee") !== -1 || s.indexOf("bend") !== -1 || s.indexOf("legs") !== -1) && fa.knee_angle != null)
        return "Your average knee bend is " + Math.round(num(fa.knee_angle)) + "° (ideal dip 110–140°).";
      if ((s.indexOf("lean") !== -1 || s.indexOf("balance") !== -1) && fa.lean_deg != null)
        return "Average body lean " + Math.round(num(fa.lean_deg)) + "° — keep it under 12°.";
      if ((s.indexOf("follow") !== -1) && fa.follow_through_pct != null)
        return "You held your follow-through on " + Math.round(num(fa.follow_through_pct)) + "% of shots.";
      return "";
    }

    return {
      init() {
        // "Log this drill" is in the dynamic focus hero — delegate from the view root
        $("#coachContent").addEventListener("click", (e) => {
          if (e.target.closest("#logDrillBtn")) logDrill();
        });
      },
      enter() { load(); }
    };
  })();

  /* ============================================================
     FORM GALLERY VIEW + lightbox
     Thumbnail grid of release frames (with skeleton/angles already
     drawn server-side). Filter All/Makes/Misses; click -> lightbox
     with the full-size image, this shot's form metrics + a link to
     the session. Empty until the user processes new video.
     ============================================================ */
  function openLightbox(shot) {
    const overlay = $("#formLightbox");
    const body = $("#lightboxBody");
    const isMake = shot.result === "make";
    const rows = formMetricRows(shot.form);
    const metaBits = [];
    if (shot.zone) metaBits.push(esc(cap(shot.zone)));
    if (shot.date) metaBits.push(esc(shot.date));
    if (shot.t != null) metaBits.push("@" + Number(shot.t).toFixed(1) + "s");

    let html = '<div class="lb-figure"><img src="' + esc(shot.image) + '" alt="Release frame for shot #' + num(shot.id) +
      '" loading="lazy" onerror="this.classList.add(\'broken\')" /></div>';
    html += '<div class="lb-info">';
    html += '<div class="lb-info-head"><span class="log-res ' + (isMake ? "make" : "miss") + '">' +
      (isMake ? "MAKE" : "MISS") + '</span>' +
      (metaBits.length ? '<span class="lb-meta">' + metaBits.join(" &middot; ") + '</span>' : '') + '</div>';
    if (rows.length) {
      html += '<div class="lb-metrics">' + rows.map((r) =>
        '<div class="lb-metric"><span class="k">' + esc(r.label) + '</span><span class="v">' + esc(r.text) + '</span></div>'
      ).join("") + '</div>';
    } else {
      html += '<div class="muted" style="font-size:13px">No form metrics recorded for this shot.</div>';
    }
    if (shot.session_id != null) {
      html += '<div class="lb-actions"><button class="btn sm" id="lbOpenSession" data-sid="' + num(shot.session_id) + '">Open session</button></div>';
    }
    html += '</div>';

    body.innerHTML = html;
    overlay.classList.add("open");
    document.body.style.overflow = "hidden";
  }

  function closeLightbox() {
    $("#formLightbox").classList.remove("open");
    // don't unlock scroll if the session modal is still open underneath
    if (!$("#sessionModal").classList.contains("open")) document.body.style.overflow = "";
  }

  Views.forms = (function () {
    let shots = [];          // cache of the last fetch
    let filter = "all";      // all | make | miss
    let loaded = false;

    function filtered() {
      if (filter === "all") return shots;
      return shots.filter((s) => s.result === filter);
    }

    function render() {
      const root = $("#formsContent");
      if (!shots.length) {
        root.innerHTML = emptyState("&#128247;", "No form snapshots yet",
          "Track a live session or import a video and your release frames show up here.");
        return;
      }
      const list = filtered();
      if (!list.length) {
        root.innerHTML = emptyState("&#128247;",
          filter === "make" ? "No makes captured yet" : "No misses captured yet",
          "Switch the filter, or track more shots to fill this in.");
        return;
      }
      root.innerHTML = '<div class="form-grid">' + list.map(cardHtml).join("") + '</div>';
    }

    function cardHtml(s) {
      const isMake = s.result === "make";
      const f = s.form || {};
      // a couple of key numbers on the card face
      const bits = [];
      if (num(f.elbow_angle, null) != null) bits.push('<span class="fg-stat">Elbow ' + Math.round(num(f.elbow_angle)) + '&deg;</span>');
      if (num(f.release_angle, null) != null) bits.push('<span class="fg-stat">Release ' + Math.round(num(f.release_angle)) + '&deg;</span>');
      else if (num(f.knee_bend, null) != null) bits.push('<span class="fg-stat">Knee ' + Math.round(num(f.knee_bend)) + '&deg;</span>');
      return '<button class="form-card" data-id="' + num(s.id) + '" aria-label="Open release frame for shot #' + num(s.id) + '">' +
        '<div class="fc-thumb"><img src="' + esc(s.image) + '" loading="lazy" alt="Release frame, shot #' + num(s.id) +
          '" onerror="this.classList.add(\'broken\')" />' +
          '<span class="fc-chip ' + (isMake ? "make" : "miss") + '">' + (isMake ? "MAKE" : "MISS") + '</span></div>' +
        '<div class="fc-foot"><span class="fc-zone">' + esc(cap(s.zone || "—")) + '</span>' +
          (bits.length ? '<span class="fc-stats">' + bits.join("") + '</span>' : '') + '</div>' +
      '</button>';
    }

    async function load() {
      const root = $("#formsContent");
      // keep the filter buttons in sync with state (e.g. when re-entering the view)
      $$("#formsFilter .filter-btn").forEach((b) => b.classList.toggle("active", b.dataset.filter === filter));
      root.innerHTML = '<div class="empty-state"><div class="spinner"></div></div>';
      try {
        const d = await api("/api/forms");
        shots = (d && d.shots) || [];
        loaded = true;
        render();
      } catch (e) {
        root.innerHTML = emptyState("&#9888;", "Couldn't load form snapshots", e.message);
      }
    }

    function setFilter(f) {
      if (f === filter) return;
      filter = f;
      $$("#formsFilter .filter-btn").forEach((b) => b.classList.toggle("active", b.dataset.filter === f));
      if (loaded) render();
    }

    function findShot(id) { return shots.filter((s) => num(s.id) === id)[0] || null; }

    return {
      init() {
        $("#formsFilter").addEventListener("click", (e) => {
          const b = e.target.closest(".filter-btn");
          if (b) setFilter(b.dataset.filter);
        });
        $("#formsContent").addEventListener("click", (e) => {
          const card = e.target.closest(".form-card");
          if (card) { const sh = findShot(+card.dataset.id); if (sh) openLightbox(sh); }
        });
        // lightbox: "Open session" jumps to the session modal
        $("#lightboxBody").addEventListener("click", (e) => {
          const b = e.target.closest("#lbOpenSession");
          if (b) { closeLightbox(); show("sessions"); openSession(+b.dataset.sid); }
        });
      },
      enter() { load(); }
    };
  })();

  /* ============================================================
     BOOT
     ============================================================ */
  function boot() {
    // nav
    $("#nav").addEventListener("click", (e) => {
      const b = e.target.closest("button[data-view]");
      if (b) show(b.dataset.view);
    });

    // init every view once (wires listeners)
    Object.keys(Views).forEach((k) => { if (Views[k].init) try { Views[k].init(); } catch (e) { console.error(e); } });

    // modal close
    $("#modalClose").addEventListener("click", closeModal);
    $("#sessionModal").addEventListener("click", (e) => { if (e.target === $("#sessionModal")) closeModal(); });

    // form lightbox close
    $("#lightboxClose").addEventListener("click", closeLightbox);
    $("#formLightbox").addEventListener("click", (e) => { if (e.target === $("#formLightbox")) closeLightbox(); });

    // Escape closes the lightbox first (it sits above the session modal), then the modal
    document.addEventListener("keydown", (e) => {
      if (e.key !== "Escape") return;
      if ($("#formLightbox").classList.contains("open")) { closeLightbox(); return; }
      if ($("#sessionModal").classList.contains("open")) closeModal();
    });

    // expose for cross-view calls
    window.HoopTracker = { show: show, openSession: openSession };

    // enter the default (Live) view
    currentView = null;
    show("live");
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", boot);
  else boot();

})();

/* HoopTracker mobile - on-device live make/miss counter + recorder.
   Detection: YOLO (ball/rim/person) via ONNX Runtime Web (wasm).
   Make/miss: window.HoopShot.ShotTracker (trajectory). */
(function () {
  const MODEL = 'models/detector.onnx';
  const INPUT = 416;
  const PREFILTER = 0.20;          // low pre-threshold; ShotTracker applies per-class conf
  const NMS_IOU = 0.45;
  const CLASS_KIND = { 0: 'ball', 1: 'rim', 2: 'person' };  // merged data.yaml order

  const $ = (id) => document.getElementById(id);
  const video = $('cam'), overlay = $('overlay'), octx = overlay.getContext('2d');
  const elMakes = $('makes'), elAtt = $('attempts'), elPct = $('pct'),
        elFlash = $('flash'), elStatus = $('status'), recBtn = $('recBtn'), resetBtn = $('resetBtn'),
        voiceBtn = $('voiceBtn'), setupBtn = $('setupBtn'), elSetup = $('setup'),
        chkHoop = $('chkHoop'), chkArc = $('chkArc'), chkYou = $('chkYou'), setupVerdict = $('setupVerdict');

  const pre = document.createElement('canvas'); pre.width = INPUT; pre.height = INPUT;
  const pctx = pre.getContext('2d', { willReadFrequently: true });

  let session = null, inName = null, outName = null;
  let tracker = new HoopShot.ShotTracker();
  let frameIdx = 0, active = false;
  let vw = 0, vh = 0, lb = { scale: 1, padX: 0, padY: 0 };
  let stream = null, recorder = null, chunks = [], recording = false;

  const setStatus = (t) => { elStatus.textContent = t; };

  // ---------- voice callout (offline, on-device text-to-speech) ----------
  let voiceOn = localStorage.getItem('hoop_voice') !== '0';   // default on
  function speak(text, force) {
    // force=true bypasses the voice toggle (used by Setup, where audio is the point)
    if ((!voiceOn && !force) || !('speechSynthesis' in window)) return;
    try {
      speechSynthesis.cancel();                 // always announce the latest
      const u = new SpeechSynthesisUtterance(text);
      u.rate = 1.05; u.lang = 'en-US';
      speechSynthesis.speak(u);
    } catch (e) {}
  }
  function updateVoiceBtn() {
    if (!voiceBtn) return;
    voiceBtn.textContent = voiceOn ? '🔊' : '🔇';
    voiceBtn.classList.toggle('off', !voiceOn);
  }

  // ---------- setup / framing check (works from any tripod spot) ----------
  let setupOn = false, lastSetupMsg = '';
  function setCheck(elc, state) {
    elc.className = 'chk ' + (state === true ? 'ok' : state === false ? 'bad' : 'wait');
    elc.querySelector('.ic').textContent = state === true ? '✓' : state === false ? '✕' : '…';
  }
  function evalSetup(dets) {
    const rim = dets.find(d => d.kind === 'rim');
    let person = null;
    for (const d of dets) if (d.kind === 'person' && (!person || d.w * d.h > person.w * person.h)) person = d;
    const rimOk = !!rim;
    const headOk = rim ? (rim.c[1] - rim.h / 2) > vh * 0.12 : false;   // space above the rim for the arc
    let youOk = null;
    if (person) {                                                        // not cropped at any edge
      const L = person.c[0] - person.w / 2, R = person.c[0] + person.w / 2,
            T = person.c[1] - person.h / 2, B = person.c[1] + person.h / 2;
      youOk = L > vw * 0.02 && R < vw * 0.98 && T > -vh * 0.01 && B < vh * 0.985;
    }
    setCheck(chkHoop, rimOk);
    setCheck(chkArc, rimOk ? headOk : 'wait');
    setCheck(chkYou, youOk === null ? 'wait' : youOk);
    let msg, good = false;
    if (!rimOk) msg = 'Point the camera at the hoop';
    else if (!headOk) msg = 'Leave room above the rim';
    else if (person && !youOk) msg = "You're cut off — move the camera back";
    else if (!person) msg = 'Hoop looks good — step into your spot';
    else { msg = 'Good to go'; good = true; }
    setupVerdict.textContent = good ? '✓ Good to go' : msg;
    setupVerdict.className = 'setup-verdict ' + (good ? 'good' : 'bad');
    if (msg !== lastSetupMsg) { lastSetupMsg = msg; speak(msg, true); } // setup always speaks (hear it from your spot)
  }
  function updateSetupBtn() { setupBtn.classList.toggle('on', setupOn); }

  async function initModel() {
    ort.env.wasm.numThreads = 1;     // WebView has no SharedArrayBuffer
    ort.env.wasm.simd = true;
    ort.env.wasm.wasmPaths = 'ort/';
    session = await ort.InferenceSession.create(MODEL, {
      executionProviders: ['wasm'], graphOptimizationLevel: 'all'
    });
    inName = session.inputNames[0];
    outName = session.outputNames[0];
  }

  async function initCamera() {
    stream = await navigator.mediaDevices.getUserMedia({
      audio: false,
      video: { facingMode: { ideal: 'environment' }, width: { ideal: 1280 }, height: { ideal: 720 } }
    });
    video.srcObject = stream;
    await video.play();
    if (!video.videoWidth) await new Promise(r => { video.onloadedmetadata = r; });
    vw = video.videoWidth; vh = video.videoHeight;
    overlay.width = vw; overlay.height = vh;
    lb.scale = Math.min(INPUT / vw, INPUT / vh);
    lb.padX = (INPUT - vw * lb.scale) / 2;
    lb.padY = (INPUT - vh * lb.scale) / 2;
  }

  function preprocess() {
    pctx.fillStyle = '#727272'; pctx.fillRect(0, 0, INPUT, INPUT);
    pctx.drawImage(video, lb.padX, lb.padY, vw * lb.scale, vh * lb.scale);
    const d = pctx.getImageData(0, 0, INPUT, INPUT).data;
    const n = INPUT * INPUT, f = new Float32Array(3 * n);
    for (let i = 0; i < n; i++) {
      f[i] = d[i * 4] / 255; f[n + i] = d[i * 4 + 1] / 255; f[2 * n + i] = d[i * 4 + 2] / 255;
    }
    return new ort.Tensor('float32', f, [1, 3, INPUT, INPUT]);
  }

  function iou(a, b) {
    const ix = Math.max(0, Math.min(a.cx + a.w / 2, b.cx + b.w / 2) - Math.max(a.cx - a.w / 2, b.cx - b.w / 2));
    const iy = Math.max(0, Math.min(a.cy + a.h / 2, b.cy + b.h / 2) - Math.max(a.cy - a.h / 2, b.cy - b.h / 2));
    const inter = ix * iy, uni = a.w * a.h + b.w * b.h - inter;
    return uni > 0 ? inter / uni : 0;
  }

  function parse(out) {
    const data = out.data, dims = out.dims, nc = dims[1] - 4, num = dims[2];
    const raw = [];
    for (let i = 0; i < num; i++) {
      let bc = -1, bs = 0;
      for (let c = 0; c < nc; c++) { const s = data[(4 + c) * num + i]; if (s > bs) { bs = s; bc = c; } }
      if (bs < PREFILTER) continue;
      raw.push({ c: bc, s: bs, cx: data[i], cy: data[num + i], w: data[2 * num + i], h: data[3 * num + i] });
    }
    raw.sort((a, b) => b.s - a.s);
    const keep = [];
    for (const d of raw) if (!keep.some(k => k.c === d.c && iou(k, d) > NMS_IOU)) keep.push(d);
    return keep.map(d => ({
      kind: CLASS_KIND[d.c], conf: d.s,
      c: [(d.cx - lb.padX) / lb.scale, (d.cy - lb.padY) / lb.scale],
      w: d.w / lb.scale, h: d.h / lb.scale
    }));
  }

  function draw(dets) {
    octx.clearRect(0, 0, vw, vh);
    for (const d of dets) {
      if (d.kind === 'person') continue;
      octx.strokeStyle = d.kind === 'ball' ? '#ffa61a' : '#ff4d4d';
      octx.lineWidth = Math.max(2, vw / 240);
      octx.strokeRect(d.c[0] - d.w / 2, d.c[1] - d.h / 2, d.w, d.h);
    }
    octx.fillStyle = '#ffa61a';
    for (const p of tracker.ball) { octx.beginPath(); octx.arc(p.c[0], p.c[1], Math.max(3, vw / 220), 0, 7); octx.fill(); }
  }

  function flash(result) {
    elFlash.className = 'show ' + result;
    elFlash.textContent = result === 'make' ? 'MAKE' : 'MISS';
    setTimeout(() => { elFlash.className = ''; }, 650);
  }

  function updateHUD() {
    elMakes.textContent = tracker.makes;
    elAtt.textContent = tracker.attempts;
    elPct.textContent = Math.round(tracker.fgPct) + '%';
  }

  async function tick() {
    if (!active) return;
    try {
      const res = await session.run({ [inName]: preprocess() });
      const dets = parse(res[outName]);
      const evt = tracker.update(dets, frameIdx++);
      draw(dets);
      if (setupOn) evalSetup(dets);
      if (evt) {
        flash(evt.result);
        updateHUD();
        speak(tracker.makes + ' of ' + tracker.attempts + ', ' + Math.round(tracker.fgPct) + ' percent');
      }
    } catch (e) { setStatus('err: ' + (e && e.message ? e.message : e)); }
    requestAnimationFrame(tick);
  }

  // ---------- recording (raw camera -> saved for desktop import) ----------
  function pickMime() {
    for (const m of ['video/webm;codecs=vp9', 'video/webm;codecs=vp8', 'video/webm', 'video/mp4'])
      if (window.MediaRecorder && MediaRecorder.isTypeSupported(m)) return m;
    return '';
  }

  async function saveBlob(blob, name) {
    const cap = window.Capacitor;
    if (cap && cap.Plugins && cap.Plugins.Filesystem) {
      const b64 = await new Promise(r => { const fr = new FileReader(); fr.onloadend = () => r(String(fr.result).split(',')[1]); fr.readAsDataURL(blob); });
      await cap.Plugins.Filesystem.writeFile({ path: 'HoopTracker/' + name, data: b64, directory: 'DOCUMENTS', recursive: true });
      setStatus('Saved: Documents/HoopTracker/' + name);
    } else {
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a'); a.href = url; a.download = name;
      document.body.appendChild(a); a.click();
      setTimeout(() => { URL.revokeObjectURL(url); a.remove(); }, 1500);
      setStatus('Downloaded ' + name);
    }
  }

  function startRec() {
    const mime = pickMime(); chunks = [];
    recorder = new MediaRecorder(stream, mime ? { mimeType: mime } : undefined);
    recorder.ondataavailable = e => { if (e.data && e.data.size) chunks.push(e.data); };
    recorder.onstop = async () => {
      const type = chunks[0] ? chunks[0].type : 'video/webm';
      const ext = type.indexOf('mp4') >= 0 ? 'mp4' : 'webm';
      await saveBlob(new Blob(chunks, { type }), 'session_' + Date.now() + '.' + ext);
    };
    recorder.start(1000);
    recording = true;
    recBtn.classList.add('recording');
    recBtn.innerHTML = '<span class="rec-dot"></span>Stop';
  }

  function stopRec() {
    if (recorder && recording) recorder.stop();
    recording = false;
    recBtn.classList.remove('recording');
    recBtn.innerHTML = '<span class="rec-dot"></span>Start';
  }

  recBtn.addEventListener('click', () => { recording ? stopRec() : startRec(); });
  resetBtn.addEventListener('click', () => {
    tracker = new HoopShot.ShotTracker(); frameIdx = 0; updateHUD(); octx.clearRect(0, 0, vw, vh);
    if ('speechSynthesis' in window) speechSynthesis.cancel();
  });
  voiceBtn.addEventListener('click', () => {
    voiceOn = !voiceOn;
    localStorage.setItem('hoop_voice', voiceOn ? '1' : '0');
    if (!voiceOn && 'speechSynthesis' in window) speechSynthesis.cancel();
    updateVoiceBtn();
    setStatus(voiceOn ? 'Voice on — calling your count after each shot.' : 'Voice off.');
  });
  updateVoiceBtn();

  setupBtn.addEventListener('click', () => {
    setupOn = !setupOn;
    elSetup.hidden = !setupOn;
    if (!setupOn) { lastSetupMsg = ''; if ('speechSynthesis' in window) speechSynthesis.cancel(); }
    updateSetupBtn();
    if (setupOn) setStatus('Setup check — aim at the hoop, then step into your spot.');
  });
  updateSetupBtn();

  (async function boot() {
    try {
      setStatus('Loading detector…');
      await initModel();
      setStatus('Starting camera…');
      await initCamera();
      active = true; recBtn.disabled = false;
      setStatus('Point at the hoop - counting live. Tap Start to also record.');
      requestAnimationFrame(tick);
    } catch (e) {
      setStatus('Startup error: ' + (e && e.message ? e.message : e));
    }
  })();
})();

# HoopTracker 🏀

A local PC app that tracks your basketball shooting **in real time** (webcam) and from **imported video**, then helps you improve with **progress analytics** and **shooting-form coaching**.

Detection runs on the GPU (RTX 5070 Ti, ~7 ms/frame — ~145 FPS of headroom), so it can run the detector + pose + court models together and stay well above real time.

## What it does
- **Live mode** — point a webcam (or your phone as a wireless webcam) at the hoop; get real-time make/miss calls and a running FG%.
- **Import mode** — drop in a clip; get an annotated video + a full shot breakdown.
- **Progress** — per-session & lifetime FG%, trends over time, by-zone shot chart, streaks, hot/cold, personal bests.
- **Form coaching** — per-shot body-mechanics read (release, elbow, knee bend, balance, follow-through) via YOLO-pose, with data-driven tips.

## How it's built — a fusion of two reference projects + a new analytics layer
| Piece | Origin |
|---|---|
| Make/miss via ball-arc trajectory → rim plane | adapted from `avishah3/AI-Basketball-Shot-Detection-Tracker` |
| Live webcam realtime loop | avishah3 |
| Video upload → process → annotate flow, HUD/overlays | inspired by `sPappalard/SwishAI` |
| Custom detector (ball / hoop / player) | trained here on 3 Roboflow datasets |
| Court-zone mapping (shot chart) | trained here on `basketball-courts-class` |
| Shooting-form / pose coaching | YOLO-pose (Ultralytics) |
| Progress DB, trends, coaching tips | new in this project |

## Project structure
```
HoopTracker/
├── app.py              # Flask app: live webcam (MJPEG), video upload, routes
├── config.py           # paths, model files, thresholds, port
├── detection/          # YOLO engine, trajectory make/miss, pose, court mapping
├── stats/              # SQLite persistence + insights/coaching engine
├── training/           # dataset download + merge + train scripts
├── models/             # trained weights (.pt)  [gitignored]
├── static/  templates/ # dashboard frontend
├── uploads/ processed/ # user videos + annotated output  [gitignored]
└── data/               # hooptracker.db  [gitignored]
```

## Models
- `models/detector.pt` — custom ball/hoop/player detector (trained on the 5070 Ti)
- `models/court.pt` — court-zone model (shot chart)
- `models/yolo11s-pose.pt` — pretrained pose (shooting form)
- `models/best_fallback.pt` — avishah's 2-class model; loads + runs today as a fallback

## Setup
Reuses the system Python env (already has `torch 2.11.0+cu128` + `ultralytics`).
```
pip install -r requirements.txt
```
(Read the note at the top of `requirements.txt` so pip doesn't replace your CUDA torch.)

## Run
```
start.bat        # or:  python app.py   →   http://127.0.0.1:8791
```

## Credits & license
- Shot-detection algorithm adapted from **avishah3/AI-Basketball-Shot-Detection-Tracker** (with credit).
- App/UX patterns inspired by **sPappalard/SwishAI** (AGPL-3.0).
- Detection & pose powered by **Ultralytics YOLO** (AGPL-3.0) — fine for personal/local use; publicly **hosting** it as a service would trigger AGPL source-disclosure.
- Datasets via Roboflow Universe — per-dataset citations & licenses live in `training/`.

## Status
Work in progress. Current phase: dataset spec → unified training set → train detector.

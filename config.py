"""HoopTracker — central configuration.

Values marked (finalized after dataset spec) get tuned once the unified
class schema from the 4 Roboflow datasets is known.
"""
from pathlib import Path

# --- Paths ---
ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"
UPLOAD_DIR = ROOT / "uploads"
PROCESSED_DIR = ROOT / "processed"
DATA_DIR = ROOT / "data"
DB_PATH = DATA_DIR / "hooptracker.db"

for _d in (MODELS_DIR, UPLOAD_DIR, PROCESSED_DIR, DATA_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --- Model files ---
DETECTOR_WEIGHTS = MODELS_DIR / "detector.pt"       # custom ball/hoop/player detector
POSE_WEIGHTS     = MODELS_DIR / "yolo11s-pose.pt"   # pretrained pose -> shooting form
COURT_WEIGHTS    = MODELS_DIR / "court.pt"          # court-zone model (dataset 4)
FALLBACK_WEIGHTS = MODELS_DIR / "best_fallback.pt"  # avishah 2-class model (works now)

# --- Inference ---
DEVICE = 0          # CUDA device index; set to "cpu" to force CPU
IMG_SIZE = 640
HALF = True         # FP16 inference (fine on Blackwell)

# Per-class confidence thresholds (unified schema; finalized after dataset spec)
CONF = {
    "ball": 0.35,
    "hoop": 0.45,
    "player": 0.50,
}

# --- Shot detection (ported from avishah trajectory logic) ---
BALL_TRACK_MAX_FRAMES = 30   # frames a ball point stays in the buffer
SHOT_COOLDOWN_S = 1.2        # min seconds between counted attempts

# --- Server ---
HOST = "127.0.0.1"
PORT = 8791

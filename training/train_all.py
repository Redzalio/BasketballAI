"""Train the detector then the court model sequentially on the GPU.

Run in the background:  python training/train_all.py
Outputs: models/detector.pt  +  models/court.pt
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable
SCRIPTS = ["train_detector.py", "train_court.py"]

for s in SCRIPTS:
    print(f"\n{'=' * 60}\nRUNNING {s}\n{'=' * 60}", flush=True)
    code = subprocess.run([PY, str(ROOT / "training" / s)]).returncode
    if code != 0:
        print(f"!! {s} exited with code {code}; stopping pipeline.", flush=True)
        sys.exit(code)

det = ROOT / "models" / "detector.pt"
court = ROOT / "models" / "court.pt"
print(f"\n{'=' * 60}\nALL TRAINING DONE\n"
      f"  detector.pt present: {det.exists()}\n"
      f"  court.pt present:    {court.exists()}\n{'=' * 60}", flush=True)

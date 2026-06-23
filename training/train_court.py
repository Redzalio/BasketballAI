"""Train the court-geometry model (court lines + hoop hardware) on dataset #4 (v11).

Detects Three Point Line / Free Throw Line / Half-Court-Line (+ rim/backboard/net)
so the app can map a shot's location to a court zone for the shot chart.

Run after download_datasets.py.  Output: models/court.pt
"""
from pathlib import Path
import shutil
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
DS = ROOT / "datasets" / "courts"
MODELS = ROOT / "models"
BASE = "yolo11s.pt"


def find_data_yaml():
    if (DS / "data.yaml").exists():
        return DS / "data.yaml"
    nested = list(DS.glob("*/data.yaml"))
    return nested[0] if nested else None


def main():
    data = find_data_yaml()
    assert data, f"Missing data.yaml under {DS}. Run download_datasets.py first."
    # Dataset #4 has a tiny valid/test split — fine for static court registration.
    model = YOLO(BASE)
    model.train(
        data=str(data),
        epochs=120,
        imgsz=640,
        batch=32,
        device=0,
        workers=8,         # validated OK on Windows/Py3.14 in the detector smoke test
        optimizer="AdamW",
        lr0=0.001, cos_lr=True,
        patience=30,
        # court lines are long/thin -> gentler geometric aug, keep perspective
        degrees=0.0, shear=1.0, perspective=0.0005,
        flipud=0.0, fliplr=0.5, mosaic=1.0, mixup=0.0,
        project=str(ROOT / "runs"), name="court", exist_ok=True,
        plots=True, seed=42,
    )
    best = ROOT / "runs" / "court" / "weights" / "best.pt"
    if best.exists():
        MODELS.mkdir(exist_ok=True)
        shutil.copy2(best, MODELS / "court.pt")
        print(f"\n[OK] Court model saved -> {MODELS / 'court.pt'}")
    else:
        print("!! best.pt not found under runs/court/weights/")


if __name__ == "__main__":
    main()

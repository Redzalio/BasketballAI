"""Train the unified ball/rim/person detector on the merged dataset (RTX 5070 Ti).

Run after download_datasets.py + merge_datasets.py.
Output: models/detector.pt
"""
from pathlib import Path
import shutil
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "datasets" / "merged_detector" / "data.yaml"
MODELS = ROOT / "models"
BASE = "yolo11s.pt"   # small variant: great accuracy + fast on this GPU


def main():
    assert DATA.exists(), f"Missing {DATA}. Run download_datasets.py then merge_datasets.py first."
    model = YOLO(BASE)
    model.train(
        data=str(DATA),
        epochs=100,
        imgsz=640,
        batch=32,          # ~8 GB VRAM (smoke test: 4 GB at bs16); fast on the 5070 Ti
        device=0,
        workers=8,         # if you hit a Windows multiprocessing error, set to 0
        optimizer="AdamW",
        lr0=0.001, cos_lr=True,
        patience=25,       # early stop when converged
        close_mosaic=10,
        # sports-tuned augmentation (orange ball, varied light, court angles)
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
        degrees=0.0, translate=0.1, scale=0.5, shear=2.0, perspective=0.0005,
        flipud=0.0, fliplr=0.5, mosaic=1.0, mixup=0.1,
        project=str(ROOT / "runs"), name="detector", exist_ok=True,
        plots=True, seed=42,
    )
    best = ROOT / "runs" / "detector" / "weights" / "best.pt"
    if best.exists():
        MODELS.mkdir(exist_ok=True)
        shutil.copy2(best, MODELS / "detector.pt")
        print(f"\n[OK] Detector saved -> {MODELS / 'detector.pt'}")
    else:
        print("!! best.pt not found under runs/detector/weights/")


if __name__ == "__main__":
    main()

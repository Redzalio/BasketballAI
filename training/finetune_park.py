"""Fine-tune the ball/rim/person detectors on the user's park court (domain adaptation).

Transfer-learns FROM the existing trained models (which already know the 3 classes)
on the combined merged+park dataset, so the model adapts to the outdoor court/lighting
without forgetting general detection. Originals are NOT overwritten — outputs go to
models/*_ft.pt for evaluation before promotion.

  s : models/detector.pt   (YOLO11s @ 640) -> models/detector_ft.pt     (PC app)
  n : models/detector_n.pt (YOLO11n @ 416) -> models/detector_n_ft.pt   (mobile / APK)

Run:  python training/finetune_park.py        (nano then 11s — nano is the mobile fix)
      python training/finetune_park.py n      (just one)
"""
import sys, os
# make per-user site-packages importable no matter how this is launched (overlay/AV-proof)
sys.path.insert(0, os.path.join(os.path.expanduser("~"),
                                "AppData", "Roaming", "Python", "Python314", "site-packages"))
from pathlib import Path
import shutil
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "datasets" / "finetune_detector.yaml"   # train=combined, val=park
PARK = ROOT / "datasets" / "park_only.yaml"
MODELS = ROOT / "models"

VARIANTS = {
    "n": dict(base="detector_n.pt", out="detector_n_ft.pt", imgsz=416, batch=64, epochs=60),
    "s": dict(base="detector.pt",   out="detector_ft.pt",   imgsz=640, batch=32, epochs=50),
}


def run(key, cfg):
    base = MODELS / cfg["base"]
    assert base.exists(), f"missing {base}"
    print(f"\n===== fine-tune [{key}] from {cfg['base']} @ {cfg['imgsz']} "
          f"(bs={cfg['batch']}, {cfg['epochs']} ep) =====", flush=True)
    model = YOLO(str(base))
    model.train(
        data=str(DATA),
        epochs=cfg["epochs"], imgsz=cfg["imgsz"], batch=cfg["batch"],
        device=0, workers=8,
        optimizer="AdamW", lr0=0.001, cos_lr=True,
        patience=20, close_mosaic=10,
        # sports-tuned aug; scale bumped to 0.6 to help the small/distant outdoor ball
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
        degrees=0.0, translate=0.1, scale=0.6, shear=2.0, perspective=0.0005,
        flipud=0.0, fliplr=0.5, mosaic=1.0, mixup=0.1,
        project=str(ROOT / "runs"), name=f"finetune_{key}", exist_ok=True,
        plots=True, seed=42,
    )
    best = ROOT / "runs" / f"finetune_{key}" / "weights" / "best.pt"
    if best.exists():
        shutil.copy2(best, MODELS / cfg["out"])
        print(f"[OK] {key}: saved -> {MODELS / cfg['out']}", flush=True)
        m = YOLO(str(MODELS / cfg["out"]))
        r = m.val(data=str(PARK), imgsz=cfg["imgsz"], device=0, workers=4,
                  plots=False, verbose=False)
        print(f"[{key}] PARK val  mAP50={r.box.map50:.3f}  mAP50-95={r.box.map:.3f}", flush=True)
    else:
        print(f"!! best.pt missing for {key}", flush=True)


def main():
    keys = [a for a in sys.argv[1:] if a in VARIANTS] or ["n", "s"]
    print(f"fine-tuning variants: {keys}", flush=True)
    for k in keys:
        run(k, VARIANTS[k])
    print("\nDONE. Next: python training/eval_finetune.py", flush=True)


if __name__ == "__main__":
    main()

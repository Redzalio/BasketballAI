"""Promote the park-fine-tuned detectors and export the mobile ONNX.

Run only AFTER eval_finetune.py confirms park mAP improved without forgetting.
  - backs up the current models as *_pre_park.{pt,onnx}
  - promotes detector_ft.pt -> detector.pt and detector_n_ft.pt -> detector_n.pt
  - re-exports ONNX (nano @416 for mobile, 11s @640 for parity)
  - copies the nano ONNX into mobile/www/models/detector.onnx (what the APK loads)

Model files are gitignored, so this only touches the user's real disk (no commit).
"""
import sys, os, shutil
sys.path.insert(0, os.path.join(os.path.expanduser("~"),
                                "AppData", "Roaming", "Python", "Python314", "site-packages"))
from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
M = ROOT / "models"
MOBILE = ROOT / "mobile" / "www" / "models"

def backup(name):
    src = M / name
    if src.exists():
        dst = M / f"{src.stem}_pre_park{src.suffix}"
        shutil.copy2(src, dst)
        print(f"  backup {name} -> {dst.name}")

def main():
    for f in ("detector.pt", "detector_n.pt", "detector.onnx", "detector_n.onnx"):
        backup(f)

    assert (M / "detector_ft.pt").exists() and (M / "detector_n_ft.pt").exists(), "fine-tuned weights missing"
    shutil.copy2(M / "detector_ft.pt", M / "detector.pt")
    shutil.copy2(M / "detector_n_ft.pt", M / "detector_n.pt")
    print("[OK] promoted detector.pt + detector_n.pt")

    # nano ONNX (mobile) — critical
    YOLO(str(M / "detector_n.pt")).export(format="onnx", opset=17, imgsz=416)
    MOBILE.mkdir(parents=True, exist_ok=True)
    shutil.copy2(M / "detector_n.onnx", MOBILE / "detector.onnx")
    print(f"[OK] nano ONNX -> {MOBILE / 'detector.onnx'}")

    # 11s ONNX (parity; PC app uses the .pt directly so this is just housekeeping)
    try:
        YOLO(str(M / "detector.pt")).export(format="onnx", opset=17, imgsz=640)
        print("[OK] 11s ONNX re-exported")
    except Exception as e:
        print(f"[warn] 11s ONNX export skipped: {e}")

    print("\nDONE. Rebuild the APK (mobile/build_apk.bat) to ship the new nano detector.")

if __name__ == "__main__":
    main()

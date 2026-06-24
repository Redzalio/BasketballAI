"""Before/after comparison for the park fine-tune.

For each variant, val the ORIGINAL and the FINE-TUNED model on:
  - park   (the user's court — should go UP)
  - merged (the original 10k val — should NOT drop much = no catastrophic forgetting)

Prints a 2x2 table per variant so we can decide whether to promote.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.expanduser("~"),
                                "AppData", "Roaming", "Python", "Python314", "site-packages"))
from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"
PARK = ROOT / "datasets" / "park_only.yaml"
MERGED = ROOT / "datasets" / "merged_only.yaml"

PAIRS = {
    "n (mobile, 416)": ("detector_n.pt", "detector_n_ft.pt", 416),
    "s (PC, 640)":     ("detector.pt",   "detector_ft.pt",   640),
}


def val(model_path, data, imgsz):
    if not (MODELS / model_path).exists():
        return None
    m = YOLO(str(MODELS / model_path))
    r = m.val(data=str(data), imgsz=imgsz, device=0, workers=4, plots=False, verbose=False)
    return r.box.map50, r.box.map


def fmt(x):
    return "  --  " if x is None else f"{x[0]:.3f}/{x[1]:.3f}"


def main():
    print(f"{'variant':<18}{'set':<8}{'ORIGINAL':<16}{'FINE-TUNED':<16}{'Δ mAP50'}")
    print("-" * 64)
    for label, (orig, ft, imgsz) in PAIRS.items():
        for ds_name, ds in (("park", PARK), ("merged", MERGED)):
            o = val(orig, ds, imgsz)
            f = val(ft, ds, imgsz)
            delta = "" if (o is None or f is None) else f"{(f[0]-o[0]):+.3f}"
            print(f"{label:<18}{ds_name:<8}{fmt(o):<16}{fmt(f):<16}{delta}")
        print("-" * 64)
    print("(cells show mAP50/mAP50-95.  park should rise; merged should hold.)")


if __name__ == "__main__":
    main()

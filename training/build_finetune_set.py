"""Build a fine-tune dataset = original merged_detector + the user's park_court frames.

park_court is exported from Roboflow with two quirks this script normalizes:
  1. class order is ball=0, person=1, rim=2  -> remapped by NAME to the canonical
     ball=0, rim=1, person=2 (same order as merged_detector / the live model).
  2. labels are POLYGON / segmentation (variable-length coord lists, from SAM
     Label Assist) -> converted to YOLO detection bboxes (min/max of the polygon).

Park *train* images are oversampled (duplicated) so the small ~117-image court set
carries real weight against the ~10k original images during fine-tuning. We use a
path-LIST data.yaml so merged_detector does NOT need to be copied.

Outputs:
  datasets/park_remapped/{train,valid,test}/{images,labels}
  datasets/finetune_detector.yaml   (merged + park — used for training)
  datasets/park_only.yaml           (park val only — used for before/after reporting)
"""
from pathlib import Path
import shutil
import yaml

ROOT = Path(__file__).resolve().parent.parent
DS = ROOT / "datasets"
SRC = DS / "park_court"
DST = DS / "park_remapped"

UNIFIED = ["ball", "rim", "person"]          # canonical order — matches merged_detector
UID = {n: i for i, n in enumerate(UNIFIED)}
OVERSAMPLE_TRAIN = 15                         # ~117 -> ~1755 train copies (~15% of combined)


def src_names():
    with open(SRC / "data.yaml") as f:
        d = yaml.safe_load(f)
    names = d["names"]
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names, key=lambda k: int(k))]
    return [str(n).strip().lower() for n in names]


def convert_line(line, names):
    """Remap class by name + convert polygon->bbox. Returns 'cls cx cy w h' or None."""
    parts = line.split()
    if len(parts) < 5:
        return None
    cid = int(float(parts[0]))
    if not (0 <= cid < len(names)):
        return None
    uni = names[cid]
    if uni not in UID:
        return None
    coords = [float(x) for x in parts[1:]]
    if len(coords) == 4:
        cx, cy, w, h = coords
    else:                                     # polygon: x1 y1 x2 y2 ... -> bbox
        xs, ys = coords[0::2], coords[1::2]
        xmin, xmax, ymin, ymax = min(xs), max(xs), min(ys), max(ys)
        cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2
        w, h = xmax - xmin, ymax - ymin
    cx, cy = min(max(cx, 0), 1), min(max(cy, 0), 1)
    w, h = min(max(w, 0), 1), min(max(h, 0), 1)
    if w <= 0 or h <= 0:
        return None
    return f"{UID[uni]} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


def remap_file(src_txt, names):
    out = []
    for line in src_txt.read_text().splitlines():
        line = line.strip()
        if line:
            r = convert_line(line, names)
            if r:
                out.append(r)
    return out


def split_src(split):
    cands = ["valid", "val"] if split == "valid" else [split]
    for c in cands:
        if (SRC / c / "images").exists():
            return SRC / c
    return None


def main():
    names = src_names()
    print("park source class order:", names, "-> canonical:", UNIFIED)
    if DST.exists():
        shutil.rmtree(DST)

    for split in ("train", "valid", "test"):
        s = split_src(split)
        (DST / split / "images").mkdir(parents=True, exist_ok=True)
        (DST / split / "labels").mkdir(parents=True, exist_ok=True)
        if not s:
            continue
        reps = OVERSAMPLE_TRAIN if split == "train" else 1
        tally = {n: 0 for n in UNIFIED}
        imgs = poly = 0
        for img in (s / "images").iterdir():
            if img.suffix.lower() not in (".jpg", ".jpeg", ".png", ".bmp"):
                continue
            slbl = s / "labels" / f"{img.stem}.txt"
            if slbl.exists():
                raw = [l for l in slbl.read_text().splitlines() if l.strip()]
                poly += sum(1 for l in raw if len(l.split()) > 5)
            lines = remap_file(slbl, names) if slbl.exists() else []
            for l in lines:
                tally[UNIFIED[int(l.split()[0])]] += 1
            imgs += 1
            for k in range(reps):
                stem = f"{img.stem}_c{k}" if reps > 1 else img.stem
                shutil.copy2(img, DST / split / "images" / f"{stem}{img.suffix}")
                (DST / split / "labels" / f"{stem}.txt").write_text(
                    "\n".join(lines) + ("\n" if lines else ""))
        print(f"  [{split}] {imgs} src images x{reps} | instances {tally} | polygons converted: {poly}")

    ds_posix = DS.as_posix()
    PARK_VAL = ["park_remapped/valid/images", "park_remapped/test/images"]  # 17 imgs
    # Train on combined (so general detection is retained); validate on PARK only
    # (so checkpoint selection / early-stop tracks the actual adaptation goal).
    finetune_yaml = {
        "path": ds_posix,
        "train": ["merged_detector/train/images", "park_remapped/train/images"],
        "val": PARK_VAL,
        "nc": len(UNIFIED),
        "names": {i: n for i, n in enumerate(UNIFIED)},
    }
    park_yaml = {
        "path": ds_posix,
        "train": "park_remapped/train/images",
        "val": PARK_VAL,
        "nc": len(UNIFIED),
        "names": {i: n for i, n in enumerate(UNIFIED)},
    }
    merged_yaml = {                              # for the catastrophic-forgetting check
        "path": ds_posix,
        "train": "merged_detector/train/images",
        "val": "merged_detector/valid/images",
        "nc": len(UNIFIED),
        "names": {i: n for i, n in enumerate(UNIFIED)},
    }
    with open(DS / "finetune_detector.yaml", "w") as f:
        yaml.safe_dump(finetune_yaml, f, sort_keys=False)
    with open(DS / "park_only.yaml", "w") as f:
        yaml.safe_dump(park_yaml, f, sort_keys=False)
    with open(DS / "merged_only.yaml", "w") as f:
        yaml.safe_dump(merged_yaml, f, sort_keys=False)

    print(f"\nWrote {DS / 'finetune_detector.yaml'}  (train=combined, val=park)")
    print(f"Wrote {DS / 'park_only.yaml'}  (val=park, 17 imgs)")
    print(f"Wrote {DS / 'merged_only.yaml'}  (val=original, forgetting check)")
    print("Next: python training/finetune_park.py")


if __name__ == "__main__":
    main()

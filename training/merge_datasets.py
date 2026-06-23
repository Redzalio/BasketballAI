"""Merge datasets 1-3 into one unified ball/rim/person detector dataset.

Robust to messy source classes: remaps by class *name* (not index), so dupes
(ball/basketball, people/person), junk classes (literal "0"), and dropped/extra
classes are all handled automatically. Court dataset (#4) is NOT merged here —
it's trained separately by train_court.py.
"""
from pathlib import Path
import shutil, sys
import yaml

ROOT = Path(__file__).resolve().parent.parent
DATASETS_DIR = ROOT / "datasets"
OUT = DATASETS_DIR / "merged_detector"

# shooting_robot is EXCLUDED: its v1 export collapsed to rim-only (nc:1), so its
# images contain unlabeled balls/people that would poison those classes if merged.
SOURCES = ["ownprojects", "people_hoops"]

UNIFIED = ["ball", "rim", "person"]
UID = {name: i for i, name in enumerate(UNIFIED)}

# source class name (lowercased) -> unified name, or None to DROP
NAME_MAP = {
    "ball": "ball", "basketball": "ball",
    "rim": "rim", "hoop": "rim", "basket": "rim",
    "basketball rim": "rim", "basketball hoop": "rim",
    "person": "person", "people": "person", "player": "person", "players": "person",
    # explicit drops (weak event-classes, junk, court hardware/lines)
    "made": None, "shoot": None, "shooting": None, "0": None,
    "basketball backboard": None, "backboard": None,
    "basketball net": None, "net": None,
    "free throw line": None, "half-court-line": None, "half court line": None,
    "three point line": None, "3-point line": None,
}


def load_names(yaml_path):
    with open(yaml_path) as f:
        data = yaml.safe_load(f)
    names = data.get("names")
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names, key=lambda k: int(k))]
    return names


def remap_label(src_txt, dst_txt, src_names, dropped, kept):
    out = []
    for line in src_txt.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        cid = int(float(parts[0]))
        if not (0 <= cid < len(src_names)):
            continue
        name = str(src_names[cid]).strip().lower()
        unified = NAME_MAP.get(name, "__UNKNOWN__")
        if unified is None:
            dropped[name] = dropped.get(name, 0) + 1
            continue
        if unified == "__UNKNOWN__":
            dropped[f"?{name}"] = dropped.get(f"?{name}", 0) + 1
            continue
        parts[0] = str(UID[unified])
        kept[unified] = kept.get(unified, 0) + 1
        out.append(" ".join(parts))
    dst_txt.write_text("\n".join(out) + ("\n" if out else ""))


def split_dir(base, split):
    cands = ["valid", "val"] if split == "valid" else [split]
    for c in cands:
        if (base / c).exists():
            return base / c
    return None


def resolve_base(name):
    base = DATASETS_DIR / name
    if (base / "data.yaml").exists():
        return base
    nested = list(base.glob("*/data.yaml"))
    return nested[0].parent if nested else base


def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    for split in ("train", "valid", "test"):
        (OUT / split / "images").mkdir(parents=True, exist_ok=True)
        (OUT / split / "labels").mkdir(parents=True, exist_ok=True)

    dropped, kept = {}, {}
    total = 0
    for src in SOURCES:
        base = resolve_base(src)
        yml = base / "data.yaml"
        if not yml.exists():
            print(f"!! missing {yml} — run download_datasets.py first; skipping {src}")
            continue
        names = load_names(yml)
        print(f"[{src}] source classes: {names}")
        for split in ("train", "valid", "test"):
            sdir = split_dir(base, split)
            if not sdir or not (sdir / "images").exists():
                continue
            for img in (sdir / "images").iterdir():
                if img.suffix.lower() not in (".jpg", ".jpeg", ".png", ".bmp"):
                    continue
                shutil.copy2(img, OUT / split / "images" / f"{src}__{img.name}")
                slbl = sdir / "labels" / f"{img.stem}.txt"
                dlbl = OUT / split / "labels" / f"{src}__{img.stem}.txt"
                if slbl.exists():
                    remap_label(slbl, dlbl, names, dropped, kept)
                else:
                    dlbl.write_text("")  # background frame
                total += 1

    data_yaml = {
        "train": (OUT / "train" / "images").as_posix(),
        "val": (OUT / "valid" / "images").as_posix(),
        "test": (OUT / "test" / "images").as_posix(),
        "nc": len(UNIFIED),
        "names": UNIFIED,
    }
    with open(OUT / "data.yaml", "w") as f:
        yaml.safe_dump(data_yaml, f, sort_keys=False)

    print(f"\nMerged {total} images -> {OUT}")
    print("Kept (unified):", kept)
    print("Dropped (source name -> count):", dropped)
    print(f"\ndata.yaml: {OUT / 'data.yaml'}")
    print("Next: python training/train_detector.py")


if __name__ == "__main__":
    main()

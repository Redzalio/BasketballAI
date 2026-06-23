"""Download the 4 Roboflow datasets as YOLO exports.

Uses the Roboflow REST export API via `requests` so the validated torch/CUDA
env stays untouched (no roboflow SDK / dependency churn).

API key (free): roboflow.com -> Settings -> Roboflow API Key (Private).
Provide it via --key, the ROBOFLOW_API_KEY env var, or a line
    ROBOFLOW_API_KEY=xxxxxxxx
in HoopTracker/.env  (gitignored).

Test one first:   python training/download_datasets.py --only people_hoops
Then all:         python training/download_datasets.py
"""
from pathlib import Path
import argparse, io, os, sys, time, zipfile
import requests

ROOT = Path(__file__).resolve().parent.parent
DATASETS_DIR = ROOT / "datasets"
FMT = "yolov11"  # YOLO TXT format (same layout as yolov8)

DATASETS = [
    dict(name="shooting_robot", ws="the-university-of-arizona-th1yv", proj="basketball-shooting-robot", ver=1),
    dict(name="ownprojects",    ws="ownprojects",                     proj="basketball-w2xcw",          ver=2),
    dict(name="people_hoops",   ws="mytem",                           proj="people_basketball_hoops",   ver=4),
    dict(name="courts",         ws="shotanalyzer-workspsace",         proj="basketball-courts-class",   ver=11),
]


def get_key(cli_key):
    if cli_key:
        return cli_key.strip()
    if os.environ.get("ROBOFLOW_API_KEY"):
        return os.environ["ROBOFLOW_API_KEY"].strip()
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            if line.strip().startswith("ROBOFLOW_API_KEY"):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    sys.exit("No API key. Use --key, ROBOFLOW_API_KEY env var, or HoopTracker/.env")


def export_link(ds, key, tries=40):
    """Ask Roboflow for the export zip link, polling while it generates."""
    url = f"https://api.roboflow.com/{ds['ws']}/{ds['proj']}/{ds['ver']}/{FMT}"
    for i in range(tries):
        r = requests.get(url, params={"api_key": key}, timeout=60)
        if r.status_code == 401:
            sys.exit("401 Unauthorized — bad/blank API key.")
        try:
            j = r.json()
        except Exception:
            time.sleep(5)
            continue
        link = (j.get("export") or {}).get("link") or j.get("link")
        if link:
            return link
        if i == 0:
            print(f"   export generating... (response keys: {list(j.keys())})")
        time.sleep(min(5 + i, 20))
    sys.exit(f"Timed out waiting for export of {ds['name']}.")


def download(ds, key):
    dest = DATASETS_DIR / ds["name"]
    if (dest / "data.yaml").exists() or list(dest.glob("*/data.yaml")):
        print(f"[{ds['name']}] already present, skipping.")
        return
    print(f"[{ds['name']}] requesting export ({ds['ws']}/{ds['proj']} v{ds['ver']})...")
    link = export_link(ds, key)
    print(f"[{ds['name']}] downloading zip...")
    z = requests.get(link, timeout=600)
    z.raise_for_status()
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(z.content)) as zf:
        zf.extractall(dest)
    n = len(list((dest).rglob("*.jpg"))) + len(list((dest).rglob("*.png")))
    print(f"[{ds['name']}] extracted ~{n} images -> {dest}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--key", default=None)
    ap.add_argument("--only", default=None, help="comma list of dataset names")
    args = ap.parse_args()
    key = get_key(args.key)
    DATASETS_DIR.mkdir(exist_ok=True)
    targets = DATASETS
    if args.only:
        want = set(args.only.split(","))
        targets = [d for d in DATASETS if d["name"] in want]
    for ds in targets:
        download(ds, key)
    print("\nDone. Next: python training/merge_datasets.py")


if __name__ == "__main__":
    main()

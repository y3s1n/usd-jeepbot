"""
JeepBot - draft boxes for Label Studio

Usage:
    python prelabel.py --stride 16 --offset 0     # one annotator's file
    python prelabel.py --only _A_                 # just section A folders
    python prelabel.py --no-hydrant               # person drafts only

Runs the pretrained COCO detector over a spread-out batch of frames and
writes a Label Studio import file with DRAFT boxes for person and
fire_hydrant. COCO has no trash can class - those are drawn by hand.

Every draft must be checked by a person: fix it, delete it, or add what
it missed. Read-only on the dataset.

Frames inside each file are shuffled (seeded by the offset, so the same
command always gives the same order). Any partly finished file is
therefore a random spread across all folders, lighting and sections.
"""

import argparse
import json
import random
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "dataset"
OUT_DIR = ROOT / "labelstudio"

MODEL = str(ROOT / "yolov8s.pt")
IMGSZ = 1280

# COCO class id -> (our label, minimum confidence for a draft box)
COCO = {
    0: ("person", 0.35),
    10: ("fire_hydrant", 0.30),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--only", default=None,
                    help="only folders whose name contains this text")
    ap.add_argument("--no-hydrant", action="store_true")
    args = ap.parse_args()

    classes = dict(COCO)
    if args.no_hydrant:
        classes.pop(10)

    folders = sorted(p for p in DATASET.iterdir()
                     if p.is_dir() and not p.name.startswith("_")
                     and (args.only is None or args.only in p.name))
    files = []
    for d in folders:
        files += sorted(d.glob("*.jpg"))[args.offset::args.stride]
    random.Random(args.offset).shuffle(files)

    if not files:
        print("no frames matched - check --only and the dataset path")
        return
    print(f"{len(files)} frames from {len(folders)} folders "
          f"(stride {args.stride}, offset {args.offset})")

    model = YOLO(MODEL)
    min_conf = min(c for _, c in classes.values())
    counts = {name: 0 for name, _ in classes.values()}
    tasks = []

    for i, f in enumerate(files):
        r = model.predict(source=str(f), imgsz=IMGSZ, conf=min_conf,
                          classes=list(classes), verbose=False)[0]
        h, w = r.orig_shape

        result = []
        for b in r.boxes:
            cls, conf = int(b.cls[0]), float(b.conf[0])
            name, threshold = classes[cls]
            if conf < threshold:
                continue
            x1, y1, x2, y2 = b.xyxy[0].tolist()
            x1, y1 = max(0.0, x1), max(0.0, y1)
            x2, y2 = min(float(w), x2), min(float(h), y2)
            result.append({
                "from_name": "label",
                "to_name": "image",
                "type": "rectanglelabels",
                "original_width": w,
                "original_height": h,
                "score": round(conf, 3),
                "value": {
                    "x": 100 * x1 / w,
                    "y": 100 * y1 / h,
                    "width": 100 * (x2 - x1) / w,
                    "height": 100 * (y2 - y1) / h,
                    "rotation": 0,
                    "rectanglelabels": [name],
                },
            })
            counts[name] += 1

        rel = f.relative_to(ROOT).as_posix()
        tasks.append({
            "data": {"image": f"/data/local-files/?d={rel}"},
            "predictions": [{"model_version": "yolov8s-coco-draft",
                             "result": result}],
        })

        if (i + 1) % 200 == 0:
            print(f"  {i + 1} / {len(files)}")

    OUT_DIR.mkdir(exist_ok=True)
    tag = f"s{args.stride}_o{args.offset}" + (f"_{args.only.strip('_')}" if args.only else "")
    out = OUT_DIR / f"tasks_{tag}.json"
    with open(out, "w") as fh:
        json.dump(tasks, fh)

    print(f"\nwrote {len(tasks)} tasks to {out}")
    for name, n in counts.items():
        print(f"  draft {name:<14}{n:>6} boxes")
    print("  light_pole: none - draw these by hand")


if __name__ == "__main__":
    main()

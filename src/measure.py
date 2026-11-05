"""
JeepBot - measure person sizes in every frame

Usage:
    python measure.py

Runs a pretrained person detector over every frame in dataset\\ and records,
per frame, how many people it found and the pixel height of the tallest one.
Nothing is moved or labelled - it only measures.

Writes:  measurements.csv
Prints:  how the labels WOULD change at several different thresholds,
         so you can see how sensitive your dataset is to that choice.
"""

import csv
import sys
from collections import defaultdict
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "dataset"
OUT_CSV = ROOT / "measurements.csv"
ONLY = sys.argv[1] if len(sys.argv) > 1 else None

MODEL = str(ROOT / "yolov8s.pt")   # downloads automatically the first time
IMGSZ = 1280              # full resolution - so small distant people are found
CONF = 0.25               # low on purpose: we want to catch faint people
THRESHOLDS = [25, 40, 60, 80, 100]


def main():
    folders = sorted(p for p in DATASET.iterdir()
                     if p.is_dir() and not p.name.startswith("_")
                     and (ONLY is None or ONLY in p.name))
    files = [f for d in folders for f in sorted(d.glob("*.jpg"))]
    print(f"{len(files)} frames to measure")

    model = YOLO(MODEL)
    rows = []

    for i, f in enumerate(files):
        result = model.predict(source=str(f), imgsz=IMGSZ, conf=CONF,
                               classes=[0], verbose=False)[0]

        heights, confs = [], []
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            heights.append(y2 - y1)
            confs.append(float(box.conf[0]))

        rows.append({
            "path": str(f.relative_to(DATASET)),
            "folder": f.parent.name,
            "n_people": len(heights),
            "max_h": round(max(heights), 1) if heights else 0,
            "max_conf": round(max(confs), 3) if confs else 0,
        })

        if (i + 1) % 250 == 0:
            print(f"  {i + 1} / {len(files)}")

    all_rows = rows
    if ONLY and OUT_CSV.exists():
        with open(OUT_CSV, newline="") as fh:
            old = [r for r in csv.DictReader(fh) if ONLY not in r["folder"]]
        all_rows = old + rows

    with open(OUT_CSV, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(all_rows)

    # ---------------------------------------------------------------- report
    empty = [r for r in rows if r["folder"].startswith("empty")]
    person = [r for r in rows if r["folder"].startswith("person")]

    print("\n" + "=" * 66)
    print("HOW MANY FRAMES ARE MISLABELLED - AT EACH POSSIBLE THRESHOLD")
    print("=" * 66)
    print(f"{'threshold':>10}{'empty folder, person >= T':>28}"
          f"{'person folder, no one >= T':>28}")
    for t in THRESHOLDS:
        bad_empty = sum(1 for r in empty if r["max_h"] >= t)
        bad_person = sum(1 for r in person if r["max_h"] < t)
        print(f"{t:>8}px{bad_empty:>20} ({100*bad_empty/max(1,len(empty)):4.1f}%)"
              f"{bad_person:>20} ({100*bad_person/max(1,len(person)):4.1f}%)")

    print("\nIf these numbers barely change between 40 and 80, your dataset is")
    print("insensitive to the threshold and the choice doesn't matter much.")
    print("If they swing a lot, the in-between band is large - exclude it.")

    print("\n" + "=" * 66)
    print("TALLEST-PERSON HEIGHT DISTRIBUTION  (person folders)")
    print("=" * 66)
    buckets = [(0, 1), (1, 25), (25, 40), (40, 60), (60, 100),
               (100, 200), (200, 400), (400, 9999)]
    for lo, hi in buckets:
        n = sum(1 for r in person if lo <= r["max_h"] < hi)
        label = "none found" if lo == 0 else f"{lo}-{hi}px" if hi < 9999 else f"{lo}px+"
        print(f"  {label:<12}{n:>6}  {'#' * (n // 10)}")

    print("\nPer folder: frames where the detector found nobody")
    per = defaultdict(lambda: [0, 0])
    for r in rows:
        per[r["folder"]][0] += 1
        if r["n_people"] == 0:
            per[r["folder"]][1] += 1
    for folder, (n, z) in sorted(per.items()):
        print(f"  {folder:<26}{z:>5} / {n}")


if __name__ == "__main__":
    main()

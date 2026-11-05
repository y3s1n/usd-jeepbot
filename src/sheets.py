"""
JeepBot - contact sheets

Usage:
    python sheets.py            # 10 person + 10 empty per run
    python sheets.py 20         # 20 of each instead

Builds one PNG per run showing what each class actually looks like in that
condition, sampled evenly across the folder. Writes to  contact_sheets\\
Read-only on the dataset.
"""

import csv
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "dataset"
MEAS_CSV = ROOT / "measurements.csv"
OUT_DIR = ROOT / "contact_sheets"

PER_CLASS = int(sys.argv[1]) if len(sys.argv) > 1 else 10
COLS = 5
TW, TH = 320, 200          # tile size - 1.6 ratio, same as 1280x800
HEADER = 34
PAD = 4


def load_heights():
    """basename -> max person height in px, so it survives apply.py moves"""
    out = {}
    if MEAS_CSV.exists():
        with open(MEAS_CSV, newline="") as fh:
            for r in csv.DictReader(fh):
                out[Path(r["path"]).name] = float(r["max_h"])
    return out


def sample(folder, n):
    files = sorted(folder.glob("*.jpg"))
    if not files:
        return []
    if len(files) <= n:
        return files
    step = len(files) / n
    return [files[int(i * step)] for i in range(n)]


def grid(files, heights, label, width):
    rows = max(1, -(-len(files) // COLS))
    h = HEADER + rows * (TH + PAD)
    canvas = np.full((h, width, 3), 24, dtype=np.uint8)

    cv2.putText(canvas, label, (10, 23), cv2.FONT_HERSHEY_SIMPLEX,
                0.62, (235, 235, 235), 1)

    for i, f in enumerate(files):
        img = cv2.imread(str(f))
        if img is None:
            continue
        tile = cv2.resize(img, (TW, TH))

        px = heights.get(f.name)
        if px is not None:
            txt = f"{px:.0f}px" if px > 0 else "none"
            colour = (60, 220, 60) if px >= 60 else \
                     (60, 200, 240) if px >= 25 else (150, 150, 150)
            cv2.putText(tile, txt, (8, TH - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 0, 0), 3)
            cv2.putText(tile, txt, (8, TH - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, colour, 1)

        r, c = divmod(i, COLS)
        y = HEADER + r * (TH + PAD)
        x = c * (TW + PAD)
        canvas[y:y + TH, x:x + TW] = tile

    return canvas


def main():
    if not DATASET.exists():
        print(f"no dataset at {DATASET}")
        return

    heights = load_heights()
    OUT_DIR.mkdir(exist_ok=True)

    runs = {}
    for p in sorted(DATASET.iterdir()):
        if not p.is_dir() or p.name.startswith("_"):
            continue
        if p.name.startswith("person_"):
            runs.setdefault(p.name[7:], {})["person"] = p
        elif p.name.startswith("empty_"):
            runs.setdefault(p.name[6:], {})["empty"] = p
        else:
            runs.setdefault(p.name, {})["other"] = p

    width = COLS * (TW + PAD)

    for run, folders in sorted(runs.items()):
        blocks = []
        for cls in ("person", "empty", "other"):
            folder = folders.get(cls)
            if folder is None:
                continue
            files = sample(folder, PER_CLASS)
            if not files:
                continue
            total = len(list(folder.glob("*.jpg")))
            blocks.append(grid(files, heights,
                               f"{cls.upper()}  -  {folder.name}  "
                               f"({len(files)} of {total})", width))

        if not blocks:
            continue

        sheet = np.vstack(blocks)
        out = OUT_DIR / f"{run}.png"
        cv2.imwrite(str(out), sheet)
        print(f"  {out.name:<20}{sheet.shape[1]}x{sheet.shape[0]}")

    print(f"\nwrote {len(list(OUT_DIR.glob('*.png')))} sheets to {OUT_DIR}")
    print("\nGreen label = person >= 60px, amber = 25-60px, grey = under 25px")
    print("Labels come from the detector, so treat them as a hint not a verdict.")


if __name__ == "__main__":
    main()

"""view_labels.py - draw the merged labels on their photos, to check what training actually sees.

    python view_labels.py --root .                          # every labeled frame, as contact sheets
    python view_labels.py --root . --single                 # one full-size image per frame instead
    python view_labels.py --root . --cls fire_hydrant       # only frames containing a hydrant
    python view_labels.py --root . --export angie           # only frames from exports whose name contains "angie"
    python view_labels.py --root . --split val --folder dusk

Writes to <root>/label_view/ (deleted and rebuilt each run).
Colors: person orange, fire_hydrant red, trash_can cyan, ignore gray.
"""
import argparse
import csv
import shutil
from pathlib import Path

from PIL import Image, ImageDraw

from dataset import read_boxes

ROOT = Path(__file__).resolve().parent.parent

NAMES = {0: "person", 1: "fire_hydrant", 2: "trash_can", -1: "ignore"}
COLORS = {0: "orange", 1: "red", 2: "cyan", -1: "gray"}
THUMB_W, THUMB_H = 480, 300
COLS, ROWS = 4, 4


def draw_frame(root, folder, file, width):
    img = Image.open(Path(root) / "dataset" / folder / file).convert("RGB")
    scale = width / img.width
    img = img.resize((width, int(img.height * scale)), Image.BILINEAR)
    d = ImageDraw.Draw(img)
    line = 3 if width > 800 else 2
    for c, cx, cy, w, h in read_boxes(root, folder, file):
        box = [(cx - w / 2) * img.width, (cy - h / 2) * img.height,
               (cx + w / 2) * img.width, (cy + h / 2) * img.height]
        d.rectangle(box, outline=COLORS[c], width=line)
        d.text((box[0] + 3, box[1] + 2), NAMES[c], fill=COLORS[c])
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(ROOT))
    ap.add_argument("--split", choices=["train", "val"], help="only this split")
    ap.add_argument("--cls", choices=list(NAMES.values()), help="only frames containing this label")
    ap.add_argument("--export", help="only frames whose export file name contains this text")
    ap.add_argument("--folder", help="only folders whose name contains this text, e.g. dusk or empty_B")
    ap.add_argument("--single", action="store_true", help="full-size single images instead of sheets")
    args = ap.parse_args()
    root = Path(args.root)

    with open(root / "split.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    want = {v: k for k, v in NAMES.items()}.get(args.cls)
    picked = []
    for r in rows:
        if args.split and r["split"] != args.split:
            continue
        if args.export and args.export.lower() not in r["export"].lower():
            continue
        if args.folder and args.folder.lower() not in r["folder"].lower():
            continue
        if want is not None and not any(b[0] == want for b in read_boxes(root, r["folder"], r["file"])):
            continue
        picked.append(r)

    out = root / "label_view"
    if out.exists():
        shutil.rmtree(out)
    out.mkdir()
    print(f"{len(picked)} frames match. Writing to {out} ...")

    if args.single:
        for r in picked:
            img = draw_frame(root, r["folder"], r["file"], 1280)
            img.save(out / f"{r['split']}_{Path(r['file']).stem}.jpg", quality=90)
    else:
        per_sheet = COLS * ROWS
        for s in range(0, len(picked), per_sheet):
            sheet = Image.new("RGB", (COLS * THUMB_W, ROWS * (THUMB_H + 16)), "black")
            d = ImageDraw.Draw(sheet)
            for k, r in enumerate(picked[s:s + per_sheet]):
                x, y = (k % COLS) * THUMB_W, (k // COLS) * (THUMB_H + 16)
                sheet.paste(draw_frame(root, r["folder"], r["file"], THUMB_W), (x, y))
                d.text((x + 3, y + THUMB_H + 2), f"{r['file']}  [{r['split']}]  {r['export'][:24]}",
                       fill="white")
            sheet.save(out / f"sheet_{s // per_sheet + 1:03d}.jpg", quality=88)
        print(f"{(len(picked) + per_sheet - 1) // per_sheet} contact sheets, {per_sheet} frames each")
    print("Done.")


if __name__ == "__main__":
    main()

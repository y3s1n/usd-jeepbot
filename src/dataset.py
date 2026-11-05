"""dataset.py - Stage 2, steps 2-5: resize, grid targets, augmentation, normalization.

Run once to compute normalization stats and save visual checks:

    python dataset.py --root .

Reads frames from two places:
    split.csv         campus frames   -> dataset/<folder>/<file>,  labels/<folder>/<stem>.txt
    split_public.csv  public frames   -> public/<name>/images/<file>, public/<name>/labels/<stem>.txt
                                         (folder column is "public_<name>", train only)

Writes:
    <root>/stats.json        per-channel mean and std of the training images
    <root>/check/*.png       a few training frames with their boxes and responsible cells drawn
"""
import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parent.parent

# Step 2: resize. 1280x800 -> 384x240 keeps the same shape (both are 1.6 wide per 1 tall).
IMG_W, IMG_H = 384, 240
CELL = 16                                        # four 2x2 poolings shrink the image by 16
GRID_W, GRID_H = IMG_W // CELL, IMG_H // CELL   # 24 x 15 grid
CLASS_NAMES = ["person", "fire_hydrant", "trash_can"]   # target class = index + 1; 0 = background

PUBLIC_PREFIX = "public_"


def read_split(root, split):
    """Frames in this split, from the campus CSV and the public CSV if it exists."""
    items = []
    for name in ("split.csv", "split_public.csv"):
        path = Path(root) / name
        if not path.exists():
            continue
        with open(path, newline="") as f:
            items += [(r["folder"], r["file"]) for r in csv.DictReader(f) if r["split"] == split]
    return items


def image_path(root, folder, file):
    if folder.startswith(PUBLIC_PREFIX):
        return Path(root) / "public" / folder[len(PUBLIC_PREFIX):] / "images" / file
    return Path(root) / "dataset" / folder / file


def label_path(root, folder, file):
    stem = Path(file).stem + ".txt"
    if folder.startswith(PUBLIC_PREFIX):
        return Path(root) / "public" / folder[len(PUBLIC_PREFIX):] / "labels" / stem
    return Path(root) / "labels" / folder / stem


def read_boxes(root, folder, file):
    path = label_path(root, folder, file)
    boxes = []
    for line in path.read_text().splitlines():
        if line.strip():
            c, cx, cy, w, h = line.split()
            boxes.append((int(c), float(cx), float(cy), float(w), float(h)))
    return boxes


def overfit_items(root, n):
    """The first n training frames that contain at least one box (for the overfit sanity test)."""
    return [(f, x) for f, x in read_split(root, "train") if read_boxes(root, f, x)][:n]


def build_targets(boxes):
    """Step 3: convert boxes to grid targets.

    cls  [15, 24]     0 = background, 1..3 = person / fire_hydrant / trash_can,
                      -1 = ignore (a neighbour of a responsible cell that lies inside the box:
                      it sees nearly the same pixels, so it is not punished for firing)
    box  [15, 24, 4]  (ox, oy, w, h): centre inside the cell, size as fraction of the image
    mask [15, 24]     1 where a cell is responsible for an object
    """
    cls = torch.zeros(GRID_H, GRID_W, dtype=torch.long)
    box = torch.zeros(GRID_H, GRID_W, 4)
    mask = torch.zeros(GRID_H, GRID_W)
    area = torch.zeros(GRID_H, GRID_W)
    for c, cx, cy, w, h in boxes:
        if c == -1:
            continue                                # ignore boxes are handled below
        col = min(int(cx * GRID_W), GRID_W - 1)
        row = min(int(cy * GRID_H), GRID_H - 1)
        if mask[row, col] and area[row, col] >= w * h:
            continue                                # two centres in one cell: keep the larger box
        cls[row, col] = c + 1
        box[row, col] = torch.tensor([cx * GRID_W - col, cy * GRID_H - row, w, h])
        mask[row, col] = 1
        area[row, col] = w * h

    # Ignore-labeled boxes (class -1): every background cell whose centre lies inside them.
    for c, cx, cy, w, h in boxes:
        if c != -1:
            continue
        for r in range(GRID_H):
            for q in range(GRID_W):
                cell_cx, cell_cy = (q + 0.5) / GRID_W, (r + 0.5) / GRID_H
                if abs(cell_cx - cx) <= w / 2 and abs(cell_cy - cy) <= h / 2 and cls[r, q] == 0:
                    cls[r, q] = -1
        # also the cell holding its centre, so even a box smaller than one cell is ignored
        row, col = min(int(cy * GRID_H), GRID_H - 1), min(int(cx * GRID_W), GRID_W - 1)
        if cls[row, col] == 0:
            cls[row, col] = -1

    # Ignore the 3x3 neighbours of each responsible cell that fall inside its box.
    for c, cx, cy, w, h in boxes:
        if c == -1:
            continue
        col = min(int(cx * GRID_W), GRID_W - 1)
        row = min(int(cy * GRID_H), GRID_H - 1)
        for r in range(max(row - 1, 0), min(row + 2, GRID_H)):
            for q in range(max(col - 1, 0), min(col + 2, GRID_W)):
                cell_cx, cell_cy = (q + 0.5) / GRID_W, (r + 0.5) / GRID_H
                inside = abs(cell_cx - cx) <= w / 2 and abs(cell_cy - cy) <= h / 2
                if inside and cls[r, q] == 0:
                    cls[r, q] = -1
    return cls, box, mask


def load_image(root, folder, file):
    img = Image.open(image_path(root, folder, file)).convert("RGB")
    img = img.resize((IMG_W, IMG_H), Image.BILINEAR)
    x = torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0)   # [240, 384, 3] in 0..1
    return x.permute(2, 0, 1).contiguous()                             # [3, 240, 384]


class DetectionDataset(Dataset):
    def __init__(self, root, split, augment=False, stats=None):
        self.root = Path(root)
        self.items = read_split(root, split)
        self.augment = augment
        self.mean = torch.tensor(stats["mean"]).view(3, 1, 1) if stats else None
        self.std = torch.tensor(stats["std"]).view(3, 1, 1) if stats else None

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        folder, file = self.items[i]
        x = load_image(self.root, folder, file)
        boxes = read_boxes(self.root, folder, file)

        # Step 4: augment (training only, fresh every time).
        if self.augment:
            if random.random() < 0.5:
                x = torch.flip(x, dims=[2])
                boxes = [(c, 1.0 - cx, cy, w, h) for c, cx, cy, w, h in boxes]
            x = (x * random.uniform(0.6, 1.4)).clamp(0.0, 1.0)

        # Step 5: normalize per channel.
        if self.mean is not None:
            x = (x - self.mean) / self.std

        cls, box, mask = build_targets(boxes)
        return x, cls, box, mask


def compute_stats(root):
    """Per-channel mean and std over the training images (no augmentation)."""
    total = torch.zeros(3)
    total_sq = torch.zeros(3)
    n = 0
    for folder, file in read_split(root, "train"):
        x = load_image(root, folder, file)
        total += x.sum(dim=(1, 2))
        total_sq += (x ** 2).sum(dim=(1, 2))
        n += x.shape[1] * x.shape[2]
    mean = total / n
    std = (total_sq / n - mean ** 2).sqrt()
    return {"mean": mean.tolist(), "std": std.tolist()}


def save_checks(root, count=6):
    """Draw boxes (green) and their responsible cells (red) on a few training frames."""
    out_dir = Path(root) / "check"
    out_dir.mkdir(exist_ok=True)
    items = read_split(root, "train")
    random.seed(0)
    for folder, file in random.sample(items, min(count, len(items))):
        img = Image.open(image_path(root, folder, file)).convert("RGB")
        img = img.resize((IMG_W, IMG_H), Image.BILINEAR)
        draw = ImageDraw.Draw(img)
        boxes = read_boxes(root, folder, file)
        _, _, mask = build_targets(boxes)
        for row, col in mask.nonzero().tolist():
            draw.rectangle([col * CELL, row * CELL, (col + 1) * CELL, (row + 1) * CELL], outline="red")
        for c, cx, cy, w, h in boxes:
            x0, y0 = (cx - w / 2) * IMG_W, (cy - h / 2) * IMG_H
            x1, y1 = (cx + w / 2) * IMG_W, (cy + h / 2) * IMG_H
            color = "gray" if c == -1 else "lime"
            draw.rectangle([x0, y0, x1, y1], outline=color)
            draw.text((x0 + 2, y0 + 2), "ignore" if c == -1 else CLASS_NAMES[c], fill=color)
        img.save(out_dir / f"{folder}_{Path(file).stem}.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(ROOT))
    args = ap.parse_args()

    campus = sum(1 for f, _ in read_split(args.root, "train") if not f.startswith(PUBLIC_PREFIX))
    public = sum(1 for f, _ in read_split(args.root, "train") if f.startswith(PUBLIC_PREFIX))
    print(f"train frames: {campus} campus + {public} public = {campus + public}")
    print(f"val frames:   {len(read_split(args.root, 'val'))} (campus only)")

    stats = compute_stats(args.root)
    (Path(args.root) / "stats.json").write_text(json.dumps(stats, indent=2))
    print("mean:", [round(v, 4) for v in stats["mean"]], " std:", [round(v, 4) for v in stats["std"]])

    ds = DetectionDataset(args.root, "train", augment=True, stats=stats)
    x, cls, box, mask = ds[0]
    print("image", tuple(x.shape), " cls", tuple(cls.shape), " box", tuple(box.shape),
          " mask", tuple(mask.shape), " object cells in sample:", int(mask.sum()))

    save_checks(args.root)
    print("Saved visual checks to", Path(args.root) / "check")


if __name__ == "__main__":
    main()
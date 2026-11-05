"""train.py - Stage 2, steps 7-8: the cost and the training loop.

    python train.py --root .
    python train.py --root . --prefix person                  # campus person folders only
    python train.py --root . --prefix person public_person    # plus the COCO people
    python train.py --root . --prefix public_person --val-prefix person --val-split all
                                         # COCO only, scored on every campus person frame

Writes:
    <root>/checkpoints/best.pt   weights with the lowest validation cost so far
    <root>/train_log.csv         per-epoch costs, for comparing retrains
"""
import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import DetectionDataset, overfit_items, read_split
from model import Detector

ROOT = Path(__file__).resolve().parent.parent

OBJECT_WEIGHT = 20.0  # class error on object cells counts 20x a background cell (5 was too weak)
BOX_WEIGHT = 5.0      # box term counts 5x the class term
LEARNING_RATE = 1e-3
BATCH_SIZE = 32
MAX_EPOCHS = 100
PATIENCE = 10         # stop after this many epochs without a new lowest validation cost


def detection_cost(class_scores, box_raw, cls_t, box_t, mask):
    """Step 7.

    class_scores [B, 4, 15, 24]   box_raw [B, 4, 15, 24]
    cls_t [B, 15, 24]   box_t [B, 15, 24, 4]   mask [B, 15, 24]
    """
    # Class term: cross-entropy at every cell, object cells weighted more.
    ce = F.cross_entropy(class_scores, cls_t, reduction="none", ignore_index=-1)   # 0 at ignored cells
    weights = torch.where(cls_t > 0, OBJECT_WEIGHT, 1.0)
    class_term = (ce * weights).mean()

    # Box term: squared error after sigmoid, only on responsible cells (masking).
    pred = torch.sigmoid(box_raw).permute(0, 2, 3, 1)                     # [B, 15, 24, 4]
    sq_err = ((pred - box_t) ** 2).sum(dim=-1)                            # [B, 15, 24]
    box_term = (sq_err * mask).sum() / mask.sum().clamp(min=1)

    return class_term + BOX_WEIGHT * box_term, class_term, box_term


def run_epoch(model, loader, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    totals = np.zeros(3)
    batches = 0
    with torch.set_grad_enabled(training):
        for x, cls_t, box_t, mask in loader:
            x, cls_t, box_t, mask = x.to(device), cls_t.to(device), box_t.to(device), mask.to(device)
            scores, box_raw = model(x)
            cost, class_term, box_term = detection_cost(scores, box_raw, cls_t, box_t, mask)
            if training:
                optimizer.zero_grad()
                cost.backward()
                optimizer.step()
            totals += [cost.item(), class_term.item(), box_term.item()]
            batches += 1
    return totals / max(batches, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(ROOT))
    ap.add_argument("--workers", type=int, default=0)   # 0 is safest on Windows
    ap.add_argument("--object-weight", type=float, default=OBJECT_WEIGHT,
                    help="how much more an object cell's class error counts than a background cell's")
    ap.add_argument("--prefix", nargs="+", default=[],
                    help="only use folders whose name starts with one of these, e.g. "
                         "--prefix person public_person  (campus people + COCO people)")
    ap.add_argument("--val-prefix", nargs="+", default=None,
                    help="campus folders to validate on (default: same as --prefix). "
                         "e.g. --prefix public_person --val-prefix person")
    ap.add_argument("--val-split", default="val", choices=["val", "all"],
                    help="all = validate on every labeled frame of those folders (train + val rows). "
                         "Only allowed when the model does not train on those folders.")
    ap.add_argument("--overfit", type=int, default=0,
                    help="sanity test: train on N frames with no augmentation and no early stopping, "
                         "save checkpoints/overfit.pt. Working code should memorize them.")
    args = ap.parse_args()
    root = Path(args.root)
    globals()["OBJECT_WEIGHT"] = args.object_weight
    print("object weight:", OBJECT_WEIGHT)

    torch.manual_seed(0)
    random.seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    stats = json.loads((root / "stats.json").read_text())
    train_ds = DetectionDataset(root, "train", augment=True, stats=stats)
    val_ds = DetectionDataset(root, "val", augment=False, stats=stats)

    if args.prefix:
        prefixes = tuple(args.prefix)
        train_ds.items = [(f, x) for f, x in train_ds.items if f.startswith(prefixes)]
        n_public = sum(f.startswith("public_") for f, _ in train_ds.items)
        print(f"train folders: {', '.join(sorted({f for f, _ in train_ds.items}))}")
        print(f"train frames: {len(train_ds.items) - n_public} campus + {n_public} public")

    val_prefixes = tuple(args.val_prefix or args.prefix)
    if args.val_split == "all":
        val_ds.items = read_split(root, "train") + read_split(root, "val")
    val_ds.items = [(f, x) for f, x in val_ds.items if not f.startswith("public_")
                    and (not val_prefixes or f.startswith(val_prefixes))]
    print(f"val ({args.val_split}) folders: {', '.join(sorted({f for f, _ in val_ds.items}))}")

    overlap = {f for f, _ in train_ds.items} & {f for f, _ in val_ds.items}
    if args.val_split == "all" and overlap:
        raise SystemExit(f"--val-split all would score frames the model trains on: {sorted(overlap)}")
    if not train_ds.items or not val_ds.items:
        raise SystemExit("No frames left after the prefix filter. Check the folder names in split.csv.")

    ckpt_name, max_epochs, patience = "best.pt", MAX_EPOCHS, PATIENCE
    if args.overfit:
        overfit = [(f, x) for f, x in overfit_items(root, 10 ** 9)
                   if not args.prefix or f.startswith(tuple(args.prefix))][: args.overfit]
        train_ds = DetectionDataset(root, "train", augment=False, stats=stats)
        train_ds.items = overfit
        val_ds = train_ds                       # "val" = the same frames: we only test memorizing
        ckpt_name, max_epochs, patience = "overfit.pt", 300, 300
        print(f"OVERFIT TEST on {len(train_ds.items)} frames, 300 epochs")
    print(f"train frames: {len(train_ds)}  val frames: {len(val_ds)}")
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=args.workers)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=args.workers)

    model = Detector().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    (root / "checkpoints").mkdir(exist_ok=True)
    best_val, since_best = float("inf"), 0
    with open(root / "train_log.csv", "w", newline="") as log_file:
        log = csv.writer(log_file)
        log.writerow(["epoch", "train_cost", "train_class", "train_box", "val_cost", "val_class", "val_box", "seconds"])

        for epoch in range(1, max_epochs + 1):
            start = time.time()
            tr = run_epoch(model, train_loader, device, optimizer)
            va = run_epoch(model, val_loader, device)
            secs = time.time() - start
            log.writerow([epoch, *[round(v, 5) for v in tr], *[round(v, 5) for v in va], round(secs, 1)])
            log_file.flush()

            marker = ""
            if va[0] < best_val:
                best_val, since_best = va[0], 0
                torch.save({"model": model.state_dict(), "stats": stats, "epoch": epoch},
                           root / "checkpoints" / ckpt_name)
                marker = "  <- saved"
            else:
                since_best += 1
            print(f"epoch {epoch:3d}  train {tr[0]:.4f}  val {va[0]:.4f} "
                  f"(class {va[1]:.4f}, box {va[2]:.4f})  {secs:.0f}s{marker}")

            if since_best >= patience:
                print(f"No improvement for {patience} epochs. Best val cost {best_val:.4f}.")
                break


if __name__ == "__main__":
    main()
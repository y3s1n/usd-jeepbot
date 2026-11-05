"""evaluate.py - Stage 2, step 10: score the trained model on the validation frames.

    python evaluate.py --root .

A detection is a HIT if it has the same class as an unmatched labeled box and IoU >= 0.5.
Each labeled box can be matched once. Unmatched detections are FALSE ALARMS,
unmatched labeled boxes are MISSES.

Prints precision and recall per class at several confidence thresholds, and saves
val frames with boxes drawn to <root>/preds/ (green = label, colored = model).
"""
import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from dataset import CLASS_NAMES, build_targets, overfit_items, read_boxes, read_split
from detect import FRAME_H, FRAME_W, decode, iou, load_model, preprocess

ROOT = Path(__file__).resolve().parent.parent

HIT_IOU = 0.5
THRESHOLDS = [0.5, 0.7, 0.8, 0.9, 0.95]
MODES = {"plain": False}     # add "peaks": True to compare peak-picking (it lost hits on our val set)
SIZE_BUCKETS = [(0, 60), (60, 150), (150, 300), (300, 100000)]   # labeled box height, camera pixels
COLORS = {0: "orange", 1: "red", 2: "cyan"}


def label_boxes(root, folder, file):
    """Labeled boxes as (class, x0, y0, x1, y1) in camera pixels."""
    out = []
    for c, cx, cy, w, h in read_boxes(root, folder, file):
        out.append((c, (cx - w / 2) * FRAME_W, (cy - h / 2) * FRAME_H,
                    (cx + w / 2) * FRAME_W, (cy + h / 2) * FRAME_H))
    return out


def match(detections, labels):
    """Greedy: most confident detection first. Returns per-class hit/false-alarm/miss counts.

    Boxes labeled 'ignore' (class -1) are never misses, and a detection centred on one is
    neither a hit nor a false alarm.
    """
    ignore = [lb for lc, *lb in labels if lc == -1]
    labels = [l for l in labels if l[0] != -1]
    kept = []
    for d in detections:
        mx, my = (d[2] + d[4]) / 2, (d[3] + d[5]) / 2
        if not any(b[0] <= mx <= b[2] and b[1] <= my <= b[3] for b in ignore):
            kept.append(d)
    detections = kept
    hits = np.zeros(3, int)
    false_alarms = np.zeros(3, int)
    on_object = np.zeros(3, int)
    used = [False] * len(labels)
    for c, conf, *b in sorted(detections, key=lambda d: -d[1]):
        best_iou, best_j = 0.0, -1
        for j, (lc, *lb) in enumerate(labels):
            if used[j] or lc != c:
                continue
            o = iou(np.array(b), np.array([lb]))[0]
            if o > best_iou:
                best_iou, best_j = o, j
        if best_iou >= HIT_IOU:
            used[best_j] = True
            hits[c] += 1
        else:
            false_alarms[c] += 1
            # Is this false alarm sitting ON a labeled object of its class (centre inside the box)?
            mx, my = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
            if any(lc == c and lb[0] <= mx <= lb[2] and lb[1] <= my <= lb[3] for lc, *lb in labels):
                on_object[c] += 1
    misses = np.zeros(3, int)
    for j, (lc, *_) in enumerate(labels):
        if not used[j]:
            misses[lc] += 1
    return hits, false_alarms, misses, on_object


def draw(image, labels, detections, path):
    img = image.convert("RGB").copy()
    d = ImageDraw.Draw(img)
    for c, *b in labels:
        d.rectangle(b, outline="lime", width=2)
    for c, conf, *b in detections:
        d.rectangle(b, outline=COLORS[c], width=3)
        d.text((b[0] + 4, b[1] + 4), f"{CLASS_NAMES[c]} {conf:.2f}", fill=COLORS[c])
    try:
        img.save(path)
    except OSError as e:                      # file open elsewhere, or a sync tool holding it
        print(f"  (could not save {path.name}: {e.strerror}; skipped)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(ROOT))
    ap.add_argument("--save", type=int, default=30, help="how many frames to draw")
    ap.add_argument("--split", default="val", choices=["val", "train"],
                    help="train = score frames the model learned from (a check for bugs vs too little data)")
    ap.add_argument("--overfit", type=int, default=0,
                    help="score checkpoints/overfit.pt on the same N frames train.py --overfit N used")
    ap.add_argument("--draw-conf", type=float, default=0.9, choices=THRESHOLDS,
                    help="threshold used for the drawn frames and eval_frames.csv")
    args = ap.parse_args()
    root = Path(args.root)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, stats, epoch = load_model(root, device, "overfit.pt" if args.overfit else "best.pt")
    print(f"model from epoch {epoch}, device {device}")

    items = read_split(root, args.split)
    if args.overfit:
        from dataset import overfit_items
        items, args.split = overfit_items(root, args.overfit), "overfit"
    totals = {(m, t): [np.zeros(3, int) for _ in range(4)] for m in MODES for t in THRESHOLDS}
    preds_dir = root / {"val": "preds", "train": "preds_train", "overfit": "preds_overfit"}[args.split]
    preds_dir.mkdir(exist_ok=True)
    rows = []
    right_prob, background_max = [], []
    size_stats = np.zeros((3, len(SIZE_BUCKETS), 3), int)   # per class, bucket: boxes, found@.5, found@.3

    for i, (folder, file) in enumerate(items):
        image = Image.open(root / "dataset" / folder / file)
        with torch.no_grad():
            scores, box_raw = model(preprocess(image, stats).to(device))
        labels = label_boxes(root, folder, file)

        # Diagnostic: how confident is the model at the cells that SHOULD fire, vs everywhere else?
        probs = torch.softmax(scores[0], dim=0).cpu()
        cls_t, _, mask = build_targets(read_boxes(root, folder, file))
        for r, c in mask.nonzero().tolist():
            right_prob.append(float(probs[cls_t[r, c], r, c]))
        other = probs[1:].max(dim=0).values[cls_t == 0]      # true background, not ignored neighbours
        if len(other):
            background_max.append(float(other.max()))
        for mode, peaks in MODES.items():
            for t in THRESHOLDS:
                dets = decode(scores[0], box_raw[0], conf=t, peaks=peaks)
                for k, arr in enumerate(match(dets, labels)):
                    totals[(mode, t)][k] += arr
                if not peaks and t == args.draw_conf:
                    for lc, *lb in labels:
                        if lc == -1:
                            continue
                        hpx = lb[3] - lb[1]
                        k = next(j for j, (lo, hi) in enumerate(SIZE_BUCKETS) if lo <= hpx < hi)
                        same = np.array([d[2:] for d in dets if d[0] == lc])
                        best = iou(np.array(lb), same).max() if len(same) else 0.0
                        size_stats[lc, k] += [1, best >= 0.5, best >= 0.3]
                    h, f, m, _ = match(dets, labels)
                    rows.append([folder, file, int(h.sum()), int(f.sum()), int(m.sum())])
                    if i < args.save:
                        draw(image, labels, dets, preds_dir / (Path(file).stem + ".png"))

    rp, bm = np.array(right_prob), np.array(background_max)
    print("\nDiagnostic (probabilities after softmax):")
    print(f"  at responsible cells, prob of the CORRECT class: median {np.median(rp):.3f}  "
          f"90th pct {np.percentile(rp, 90):.3f}  max {rp.max():.3f}")
    print(f"  at all other cells, highest object prob per frame: median {np.median(bm):.3f}  "
          f"max {bm.max():.3f}")

    n_frames = max(len(items), 1)
    labeled = totals[("plain", THRESHOLDS[0])][0] + totals[("plain", THRESHOLDS[0])][2]
    print(f"\n{args.split} frames: {n_frames}   a hit = same class and IoU >= {HIT_IOU}")
    print("labeled boxes: " + "  ".join(f"{n} {labeled[c]}" for c, n in enumerate(CLASS_NAMES)))
    print("each cell: hits / false alarms (on bg)  recall  precision\n")
    print(f"{'mode':<7}{'conf':>5}  " + "".join(f"{n:<34}" for n in CLASS_NAMES) + "false/frame")
    for mode in MODES:
        for t in THRESHOLDS:
            hits, fas, misses, on_obj = totals[(mode, t)]
            cells = []
            for c in range(len(CLASS_NAMES)):
                r = hits[c] / max(hits[c] + misses[c], 1)
                p = hits[c] / max(hits[c] + fas[c], 1)
                cells.append(f"{hits[c]:>4} / {fas[c]:>5} ({fas[c] - on_obj[c]:>5})  "
                             f"R {r:.2f} P {p:.2f}")
            print(f"{mode:<7}{t:>5}  " + "  ".join(f"{s:<32}" for s in cells) +
                  f"{fas.sum() / n_frames:>6.1f}")
        print()

    print(f"Recall by labeled box height in the 1280x800 frame (plain, conf {args.draw_conf})")
    print(f"  {'class':<14}{'height px':<12}{'boxes':>6}{'found @IoU 0.5':>17}{'found @IoU 0.3':>17}")
    for c, name in enumerate(CLASS_NAMES):
        for k, (lo, hi) in enumerate(SIZE_BUCKETS):
            n, f5, f3 = size_stats[c, k]
            if n == 0:
                continue
            rng = f"{lo}-{hi}" if hi < 100000 else f"{lo}+"
            print(f"  {name:<14}{rng:<12}{n:>6}{f5:>9} ({f5 / n:.2f}){f3:>10} ({f3 / n:.2f})")
    print()

    with open(root / "eval_frames.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["folder", "file", "hits", "false_alarms", "misses"])
        w.writerows(rows)
    print(f"Drew {min(args.save, len(items))} val frames to {preds_dir}")
    print(f"Drawn and per-frame results use conf {args.draw_conf} ({root / 'eval_frames.csv'})")


if __name__ == "__main__":
    main()

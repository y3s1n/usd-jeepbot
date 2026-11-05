"""metrics.py - the report card: standard detection metrics and graphs on the validation set.

    pip install matplotlib        (once)
    python metrics.py --root .
    python metrics.py --root . --ckpt person_only.pt --prefix person     # person folders only

Validation = the last 20% of each campus folder (the fixed split rule), never seen in training.

Writes to <root>/metrics/<checkpoint name>/  (so runs don't overwrite each other):
    summary.csv            every number below, one row each
    pr_curves.png          precision vs recall per class, at IoU 0.5 and 0.3
    recall_by_size.png     how often each object is found, by how tall it is
    ap_by_condition.png    sun vs dusk
    training_curve.png     train and val cost per epoch (from train_log.csv)

Metrics:
    AP   (average precision) = area under the precision-recall curve. Sweeps the confidence
         threshold from high to low, so it doesn't depend on picking one. 1.0 = perfect.
    mAP  = mean of the per-class APs (the single "overall" number; YOLO reports the same one).
    IoU 0.5 = box must overlap the label well (standard). IoU 0.3 = "found it, box is rough".
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from dataset import CLASS_NAMES, read_boxes, read_split
from detect import FRAME_H, FRAME_W, decode, iou, load_model, preprocess

ROOT = Path(__file__).resolve().parent.parent

SIZE_BUCKETS = [(0, 60), (60, 150), (150, 300), (300, 100000)]   # labeled box height, camera px
IOUS = [0.5, 0.3]
COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]          # person, fire_hydrant, trash_can (fixed order)
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#2b2b29", "#6b6a64", "#e6e5e0"


# ---------------------------------------------------------------- collecting

def frame_labels(root, folder, file):
    """(class, x0, y0, x1, y1) in camera pixels. class -1 = ignore."""
    out = []
    for c, cx, cy, w, h in read_boxes(root, folder, file):
        out.append((c, (cx - w / 2) * FRAME_W, (cy - h / 2) * FRAME_H,
                    (cx + w / 2) * FRAME_W, (cy + h / 2) * FRAME_H))
    return out


def collect(model, stats, root, items, device):
    """Run the model once per frame; keep every detection down to confidence 0.01."""
    frames = []
    for n, (folder, file) in enumerate(items):
        image = Image.open(root / "dataset" / folder / file)
        with torch.no_grad():
            scores, box_raw = model(preprocess(image, stats).to(device))
        dets = decode(scores[0], box_raw[0], conf=0.01)
        frames.append({"folder": folder, "file": file,
                       "labels": frame_labels(root, folder, file), "dets": dets})
        if (n + 1) % 100 == 0:
            print(f"  {n + 1}/{len(items)} frames")
    return frames


# ---------------------------------------------------------------- matching and AP

def match_frame(frame, c, thr):
    """For one frame and class: [(confidence, is_hit)], number of real labeled boxes,
    and for each labeled box the confidence of the detection that hit it (or None)."""
    ignore = [l[1:] for l in frame["labels"] if l[0] == -1]
    gts = [l[1:] for l in frame["labels"] if l[0] == c]
    dets = []
    for d in frame["dets"]:
        if d[0] != c:
            continue
        mx, my = (d[2] + d[4]) / 2, (d[3] + d[5]) / 2
        if any(b[0] <= mx <= b[2] and b[1] <= my <= b[3] for b in ignore):
            continue                                   # on an ignore box: neither hit nor false alarm
        dets.append(d)
    dets.sort(key=lambda d: -d[1])
    used = [None] * len(gts)
    scored = []
    for d in dets:
        best, best_j = 0.0, -1
        if gts:
            overlaps = iou(np.array(d[2:]), np.array(gts))
            for j in np.argsort(-overlaps):
                if used[j] is None:
                    best, best_j = overlaps[j], j
                    break
        if best >= thr:
            used[best_j] = d[1]
            scored.append((d[1], True))
        else:
            scored.append((d[1], False))
    return scored, gts, used


def average_precision(scored, n_gt):
    """Area under the precision-recall curve (all-point interpolation, as in VOC/COCO)."""
    if n_gt == 0:
        return None, np.array([]), np.array([])
    scored = sorted(scored, key=lambda s: -s[0])
    hits = np.array([s[1] for s in scored], dtype=float)
    tp, fp = np.cumsum(hits), np.cumsum(1 - hits)
    recall = tp / n_gt if len(hits) else np.array([])
    precision = tp / np.maximum(tp + fp, 1e-9) if len(hits) else np.array([])
    mrec = np.concatenate([[0.0], recall, [1.0]])
    mpre = np.concatenate([[0.0], precision, [0.0]])
    for i in range(len(mpre) - 2, -1, -1):             # make precision never increase with recall
        mpre[i] = max(mpre[i], mpre[i + 1])
    steps = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[steps + 1] - mrec[steps]) * mpre[steps + 1])), recall, precision


def class_ap(frames, c, thr):
    scored, n_gt = [], 0
    for f in frames:
        s, gts, _ = match_frame(f, c, thr)
        scored += s
        n_gt += len(gts)
    ap, rec, prec = average_precision(scored, n_gt)
    return ap, rec, prec, n_gt


def recall_by_size(frames, c, thr, conf):
    """Per size bucket: (labeled boxes, found at this confidence threshold)."""
    out = np.zeros((len(SIZE_BUCKETS), 2), int)
    for f in frames:
        _, gts, used = match_frame(f, c, thr)
        for g, u in zip(gts, used):
            k = next(j for j, (lo, hi) in enumerate(SIZE_BUCKETS) if lo <= g[3] - g[1] < hi)
            out[k, 0] += 1
            out[k, 1] += int(u is not None and u >= conf)
    return out


def condition(folder):
    return "dusk" if "dusk" in folder else "sun" if "sun" in folder else "other"


# ---------------------------------------------------------------- graphs

def style(ax, title):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, color=INK, fontsize=11, loc="left")
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.xaxis.label.set_color(INK)
    ax.yaxis.label.set_color(INK)


def plot_pr(curves, out):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), facecolor=SURFACE, sharey=True)
    for ax, thr in zip(axes, IOUS):
        style(ax, f"Precision vs recall, hit = IoU ≥ {thr}")
        for c, name in enumerate(CLASS_NAMES):
            ap, rec, prec, n = curves[(c, thr)]
            if ap is None or len(rec) == 0:
                continue
            ax.plot(rec, prec, color=COLORS[c], linewidth=2, label=f"{name}   AP {ap:.2f}  (n={n})")
        ax.legend(frameon=False, labelcolor=INK, fontsize=9, loc="upper right")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("recall (share of labeled objects found)")
    axes[0].set_ylabel("precision (share of detections that are right)")
    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_sizes(sizes, conf, out):
    import matplotlib.pyplot as plt
    labels = [f"{lo}-{hi} px" if hi < 100000 else f"{lo}+ px" for lo, hi in SIZE_BUCKETS]
    x = np.arange(len(SIZE_BUCKETS))
    width = 0.26
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), facecolor=SURFACE, sharey=True)
    for ax, thr in zip(axes, IOUS):
        style(ax, f"Found at confidence ≥ {conf}, hit = IoU ≥ {thr}")
        for c, name in enumerate(CLASS_NAMES):
            counts = sizes[(c, thr)]
            r = np.where(counts[:, 0] > 0, counts[:, 1] / np.maximum(counts[:, 0], 1), np.nan)
            pos = x + (c - 1) * width
            ax.bar(pos, np.nan_to_num(r), width - 0.03, color=COLORS[c], label=name)
            for p, val, n in zip(pos, r, counts[:, 0]):
                if n:
                    ax.text(p, (0 if np.isnan(val) else val) + 0.015, f"{val:.2f}\nn={n}",
                            ha="center", va="bottom", fontsize=7, color=MUTED)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_xlabel("labeled box height in the 1280x800 camera frame")
        ax.set_ylim(0, 1.1)
    axes[0].set_ylabel("recall")
    axes[0].legend(frameon=False, labelcolor=INK, fontsize=9, loc="upper left")
    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_conditions(cond_ap, out):
    import matplotlib.pyplot as plt
    conds = [k for k in ("sun", "dusk") if any((k, c) in cond_ap for c in range(3))]
    x = np.arange(len(conds))
    width = 0.26
    fig, ax = plt.subplots(figsize=(7, 4.5), facecolor=SURFACE)
    style(ax, "Average precision by lighting, IoU ≥ 0.5")
    for c, name in enumerate(CLASS_NAMES):
        vals = [cond_ap.get((k, c), (None, 0)) for k in conds]
        pos = x + (c - 1) * width
        ax.bar(pos, [v[0] or 0 for v in vals], width - 0.03, color=COLORS[c], label=name)
        for p, (ap, n) in zip(pos, vals):
            ax.text(p, (ap or 0) + 0.01, "n/a" if ap is None else f"{ap:.2f}\nn={n}",
                    ha="center", va="bottom", fontsize=7, color=MUTED)
    ax.set_xticks(x)
    ax.set_xticklabels(conds)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("AP")
    ax.legend(frameon=False, labelcolor=INK, fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_training(log_path, out):
    import matplotlib.pyplot as plt
    with open(log_path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None
    ep = [int(r["epoch"]) for r in rows]
    tr = [float(r["train_cost"]) for r in rows]
    va = [float(r["val_cost"]) for r in rows]
    best = int(np.argmin(va))
    fig, ax = plt.subplots(figsize=(8, 4.5), facecolor=SURFACE)
    style(ax, "Training: cost per epoch (lower is better)")
    ax.plot(ep, tr, color=COLORS[0], linewidth=2)
    ax.plot(ep, va, color=COLORS[1], linewidth=2)
    ax.annotate("train", (ep[-1], tr[-1]), xytext=(6, 0), textcoords="offset points", color=INK, fontsize=9)
    ax.annotate("validation", (ep[-1], va[-1]), xytext=(6, 0), textcoords="offset points", color=INK, fontsize=9)
    ax.scatter([ep[best]], [va[best]], s=60, color=COLORS[1], edgecolor=SURFACE, linewidth=2, zorder=3)
    ax.annotate(f"kept: epoch {ep[best]}", (ep[best], va[best]), xytext=(0, -18),
                textcoords="offset points", ha="center", color=INK, fontsize=9)
    from matplotlib.ticker import MaxNLocator
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel("epoch")
    ax.set_ylabel("cost")
    ax.set_xlim(ep[0], ep[-1] + max(2, len(ep) // 8))
    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return ep[best]


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(ROOT))
    ap.add_argument("--ckpt", default="best.pt", help="checkpoint in <root>/checkpoints/")
    ap.add_argument("--split", default="val", choices=["val", "train", "all"],
                    help="all = train + val rows; only fair for a model that never trained on those folders")
    ap.add_argument("--conf", type=float, default=0.9, help="threshold for the recall-by-size graph")
    ap.add_argument("--prefix", nargs="+", default=[],
                    help="only score campus folders whose name starts with one of these, e.g. person")
    ap.add_argument("--log", default="", help="train_log.csv to plot (default: <root>/train_log.csv)")
    args = ap.parse_args()
    root = Path(args.root)

    tag = Path(args.ckpt).stem + (f"_{'_'.join(args.prefix)}" if args.prefix else "_all") \
        + ("" if args.split == "val" else f"_{args.split}split")
    out = root / "metrics" / tag
    out.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, stats, epoch = load_model(root, device, args.ckpt)
    splits = ["train", "val"] if args.split == "all" else [args.split]
    items = [(f, x) for s in splits for f, x in read_split(root, s) if not f.startswith("public_")]
    if args.prefix:
        items = [(f, x) for f, x in items if f.startswith(tuple(args.prefix))]
    if not items:
        raise SystemExit("No frames to score. Check --prefix against the folder names in split.csv.")
    folders = sorted({f for f, _ in items})
    print(f"model {args.ckpt} (epoch {epoch}) on {len(items)} {args.split} frames "
          f"from {len(folders)} folders, device {device}")
    print("folders:", ", ".join(folders))
    frames = collect(model, stats, root, items, device)

    rows = []
    curves, sizes, cond_ap = {}, {}, {}
    for thr in IOUS:
        aps = []
        for c, name in enumerate(CLASS_NAMES):
            curves[(c, thr)] = class_ap(frames, c, thr)
            a, _, _, n = curves[(c, thr)]
            sizes[(c, thr)] = recall_by_size(frames, c, thr, args.conf)
            rows.append(["AP", name, thr, "all", "", "" if a is None else round(a, 4), n])
            if a is not None:
                aps.append(a)
        rows.append(["mAP", "all classes", thr, "all", "", round(float(np.mean(aps)), 4) if aps else "",
                     sum(curves[(c, thr)][3] for c in range(3))])
    for k in ("sun", "dusk"):
        sub = [f for f in frames if condition(f["folder"]) == k]
        if not sub:
            continue
        for c, name in enumerate(CLASS_NAMES):
            a, _, _, n = class_ap(sub, c, 0.5)
            cond_ap[(k, c)] = (a, n)
            rows.append(["AP", name, 0.5, k, "", "" if a is None else round(a, 4), n])
    for thr in IOUS:
        for c, name in enumerate(CLASS_NAMES):
            for (lo, hi), (n, found) in zip(SIZE_BUCKETS, sizes[(c, thr)]):
                if n:
                    rng = f"{lo}-{hi}" if hi < 100000 else f"{lo}+"
                    rows.append([f"recall@conf{args.conf}", name, thr, "all", rng, round(found / n, 4), n])

    with open(out / "summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "class", "iou", "condition", "height_px", "value", "labeled_boxes"])
        w.writerows(rows)

    # ---- print the headline numbers
    print(f"\n{'':<14}{'AP@0.5':>8}{'AP@0.3':>8}{'boxes':>8}")
    for c, name in enumerate(CLASS_NAMES):
        a5, a3, n = curves[(c, 0.5)][0], curves[(c, 0.3)][0], curves[(c, 0.5)][3]
        fmt = lambda v: "  n/a" if v is None else f"{v:.3f}"
        print(f"{name:<14}{fmt(a5):>8}{fmt(a3):>8}{n:>8}")
    m = {thr: [curves[(c, thr)][0] for c in range(3) if curves[(c, thr)][0] is not None] for thr in IOUS}
    print(f"{'mAP':<14}{np.mean(m[0.5]) if m[0.5] else 0:>8.3f}{np.mean(m[0.3]) if m[0.3] else 0:>8.3f}")
    if cond_ap:
        print(f"\nAP@0.5 by lighting")
        for k in ("sun", "dusk"):
            vals = [cond_ap.get((k, c)) for c in range(3)]
            if any(vals):
                print(f"  {k:<6}" + "  ".join(f"{CLASS_NAMES[c]} "
                      f"{'n/a' if v is None or v[0] is None else f'{v[0]:.3f}'} (n={0 if v is None else v[1]})"
                      for c, v in enumerate(vals)))

    # ---- graphs
    try:
        import matplotlib
        matplotlib.use("Agg")
    except ImportError:
        print("\nmatplotlib not installed: numbers saved, graphs skipped. Run: pip install matplotlib")
        return
    plot_pr(curves, out / "pr_curves.png")
    plot_sizes(sizes, args.conf, out / "recall_by_size.png")
    if cond_ap:
        plot_conditions(cond_ap, out / "ap_by_condition.png")
    log_path = Path(args.log) if args.log else root / "train_log.csv"
    if log_path.exists():
        plot_training(log_path, out / "training_curve.png")
    print(f"\nGraphs and summary.csv written to {out}")


if __name__ == "__main__":
    main()
"""
JeepBot dataset audit

Usage:
    python audit.py

Reads every folder under dataset\\ and reports:
  * frame counts per run and class
  * brightness distribution, and whether brightness alone predicts the label
  * how many distinct clips are in each folder (scene-change detection)
  * a redundancy estimate - how many near-duplicate frames you have
  * a sample of empty-class frames copied out for you to eyeball
"""

import shutil
import statistics
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "dataset"
THUMB = 64                    # downscale for speed
SCENE_CUT = 18.0              # mean abs diff above this = new clip
DUP_THRESHOLD = 3.0           # mean abs diff below this = near-duplicate


def load_folder(folder):
    """Return (filenames, small grayscale arrays) in filename order."""
    thumbs, kept = [], []
    for f in sorted(folder.glob("*.jpg")):
        img = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        thumbs.append(cv2.resize(img, (THUMB, THUMB)).astype(np.float32))
        kept.append(f)
    return kept, thumbs


def analyse(folder):
    files, thumbs = load_folder(folder)
    if not thumbs:
        return None

    brightness = [float(t.mean()) for t in thumbs]
    diffs = [float(np.abs(thumbs[i] - thumbs[i - 1]).mean())
             for i in range(1, len(thumbs))]

    clips = 1 + sum(1 for d in diffs if d > SCENE_CUT)
    dups = sum(1 for d in diffs if d < DUP_THRESHOLD)

    return {
        "folder": folder.name,
        "n": len(thumbs),
        "bright_med": statistics.median(brightness),
        "clips": clips,
        "dups": dups,
        "dup_pct": 100.0 * dups / max(1, len(diffs)),
        "brightness": brightness,
        "files": files,
    }


def main():
    if not DATASET.exists():
        print(f"no dataset folder at {DATASET}")
        return

    folders = sorted(p for p in DATASET.iterdir() if p.is_dir())
    results = []

    print(f"scanning {len(folders)} folders in {DATASET}\n")
    for f in folders:
        r = analyse(f)
        if r:
            results.append(r)
            print(f"  {r['folder']:<28} {r['n']:>5} frames")

    if not results:
        print("no frames found")
        return

    # ------------------------------------------------------------- counts
    print("\n" + "=" * 72)
    print("FRAME COUNTS")
    print("=" * 72)
    print(f"{'folder':<28}{'frames':>8}{'clips':>8}{'near-dup':>10}{'med brt':>10}")
    print("-" * 72)

    totals = defaultdict(int)
    for r in results:
        cls = ("person" if r["folder"].startswith("person")
               else "empty" if r["folder"].startswith("empty")
               else "other")
        totals[cls] += r["n"]
        print(f"{r['folder']:<28}{r['n']:>8}{r['clips']:>8}"
              f"{r['dup_pct']:>9.0f}%{r['bright_med']:>10.1f}")

    print("-" * 72)
    for cls, n in sorted(totals.items()):
        print(f"{cls:<28}{n:>8}")

    p_tot = totals.get("person", 0)
    e_tot = totals.get("empty", 0) + totals.get("other", 0)
    if p_tot and e_tot:
        ratio = max(p_tot, e_tot) / min(p_tot, e_tot)
        print(f"\nclass balance: {ratio:.2f} : 1")
        if ratio > 1.5:
            print("  ^ imbalanced enough to matter - weight the loss or subsample")

    # --------------------------------------------- brightness vs class
    print("\n" + "=" * 72)
    print("BRIGHTNESS vs CLASS   (the shortcut check)")
    print("=" * 72)

    pb = [b for r in results if r["folder"].startswith("person")
          for b in r["brightness"]]
    eb = [b for r in results if not r["folder"].startswith("person")
          for b in r["brightness"]]

    if pb and eb:
        print(f"person : median {statistics.median(pb):6.1f}   "
              f"range {min(pb):5.1f}-{max(pb):5.1f}   n={len(pb)}")
        print(f"empty  : median {statistics.median(eb):6.1f}   "
              f"range {min(eb):5.1f}-{max(eb):5.1f}   n={len(eb)}")

        best = 0.0
        for t in np.linspace(min(pb + eb), max(pb + eb), 300):
            acc = (sum(1 for b in pb if b > t) +
                   sum(1 for b in eb if b <= t)) / (len(pb) + len(eb))
            best = max(best, acc, 1.0 - acc)

        print(f"\nbest accuracy from BRIGHTNESS ALONE: {100 * best:.1f}%")
        if best > 0.70:
            print("  !! PROBLEM. A network can score this well without seeing a person.")
            print("     Balance person/empty counts within each lighting condition,")
            print("     and use strong brightness/contrast augmentation.")
        elif best > 0.60:
            print("  ^ mild shortcut available. Augmentation should cover it.")
        else:
            print("  OK - brightness is not a usable shortcut.")

    # ------------------------------------------- per-run class balance
    print("\n" + "=" * 72)
    print("PER-RUN BALANCE   (every run needs both classes)")
    print("=" * 72)

    runs = defaultdict(dict)
    for r in results:
        name = r["folder"]
        if name.startswith("person_"):
            runs[name[7:]]["person"] = r["n"]
        elif name.startswith("empty_"):
            runs[name[6:]]["empty"] = r["n"]
        else:
            runs[name]["other"] = r["n"]

    print(f"{'run':<24}{'person':>9}{'empty':>9}{'ratio':>10}")
    print("-" * 72)
    for run, c in sorted(runs.items()):
        p = c.get("person", 0)
        e = c.get("empty", 0) or c.get("other", 0)
        txt = f"{max(p, e) / min(p, e):.1f}:1" if (p and e) else "MISSING"
        bad = (not p or not e or max(p, e) / max(1, min(p, e)) > 3)
        print(f"{run:<24}{p:>9}{e:>9}{txt:>10}{'   <-- fix' if bad else ''}")

    # ---------------------------------------------- effective dataset
    print("\n" + "=" * 72)
    print("EFFECTIVE DATASET SIZE")
    print("=" * 72)

    frames = sum(r["n"] for r in results)
    clips = sum(r["clips"] for r in results)
    dups = sum(r["dups"] for r in results)

    print(f"frames on disk        : {frames}")
    print(f"distinct clips        : {clips}")
    print(f"near-duplicate frames : {dups}  ({100 * dups / max(1, frames):.0f}%)")
    print(f"frames per clip (avg) : {frames / max(1, clips):.0f}")
    print("\nSplit train/validation by CLIP, never by frame.")
    print(f"With {clips} clips, hold out roughly 20% of them for validation.")

    # ---------------------------------------------- review sample
    print("\n" + "=" * 72)
    print("LABEL-NOISE REVIEW SAMPLE")
    print("=" * 72)

    review = ROOT / "review_sample"
    review.mkdir(exist_ok=True)
    for old in review.glob("*.jpg"):
        old.unlink()

    n_each = 12
    for r in results:
        if r["folder"].startswith("person"):
            continue
        step = max(1, len(r["files"]) // n_each)
        for f in r["files"][::step][:n_each]:
            shutil.copy(f, review / f"{r['folder']}__{f.name}")

    n = len(list(review.glob("*.jpg")))
    print(f"copied {n} empty-class frames to:")
    print(f"  {review}")
    print("\nOpen that folder, view them large, and count how many contain a")
    print("person anywhere - including distant ones and people at the edges.")
    print("Report the count back. That fraction is your label-noise rate.")


if __name__ == "__main__":
    main()

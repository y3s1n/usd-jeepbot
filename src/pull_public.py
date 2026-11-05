"""pull_public.py - labeled person + fire hydrant images from COCO.

Writes, in the same format merge.py produces:
    public/person/images/*.jpg   public/person/labels/*.txt
    public/hydrant/images/*.jpg  public/hydrant/labels/*.txt
    split_public.csv             every row split=train

Classes match merge.py: 0 person, 1 fire_hydrant, 2 trash_can
The two folders are disjoint: any image containing a hydrant goes to
the hydrant folder, so no image ever carries an unlabeled hydrant.
"""
import csv
import shutil
from pathlib import Path

import fiftyone as fo
import fiftyone.zoo as foz
from fiftyone import ViewField as F

ROOT = Path(__file__).resolve().parent.parent
PUBLIC = ROOT / "public"
CLASSES = {"person": 0, "fire_hydrant": 1}
WANT = {"person": 6000, "fire hydrant": 6000}   # over-request; aspect filter drops some
AR_MIN, AR_MAX = 1.3, 1.9                       # campus frames are 1280x800 = 1.60


def fresh(name):
    if name in fo.list_datasets():
        fo.delete_dataset(name)
    return name


parts = []
for coco_name, n in WANT.items():
    for split in ("train", "validation"):
        tag = fresh(f"coco_{coco_name.replace(' ', '_')}_{split}")
        ds = foz.load_zoo_dataset(
            "coco-2017", split=split, label_types=["detections"],
            classes=[coco_name], max_samples=n, only_matching=False,
            dataset_name=tag,
        )
        print(f"  {coco_name:<14}{split:<12}{len(ds):>6}")
        parts.append(ds)

merged = fo.Dataset(fresh("jeepbot_public"))
for p in parts:
    merged.add_samples(p)

clean = (
    merged.filter_labels("ground_truth",
                         F("label").is_in(["person", "fire hydrant"]),
                         only_matches=True)
    .map_labels("ground_truth", {"fire hydrant": "fire_hydrant"})
    .clone(fresh("jeepbot_public_clean"))
)
clean.compute_metadata()

ar = F("metadata.width") / F("metadata.height")
shaped = clean.match((ar >= AR_MIN) & (ar <= AR_MAX))
print(f"\naspect filter kept {len(shaped)} of {len(clean)}")

has_hyd = F("ground_truth.detections.label").contains("fire_hydrant")
sets = {"hydrant": shaped.match(has_hyd), "person": shaped.match(~has_hyd)}

rows = []
for name, view in sets.items():
    img_dir, lbl_dir = PUBLIC / name / "images", PUBLIC / name / "labels"
    if (PUBLIC / name).exists():
        shutil.rmtree(PUBLIC / name)
    img_dir.mkdir(parents=True), lbl_dir.mkdir(parents=True)

    boxes = 0
    for s in view:
        src = Path(s.filepath)
        out_name = f"{name}_{src.stem}{src.suffix}"
        shutil.copy2(src, img_dir / out_name)
        lines = []
        for d in s.ground_truth.detections:
            x, y, w, h = d.bounding_box          # relative, top-left corner
            lines.append(f"{CLASSES[d.label]} {x + w / 2:.6f} {y + h / 2:.6f} {w:.6f} {h:.6f}")
        (lbl_dir / f"{Path(out_name).stem}.txt").write_text("\n".join(lines) + "\n")
        boxes += len(lines)
        rows.append([f"public_{name}", out_name, "train", len(lines), "coco-2017"])
    print(f"{name:<10}{len(view):>6} images{boxes:>8} boxes")

with open(ROOT / "split_public.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["folder", "file", "split", "n_boxes", "export"])
    w.writerows(sorted(rows))
print(f"\nwrote {ROOT / 'split_public.csv'}")
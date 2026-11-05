# JeepBot — campus object detector

A single-shot object detector built from scratch in PyTorch, trained on frames
captured from an OAK-D Pro W camera driving a route around campus. It finds
three classes:

| id | class          |
|----|----------------|
| 0  | `person`       |
| 1  | `fire_hydrant` |
| 2  | `trash_can`    |

No detection library is used for the model itself — the network, the grid
targets, the loss, the decoder and non-max suppression are all in this repo.
A pretrained YOLOv8 is used only as a *labelling aid* (drafting boxes and
measuring person heights), never as the model being trained.

## How it works

The network takes a `384x240` image and four 2x2 poolings reduce it to a
`24x15` grid, one cell per 16 pixels. Each cell predicts 4 class scores
(3 classes + background) and 4 box numbers, so the head is a single `1x1`
convolution with 8 output channels. `src/detect.py` turns that grid back into
pixel boxes in the `1280x800` camera frame and applies per-class NMS.

## Repository layout

Only code and small result files are tracked. **Images, labels, datasets and
model weights are intentionally not in this repository** — see
[Data](#data-not-in-this-repository).

```
src/                    all source (every script takes --root, defaulting to the repo root)
  capture.py            record frames from the OAK-D, sorted into person/ and empty/ runs
  audit.py              dataset health report: counts, brightness, near-duplicates, clips
  measure.py            YOLOv8 pass recording the tallest person height per frame
  sheets.py             contact sheets, one PNG per capture run
  verify.py             review only the frames where a wrong call matters
  apply.py              act on verify.py's decisions (move / reject; never deletes)
  prelabel.py           draft boxes for Label Studio from the pretrained COCO detector
  merge.py              rebuild labels/ and split.csv from Label Studio exports
  pull_public.py        pull a labelled person + hydrant subset out of COCO-2017
  view_labels.py        draw merged labels on their photos to see what training sees
  dataset.py            resize, grid targets, augmentation, normalization stats
  model.py              the network
  train.py              loss and training loop
  detect.py             grid -> boxes: softmax, threshold, decode, scale, NMS
  evaluate.py           precision / recall per class at several thresholds
  metrics.py            report card: AP, PR curves, recall by size, sun vs dusk
  live.py               run the trained detector live on the OAK-D or a webcam

metrics/                generated report cards (summary.csv + plots), one dir per checkpoint
split.csv               every labelled campus frame and its train/val split
split_public.csv        the COCO rows (all train)
measurements.csv        per-frame person-height measurements from measure.py
review_decisions.csv    decisions recorded by verify.py
eval_frames.csv         per-frame evaluation output
stats.json              per-channel mean/std of the training images
train_log*.csv          per-epoch costs for each training run
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` lists the core training and evaluation dependencies.
Three extras are commented out because they are only needed for specific
steps: `ultralytics` (pre-labelling and measuring), `depthai` (OAK-D capture
and the live demo) and `fiftyone` (pulling the COCO subset).

## Data (not in this repository)

The scripts expect these directories beside the repo root. They are listed in
`.gitignore` and must be rebuilt or supplied locally:

```
dataset/<run>/<run>_<frame>.jpg          captured campus frames      (~3.0 GB)
labels/<run>/<frame>.txt                 YOLO-format labels from merge.py
public/<name>/images|labels/             COCO-2017 subset            (~1.1 GB)
exports/*.json                           Label Studio project exports
checkpoints/*.pt                         trained weights
yolov8s.pt                               pretrained labelling aid (auto-downloads)
```

A label file holds one line per box, `class cx cy w h`, with the coordinates
as fractions of the image. A class of `-1` marks an ignore region.

To rebuild the derived files from a dataset and Label Studio exports:

```bash
python src/merge.py            # -> labels/ and split.csv
python src/pull_public.py      # -> public/ and split_public.csv  (needs fiftyone)
python src/dataset.py          # -> stats.json and check/ visual sanity checks
```

## Training

Validation is the last 20% of each campus folder, by a fixed rule, and is
never trained on. COCO rows are train-only.

```bash
python src/train.py                                        # all classes
python src/train.py --prefix person                        # campus person folders only
python src/train.py --prefix person public_person          # plus the COCO people
```

Weights with the lowest validation cost go to `checkpoints/best.pt`, and
per-epoch costs to `train_log.csv`.

## Evaluation

```bash
python src/evaluate.py                                     # precision/recall, draws preds/
python src/metrics.py                                      # AP, PR curves, plots -> metrics/
python src/metrics.py --ckpt person_only.pt --prefix person
```

`metrics.py` writes to `metrics/<checkpoint>/` so runs do not overwrite each
other. Current committed report cards are in [metrics/](metrics/).

## Live demo

```bash
python src/live.py                 # OAK-D Wide Pro
python src/live.py --webcam 0      # laptop webcam
python src/live.py --conf 0.3      # lower threshold: more boxes, more false alarms
```

## Status

Work in progress. The detector finds people with partial reliability and the
two small static classes are still essentially unlearned — the latest run in
`metrics/best_person/summary.csv` reports person AP 0.15 and mAP 0.05 at
IoU 0.5 (person AP 0.30 at IoU 0.3), over 1206 labelled person boxes, 35
hydrants and 82 trash cans in validation. The hydrant and trash-can classes
are badly under-represented; more labelled examples of both are the next step.

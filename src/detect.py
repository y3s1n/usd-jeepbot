"""detect.py - Stage 2, step 9: turn the network's grid into boxes.

Shared by evaluate.py and live.py. Not run on its own.

Steps, in order:
  1. softmax the 4 class scores per cell -> probabilities that add to 1
  2. argmax over the 3 OBJECT classes only -> the cell's class; its probability = confidence
  3. drop cells under the confidence threshold
  4. sigmoid the 4 box numbers; centre x = (col + ox) * 16, y = (row + oy) * 16,
     w = w * 384, h = h * 240  (pixels of the 384x240 network input)
  5. scale to the camera frame (x 1280/384 = 800/240 = 3.333)
  6. non-max suppression, separately for each class
"""
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from dataset import CELL, CLASS_NAMES, IMG_H, IMG_W
from model import Detector

FRAME_W, FRAME_H = 1280, 800
SCALE_X, SCALE_Y = FRAME_W / IMG_W, FRAME_H / IMG_H   # both 3.333
CONF_THRESHOLD = 0.9
NMS_IOU = 0.5


def load_model(root, device, name="best.pt"):
    ckpt = torch.load(Path(root) / "checkpoints" / name, map_location=device)
    model = Detector().to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()                     # batch norm uses its running averages, not batch stats
    return model, ckpt["stats"], ckpt.get("epoch")


def preprocess(rgb_image, stats):
    """PIL RGB image (any size) -> normalized tensor [1, 3, 240, 384]."""
    img = rgb_image.convert("RGB").resize((IMG_W, IMG_H), Image.BILINEAR)
    x = torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0).permute(2, 0, 1)
    mean = torch.tensor(stats["mean"]).view(3, 1, 1)
    std = torch.tensor(stats["std"]).view(3, 1, 1)
    return ((x - mean) / std).unsqueeze(0)


def iou(a, b):
    """a: [4], b: [N, 4], boxes as x0, y0, x1, y1. Returns [N]."""
    ix0 = np.maximum(a[0], b[:, 0])
    iy0 = np.maximum(a[1], b[:, 1])
    ix1 = np.minimum(a[2], b[:, 2])
    iy1 = np.minimum(a[3], b[:, 3])
    inter = np.clip(ix1 - ix0, 0, None) * np.clip(iy1 - iy0, 0, None)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(area_a + area_b - inter, 1e-9)


def nms(boxes, scores, threshold=NMS_IOU):
    """Keep the most confident box, delete others overlapping it above threshold, repeat."""
    order = np.argsort(-scores)
    keep = []
    while len(order):
        best = order[0]
        keep.append(best)
        rest = order[1:]
        order = rest[iou(boxes[best], boxes[rest]) < threshold]
    return keep


@torch.no_grad()
def decode(class_scores, box_raw, conf=CONF_THRESHOLD, peaks=False):
    """One image's network output -> list of (class_index, confidence, x0, y0, x1, y1)
    in 1280x800 camera pixels. class_index: 0 person, 1 fire_hydrant, 2 trash_can.

    class_scores [4, 15, 24]   box_raw [4, 15, 24]
    """
    probs = torch.softmax(class_scores, dim=0)              # step 1
    # Step 2: best OBJECT class per cell. Background is left out of the argmax, so a cell
    # that says "60% background, 35% person" still counts as a 0.35 person candidate.
    confidence, cls = probs[1:].max(dim=0)                  # [15, 24]
    keep = confidence >= conf                               # step 3
    if peaks:
        # Keep a cell only if it is the strongest for its class in its 3x3 patch. Neighbours of an
        # object's centre are ignored in training, so they may fire with untrained boxes.
        pooled = F.max_pool2d(probs[1:].unsqueeze(0), 3, stride=1, padding=1)[0]
        is_peak = probs[1:] >= pooled                       # [3, 15, 24]
        keep &= is_peak.gather(0, cls.unsqueeze(0))[0]
    cls = cls + 1                                           # 1..3 again
    rows, cols = keep.nonzero(as_tuple=True)
    if len(rows) == 0:
        return []

    box = torch.sigmoid(box_raw)[:, rows, cols]             # [4, N]: ox, oy, w, h
    cx = (cols + box[0]) * CELL                             # step 4
    cy = (rows + box[1]) * CELL
    w = box[2] * IMG_W
    h = box[3] * IMG_H

    x0 = (cx - w / 2) * SCALE_X                             # step 5
    y0 = (cy - h / 2) * SCALE_Y
    x1 = (cx + w / 2) * SCALE_X
    y1 = (cy + h / 2) * SCALE_Y
    boxes = torch.stack([x0, y0, x1, y1], dim=1).cpu().numpy()
    scores = confidence[rows, cols].cpu().numpy()
    labels = (cls[rows, cols] - 1).cpu().numpy()            # back to 0 = person

    detections = []
    for c in range(len(CLASS_NAMES)):                       # step 6, per class
        idx = np.where(labels == c)[0]
        for k in nms(boxes[idx], scores[idx]):
            i = idx[k]
            detections.append((c, float(scores[i]), *boxes[i].tolist()))
    return detections

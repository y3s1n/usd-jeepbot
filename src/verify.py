"""
JeepBot - verify the detections that matter

Usage:
    python verify.py            # the 2 mandatory queues
    python verify.py spot       # random spot-check of the rest

Reads measurements.csv, shows ONLY the frames where a wrong call would hurt,
with the detector's boxes drawn and labelled in pixels. Decides nothing on
disk - writes review_decisions.csv for apply.py.

------------------------------------------------------------------- CONTROLS

  P   a real member of the public, at or above the threshold  -> person
  E   nobody usable in this frame                             -> empty
  X   reject - unusable frame

  C   mark CREW (you, a teammate, the cart, your shadow) and send to empty
      use this whenever the box is on one of your own team

  E / LEFT   back one      RIGHT  skip without deciding
  B          toggle the 60px reference bar
  L          toggle brightness boost (for dusk frames)
  Z          cycle zoom  fit -> 100% -> 200%   (click to recentre)
  Q          save and quit
"""

import csv
import random
import sys
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "dataset"
MEAS_CSV = ROOT / "measurements.csv"
OUT_CSV = ROOT / "review_decisions.csv"
ONLY = next((a for a in sys.argv[1:] if a != "spot"), None)

T_PERSON = 60          # at or above this -> person
T_EMPTY = 25           # below this -> empty
MODEL = str(ROOT / "yolov8s.pt")
IMGSZ = 1280
CONF = 0.25
VIEW_W = 1000
ZOOMS = [None, 1.0, 2.0]


def build_queues(spot=False):
    with open(MEAS_CSV, newline="") as fh:
        rows = [r for r in csv.DictReader(fh)]
    for r in rows:
        r["max_h"] = float(r["max_h"])
        r["n_people"] = int(r["n_people"])
    if ONLY:
        rows = [r for r in rows if ONLY in r["folder"]]

    if spot:
        suspect = {"empty_A_up_sun", "empty_A_up_dusk",
                   "empty_B_up_dusk", "empty_B_down_dusk"}
        pool = [r for r in rows if r["folder"] in suspect
                and r["max_h"] < T_EMPTY]
        random.seed(0)
        print(f"spot check: 80 of {len(pool)} no-detection frames "
              f"in the backlit/dusk empty folders")
        return random.sample(pool, min(80, len(pool)))

    q1 = [r for r in rows if r["folder"].startswith("empty")
          and r["max_h"] >= T_PERSON]
    q2 = [r for r in rows if r["folder"].startswith("person")
          and r["max_h"] < T_EMPTY]

    # biggest detections first - the most damaging ones to get wrong
    q1.sort(key=lambda r: -r["max_h"])
    print(f"queue 1: {len(q1)} empty frames with a person >= {T_PERSON}px")
    print(f"queue 2: {len(q2)} person frames with nobody >= {T_EMPTY}px")
    return q1 + q2


def load_decisions():
    out = {}
    if OUT_CSV.exists():
        with open(OUT_CSV, newline="") as fh:
            for row in csv.DictReader(fh):
                out[row["path"]] = row
    return out


def save_decisions(d):
    with open(OUT_CSV, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["path", "folder", "decision", "tags"])
        w.writeheader()
        w.writerows(d.values())


def boost(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)


class View:
    def __init__(self):
        self.centre = None
        self.zoom_i = 0

    def on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            scale, ox, oy = param
            self.centre = (int(ox + x / scale), int(oy + y / scale))

    def draw(self, img, boxes, show_bar, boosted):
        src_h, src_w = img.shape[:2]
        work = boost(img) if boosted else img.copy()

        for (x1, y1, x2, y2, conf) in boxes:
            h = y2 - y1
            colour = (60, 220, 60) if h >= T_PERSON else \
                     (60, 200, 240) if h >= T_EMPTY else (120, 120, 120)
            cv2.rectangle(work, (int(x1), int(y1)), (int(x2), int(y2)), colour, 3)
            tag = f"{h:.0f}px  {conf:.2f}"
            ty = max(int(y1) - 8, 18)
            cv2.putText(work, tag, (int(x1), ty),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4)
            cv2.putText(work, tag, (int(x1), ty),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2)

        zoom = ZOOMS[self.zoom_i]
        if zoom is None:
            scale = VIEW_W / src_w
            view = cv2.resize(work, (VIEW_W, int(src_h * scale)))
            ox = oy = 0
        else:
            scale = zoom
            vw = min(int(VIEW_W / zoom), src_w)
            vh = min(int(VIEW_W * 0.625 / zoom), src_h)
            cx, cy = self.centre or (src_w // 2, src_h // 2)
            ox = int(np.clip(cx - vw // 2, 0, src_w - vw))
            oy = int(np.clip(cy - vh // 2, 0, src_h - vh))
            view = cv2.resize(work[oy:oy + vh, ox:ox + vw],
                              (int(vw * zoom), int(vh * zoom)))

        if show_bar:
            bh = int(T_PERSON * scale)
            x0, y0 = 40, view.shape[0] - 50
            cv2.rectangle(view, (x0 - 3, y0 - bh - 3), (x0 + 11, y0 + 3), (0, 0, 0), -1)
            cv2.rectangle(view, (x0, y0 - bh), (x0 + 8, y0), (255, 255, 255), -1)
            cv2.putText(view, f"{T_PERSON}px", (x0 + 16, y0),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        return view, (scale, ox, oy)


def main():
    spot = "spot" in sys.argv
    if not MEAS_CSV.exists():
        print(f"no {MEAS_CSV} - run measure.py first")
        return

    queue = build_queues(spot)
    if not queue:
        print("nothing to verify")
        return

    decisions = load_decisions()
    model = YOLO(MODEL)
    view = View()
    show_bar, boosted = True, False
    crew = False

    idx = 0
    for i, r in enumerate(queue):
        if r["path"] not in decisions:
            idx = i
            break

    cv2.namedWindow("verify", cv2.WINDOW_AUTOSIZE)

    while 0 <= idx < len(queue):
        row = queue[idx]
        path = DATASET / row["path"]
        img = cv2.imread(str(path))
        if img is None:
            idx += 1
            continue

        det = model.predict(source=str(path), imgsz=IMGSZ, conf=CONF,
                            classes=[0], verbose=False)[0]
        boxes = [(*b.xyxy[0].tolist(), float(b.conf[0])) for b in det.boxes]

        frame, params = view.draw(img, boxes, show_bar, boosted)
        cv2.setMouseCallback("verify", view.on_mouse, params)

        in_class = "empty" if row["folder"].startswith("empty") else "person"
        prev = decisions.get(row["path"], {})
        zl = "fit" if ZOOMS[view.zoom_i] is None else f"{ZOOMS[view.zoom_i]:.0f}x"

        panel = np.full((96, frame.shape[1], 3), 28, dtype=np.uint8)
        lines = [
            f"{idx + 1} / {len(queue)}    in: {in_class}    "
            f"detector max: {row['max_h']:.0f}px    boxes: {len(boxes)}",
            f"{row['folder']} / {Path(row['path']).name}    zoom {zl}"
            f"{'   BOOSTED' if boosted else ''}",
            f"P=person  E=empty  C=crew->empty  X=reject"
            f"{'      last: ' + prev['decision'] if prev else ''}",
        ]
        for i, t in enumerate(lines):
            cv2.putText(panel, t, (12, 28 + i * 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, (225, 225, 225), 1)

        cv2.imshow("verify", np.vstack([frame, panel]))
        key = cv2.waitKey(0) & 0xFF

        def record(target_class, tag=""):
            decisions[row["path"]] = {
                "path": row["path"],
                "folder": row["folder"],
                "decision": "keep" if target_class == in_class else "move",
                "tags": tag,
            }

        if key == ord("p"):
            record("person")
            idx += 1
        elif key == ord("e"):
            record("empty")
            idx += 1
        elif key == ord("c"):
            record("empty", "crew")
            idx += 1
        elif key == ord("x"):
            decisions[row["path"]] = {"path": row["path"], "folder": row["folder"],
                                      "decision": "reject", "tags": ""}
            idx += 1
        elif key in (83, ord(".")):
            idx += 1
        elif key in (81, ord(",")):
            idx = max(0, idx - 1)
        elif key == ord("b"):
            show_bar = not show_bar
        elif key == ord("l"):
            boosted = not boosted
        elif key == ord("z"):
            view.zoom_i = (view.zoom_i + 1) % len(ZOOMS)
        elif key == ord("q"):
            break

        if len(decisions) % 25 == 0 and decisions:
            save_decisions(decisions)

    save_decisions(decisions)
    cv2.destroyAllWindows()

    print(f"\nsaved {len(decisions)} decisions to {OUT_CSV}")

    counts, crew_n, by_folder = {}, 0, {}
    for r in decisions.values():
        counts[r["decision"]] = counts.get(r["decision"], 0) + 1
        if "crew" in r["tags"]:
            crew_n += 1
            by_folder[r["folder"]] = by_folder.get(r["folder"], 0) + 1
    for k, v in sorted(counts.items()):
        print(f"  {k:<10}{v:>6}")
    if crew_n:
        print(f"\ncrew-caused false detections: {crew_n}")
        for f, n in sorted(by_folder.items(), key=lambda x: -x[1]):
            print(f"  {f:<26}{n:>5}")


if __name__ == "__main__":
    main()

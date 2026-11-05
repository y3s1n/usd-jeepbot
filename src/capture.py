"""
JeepBot dataset capture — OAK-D Pro W

Usage:
    python capture.py A_up_sun

Keys (click the preview window first so it has focus):
    1      switch to PERSON  (green)
    2      switch to EMPTY   (blue)
    SPACE  start / stop recording
    Q      quit

Frames go to  dataset\\person_<run>\\  or  dataset\\empty_<run>\\
depending on which class is selected when you record.
"""

import sys
import time
from pathlib import Path

import cv2
import depthai as dai

# ----------------------------------------------------------------- settings
WIDTH, HEIGHT = 1280, 800                                   # OV9782 max
FPS           = 30                                          # camera frame rate
SAVE_EVERY    = 15                                          # keep 1 frame in N  (15 -> 2 per second)
ROOT = Path(__file__).resolve().parent.parent
OUT_DIR       = ROOT / "dataset"
JPEG_QUALITY  = 95
CLIP_TARGET   = 45                                          # seconds, for the on-screen timer
# ---------------------------------------------------------------------------

run = sys.argv[1] if len(sys.argv) > 1 else "unlabeled"

CLASSES = {
    "person": {"dir": OUT_DIR / f"person_{run}", "colour": (80, 220, 80)},
    "empty":  {"dir": OUT_DIR / f"empty_{run}",  "colour": (240, 160, 60)},
}

# make the folders and pick up where any earlier run left off
for name, info in CLASSES.items():
    info["dir"].mkdir(parents=True, exist_ok=True)
    stem = f"{name}_{run}"
    existing = sorted(info["dir"].glob(f"{stem}_*.jpg"))
    info["count"] = int(existing[-1].stem.split("_")[-1]) + 1 if existing else 0
    info["stem"] = stem

print(f"run        : {run}")
print(f"resolution : {WIDTH}x{HEIGHT} @ {FPS} fps")
print(f"keeping    : 1 frame in {SAVE_EVERY}  ({FPS / SAVE_EVERY:.1f} per second)")
for name, info in CLASSES.items():
    print(f"  {name:<7}-> {info['dir']}  (starting at {info['count']})")
print("\n1 = person    2 = empty    SPACE = record    Q = quit")

with dai.Pipeline() as pipeline:
    cam = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
    out = cam.requestOutput((WIDTH, HEIGHT), dai.ImgFrame.Type.BGR888i, fps=FPS)
    queue = out.createOutputQueue()
    pipeline.start()

    current = "person"
    recording = False
    frame_idx = 0
    clip_started = 0.0
    clip_saved = 0

    while pipeline.isRunning():
        packet = queue.get()
        if packet is None:
            continue
        frame = packet.getCvFrame()
        info = CLASSES[current]

        if recording:
            frame_idx += 1
            if frame_idx % SAVE_EVERY == 0:
                cv2.imwrite(
                    str(info["dir"] / f"{info['stem']}_{info['count']:05d}.jpg"),
                    frame,
                    [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY],
                )
                info["count"] += 1
                clip_saved += 1

        # ---- preview -------------------------------------------------------
        preview = cv2.resize(frame, (960, 600))
        colour = info["colour"]

        # class banner across the top, always visible at a glance
        cv2.rectangle(preview, (0, 0), (960, 44), colour, -1)
        cv2.putText(preview, current.upper(), (16, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (20, 20, 20), 2)
        cv2.putText(preview, run, (200, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (40, 40, 40), 1)

        if recording:
            elapsed = time.time() - clip_started
            status = f"REC  {elapsed:4.1f}s / {CLIP_TARGET}s   +{clip_saved}"
            cv2.circle(preview, (930, 22), 11, (0, 0, 255), -1)
            # thick border so it is obvious you are rolling
            cv2.rectangle(preview, (2, 2), (958, 598), (0, 0, 255), 4)
        else:
            status = "paused"

        totals = f"person {CLASSES['person']['count']}   empty {CLASSES['empty']['count']}"
        cv2.putText(preview, status, (16, 574),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(preview, totals, (640, 574),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        cv2.imshow("capture", preview)

        # ---- keys ----------------------------------------------------------
        key = cv2.waitKey(1) & 0xFF

        if key == ord(" "):
            recording = not recording
            if recording:
                clip_started = time.time()
                clip_saved = 0
                frame_idx = 0
                print(f"  REC  [{current}]")
            else:
                print(f"  stop [{current}] — {clip_saved} frames, "
                      f"{time.time() - clip_started:.1f}s")

        elif key in (ord("1"), ord("2")):
            wanted = "person" if key == ord("1") else "empty"
            if wanted != current:
                if recording:          # never let a clip span two classes
                    recording = False
                    print(f"  stop [{current}] — {clip_saved} frames (class switch)")
                current = wanted
                print(f"  class -> {current}")

        elif key == ord("q"):
            break

cv2.destroyAllWindows()
print(f"\ndone — person {CLASSES['person']['count']}, empty {CLASSES['empty']['count']}")
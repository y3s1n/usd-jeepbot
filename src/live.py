"""live.py - run the trained detector on the OAK-D camera and show boxes live.

    python src/live.py                 # OAK-D Wide Pro
    python src/live.py --webcam 0      # laptop webcam instead, for a quick test
    python src/live.py --conf 0.3      # lower threshold: more boxes, more false alarms

Press q in the window to quit.
Needs: pip install opencv-python   (depthai is already in your oak environment)
"""
import argparse
import time
from pathlib import Path

import cv2
import torch
from PIL import Image

from dataset import CLASS_NAMES
from detect import FRAME_H, FRAME_W, decode, load_model, preprocess

ROOT = Path(__file__).resolve().parent.parent

COLORS_BGR = {0: (0, 165, 255), 1: (0, 0, 255), 2: (255, 255, 0)}   # person, hydrant, trash can


def oak_frames():
    """Yield BGR frames, 1280x800, from the OAK-D color camera. Handles depthai v2 and v3."""
    import depthai as dai
    if dai.__version__.startswith("3"):
        with dai.Pipeline() as pipeline:
            cam = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
            queue = cam.requestOutput((FRAME_W, FRAME_H), dai.ImgFrame.Type.BGR888p).createOutputQueue()
            pipeline.start()
            while pipeline.isRunning():
                yield queue.get().getCvFrame()
    else:
        pipeline = dai.Pipeline()
        cam = pipeline.create(dai.node.ColorCamera)
        cam.setBoardSocket(dai.CameraBoardSocket.CAM_A)
        cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_800_P)
        cam.setFps(30)
        out = pipeline.create(dai.node.XLinkOut)
        out.setStreamName("rgb")
        cam.isp.link(out.input)
        with dai.Device(pipeline) as device:
            queue = device.getOutputQueue("rgb", maxSize=4, blocking=False)
            while True:
                yield queue.get().getCvFrame()


def webcam_frames(index):
    cap = cv2.VideoCapture(index)
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        yield cv2.resize(frame, (FRAME_W, FRAME_H))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(ROOT))
    ap.add_argument("--conf", type=float, default=0.8)
    ap.add_argument("--webcam", type=int, default=None)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, stats, epoch = load_model(args.root, device)
    print(f"model from epoch {epoch}, device {device}. Press q to quit.")

    frames = webcam_frames(args.webcam) if args.webcam is not None else oak_frames()
    last = time.time()
    for bgr in frames:
        # The training frames were read as RGB; OpenCV gives BGR, so flip the channel order.
        rgb = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        with torch.no_grad():
            scores, box_raw = model(preprocess(rgb, stats).to(device))
        for c, conf, x0, y0, x1, y1 in decode(scores[0], box_raw[0], conf=args.conf):
            p0, p1 = (int(x0), int(y0)), (int(x1), int(y1))
            cv2.rectangle(bgr, p0, p1, COLORS_BGR[c], 3)
            cv2.putText(bgr, f"{CLASS_NAMES[c]} {conf:.2f}", (p0[0] + 4, p0[1] + 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLORS_BGR[c], 2)

        now = time.time()
        cv2.putText(bgr, f"{1 / max(now - last, 1e-6):.0f} fps", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        last = now
        cv2.imshow("jeepbot detector", bgr)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

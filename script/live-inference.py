"""
Live YOLO Inference Script
--------------------------
Requirements:
    pip install ultralytics opencv-python

Usage:
    python live_inference.py                        # webcam (default)
    python live_inference.py --source 0             # webcam index 0
    python live_inference.py --source video.mp4     # video file
    python live_inference.py --source rtsp://...    # RTSP stream

Controls:
    Q or ESC  → quit
    P         → pause / resume
    S         → save screenshot
"""

import argparse
import time
import cv2
import numpy as np
from pathlib import Path
from collections import deque
from ultralytics import YOLO

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
MODEL_PATH   = "best.pt"       # ← change if your .pt lives elsewhere
CONF_THRESH  = 0.40            # minimum confidence to show a box
IOU_THRESH   = 0.45            # NMS IoU threshold
IMG_SIZE     = 640             # inference resolution
DEVICE       = ""              # "" = auto (GPU if available, else CPU)
FPS_WINDOW   = 30              # rolling average over N frames

# Colour palette (BGR) — one per class, cycling if more classes than colours
PALETTE = [
    (0,   200, 255), (0,   255, 128), (255, 80,  80),
    (80,  80,  255), (255, 200, 0  ), (180, 0,   255),
    (0,   255, 200), (255, 128, 0  ), (0,   180, 255),
    (128, 255, 0  ),
]

def color_for(class_id: int):
    return PALETTE[class_id % len(PALETTE)]


def draw_box(frame, x1, y1, x2, y2, label: str, conf: float, class_id: int):
    """Draw a filled-header bounding box."""
    color = color_for(class_id)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    text  = f"{label}  {conf:.0%}"
    font  = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.55
    thick = 1
    (tw, th), _ = cv2.getTextSize(text, font, scale, thick)

    # header background
    by1 = max(0, y1 - th - 8)
    by2 = y1
    cv2.rectangle(frame, (x1, by1), (x1 + tw + 8, by2), color, -1)
    cv2.putText(frame, text, (x1 + 4, by2 - 4),
                font, scale, (0, 0, 0), thick, cv2.LINE_AA)


def draw_hud(frame, fps: float, det_count: int, paused: bool,
             model_name: str, width: int, height: int):
    """Draw semi-transparent HUD bar at the top of the frame."""
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 38), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    font  = cv2.FONT_HERSHEY_SIMPLEX
    left  = f"Model: {model_name}   {width}x{height}"
    right = f"{'⏸ PAUSED   ' if paused else ''}FPS: {fps:5.1f}   Detections: {det_count}"

    cv2.putText(frame, left,  (10, 26), font, 0.58, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(frame, right, (w - 10 - cv2.getTextSize(right, font, 0.58, 1)[0][0], 26),
                font, 0.58, (0, 220, 255), 1, cv2.LINE_AA)


def draw_class_summary(frame, class_counts: dict, names: dict):
    """Draw a small legend showing detected classes and counts."""
    if not class_counts:
        return
    h, w = frame.shape[:2]
    x, y = 10, 50
    for cls_id, cnt in sorted(class_counts.items()):
        label = names.get(cls_id, str(cls_id))
        text  = f"{label}: {cnt}"
        color = color_for(cls_id)
        cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, color, 2, cv2.LINE_AA)
        y += 22


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def run(source, model_path, conf, iou, imgsz, device):
    model  = YOLO(model_path)
    names  = model.names          # {0: 'cat', 1: 'dog', ...}
    mname  = Path(model_path).stem

    # Try to open source
    src = int(source) if str(source).isdigit() else source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source: {source}")

    fps_times: deque = deque(maxlen=FPS_WINDOW)
    paused    = False
    frame_w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    last_results = None

    print(f"\n✅  Model   : {model_path}")
    print(f"   Classes : {list(names.values())}")
    print(f"   Source  : {source}")
    print(f"   Conf    : {conf}   IoU: {iou}   Size: {imgsz}")
    print(f"\n   Q / ESC → quit  |  P → pause  |  S → screenshot\n")

    screenshot_idx = 0

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):    # Q or ESC
            break
        if key == ord('p'):
            paused = not paused
        if key == ord('s'):
            fname = f"screenshot_{screenshot_idx:04d}.jpg"
            cv2.imwrite(fname, frame)
            print(f"📸  Saved {fname}")
            screenshot_idx += 1

        if paused and last_results is not None:
            cv2.imshow("YOLO Live Inference", frame)
            continue

        ret, frame = cap.read()
        if not ret:
            print("⚠️  Stream ended or frame read failed.")
            break

        t0 = time.perf_counter()

        results = model.predict(
            source    = frame,
            conf      = conf,
            iou       = iou,
            imgsz     = imgsz,
            device    = device,
            verbose   = False,
        )[0]

        t1 = time.perf_counter()
        fps_times.append(t1 - t0)
        fps = 1.0 / (sum(fps_times) / len(fps_times))

        # ── Draw detections ──────────────────────────────
        class_counts = {}
        if results.boxes is not None:
            for box in results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf_val = float(box.conf[0])
                cls_id   = int(box.cls[0])
                label    = names.get(cls_id, str(cls_id))
                draw_box(frame, x1, y1, x2, y2, label, conf_val, cls_id)
                class_counts[cls_id] = class_counts.get(cls_id, 0) + 1

        det_count = sum(class_counts.values())
        draw_hud(frame, fps, det_count, paused, mname, frame_w, frame_h)
        draw_class_summary(frame, class_counts, names)

        last_results = results
        cv2.imshow("YOLO Live Inference", frame)

    cap.release()
    cv2.destroyAllWindows()
    print("👋  Inference stopped.")


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Live YOLO inference")
    parser.add_argument("--source",  default="0",        help="Camera index, video path, or RTSP URL")
    parser.add_argument("--model",   default=MODEL_PATH, help="Path to .pt weights file")
    parser.add_argument("--conf",    default=CONF_THRESH, type=float)
    parser.add_argument("--iou",     default=IOU_THRESH,  type=float)
    parser.add_argument("--imgsz",   default=IMG_SIZE,    type=int)
    parser.add_argument("--device",  default=DEVICE,      help="cpu / 0 / 0,1 / cuda:0")
    args = parser.parse_args()

    run(
        source     = args.source,
        model_path = args.model,
        conf       = args.conf,
        iou        = args.iou,
        imgsz      = args.imgsz,
        device     = args.device,
    )
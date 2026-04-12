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

from memory_config import meta_dtype, NUM_CLASSES, dtype, COLOR_TO_ID
from multiprocessing import shared_memory
red_lower = np.array([136, 87, 111], np.uint8)
red_upper = np.array([180, 255, 255], np.uint8)

from mss import mss
# define mask

#green color
green_lower = np.array([25, 52, 72], np.uint8)
green_upper = np.array([102, 255, 255], np.uint8)

#blue color
blue_lower = np.array([94, 80, 2], np.uint8)
blue_upper = np.array([120, 255, 255], np.uint8)

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
MODEL_PATH   = "rdk/model/best.pt"       # ← change if your .pt lives elsewhere
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

def get_shm_writer(name, size):
    try:
        shm = shared_memory.SharedMemory(create=True, size=size, name=name)
    except FileExistsError:
        # leftover from previous crash — clean it up and recreate
        old = shared_memory.SharedMemory(name=name)
        old.unlink()
        old.close()
        shm = shared_memory.SharedMemory(create=True, size=size, name=name)
    return shm

shm = get_shm_writer("yolo", meta_dtype.itemsize)
mem = np.ndarray((), dtype=meta_dtype, buffer=shm.buf)
mem['seq'] = 0


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

from matplotlib import pyplot as plt
def color_det(x1, y1, x2, y2, image, class_id):
    img = image[y1:y2, x1:x2]
    img = cv2.resize(img, (64, 64), interpolation=cv2.INTER_AREA)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    pixels = hsv.reshape(-1, 3)
    h_bins = (pixels[:, 0] // 10).astype(np.uint8)  # quantize hue
    dominant_bin = np.bincount(h_bins).argmax()
    
    mask = h_bins == dominant_bin
    dominant_hsv = pixels[mask].mean(axis=0)
    
    h, s, v = dominant_hsv
    if s < 40:
        if v < 60:   return "hitam"
        if v > 200:  return "putih"
        return "abu-abu"
    
    if h < 10 or h >= 160:   return "merah"
    if 10 <= h < 25:         return "oranye"
    if 25 <= h < 35:         return "kuning"
    if 35 <= h < 85:         return "hijau"
    if 85 <= h < 130:        return "biru"
    if 130 <= h < 160:       return "ungu"
    
    return "tidak diketahui"


def unique_count_app(a):
    colors, count = np.unique(a.reshape(-1,a.shape[-1]), axis=0, return_counts=True)
    return colors[count.argmax()]



# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def run(source, model_path, conf, iou, imgsz, device, gui):
    model  = YOLO(model_path)
    names  = model.names
    mname  = Path(model_path).stem

    src = int(source) if str(source).isdigit() else source

    # ── Setup source ──────────────────────────────────────
    is_screen = (source == "scr")
    sct = None
    monitor = None
    cap = None

    if is_screen:
        print("Capturing screen...")
        sct     = mss()
        monitor = sct.monitors[1]  # primary monitor
        frame_w = monitor['width']
        frame_h = monitor['height']
    else:
        cap = cv2.VideoCapture(src)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open source: {source}")
        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fps_times: deque = deque(maxlen=FPS_WINDOW)
    paused       = False
    last_results = None
    screenshot_idx = 0

    print(f"\n  Model   : {model_path}")
    print(f"  Classes : {list(names.values())}")
    print(f"  Source  : {'screen' if is_screen else source}")
    print(f"  Conf    : {conf}   IoU: {iou}   Size: {imgsz}")
    print(f"\n  Q / ESC → quit  |  P → pause  |  S → screenshot\n")

    try:
        while True:
            # ── Read frame ───────────────────────────────────
            if is_screen:
                raw   = sct.grab(monitor)           # <-- grab tiap iterasi
                frame = np.array(raw)[:, :, :3]
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                frame = cv2.resize(frame, (frame_w//3, frame_h//3), interpolation=cv2.INTER_AREA)
                ret   = True
            else:
                ret, frame = cap.read()

            # ── Key handling ─────────────────────────────────
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                break
            if key == ord('p'):
                paused = not paused
            if key == ord('s'):
                fname = f"screenshot_{screenshot_idx:04d}.jpg"
                cv2.imwrite(fname, frame)
                print(f"Saved {fname}")
                screenshot_idx += 1

            if paused and last_results is not None:
                cv2.imshow("YOLO Live Inference", frame)
                continue

            if not ret:
                print("  Stream ended or frame read failed.")
                break

            # ── Inference & draw (tidak berubah) ─────────────
            t0 = time.perf_counter()
            results = model.predict(
                source  = frame,
                conf    = conf,
                iou     = iou,
                imgsz   = imgsz,
                device  = device,
                verbose = False,
            )[0]
            t1 = time.perf_counter()
            fps_times.append(t1 - t0)
            fps = 1.0 / (sum(fps_times) / len(fps_times))

            best: dict[int, dict] = {}


            class_counts = {}
            if results.boxes is not None:
                for box in results.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    conf_val = float(box.conf[0])
                    cls_id   = int(box.cls[0])

                    if cls_id not in best or conf_val > best[cls_id]['conf']:
                        color_str = color_det(x1, y1, x2, y2, frame, cls_id)
                        color_id = COLOR_TO_ID.get(color_str, 9) # 9 is default "tidak diketahui"
                        label = f"{names.get(cls_id, str(cls_id))}-{color_str}"

                        # draw_box(frame, x1, y1, x2, y2, label, conf_val, cls_id)
                        cx = (x1 + x2) / 2
                        cy = (y1 + y2) / 2
                        r  = max(x2 - x1, y2 - y1) / 2
                        cv2.circle(frame, (int(cx), int(cy)), int(r), color=(0, 255, 255), thickness=2)
                        cv2.drawMarker(frame, (int(cx), int(cy)), color=(0, 255, 255), markerType=cv2.MARKER_CROSS, markerSize=10, thickness=1)

                        best[cls_id] = {'x1': cx, 'y1': cy, 'r': r, 'conf': conf_val, 'color_id': color_id}

                    class_counts[cls_id] = class_counts.get(cls_id, 0) + 1

            # --- SEQ LOCK WRITER ---
            mem['seq'] += 1 # Make odd
            
            mem['count'] = len(best)
            mem['ts']    = time.monotonic()

            for i in range(NUM_CLASSES):
                if i in best:
                    obj = best[i]
                    mem['objects'][i]['x1']       = obj['x1']
                    mem['objects'][i]['y1']       = obj['y1']
                    mem['objects'][i]['r']        = obj['r']
                    mem['objects'][i]['conf']     = obj['conf']
                    mem['objects'][i]['class_id'] = i
                    mem['objects'][i]['color_id'] = obj['color_id']
                else:
                    mem['objects'][i]['conf']     = 0.0

            mem['seq'] += 1 # Make even
            # -----------------------

            det_count = sum(class_counts.values())
            draw_hud(frame, fps, det_count, paused, mname, frame_w, frame_h)
            draw_class_summary(frame, class_counts, names)
            last_results = results

            if gui:
                cv2.imshow("YOLO Live Inference", frame)

    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        # ── Cleanup ───────────────────────────────────────────
        try:
            shm.close()
            shm.unlink()
        except FileNotFoundError:
            pass
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()
        print("Inference stopped.")

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
    parser.add_argument("--gui", default=False, type=bool)
    # parser.add_argument("--mem-share", default=False, help="Enable shared memory output")
    args = parser.parse_args()

    run(
        source     = args.source,
        model_path = args.model,
        conf       = args.conf,
        iou        = args.iou,
        imgsz      = args.imgsz,
        device     = args.device,
        gui       = args.gui,
    )
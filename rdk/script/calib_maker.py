import cv2
import numpy as np
import os
import glob

# --- config ---
INPUT_DIR  = "rdk_val_images"  
OUTPUT_DIR = "rdk_calibration"
IMG_SIZE   = 640
N_IMAGES   = 110
# --------------

os.makedirs(OUTPUT_DIR, exist_ok=True)

def letterbox(img, size=640):
    h, w = img.shape[:2]
    scale = size / max(h, w)
    nh, nw = int(h * scale), int(w * scale)
    img = cv2.resize(img, (nw, nh))
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    pad_top  = (size - nh) // 2
    pad_left = (size - nw) // 2
    canvas[pad_top:pad_top+nh, pad_left:pad_left+nw] = img
    return canvas

image_paths = sorted(glob.glob(os.path.join(INPUT_DIR, "*.png")))[:N_IMAGES]

for i, path in enumerate(image_paths):
    img = cv2.imread(path)                    # BGR, HWC, uint8
    img = letterbox(img, IMG_SIZE)            # 640x640, still BGR
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # BGR → RGB (YOLOv8 trains on RGB)
    img = img.astype(np.float32) / 255.0      # normalize to [0, 1]
    img = img.transpose(2, 0, 1)              # HWC → CHW
    # img = np.expand_dims(img, 0)              # CHW → NCHW (1,3,640,640)
    
    out_path = os.path.join(OUTPUT_DIR, f"{i:06d}.npy")
    img.tofile(out_path)                      # save as raw binary float32
    print(f"Saved {out_path}")

print(f"Done. {len(image_paths)} calibration files ready.")
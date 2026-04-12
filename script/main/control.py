from multiprocessing import shared_memory
import numpy as np, time
from memory_config import meta_dtype, NUM_CLASSES, COLOR_DICT, CLASS_DICT
import cv2 as cv

img = np.zeros((644, 644, 3), np.uint8)

STALE_THRESHOLD = 0.05  # 50ms

def get_shm_reader(name, retries=20, delay=0.5):
    for i in range(retries):
        try:
            return shared_memory.SharedMemory(name=name)
        except FileNotFoundError:
            print(f"[wait] shm '{name}' not ready, retrying ({i+1}/{retries})...")
            time.sleep(delay)
    raise RuntimeError(f"Shared memory '{name}' not found after {retries} retries")

shm = get_shm_reader("yolo")

frame = np.ndarray((), dtype=meta_dtype, buffer=shm.buf)

try:
    while True:
        # 1. Read sequence lock start
        seq1 = frame['seq']
        if seq1 % 2 == 1:
            continue  # Writer is modifying data
            
        # 2. Read data
        ts = float(frame['ts'])
        count = int(frame['count'])
        objects = frame['objects'].copy()
        
        # 3. Read sequence lock end
        seq2 = frame['seq']
        if seq1 != seq2:
            continue  # Data was modified while reading, discard
            
        # 4. Check for staleness based on copied ts
        age = time.monotonic() - ts
        if age > STALE_THRESHOLD:
            continue

        for i in range(NUM_CLASSES):
            obj = objects[i]
            if obj['conf'] > 0:
                class_name = CLASS_DICT.get(obj['class_id'], str(obj['class_id']))
                color_name = COLOR_DICT.get(obj['color_id'], "Unknown")
                x = int(obj['x1'])
                y = int(obj['y1'])
                cv.drawMarker(img, (x, y), color=(0, 255, 255), markerType=cv.MARKER_CROSS, markerSize=10, thickness=1)
                # cv.drawMarker(frame, (x, y), (0, 255, 0), cv.MARKER_CROSS, markerSize=20, thickness=2)
                print(f"  {class_name} ({color_name})  conf {obj['conf']:.2f}  x {obj['x1']:.1f}  y {obj['y1']:.1f}  r {obj['r']:.1f}")
        cv.imshow("Control View", img)
        if cv.waitKey(1) == 27:  # ESC key
            break
except KeyboardInterrupt:
    print("\nInterrupted by user.")
    shm.close()

finally:
    try:
        shm.close()
    except Exception as e:
        pass
    print("Control reader closed safely.")
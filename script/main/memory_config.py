from multiprocessing import shared_memory

import numpy as np, time

#every object is processed as circle

#there would be 4 classes, there would only be 4 objects passed to shared memory (guidebook didnt say there will be multiple object with the same class)

CLASS_DICT = {0: 'square', 1: 'hexagon', 2: 'triangle', 3: 'circle'}

# Color lookup dictionary
COLOR_DICT = {
    0: "hitam",
    1: "putih",
    2: "abu-abu",
    3: "merah",
    4: "oranye",
    5: "kuning",
    6: "hijau",
    7: "biru",
    8: "ungu",
    9: "tidak diketahui"
}

# Inverse lookup for the inference script
COLOR_TO_ID = {v: k for k, v in COLOR_DICT.items()}

dtype = np.dtype([
    ('x1', np.float32),
    ('y1', np.float32),
    ('r', np.float32), #radius
    ('conf', np.float32),
    ('class_id', np.int32),
    ('color_id', np.int32)
])

NUM_CLASSES = len(CLASS_DICT)

meta_dtype = np.dtype([
    ('seq', np.uint32),   # Sequence lock counter
    ('count', np.int32),
    ('ts', np.float64),
    ('objects', dtype, NUM_CLASSES)
])


import os
import json
import cv2
import numpy as np
import shutil
import yaml
from glob import glob
from sklearn.model_selection import train_test_split
from ultralytics import YOLO
import matplotlib.pyplot as plt

# ══════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════

BLENDER_OUTPUT_DIR = "./syn_dataset/output"
YOLO_DATASET_DIR   = os.path.abspath("./syn_dataset/yolo_drone_dataset")
IMG_W, IMG_H       = 640, 640

# Must exactly match the Blender script
INSTANCE_PALETTE =[
    (1.0, 0.15, 0.15), (0.15, 1.0, 0.15), (0.15, 0.15, 1.0),
    (1.0, 1.0, 0.15),  (1.0, 0.15, 1.0),  (0.15, 1.0, 1.0),
    (1.0, 0.55, 0.15), (0.55, 0.15, 1.0), (0.15, 0.75, 1.0),
    (1.0, 0.35, 0.65), (0.65, 1.0, 0.15), (1.0, 0.80, 0.15),
]

CLASSES =[
    "blue_square", "red_square", "blue_hexagon", "red_hexagon",
    "red_triangle", "red_circle", "blue_circle", "blue_triangle"
]

# ══════════════════════════════════════════════════════════════════════
# 1. DATA CONVERSION (Instance Mask -> YOLO Bounding Boxes)
# ══════════════════════════════════════════════════════════════════════

def extract_yolo_labels():
    print("🚀 Extracting Bounding Boxes from Instance Masks...")
    
    # Create YOLO directory structure
    for split in ['train', 'val']:
        os.makedirs(os.path.join(YOLO_DATASET_DIR, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(YOLO_DATASET_DIR, 'labels', split), exist_ok=True)

    json_files = glob(os.path.join(BLENDER_OUTPUT_DIR, "annotations", "*.json"))
    dataset_records =[]

    for json_path in json_files:
        with open(json_path, 'r') as f:
            data = json.load(f)
            
        idx = data['index']
        img_name = f"{idx:06d}.png"
        
        img_path = os.path.join(BLENDER_OUTPUT_DIR, "images", img_name)
        inst_path = os.path.join(BLENDER_OUTPUT_DIR, "instances", img_name)
        
        if not os.path.exists(img_path) or not os.path.exists(inst_path):
            continue

        # Read instance mask (convert to RGB to match our palette)
        mask = cv2.imread(inst_path)
        if mask is None: continue
        mask_rgb = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)

        yolo_labels =[]

        # Iterate through the instances in the JSON metadata
        for inst in data.get('instances',[]):
            inst_id = inst['inst_id']
            class_id = inst['class_id']
            
            # Reconstruct the 8-bit color Blender used for this specific instance
            col_float = INSTANCE_PALETTE[(inst_id - 1) % len(INSTANCE_PALETTE)]
            col_8bit = np.array([int(round(c * 255)) for c in col_float])
            
            # Find pixels matching this color (with small tolerance for anti-aliasing/compression)
            color_dist = np.linalg.norm(mask_rgb - col_8bit, axis=-1)
            y_idx, x_idx = np.where(color_dist < 15)
            
            # If object is visible (at least 10 pixels)
            if len(y_idx) > 10:
                xmin, xmax = int(x_idx.min()), int(x_idx.max())
                ymin, ymax = int(y_idx.min()), int(y_idx.max())
                
                # Convert to normalized YOLO format (x_center, y_center, width, height)
                x_c = ((xmin + xmax) / 2.0) / IMG_W
                y_c = ((ymin + ymax) / 2.0) / IMG_H
                w = (xmax - xmin) / IMG_W
                h = (ymax - ymin) / IMG_H
                
                # Clip values strictly between 0 and 1
                x_c, y_c = np.clip([x_c, y_c], 0, 1)
                w, h = np.clip([w, h], 0, 1)
                
                yolo_labels.append(f"{class_id} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}")

        dataset_records.append((img_path, yolo_labels, f"{idx:06d}"))

    return dataset_records

# ══════════════════════════════════════════════════════════════════════
# 2. DATA SPLITTING & YAML GENERATION
# ══════════════════════════════════════════════════════════════════════

def prepare_yolo_dataset(records):
    print(f"🔀 Splitting {len(records)} images into Train/Val...")
    
    # 80% Train, 20% Val split
    train_recs, val_recs = train_test_split(records, test_size=0.2, random_state=42)
    
    def copy_split(split_recs, split_name):
        for img_path, labels, base_name in split_recs:
            # Copy Image
            dest_img = os.path.join(YOLO_DATASET_DIR, 'images', split_name, f"{base_name}.png")
            shutil.copy(img_path, dest_img)
            
            # Write Label TXT
            dest_txt = os.path.join(YOLO_DATASET_DIR, 'labels', split_name, f"{base_name}.txt")
            with open(dest_txt, 'w') as f:
                f.write("\n".join(labels))

    copy_split(train_recs, 'train')
    copy_split(val_recs, 'val')

    # Create dataset.yaml
    yaml_path = os.path.join(YOLO_DATASET_DIR, 'dataset.yaml')
    yaml_content = {
        'path': YOLO_DATASET_DIR,
        'train': 'images/train',
        'val': 'images/val',
        'nc': len(CLASSES),
        'names': {i: name for i, name in enumerate(CLASSES)}
    }
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_content, f, sort_keys=False)
        
    print(f"✅ YOLO dataset ready at: {YOLO_DATASET_DIR}")
    return yaml_path

# ══════════════════════════════════════════════════════════════════════
# 3. TRAINING YOLOv8
# ══════════════════════════════════════════════════════════════════════

def train_and_evaluate(yaml_path):
    print("\n🔥 Initializing YOLOv8 Nano model...")
    # Load a pretrained YOLOv8 nano model (fastest, great for synthetics)
    model = YOLO("yolov8n.pt")
    
    print("\n⚙️ Starting Training...")
    # Train the model
    # Note: Increase epochs (e.g., 50-100) if you generate more than 100 images
    results = model.train(
        data=yaml_path,
        epochs=30,           # Set to 50-100 for a larger generated dataset
        imgsz=IMG_W,
        batch=16,
        project="drone_vision",
        name="shape_detector",
        plots=True           # Generates performance plots automatically
    )
    
    print("\n📊 Validating Model...")
    metrics = model.val()
    
    print("\n==============================================")
    print("📈 PERFORMANCE METRICS")
    print("==============================================")
    print(f"mAP@50:    {metrics.box.map50:.4f}")
    print(f"mAP@50-95: {metrics.box.map:.4f}")
    print("==============================================\n")
    
    return model

# ══════════════════════════════════════════════════════════════════════
# 4. INFERENCE & VISUALIZATION
# ══════════════════════════════════════════════════════════════════════

def visualize_predictions(model):
    print("🖼️ Running inference on some validation images...")
    
    # Grab 4 random validation images
    val_images = glob(os.path.join(YOLO_DATASET_DIR, 'images', 'val', '*.png'))[:4]
    
    if not val_images:
        print("No validation images found for visualization.")
        return

    # Run batched prediction
    results = model.predict(source=val_images, save=False, conf=0.25)
    
    # Plot using matplotlib
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    axes = axes.flatten()
    
    for i, r in enumerate(results):
        # r.plot() returns BGR image array with boxes overlaid
        im_bgr = r.plot()
        im_rgb = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2RGB)
        
        axes[i].imshow(im_rgb)
        axes[i].axis('off')
        axes[i].set_title(f"Validation Image {i+1}")
        
    plt.tight_layout()
    plt.savefig("inference_results.png")
    print("✅ Visualizations saved to 'inference_results.png'")

# ══════════════════════════════════════════════════════════════════════
# MAIN EXECUTION
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # 1. Extract tight bounding boxes from instance mask colors
    records = extract_yolo_labels()
    
    if not records:
        print(f"❌ No data found! Did you generate data to {BLENDER_OUTPUT_DIR} first?")
        exit(1)
        
    # 2. Format into strict YOLO folder structure + YAML
    yaml_file = prepare_yolo_dataset(records)
    
    # 3. Train the neural network
    trained_model = train_and_evaluate(yaml_file)
    
    # 4. Verify visual performance
    visualize_predictions(trained_model)
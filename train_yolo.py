"""
YOLOv8n training script for swiftlet detection.
Trains on the MarkasWalet Roboflow dataset, then exports to ONNX for RPi5 deployment.

Usage:
    python train_yolo.py
"""

from ultralytics import YOLO
import yaml, os, shutil

DATASET_ROOT = '/Users/raulilmarajasa/Desktop/Personal/Startup/Techiro/MarkasWalet/IoT Counter Burung Walet/MarkasWalet Counting.v2i.yolo26'
OUTPUT_DIR   = os.path.join(os.path.dirname(__file__), 'runs/train')
DATA_YAML    = os.path.join(os.path.dirname(__file__), 'swiftlet_data.yaml')

# Write data.yaml with absolute paths (ultralytics needs abs paths)
data = {
    'train': os.path.join(DATASET_ROOT, 'train', 'images'),
    'val':   os.path.join(DATASET_ROOT, 'valid', 'images'),
    'test':  os.path.join(DATASET_ROOT, 'test',  'images'),
    'nc':    1,
    'names': ['swiftlets'],
}
with open(DATA_YAML, 'w') as f:
    yaml.dump(data, f)
print(f'data.yaml written to {DATA_YAML}')

# Load YOLOv8n (nano — smallest, fastest on RPi5 CPU)
model = YOLO('yolov8n.pt')

# Train
results = model.train(
    data=DATA_YAML,
    epochs=100,
    imgsz=320,           # 320×320 — fast inference on RPi5 (~50-80ms/frame)
    batch=32,            # larger batch fits well on MPS
    device='mps',        # Apple Silicon GPU (3-5× faster than CPU)
    patience=20,         # early stop if no improvement for 20 epochs
    project=OUTPUT_DIR,
    name='swiftlet_yolov8n',
    exist_ok=True,
    # Augmentation tuned for indoor swiftlet footage
    hsv_h=0.005,         # minimal hue shift (indoor lighting consistent)
    hsv_s=0.3,
    hsv_v=0.4,
    degrees=10,          # birds fly at angles
    translate=0.1,
    scale=0.4,
    fliplr=0.5,
    flipud=0.1,
    mosaic=0.8,
    mixup=0.1,
    copy_paste=0.0,
)

print('\n=== Training complete ===')
best_weights = os.path.join(OUTPUT_DIR, 'swiftlet_yolov8n', 'weights', 'best.pt')
print(f'Best weights: {best_weights}')

# Export to ONNX for RPi5 (no ultralytics needed at inference — just cv2.dnn)
print('\n=== Exporting to ONNX ===')
best_model = YOLO(best_weights)
best_model.export(
    format='onnx',
    imgsz=320,
    simplify=True,       # onnx-simplifier removes redundant ops
    opset=12,            # OpenCV DNN supports up to opset 12 well
    dynamic=False,       # static batch=1 for RPi5
)

onnx_path = best_weights.replace('.pt', '.onnx')
deploy_path = os.path.join(os.path.dirname(__file__), 'markaswalet-stream', 'swiftlet_yolov8n.onnx')
shutil.copy(onnx_path, deploy_path)
print(f'ONNX model copied to: {deploy_path}')
print('\nDone! Run the stream with use_yolo=True to use the trained model.')

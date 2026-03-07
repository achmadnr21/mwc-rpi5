# Run this script on a development machine (NOT on RPi5) to export YOLO11n to OpenVINO FP16.
# Then copy the resulting 'yolo11n_openvino_model/' folder to the RPi5 under 'models/'.
#
# Usage:
#   python setup_model.py
#
# Output:
#   models/yolo11n_openvino_model/  ← copy this entire folder to the RPi5

from ultralytics import YOLO
import os

OUTPUT_DIR = "models"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Downloading YOLO11n (if not already cached)...")
model = YOLO("yolo11n.pt")

print("Exporting to OpenVINO FP16 (imgsz=640)...")
export_path = model.export(
    format="openvino",
    half=True,       # FP16 for best performance on RPi5 CPU
    imgsz=640,
)

print(f"\nExport complete: {export_path}")
print(f"\nNext steps:")
print(f"  1. Copy the exported folder to your RPi5:")
print(f"     scp -r {export_path} pi@<rpi5-ip>:/usr/local/bin/markaswalet-stream/models/")
print(f"  2. Verify config.json has: \"model_path\": \"models/yolo11n_openvino_model\"")

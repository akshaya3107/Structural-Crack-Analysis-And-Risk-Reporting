"""Train YOLOv8n-seg on ``unified_dataset``.

Results land in ``runs/segment/runs/road_defect_seg<N>/`` (Ultralytics adds the
``segment/`` prefix and a run number). Point MODEL_PATH at the new
``weights/best.pt`` to use it in the app.
"""
from dotenv import load_dotenv
import torch
import os

load_dotenv()

def main():
    from ultralytics import YOLO

    print(f"CUDA available : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU            : {torch.cuda.get_device_name(0)}")
        print(f"VRAM           : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("No GPU — training on CPU (will be slow)")

    model = YOLO("yolov8n-seg.pt")  # nano = fastest

    model.train(
        data="./unified_dataset/data.yaml",
        epochs=50,
        imgsz=416,
        batch=16,
        name="road_defect_seg",
        project="runs",
        seed=42,
        deterministic=False,
        patience=15,
        workers=2,          
        cache=True,
        device=0 if torch.cuda.is_available() else "cpu",
        hsv_h=0.015,
        hsv_s=0.5,
        hsv_v=0.3,
        fliplr=0.5,
        mosaic=0.5,
        close_mosaic=10,
        amp=torch.cuda.is_available(),  # only use AMP if GPU available
    )

if __name__ == "__main__":
    main()
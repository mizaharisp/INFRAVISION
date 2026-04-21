from ultralytics import YOLO

# 1. Load your current model
model = YOLO("basemodel.pt")

# 2. Export it to NCNN format
# This creates a folder containing the optimized model files
model.export(format="ncnn", imgsz=320)
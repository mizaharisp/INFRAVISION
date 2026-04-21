from ultralytics import YOLO

# This will download the model automatically from Hugging Face
# Repo: cazzz307/Pothole-Finetuned-YoloV8
model = YOLO("basemodel.pt")

# To test if it works on a single image
results = model.predict("image.png", save=True)

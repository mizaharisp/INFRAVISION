import os
import shutil
import random

# =========================
# CONFIGURATION
# =========================

# CHANGE THIS ONLY IF YOUR PATH IS DIFFERENT
SOURCE_IMAGES_DIR = r"C:\Users\USER\Downloads\archive\images\train\images"
SOURCE_LABELS_DIR = r"C:\Users\USER\Downloads\archive\images\train\labels"

# Where the sampled dataset should be created
DEST_BASE_DIR = r"C:\Users\USER\OneDrive\Desktop\infravision\data\sample_dataset"

DEST_IMAGES_DIR = os.path.join(DEST_BASE_DIR, "images")
DEST_LABELS_DIR = os.path.join(DEST_BASE_DIR, "labels")

# Number of samples you want
NUM_SAMPLES = 100   # you can change this

# =========================
# CREATE DESTINATION FOLDERS
# =========================

os.makedirs(DEST_IMAGES_DIR, exist_ok=True)
os.makedirs(DEST_LABELS_DIR, exist_ok=True)

# =========================
# GET IMAGE LIST
# =========================

images = [
    img for img in os.listdir(SOURCE_IMAGES_DIR)
    if img.lower().endswith((".jpg", ".jpeg", ".png"))
]

if len(images) == 0:
    raise Exception("No images found in source directory!")

# =========================
# RANDOMLY SAMPLE IMAGES
# =========================

sampled_images = random.sample(images, min(NUM_SAMPLES, len(images)))

# =========================
# COPY FILES
# =========================

for img_name in sampled_images:
    src_img_path = os.path.join(SOURCE_IMAGES_DIR, img_name)
    dst_img_path = os.path.join(DEST_IMAGES_DIR, img_name)

    label_name = os.path.splitext(img_name)[0] + ".txt"
    src_label_path = os.path.join(SOURCE_LABELS_DIR, label_name)
    dst_label_path = os.path.join(DEST_LABELS_DIR, label_name)

    shutil.copy(src_img_path, dst_img_path)

    if os.path.exists(src_label_path):
        shutil.copy(src_label_path, dst_label_path)
    else:
        print(f"⚠️ Label missing for {img_name}")

print("✅ Sample dataset created successfully!")
print(f"Images copied: {len(sampled_images)}")
print(f"Saved at: {DEST_BASE_DIR}")

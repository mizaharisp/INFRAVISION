import os
import cv2
import random

# ========================
# PATH CONFIGURATION
# ========================
BASE_DIR = "data/sample_dataset"

IMAGES_DIR = os.path.join(BASE_DIR, "images")
LABELS_DIR = os.path.join(BASE_DIR, "labels")
OUTPUT_DIR = os.path.join(BASE_DIR, "visualized_outputs")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ========================
# CLASS NAMES (edit if needed)
# ========================
# Example: 0 = pothole
CLASS_NAMES = {
    0: "Pothole"
}

# ========================
# VISUALIZATION FUNCTION
# ========================
def visualize_samples(num_samples=5):
    images = [img for img in os.listdir(IMAGES_DIR) if img.endswith((".jpg", ".png"))]

    if not images:
        print("❌ No images found!")
        return

    sampled_images = random.sample(images, min(num_samples, len(images)))

    for img_name in sampled_images:
        img_path = os.path.join(IMAGES_DIR, img_name)
        label_path = os.path.join(LABELS_DIR, os.path.splitext(img_name)[0] + ".txt")

        image = cv2.imread(img_path)
        if image is None:
            print(f"⚠️ Could not load {img_name}")
            continue

        h, w, _ = image.shape

        # Read label file
        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                lines = f.readlines()

            for line in lines:
                parts = line.strip().split()
                class_id = int(parts[0])
                x_center, y_center, bw, bh = map(float, parts[1:])

                # Convert YOLO to pixel coordinates
                x_center *= w
                y_center *= h
                bw *= w
                bh *= h

                x1 = int(x_center - bw / 2)
                y1 = int(y_center - bh / 2)
                x2 = int(x_center + bw / 2)
                y2 = int(y_center + bh / 2)

                label = CLASS_NAMES.get(class_id, "Object")

                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 0), 2)
                cv2.putText(
                    image,
                    label,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 0),
                    2,
                )

        # Save output
        output_path = os.path.join(OUTPUT_DIR, img_name)
        cv2.imwrite(output_path, image)

        print(f"✅ Visualized: {img_name}")

    print("\n🎉 Step 7.2 completed successfully!")
    print(f"📁 Outputs saved at: {OUTPUT_DIR}")

# ========================
# RUN
# ========================
if __name__ == "__main__":
    visualize_samples(num_samples=5)

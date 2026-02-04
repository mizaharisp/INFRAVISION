import cv2
import os

# Input and output paths
input_image_path = "data/road_gray.jpg"
output_folder = "data/processed_images"
output_image_path = os.path.join(output_folder, "road_final.jpg")

# Read grayscale image
image = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)

if image is None:
    print("Error: Image not found!")
    exit()

# Resize image
resized = cv2.resize(image, (640, 480))

# Remove noise using Gaussian Blur
denoised = cv2.GaussianBlur(resized, (5, 5), 0)

# Save processed image
cv2.imwrite(output_image_path, denoised)

print("Advanced preprocessing completed successfully")

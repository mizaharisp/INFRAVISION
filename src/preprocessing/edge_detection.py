import cv2
import os

# Paths
input_image_path = "data/processed_images/road_final.jpg"
output_folder = "data/processed_images/edges"
output_image_path = os.path.join(output_folder, "road_edges.jpg")

# Read image
image = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)

if image is None:
    print("Error: Processed image not found!")
    exit()

# Apply Canny Edge Detection
edges = cv2.Canny(image, threshold1=100, threshold2=200)

# Save edge-detected image
cv2.imwrite(output_image_path, edges)

print("Edge detection completed successfully")

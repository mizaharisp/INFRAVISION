import os
import cv2

def save_image(image, output_path, filename):
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    full_path = os.path.join(output_path, filename)
    cv2.imwrite(full_path, image)

    return full_path

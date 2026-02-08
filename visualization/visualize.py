import cv2
from draw_boxes import draw_boxes
from save_output import save_image

def visualize(image, detections, save=False):
    output = draw_boxes(image, detections)

    cv2.imshow("Detection Output", output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    if save:
        save_image(output, "outputs", "result.jpg")

    return output

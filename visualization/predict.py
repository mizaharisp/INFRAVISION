import cv2
import numpy as np
from tensorflow.keras.models import load_model

def predict_pothole(image_path, model_path):
    model = load_model(model_path)

    image = cv2.imread(image_path)
    original = image.copy()

    image = cv2.resize(image, (224, 224))
    image = image / 255.0
    image = np.expand_dims(image, axis=0)

    prediction = model.predict(image)[0][0]

    if prediction >= 0.5:
        label = "Pothole Detected"
        color = (0, 0, 255)
    else:
        label = "No Pothole"
        color = (0, 255, 0)

    cv2.putText(
        original,
        label,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        color,
        2
    )

    cv2.imshow("Prediction Result", original)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return label

from ultralytics import YOLO
import cv2
import os

# 1. Load the NCNN folder you just exported
# Replace this with the exact name of the folder created on your laptop
model_path = "basemodel_ncnn_model" 

if not os.path.exists(model_path):
    print(f"ERROR: Folder '{model_path}' not found! Did you export it yet?")
    exit()

model = YOLO(model_path)

def run_wsl_test():
    # Attempt to open the webcam
    # Note: This usually fails in WSL unless you used 'usbipd'
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Webcam not detected (Normal for WSL). Switching to Image Test...")
        # Change 'image.png' to the name of your pothole image
        img_path = "image2.png" 
        
        if os.path.exists(img_path):
            results = model.predict(source=img_path, save=True, imgsz=320, conf=0.25)
            print(f"Success! Result saved to: {results[0].save_dir}")
        else:
            print(f"Error: Could not find {img_path} to test.")
    else:
        print("Webcam detected! Press 'q' to quit.")
        while True:
            ret, frame = cap.read()
            if not ret: break
            
            # Using imgsz=320 to simulate the Raspberry Pi speed
            results = model.predict(frame, imgsz=320, conf=0.25, verbose=False)
            
            # Draw results
            annotated_frame = results[0].plot()
            
            # Display (Requires GWSL or Windows 11 WSLg)
            cv2.imshow("WSL Pothole Test", annotated_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    run_wsl_test()
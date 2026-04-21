from ultralytics import YOLO
import cv2
import time
import subprocess
import sys
import threading

# Load model
model_path = "basemodel_ncnn_model"
model = YOLO(model_path, task="detect")

# --- INITIALIZE GLOBAL STATES ---
tts_running = False  # FIXED: Define this here so the function can see it
alert_active = False # For the visual overlay

# --- FINE-TUNED AUDIO SYSTEM ---
def speak_async(text):
    global tts_running
    if tts_running:
        return

    tts_running = True
    
    # Python script string to run in a separate process
    tts_code = f"""
import pyttsx3
engine = pyttsx3.init()
engine.setProperty('volume', 1.0)
engine.setProperty('rate', 145)
voices = engine.getProperty('voices')
if len(voices) > 1:
    engine.setProperty('voice', voices[1].id) # voices[1] is usually clearer
engine.say('{text}')
engine.runAndWait()
"""

    def run():
        global tts_running
        # subprocess.run waits for the speech to finish before moving to the next line
        subprocess.run([sys.executable, "-c", tts_code])
        tts_running = False

    threading.Thread(target=run, daemon=True).start()

# --- CAMERA SETUP ---
cap = cv2.VideoCapture(0)
last_alert = 0
ALERT_INTERVAL = 3.5 
frame_count = 0
WARMUP_FRAMES = 10
detection_streak = 0
REQUIRED_STREAK = 2

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    # Resize for faster inference
    frame = cv2.resize(frame, (640, 480)) # Slightly larger for better visibility

    # Run YOLO
    results = model.predict(frame, imgsz=320, conf=0.25, verbose=False)
    annotated_frame = results[0].plot()

    pothole_detected = False
    if results[0].boxes is not None and len(results[0].boxes) > 0:
        for box in results[0].boxes:
            if float(box.conf[0]) > 0.6:
                pothole_detected = True
                break

    # Detection streak logic
    if pothole_detected:
        detection_streak += 1
    else:
        detection_streak = 0

    current_time = time.time()

    # --- ALERT LOGIC ---
    if (
        detection_streak >= REQUIRED_STREAK and
        frame_count > WARMUP_FRAMES and
        (current_time - last_alert > ALERT_INTERVAL)
    ):
        speak_async("Warning! Pothole Detected")
        last_alert = current_time
        alert_active = True
    
    # Turn off the visual red box after 2 seconds
    if current_time - last_alert > 2.0:
        alert_active = False

    # --- VISUAL FEEDBACK (Optional Overlay) ---
    if alert_active:
        # Draw a thick red border around the frame when a pothole is confirmed
        cv2.rectangle(annotated_frame, (0,0), (640,480), (0, 0, 255), 20)
        cv2.putText(annotated_frame, "POTHOLE DETECTED!", (150, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    cv2.imshow("Pothole Detection System", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
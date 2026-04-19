import requests
import kornia_rs as K
import torch
import cv2
import time
from ultralytics import YOLO

# --- CONFIGURATION ---
STREAM_URL = "http://172.28.251.234:8080/shot.jpg"
MODEL_NAME = "yolo11n.pt"

print(f"Loading {MODEL_NAME} to Apple Neural Engine (MPS)...")
model = YOLO(MODEL_NAME)
model.to("mps")

decoder = K.ImageDecoder()

# PRO-UPGRADE: Use a Session to keep the TCP Wi-Fi connection alive
# This prevents the Mac from having to "handshake" with the phone every single frame
session = requests.Session()

print(f"Connecting to Samsung A51 at {STREAM_URL}...")
print("Press 'q' in the video window to quit.")

# Variables for FPS calculation
prev_time = time.time()

while True:
    try:
        # 1. PULL THE FRAME (Lightning fast over active session)
        response = session.get(STREAM_URL, timeout=1.0)
        
        # 2. DECODE (Rust Zero-Copy)
        decoded_dlpack = decoder.decode(response.content)
        img_tensor = torch.from_dlpack(decoded_dlpack)
        
        # 3. PREPARE FOR OPENCV
        img_rgb = img_tensor.numpy()
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        
        # 4. RUN AI (Unrestricted classes)
        results = model(img_bgr, verbose=False)
        
        # 5. DRAW VISUALS
        annotated_frame = results[0].plot()
        
        # --- CALCULATE REAL SYSTEM FPS ---
        current_time = time.time()
        fps = 1 / (current_time - prev_time)
        prev_time = current_time
        
        # Draw the FPS counter in the top left corner
        cv2.putText(annotated_frame, f"System FPS: {int(fps)}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # 6. DISPLAY
        cv2.imshow("Zenity ROV - AI Vision Engine", annotated_frame)
        
    except requests.exceptions.RequestException as e:
        print(f"Network blip or waiting for camera: {e}")
        time.sleep(0.5)
        
    # Clean Exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Shutting down AI Engine...")
        break

# Release resources
session.close()
cv2.destroyAllWindows()
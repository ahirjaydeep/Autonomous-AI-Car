import requests
import kornia_rs as K
import torch
import cv2
import numpy as np
from ultralytics import YOLO
import time

STREAM_URL = "http://172.28.251.234:8080/shot.jpg"

print("Loading YOLO11 Nano...")
model = YOLO("yolo11n.pt")
model.to("mps") 

decoder = K.ImageDecoder()
print(f"Connecting to Samsung A51 at {STREAM_URL}...")

while True:
    start_time = time.time()
    
    try:
        response = requests.get(STREAM_URL, timeout=1)
        jpg_bytes = response.content

        decoded_dlpack = decoder.decode(jpg_bytes)
        img_tensor = torch.from_dlpack(decoded_dlpack)
        
        img_rgb = img_tensor.numpy()
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        
        results = model(img_bgr, verbose=False)
        
        latency = (time.time() - start_time) * 1000
        print(f"Real-Time Latency: {latency:.1f} ms")
        
        annotated_frame = results[0].plot()
        cv2.imshow("Zenity ROV - AI Vision", annotated_frame)
        
    except requests.exceptions.RequestException:
        print("Waiting for camera stream...")
        time.sleep(0.5)
        
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()